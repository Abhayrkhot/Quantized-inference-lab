"""AWQ: activation-aware weight quantization (Lin et al., 2023), written from the paper and
the reference procedure.

A few input channels see much larger activations than the rest, so the weights in those
columns matter most. AWQ scales those columns up before quantization, which makes their
relative rounding error smaller. It divides the scale back out of the previous operation,
so the float model is unchanged.

Per decoder block:

1. Record the inputs of every linear on the calibration set, and the block's outputs from
   the unmodified block. Those outputs feed the next block.
2. For each scaling group (layers sharing an input, plus the op before them):
   ``s = mean|x|^alpha``, normalized by ``sqrt(max(s) * min(s))``, for ``alpha`` on a grid in
   [0, 1). Pick the ``alpha`` whose fake-quantized ``W * s`` (with ``x / s``) gives the
   smallest MSE on the group's output. Then fold ``s`` in: the previous op's output is
   divided by ``s`` and the weights' input columns are multiplied by ``s``.
3. Clipping search (not for q/k, whose errors are amplified by softmax): per output channel
   and group, shrink the clip range over ``(1 - k/20)``, k = 0..9. Keep the value with the
   smallest error of ``W x`` on a token sample.
4. RTN-quantize every weight with its clip.

Scaling groups for Llama/Qwen blocks:
* ``input_layernorm`` -> q, k, v (output measured through the whole attention module);
* ``v_proj`` -> o_proj, only when the shapes match (no grouped-query attention);
* ``post_attention_layernorm`` -> gate, up (output measured through the MLP);
* ``up_proj`` -> down_proj.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .formats import QuantizedWeight, QuantSpec, fake_quant
from .model import block_linears
from .sequential import BlockInputs, _to


@dataclass
class ScaleGroup:
    prev: str  # module whose output feeds the group
    linears: tuple[str, ...]
    inspect: str | None  # module whose output the loss is measured on (None = the linears)


GROUPS = (
    ScaleGroup(
        "input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), "self_attn"
    ),
    ScaleGroup("self_attn.v_proj", ("self_attn.o_proj",), None),
    ScaleGroup("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj"), "mlp"),
    ScaleGroup("mlp.up_proj", ("mlp.down_proj",), None),
)
NO_CLIP = ("self_attn.q_proj", "self_attn.k_proj")


def _get(block: nn.Module, name: str) -> nn.Module:
    return block.get_submodule(name)


def _forward(module: nn.Module, x: torch.Tensor, kwargs: dict, chunk: int) -> torch.Tensor:
    outs = []
    for s in range(0, x.shape[0], chunk):
        y = module(x[s : s + chunk], **kwargs) if kwargs else module(x[s : s + chunk])
        outs.append(y[0] if isinstance(y, tuple) else y)
    return torch.cat(outs)


@torch.no_grad()
def search_scale(
    block: nn.Module,
    group: ScaleGroup,
    x: torch.Tensor,
    kwargs: dict,
    spec: QuantSpec,
    n_grid: int = 20,
    chunk: int = 16,
) -> tuple[torch.Tensor, float]:
    """Best per-input-channel scale for ``group`` and the alpha that produced it."""
    lins = [_get(block, n) for n in group.linears]
    inspect = _get(block, group.inspect) if group.inspect else None
    attn_kwargs = kwargs if group.inspect == "self_attn" else {}

    def out_of(xx):
        if inspect is not None:
            return _forward(inspect, xx, attn_kwargs, chunk)
        return torch.cat([_forward(lin, xx, {}, chunk) for lin in lins], dim=-1)

    target = out_of(x).float()
    x_mean = x.abs().reshape(-1, x.shape[-1]).float().mean(0)
    originals = [lin.weight.data.clone() for lin in lins]
    best = (float("inf"), None, 0.0)
    for k in range(n_grid):
        alpha = k / n_grid
        s = x_mean.pow(alpha).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()
        for lin, w in zip(lins, originals, strict=True):
            lin.weight.data = (fake_quant(w.float() * s, spec) / s).to(w.dtype)
        err = float((out_of(x).float() - target).pow(2).mean())
        if err < best[0]:
            best = (err, s.clone(), alpha)
    for lin, w in zip(lins, originals, strict=True):
        lin.weight.data = w
    return best[1], best[2]


@torch.no_grad()
def apply_scale(block: nn.Module, group: ScaleGroup, s: torch.Tensor) -> None:
    """Fold ``s``: previous op's output / s, linears' input columns * s.

    The float model is unchanged.
    """
    prev = _get(block, group.prev)
    s = s.to(prev.weight.device)
    if isinstance(prev, nn.Linear):  # divide its output rows
        prev.weight.data = (prev.weight.data.float() / s.view(-1, 1)).to(prev.weight.dtype)
        if prev.bias is not None:
            prev.bias.data = (prev.bias.data.float() / s).to(prev.bias.dtype)
    else:  # RMSNorm-like: per-channel weight
        prev.weight.data = (prev.weight.data.float() / s).to(prev.weight.dtype)
    for n in group.linears:
        lin = _get(block, n)
        lin.weight.data = (lin.weight.data.float() * s.view(1, -1)).to(lin.weight.dtype)


@torch.no_grad()
def search_clip(
    w: torch.Tensor,
    x: torch.Tensor,
    spec: QuantSpec,
    n_grid: int = 20,
    max_shrink: float = 0.5,
    n_tokens: int = 512,
    budget: int = 2**25,
) -> torch.Tensor:
    """Per (output channel, group) clip value ``[out, n_groups, 1]``.

    Chosen to minimize ``||W x - Q(clip(W)) x||`` on a sample of ``n_tokens`` inputs. Each
    group's partial product is a batched matmul over groups, so only ``[out, tokens, groups]``
    is held, never ``[out, tokens, in]``.
    """
    out_f, in_f = w.shape
    gs = spec.group_size if spec.granularity == "group" else in_f
    ng = in_f // gs
    xs = x.reshape(-1, in_f)
    step = max(1, xs.shape[0] // n_tokens)
    xs = xs[::step][:n_tokens].float().view(-1, ng, gs)  # [T, ng, gs]
    wg = w.float().view(out_f, ng, gs)
    oc = max(1, min(out_f, budget // (xs.shape[0] * ng)))
    best_all = []
    for o in range(0, out_f, oc):
        wb = wg[o : o + oc]  # [ob, ng, gs]
        org_max = wb.abs().amax(dim=-1, keepdim=True)  # [ob, ng, 1]
        target = torch.einsum("tgk,ogk->otg", xs, wb)  # [ob, T, ng]
        best_err = torch.full((wb.shape[0], ng), float("inf"), device=w.device)
        best_max = org_max.clone()
        for k in range(int(max_shrink * n_grid)):
            max_val = org_max * (1 - k / n_grid)
            cw = torch.maximum(torch.minimum(wb, max_val), -max_val)
            qw = fake_quant(cw.reshape(-1, in_f), spec).view_as(cw)
            err = (torch.einsum("tgk,ogk->otg", xs, qw) - target).pow(2).mean(dim=1)  # [ob, ng]
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_max = torch.where(better.unsqueeze(-1), max_val, best_max)
        best_all.append(best_max)
    return torch.cat(best_all)


def make_block_hook(
    spec: QuantSpec, device, n_grid: int = 20, clip: bool = True, log=None, results=None
):
    """An ``after_block`` hook for :func:`qlab.sequential.run_blocks` (AWQ one block)."""

    def hook(i: int, block: nn.Module, inputs: BlockInputs) -> None:
        lins = block_linears(block)
        feats: dict[str, list[torch.Tensor]] = {n: [] for n in lins}
        handles = [
            lins[n].register_forward_pre_hook(lambda _m, a, n=n: feats[n].append(a[0].detach()))
            for n in lins
        ]
        try:
            for h, kw in zip(inputs.hidden, inputs.kwargs, strict=True):
                block(h.to(device), **_to(kw, device))
        finally:
            for hd in handles:
                hd.remove()
        # q/k/v (and gate/up) read the same tensor: concatenate each distinct input once.
        x: dict[str, torch.Tensor] = {}
        by_id: dict[int, torch.Tensor] = {}
        for n, v in feats.items():
            key = id(v[0])
            if key not in by_id:
                by_id[key] = torch.cat(v)
            x[n] = by_id[key]
        feats.clear()
        by_id.clear()
        kwargs = _to(inputs.kwargs[0], device)
        chunk = inputs.hidden[0].shape[0]
        for g in GROUPS:
            if not all(n in lins for n in g.linears):
                continue
            prev = _get(block, g.prev)
            if (
                isinstance(prev, nn.Linear)
                and prev.weight.shape != lins[g.linears[0]].weight.t().shape
            ):
                continue  # v_proj -> o_proj under grouped-query attention: shapes differ
            xin = x[g.linears[0]]
            s, alpha = search_scale(block, g, xin, kwargs, spec, n_grid, chunk)
            apply_scale(block, g, s)
            scaled = (xin.float() / s).to(xin.dtype)  # inputs of the scaled linears shrink by s
            for n in g.linears:
                x[n] = scaled
            del xin
            if log is not None:
                log.append((f"layers.{i}.{g.prev}", alpha))
        for n, lin in lins.items():
            w = lin.weight.data
            if clip and n not in NO_CLIP:
                max_val = search_clip(w, x[n], spec)
                ng = max_val.shape[1]
                wg = w.float().view(w.shape[0], ng, -1)
                w = torch.maximum(torch.minimum(wg, max_val), -max_val).view_as(w).to(w.dtype)
            qw = QuantizedWeight.from_float(w, spec)
            lin.weight.data = qw.dequantize(w.dtype)
            if results is not None:
                results[f"layers.{i}.{n}"] = qw
        del x

    return hook
