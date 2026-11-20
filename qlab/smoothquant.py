"""SmoothQuant (Xiao et al., 2022): move activation outliers into the weights, then W8A8.

Per-token INT8 activations fail when a few channels are 100x larger than the rest: every
token's scale is set by those channels. SmoothQuant divides each input channel ``j`` by

    s_j = max|X_j|^alpha / max|W_j|^(1 - alpha)

and multiplies the matching weight column by ``s_j``. The product is unchanged; activations
get flatter and weights a little rougher. The division is folded into the RMSNorm that
produces the input, so it costs nothing at runtime. Only linears fed by a norm can be
smoothed for free: q/k/v (``input_layernorm``) and gate/up (``post_attention_layernorm``).
Weights are then quantized per channel; activations are quantized at evaluation time by
:mod:`qlab.actquant`.

Activation maxima are collected from the unmodified block on the calibration set, as in the
reference implementation.
"""

from __future__ import annotations

import torch
from torch import nn

from .awq import ScaleGroup, apply_scale
from .formats import QuantSpec, fake_quant
from .model import block_linears
from .sequential import BlockInputs, _to

GROUPS = (
    ScaleGroup(
        "input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), None
    ),
    ScaleGroup("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj"), None),
)


@torch.no_grad()
def smooth_scale(
    act_absmax: torch.Tensor, weights: list[torch.Tensor], alpha: float
) -> torch.Tensor:
    w_absmax = torch.cat([w.abs().float() for w in weights], dim=0).amax(dim=0).clamp(min=1e-5)
    s = act_absmax.float().clamp(min=1e-5).pow(alpha) / w_absmax.pow(1 - alpha)
    return s.clamp(min=1e-5)


def make_block_hook(spec: QuantSpec, device, alpha: float = 0.5):
    """An ``after_block`` hook: smooth q/k/v and gate/up, then RTN-quantize every linear."""

    def hook(i: int, block: nn.Module, inputs: BlockInputs) -> None:
        lins = block_linears(block)
        amax: dict[str, torch.Tensor] = {}

        def record(name):
            def h(_m, a):
                v = a[0].detach().abs().reshape(-1, a[0].shape[-1]).amax(0).float()
                amax[name] = torch.maximum(amax[name], v) if name in amax else v

            return h

        firsts = [g.linears[0] for g in GROUPS if g.linears[0] in lins]
        handles = [lins[n].register_forward_pre_hook(record(n)) for n in firsts]
        try:
            for h, kw in zip(inputs.hidden, inputs.kwargs, strict=True):
                block(h.to(device), **_to(kw, device))
        finally:
            for hd in handles:
                hd.remove()
        for g in GROUPS:
            if not all(n in lins for n in g.linears):
                continue
            s = smooth_scale(amax[g.linears[0]], [lins[n].weight.data for n in g.linears], alpha)
            apply_scale(block, g, s)
        for lin in lins.values():
            lin.weight.data = fake_quant(lin.weight.data, spec)

    return hook
