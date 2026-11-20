"""GPTQ (Frantar et al., 2022), written from the paper and the reference algorithm.

For one linear layer with calibration inputs X, GPTQ quantizes the weight one input column
at a time. It spreads each column's rounding error over the columns not yet quantized,
weighted by the inverse Hessian ``H = 2 X X^T`` of the layer-wise loss ``||W X - Q X||^2``:

1. Accumulate ``H`` from the layer's inputs over the calibration set.
2. Dampen: add ``percdamp * mean(diag H)`` to the diagonal. Columns whose inputs are always
   zero ("dead") get ``H_jj = 1`` and zero weight.
3. Take the upper Cholesky factor of ``H^-1``; row ``i`` of it holds the update coefficients
   for column ``i``.
4. In blocks of ``blocksize`` columns: quantize column ``i``, set
   ``err = (w_i - q_i) / U_ii`` and update the remaining columns in the block by
   ``err * U[i, i+1:]``. At the end of the block, apply the block's accumulated error to every
   later column in one matmul (the "lazy batch" update).

Options:
* group-wise scales, computed when the quantizer reaches each group's first column (from the
  already error-updated weights);
* ``act_order``: process columns by descending ``H_jj``. Groups are then formed in that order,
  and ``g_idx`` maps each original column to its group.

Across the model, blocks are processed in order on the outputs of the already-quantized
blocks. Inside a block, linears are quantized in execution order ("true sequential"), so
``o_proj`` is calibrated on the outputs of the quantized ``q/k/v`` projections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .formats import QuantSpec, compute_qparams
from .model import block_linears
from .sequential import BlockInputs, block_forward

# Linears that are quantized together because they read the same input, in execution order.
SEQUENTIAL_GROUPS = (
    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    ("self_attn.o_proj",),
    ("mlp.gate_proj", "mlp.up_proj"),
    ("mlp.down_proj",),
)


@dataclass
class GPTQResult:
    """Integer form of one quantized layer (what the checkpoint exporter writes)."""

    codes: torch.Tensor  # uint8 [out, in], original column order
    scale: torch.Tensor  # float32 [out, n_groups]
    zero: torch.Tensor  # float32 [out, n_groups]
    g_idx: torch.Tensor  # int32 [in]: group of each original column
    spec: QuantSpec

    def dequantize(self, dtype=torch.float32) -> torch.Tensor:
        g = self.g_idx.long()
        return ((self.codes.float() - self.zero[:, g]) * self.scale[:, g]).to(dtype)


class HessianAccumulator:
    """Running ``H = 2/n * sum x x^T`` over all calibration tokens, in fp32."""

    def __init__(self, columns: int, device):
        self.H = torch.zeros(columns, columns, device=device, dtype=torch.float32)
        self.n = 0

    @torch.no_grad()
    def add(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        n_new = x.shape[0]
        self.H *= self.n / (self.n + n_new)
        self.n += n_new
        x = x * math.sqrt(2 / self.n)
        self.H.addmm_(x.t(), x)


def _params_for(w_cols: torch.Tensor, spec: QuantSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row (scale, zero) for a slice of columns, as ``[rows]`` tensors."""
    if spec.granularity == "tensor":
        s, z = compute_qparams(w_cols, QuantSpec(spec.bits, "tensor", symmetric=spec.symmetric))
        return s.expand(w_cols.shape[0], 1)[:, 0], z.expand(w_cols.shape[0], 1)[:, 0]
    s, z = compute_qparams(w_cols, QuantSpec(spec.bits, "channel", symmetric=spec.symmetric))
    return s[:, 0], z[:, 0]


@torch.no_grad()
def gptq_quantize(
    weight: torch.Tensor,
    H: torch.Tensor,
    spec: QuantSpec,
    blocksize: int = 128,
    percdamp: float = 0.01,
    act_order: bool = False,
) -> tuple[torch.Tensor, GPTQResult, float]:
    """Quantize ``weight [out, in]`` given its Hessian.

    Returns (dequantized weight, integer result, loss). ``loss`` is GPTQ's own estimate of the
    layer's squared output error, ``sum (w - q)^2 / U_ii^2 / 2``.
    """
    W = weight.detach().float().clone()
    rows, cols = W.shape
    H = H.clone()
    group = spec.group_size if spec.granularity == "group" else cols
    if spec.granularity == "group":
        spec.n_groups(cols)  # validates divisibility
        blocksize = max(blocksize - blocksize % group, group) if group < blocksize else group

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    perm = torch.argsort(torch.diag(H), descending=True) if act_order else None
    if perm is not None:
        W, H = W[:, perm], H[perm][:, perm]

    damp = percdamp * torch.mean(torch.diag(H))
    H.diagonal().add_(damp)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)

    n_groups = cols // group
    scales = torch.empty(rows, n_groups, device=W.device)
    zeros = torch.empty(rows, n_groups, device=W.device)
    Q = torch.empty_like(W)
    codes = torch.empty(rows, cols, dtype=torch.uint8, device=W.device)
    loss = torch.zeros((), device=W.device)
    maxq = spec.maxq

    if spec.granularity != "group":  # one set of params, from the original weight
        s0, z0 = _params_for(W, spec)
        scales[:, 0], zeros[:, 0] = s0, z0

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.empty_like(W1)
        Err1 = torch.empty_like(W1)
        U1 = U[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            g = col // group
            if spec.granularity == "group" and col % group == 0:
                # group starts at column i of this block and ends inside it
                s, z = _params_for(W1[:, i : i + group], spec)
                scales[:, g], zeros[:, g] = s, z
            s, z = scales[:, g], zeros[:, g]
            w = W1[:, i]
            q = torch.clamp(torch.round(w / s) + z, 0, maxq)
            codes[:, col] = q.to(torch.uint8)
            dq = (q - z) * s
            Q1[:, i] = dq
            d = U1[i, i]
            loss += ((w - dq) ** 2).sum() / d**2 / 2
            err = (w - dq) / d
            W1[:, i:] -= err.unsqueeze(1) @ U1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ U[i1:i2, i2:]

    g_idx = torch.arange(cols, device=W.device, dtype=torch.int32) // group
    if perm is not None:
        inv = torch.argsort(perm)
        Q, codes = Q[:, inv], codes[:, inv]
        g_idx = g_idx[inv]
    if spec.granularity != "group":
        g_idx = torch.zeros(cols, dtype=torch.int32, device=W.device)
    result = GPTQResult(codes.cpu(), scales.cpu(), zeros.cpu(), g_idx.cpu(), spec)
    return Q.to(weight.dtype), result, float(loss)


def make_block_hook(
    spec: QuantSpec,
    device,
    results: dict[str, GPTQResult],
    blocksize: int = 128,
    percdamp: float = 0.01,
    act_order: bool = False,
    log: list | None = None,
):
    """A ``before_block`` hook for :func:`qlab.sequential.run_blocks` (GPTQ one block)."""

    def hook(i: int, block: nn.Module, inputs: BlockInputs) -> None:
        lins = block_linears(block)
        order = [g for g in SEQUENTIAL_GROUPS if all(n in lins for n in g)]
        rest = [n for n in lins if not any(n in g for g in order)]
        order += [(n,) for n in rest]
        for names in order:
            acc = {n: HessianAccumulator(lins[n].in_features, device) for n in names}
            handles = [
                lins[n].register_forward_pre_hook(lambda _m, a, h=acc[n]: h.add(a[0]))
                for n in names
            ]
            try:
                block_forward(block, inputs, device)
            finally:
                for h in handles:
                    h.remove()
            for n in names:
                lin = lins[n]
                q, res, loss = gptq_quantize(
                    lin.weight.data, acc[n].H, spec, blocksize, percdamp, act_order
                )
                lin.weight.data.copy_(q)
                results[f"layers.{i}.{n}"] = res
                if log is not None:
                    log.append((f"layers.{i}.{n}", loss))
                del acc[n]

    return hook
