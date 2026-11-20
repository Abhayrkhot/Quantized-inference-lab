"""Round-to-nearest (RTN) weight quantization: the baseline every other method is compared to."""

from __future__ import annotations

import torch
from torch import nn

from .formats import QuantSpec, fake_quant
from .model import quantizable_linears


@torch.no_grad()
def quantize_rtn(model: nn.Module, spec: QuantSpec, device: torch.device | str) -> dict:
    """Replace every decoder linear weight with its RTN fake-quantized version, in place.

    Returns the mean relative weight error ``||W - Q(W)||^2 / ||W||^2`` over the layers.
    """
    rel = []
    for lin in quantizable_linears(model).values():
        w = lin.weight.data
        wd = w.to(device)
        wq = fake_quant(wd, spec)
        rel.append(float((wd.float() - wq.float()).pow(2).sum() / wd.float().pow(2).sum()))
        w.copy_(wq.to(w.device))
    return {"weight_rel_err": sum(rel) / len(rel)}
