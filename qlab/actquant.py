"""Fake-quantized activations at the input of every decoder linear layer (W8A8 simulation).

``granularity="tensor"``: one dynamic scale per sequence (absmax over all its tokens and
channels), which is what a single per-tensor scale amounts to. ``"token"``: one dynamic scale
per token, as SmoothQuant and most INT8 serving kernels use.

Symmetric, with the same convention as symmetric weights (``qlab.formats``):
``scale = absmax / 127.5`` at 8 bits and codes -128..127.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from .model import quantizable_linears


def fake_quant_act(x: torch.Tensor, bits: int = 8, granularity: str = "token") -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    if granularity == "token":
        amax = x.abs().amax(dim=-1, keepdim=True)
    elif granularity == "tensor":
        amax = (
            x.abs().amax(dim=tuple(range(1, x.dim())), keepdim=True)
            if x.dim() > 2
            else x.abs().amax()
        )
    else:
        raise ValueError(f"activation granularity must be 'token' or 'tensor', got {granularity!r}")
    scale = (amax.float() / (qmax + 0.5)).clamp(min=1e-8)
    return ((x.float() / scale).round().clamp(-qmax - 1, qmax) * scale).to(x.dtype)


@contextmanager
def activation_quant(model: nn.Module, bits: int = 8, granularity: str = "token"):
    """Install input fake-quant hooks on every decoder linear for the duration of the block."""

    def hook(_mod, args):
        return (fake_quant_act(args[0], bits, granularity), *args[1:])

    handles = [m.register_forward_pre_hook(hook) for m in quantizable_linears(model).values()]
    try:
        yield
    finally:
        for h in handles:
            h.remove()
