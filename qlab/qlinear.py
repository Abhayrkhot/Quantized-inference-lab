"""``QuantLinear``: a drop-in nn.Linear replacement that runs the Triton W4A16 kernel."""

from __future__ import annotations

import torch
from torch import nn

from .formats import QuantizedWeight, QuantSpec
from .kernels.w4a16 import pack_for_kernel, reserve_workspace, w4a16_matmul
from .model import decoder_layers


@torch.library.custom_op("qlab::w4a16", mutates_args=())
def _w4a16_op(
    x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, group: int
) -> torch.Tensor:
    return w4a16_matmul(x, qweight, scales, qzeros, group)


@_w4a16_op.register_fake
def _(x, qweight, scales, qzeros, group):
    return x.new_empty(*x.shape[:-1], qweight.shape[1])


class QuantLinear(nn.Module):
    """4-bit group-wise weights in the GPTQ layout; ``forward`` is one kernel launch."""

    def __init__(self, qweight, scales, qzeros, group: int, bias=None):
        super().__init__()
        self.in_features = qweight.shape[0] * 8
        self.out_features = qweight.shape[1]
        self.group = group
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.register_buffer("qzeros", qzeros)
        self.register_buffer("bias", bias)

    @classmethod
    def from_quantized(cls, qw: QuantizedWeight, dtype, bias=None) -> QuantLinear:
        packed = pack_for_kernel(qw.codes, qw.scale, qw.zero, qw.spec, dtype)
        return cls(*packed, qw.spec.group_size, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _w4a16_op(x, self.qweight, self.scales, self.qzeros, self.group)
        return y + self.bias if self.bias is not None else y

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, group={self.group}"


@torch.no_grad()
def swap_in_quant_linears(
    model: nn.Module, spec: QuantSpec, quantized: dict[str, QuantizedWeight] | None = None
) -> int:
    """Replace every decoder linear with a QuantLinear (RTN-quantizing any not in ``quantized``)."""
    n = 0
    dtype = next(model.parameters()).dtype
    widest = 0
    for i, layer in enumerate(decoder_layers(model)):
        for name, mod in list(layer.named_modules()):
            if not isinstance(mod, nn.Linear):
                continue
            key = f"layers.{i}.{name}"
            qw = (quantized or {}).get(key) or QuantizedWeight.from_float(mod.weight.data, spec)
            new = QuantLinear.from_quantized(qw, dtype, mod.bias)
            parent = layer.get_submodule(name.rsplit(".", 1)[0]) if "." in name else layer
            setattr(parent, name.rsplit(".", 1)[-1], new.to(mod.weight.device))
            widest = max(widest, new.out_features)
            n += 1
    if n:
        # allocate the kernel's split-K workspace now, before any torch.compile / CUDA graph
        reserve_workspace(next(model.parameters()).device, widest)
    return n
