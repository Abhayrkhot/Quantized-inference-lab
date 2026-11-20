"""Integer weight-quantization formats.

Conventions (shared by every method in qlab and by the GPTQ checkpoint format it exports):

* Weights are 2-D ``[out_features, in_features]``, the layout of ``nn.Linear.weight``.
* Codes are unsigned integers ``q`` in ``[0, 2**bits - 1]``; the dequantized value is
  ``(q - zero) * scale``.
* Symmetric: ``zero`` is fixed at ``2**(bits - 1)`` and ``scale = absmax / (2**(bits - 1) - 1)``,
  so ``[-absmax, absmax]`` is covered exactly and the most negative code goes unused.
* Asymmetric: the range ``[min(w, 0), max(w, 0)]`` is mapped onto all ``2**bits`` codes and
  ``zero`` is an integer code, so 0.0 is always exactly representable.
* Groups run along the input dimension: group ``g`` covers input columns ``[g*G, (g+1)*G)``.
  One (scale, zero) pair is stored per output row per group.

Every value in the clipping range round-trips with error at most ``scale / 2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Granularity = Literal["tensor", "channel", "group"]
GRANULARITIES = ("tensor", "channel", "group")

SCALE_BITS = 16  # scales are stored as fp16


@dataclass(frozen=True)
class QuantSpec:
    """How to quantize a weight matrix."""

    bits: int = 4
    granularity: Granularity = "group"
    group_size: int = 128
    symmetric: bool = False

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 8:
            raise ValueError(f"bits must be in [2, 8], got {self.bits}")
        if self.granularity not in GRANULARITIES:
            raise ValueError(
                f"granularity must be one of {GRANULARITIES}, got {self.granularity!r}"
            )
        if self.granularity == "group" and self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")

    @property
    def maxq(self) -> int:
        return 2**self.bits - 1

    @property
    def label(self) -> str:
        gran = f"g{self.group_size}" if self.granularity == "group" else self.granularity
        return f"int{self.bits} {gran} {'sym' if self.symmetric else 'asym'}"

    def n_groups(self, in_features: int) -> int:
        if self.granularity != "group":
            return 1
        if in_features % self.group_size:
            raise ValueError(
                f"in_features={in_features} is not a multiple of group_size={self.group_size}"
            )
        return in_features // self.group_size

    def n_param_sets(self, shape: tuple[int, int]) -> int:
        """Number of (scale, zero) pairs stored for a weight of this shape."""
        out_features, in_features = shape
        if self.granularity == "tensor":
            return 1
        return out_features * self.n_groups(in_features)

    def overhead_bits(self, shape: tuple[int, int]) -> int:
        """Bits spent on scales and zero-points (zeros are packed at ``bits`` each)."""
        per_set = SCALE_BITS + (0 if self.symmetric else self.bits)
        return self.n_param_sets(shape) * per_set

    def bits_per_weight(self, shape: tuple[int, int]) -> float:
        """Effective storage per weight, including scales and zero-points."""
        numel = shape[0] * shape[1]
        return self.bits + self.overhead_bits(shape) / numel

    def nbytes(self, shape: tuple[int, int]) -> int:
        """Bytes to store a weight of this shape, codes densely packed plus overhead."""
        numel = shape[0] * shape[1]
        return -(-(numel * self.bits + self.overhead_bits(shape)) // 8)


def _check_2d(w: torch.Tensor) -> None:
    if w.dim() != 2:
        raise ValueError(f"expected a 2-D [out, in] weight, got shape {tuple(w.shape)}")


def _grouped(w: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    """View ``w`` as ``[rows, groups, group_len]`` so stats reduce over the last dim."""
    out_features, in_features = w.shape
    if spec.granularity == "tensor":
        return w.reshape(1, 1, -1)
    if spec.granularity == "channel":
        return w.reshape(out_features, 1, in_features)
    return w.reshape(out_features, spec.n_groups(in_features), spec.group_size)


def compute_qparams(
    w: torch.Tensor, spec: QuantSpec, clip: float | torch.Tensor = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(scale, zero)``, float32, shaped ``[rows, groups]``.

    ``clip`` shrinks the range before computing the scale (1.0 = plain min/max). It may be a
    tensor broadcastable to ``[rows, groups]`` for per-group clip search.
    """
    _check_2d(w)
    wg = _grouped(w.float(), spec)
    if spec.symmetric:
        absmax = wg.abs().amax(dim=-1) * clip
        half = 2 ** (spec.bits - 1)
        scale = absmax / (half - 1)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        zero = torch.full_like(scale, float(half))
    else:
        wmin = wg.amin(dim=-1).clamp(max=0) * clip
        wmax = wg.amax(dim=-1).clamp(min=0) * clip
        scale = (wmax - wmin) / spec.maxq
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        zero = torch.round(-wmin / scale).clamp(0, spec.maxq)
    return scale, zero


def quantize(
    w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, spec: QuantSpec
) -> torch.Tensor:
    """Map ``w`` to uint8 codes in ``[0, 2**bits - 1]`` with the given params."""
    _check_2d(w)
    wg = _grouped(w.float(), spec)
    q = torch.round(wg / scale.unsqueeze(-1)) + zero.unsqueeze(-1)
    return q.clamp(0, spec.maxq).to(torch.uint8).reshape(w.shape)


def dequantize(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    spec: QuantSpec,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct float weights from codes."""
    _check_2d(q)
    qg = _grouped(q.float(), spec)
    w = (qg - zero.unsqueeze(-1)) * scale.unsqueeze(-1)
    return w.reshape(q.shape).to(dtype)


def expand_params(p: torch.Tensor, spec: QuantSpec, shape: tuple[int, int]) -> torch.Tensor:
    """Broadcast a ``[rows, groups]`` param tensor to the full ``[out, in]`` weight shape."""
    out_features, in_features = shape
    if spec.granularity == "tensor":
        return p.reshape(1, 1).expand(out_features, in_features)
    if spec.granularity == "channel":
        return p.reshape(out_features, 1).expand(out_features, in_features)
    return p.repeat_interleave(spec.group_size, dim=1)


def fake_quant(w: torch.Tensor, spec: QuantSpec, clip: float | torch.Tensor = 1.0) -> torch.Tensor:
    """Quantize then dequantize, returning a tensor with ``w``'s dtype and shape."""
    scale, zero = compute_qparams(w, spec, clip)
    return dequantize(quantize(w, scale, zero, spec), scale, zero, spec, dtype=w.dtype)


@dataclass
class QuantizedWeight:
    """Codes plus the params needed to reconstruct them."""

    codes: torch.Tensor  # uint8 [out, in]
    scale: torch.Tensor  # float32 [rows, groups]
    zero: torch.Tensor  # float32 [rows, groups], integer-valued
    spec: QuantSpec

    @classmethod
    def from_float(cls, w: torch.Tensor, spec: QuantSpec) -> QuantizedWeight:
        scale, zero = compute_qparams(w, spec)
        return cls(quantize(w, scale, zero, spec), scale, zero, spec)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.codes.shape)  # type: ignore[return-value]

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return dequantize(self.codes, self.scale, self.zero, self.spec, dtype)

    def nbytes(self) -> int:
        return self.spec.nbytes(self.shape)
