"""Bit packing for quantized codes.

Two layouts:

* ``pack_int4`` / ``unpack_int4``: two 4-bit codes per byte, low nibble first. This is the
  storage the memory numbers in this repo assume.
* GPTQ int32 layout, which vLLM's GPTQ and Marlin loaders read:

  - ``qweight`` is ``int32 [in_features * bits / 32, out_features]``. Each int32 holds
    ``32 / bits`` consecutive *input* rows of one output column; row ``i`` of the pack sits at
    bit offset ``bits * i``. Callers pass codes as ``[in, out]`` (the transpose of
    ``nn.Linear.weight``).
  - ``qzeros`` is ``int32 [n_groups, out_features * bits / 32]``, packed along the *output*
    dimension in the same way.

Values are packed as unsigned 32-bit words and stored in int32 (two's complement), which is
what the checkpoint format holds.
"""

from __future__ import annotations

import torch

_PACKABLE_BITS = (2, 4, 8)


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack uint8 codes in ``[0, 15]`` along the last dim, two per byte (low nibble first)."""
    if codes.shape[-1] % 2:
        raise ValueError(f"last dim must be even to pack int4, got {codes.shape[-1]}")
    if codes.numel() and int(codes.max()) > 15:
        raise ValueError("int4 codes must be in [0, 15]")
    c = codes.to(torch.uint8)
    return c[..., 0::2] | (c[..., 1::2] << 4)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_int4`."""
    p = packed.to(torch.uint8)
    out = torch.empty(*p.shape[:-1], p.shape[-1] * 2, dtype=torch.uint8, device=p.device)
    out[..., 0::2] = p & 0xF
    out[..., 1::2] = p >> 4
    return out


def _check_bits(bits: int) -> int:
    if bits not in _PACKABLE_BITS:
        raise ValueError(f"int32 packing supports bits in {_PACKABLE_BITS}, got {bits}")
    return 32 // bits


def _to_int32(words: torch.Tensor) -> torch.Tensor:
    """Reinterpret unsigned 32-bit values held in int64 as int32."""
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


def _from_int32(words: torch.Tensor) -> torch.Tensor:
    return words.to(torch.int64) & 0xFFFFFFFF


def pack_rows_int32(codes: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Pack ``[K, N]`` codes along K into ``int32 [K * bits / 32, N]`` (GPTQ ``qweight``)."""
    per = _check_bits(bits)
    k, n = codes.shape
    if k % per:
        raise ValueError(f"rows ({k}) must be a multiple of {per} for {bits}-bit packing")
    c = codes.to(torch.int64).reshape(k // per, per, n)
    shifts = (torch.arange(per, device=codes.device, dtype=torch.int64) * bits).view(1, per, 1)
    return _to_int32((c << shifts).sum(dim=1))


def unpack_rows_int32(packed: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Inverse of :func:`pack_rows_int32`; returns uint8 ``[K, N]``."""
    per = _check_bits(bits)
    w = _from_int32(packed).unsqueeze(1)  # [K/per, 1, N]
    shifts = (torch.arange(per, device=packed.device, dtype=torch.int64) * bits).view(1, per, 1)
    vals = (w >> shifts) & (2**bits - 1)
    return vals.reshape(-1, packed.shape[1]).to(torch.uint8)


def pack_cols_int32(codes: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Pack ``[G, N]`` codes along N into ``int32 [G, N * bits / 32]`` (GPTQ ``qzeros``)."""
    return pack_rows_int32(codes.t().contiguous(), bits).t().contiguous()


def unpack_cols_int32(packed: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Inverse of :func:`pack_cols_int32`."""
    return unpack_rows_int32(packed.t().contiguous(), bits).t().contiguous()
