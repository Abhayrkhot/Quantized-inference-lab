"""Quantize synthetic matrices to show how granularity and outliers drive error.

A Gaussian matrix is the easy case: error follows the uniform-noise model ``scale**2 / 12``.
Real LLM weights are not like that; a few input channels carry much larger values. The
``outliers`` matrix scales 1% of the input columns by 20x to mimic that, which makes
one-scale-per-tensor and one-scale-per-row quantization fall apart while small groups survive.
"""

from __future__ import annotations

import math

import torch

from .formats import QuantSpec, fake_quant

DEFAULT_SPECS: tuple[QuantSpec, ...] = tuple(
    QuantSpec(bits=b, granularity=g, group_size=128, symmetric=s)
    for b in (8, 4, 3)
    for g in ("tensor", "channel", "group")
    for s in (True, False)
)


def make_matrix(
    shape: tuple[int, int],
    seed: int,
    outlier_frac: float = 0.0,
    outlier_scale: float = 20.0,
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(shape, generator=gen, dtype=torch.float32)
    n_out = round(shape[1] * outlier_frac)
    if n_out:
        cols = torch.randperm(shape[1], generator=gen)[:n_out]
        w[:, cols] *= outlier_scale
    return w


def evaluate(w: torch.Tensor, spec: QuantSpec) -> dict[str, object]:
    w_hat = fake_quant(w, spec)
    err = w - w_hat
    mse = float(err.pow(2).mean())
    signal = float(w.pow(2).mean())
    return {
        "format": spec.label,
        "bits": spec.bits,
        "granularity": spec.granularity,
        "group_size": spec.group_size if spec.granularity == "group" else "",
        "symmetric": spec.symmetric,
        "bits_per_weight": round(spec.bits_per_weight(tuple(w.shape)), 4),
        "bytes": spec.nbytes(tuple(w.shape)),
        "mse": mse,
        "sqnr_db": 10 * math.log10(signal / mse) if mse > 0 else math.inf,
    }


def run(
    shape: tuple[int, int] = (1024, 4096),
    seed: int = 0,
    specs: tuple[QuantSpec, ...] = DEFAULT_SPECS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    matrices = {
        "gaussian": make_matrix(shape, seed),
        "outliers": make_matrix(shape, seed, outlier_frac=0.01),
    }
    for name, w in matrices.items():
        rows.append(
            {
                "matrix": name,
                "format": "fp32",
                "bits": 32,
                "granularity": "",
                "group_size": "",
                "symmetric": "",
                "bits_per_weight": 32.0,
                "bytes": w.numel() * 4,
                "mse": 0.0,
                "sqnr_db": math.inf,
            }
        )
        rows.extend({"matrix": name, **evaluate(w, s)} for s in specs)
    return rows
