"""Quantized inference laboratory.

This script demonstrates uniform quantisation of random tensors to INT8 and
INT4.  It reports memory savings and reconstruction error relative to
the original FP32 representation.

The module exposes helper functions for quantisation and a command‑line
interface for running simulations.  It uses NumPy for numerical
computations.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from typing import Dict, List, Tuple

import numpy as np  # type: ignore


def uniform_quantize(
    tensor: np.ndarray, num_bits: int
) -> Tuple[np.ndarray, float]:
    """Uniformly quantise a float tensor to a given bit width.

    This function finds the maximum absolute value of the tensor and uses it
    to compute a scale factor so that the range maps to [-max_q, max_q], where
    ``max_q = 2^(num_bits - 1) - 1`` for signed integers.  The quantised
    tensor is returned as an int8 array (we reuse int8 for int4 values) along
    with the scale factor for dequantisation.

    Parameters
    ----------
    tensor: np.ndarray
        The input tensor of dtype float32.
    num_bits: int
        The target bit width (e.g., 8 or 4).

    Returns
    -------
    Tuple[np.ndarray, float]
        A tuple containing the quantised int array and the scale factor.
    """
    assert num_bits in (8, 4), "Only INT8 and INT4 quantisation are supported"
    max_abs = np.max(np.abs(tensor))
    if max_abs == 0:
        scale = 1.0
        quantised = np.zeros_like(tensor, dtype=np.int8)
        return quantised, scale
    max_q = 2 ** (num_bits - 1) - 1
    scale = max_abs / max_q
    # Avoid division by zero
    scaled = tensor / scale
    # Clip to representable range
    clipped = np.clip(np.round(scaled), -max_q, max_q)
    quantised = clipped.astype(np.int8)
    return quantised, scale


def dequantize(quantised: np.ndarray, scale: float) -> np.ndarray:
    """Reconstruct a float tensor from a quantised integer tensor and scale."""
    return quantised.astype(np.float32) * scale


def memory_bytes(tensor: np.ndarray, num_bits: int) -> int:
    """Compute the memory size in bytes for a tensor with given bit width."""
    n_elements = tensor.size
    return int(math.ceil(n_elements * num_bits / 8))


def simulate(shape: List[int], seed: int) -> Dict[str, object]:
    """Simulate quantisation on a random tensor.

    Parameters
    ----------
    shape: List[int]
        The shape of the random tensor to generate.
    seed: int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, object]
        A summary containing memory and error metrics for FP32, INT8 and INT4.
    """
    rng = np.random.default_rng(seed)
    tensor = rng.standard_normal(size=shape, dtype=np.float32)
    # FP32 memory
    fp32_bytes = tensor.nbytes
    # INT8 quantisation
    q8, scale8 = uniform_quantize(tensor, num_bits=8)
    deq8 = dequantize(q8, scale8)
    mse8 = float(np.mean((tensor - deq8) ** 2))
    int8_bytes = q8.nbytes  # Each element stored as int8 (1 byte)
    # INT4 quantisation: reuse int8 array but account for 4 bits per value
    q4, scale4 = uniform_quantize(tensor, num_bits=4)
    deq4 = dequantize(q4, scale4)
    mse4 = float(np.mean((tensor - deq4) ** 2))
    int4_bytes = memory_bytes(q4, num_bits=4)
    return {
        "shape": list(shape),
        "fp32_bytes": fp32_bytes,
        "int8_bytes": int8_bytes,
        "int8_mse": mse8,
        "int4_bytes": int4_bytes,
        "int4_mse": mse4,
    }


def cli(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Simulate INT8 and INT4 quantisation on a random tensor.")
    parser.add_argument("--shape", type=int, nargs="+", default=[1024, 4096], help="Shape of the random tensor (list of dimensions)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args(argv)
    result = simulate(args.shape, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()