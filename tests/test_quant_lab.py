"""Tests for the quantized inference lab."""

import importlib
import os
import sys

import numpy as np  # type: ignore

# Adjust path to import the module under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa

from quant_lab import uniform_quantize, dequantize, simulate


def test_uniform_quantize_int8_and_int4_memory_ratio():
    """INT8 should use roughly 1/4 the memory of FP32 and INT4 ~1/8."""
    # Generate random float32 tensor
    tensor = np.random.randn(100).astype(np.float32)
    fp32_bytes = tensor.nbytes
    q8, scale8 = uniform_quantize(tensor, num_bits=8)
    q4, scale4 = uniform_quantize(tensor, num_bits=4)
    int8_bytes = q8.nbytes
    # For int4 we calculate memory manually: half byte per element
    int4_bytes = (q4.size + 1) // 2
    # Check ratios
    assert abs(int8_bytes / fp32_bytes - 0.25) < 0.05
    assert abs(int4_bytes / fp32_bytes - 0.125) < 0.05


def test_simulate_summary_fields():
    """The simulate function should return keys with expected properties."""
    result = simulate([32, 32], seed=0)
    assert set(result.keys()) == {"shape", "fp32_bytes", "int8_bytes", "int8_mse", "int4_bytes", "int4_mse"}
    assert result["fp32_bytes"] > result["int8_bytes"] > result["int4_bytes"]
    assert result["int8_mse"] >= 0
    assert result["int4_mse"] >= 0
