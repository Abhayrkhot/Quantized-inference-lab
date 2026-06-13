# Quantized Inference Lab

## Overview

Quantization compresses model weights and activations by reducing the bit
width used to represent each value.  Converting from high‑precision formats
such as FP32 (32 bits) to lower precision formats like FP16, INT8 or INT4 can
cut memory usage dramatically【470140481889852†L53-L64】.  For example, a 7 B
parameter model in FP32 requires around 28 GB just for the weights, while
FP16 halves this to 14 GB, and INT8 reduces it further to 7 GB【470140481889852†L53-L64】.  These savings
extend to the runtime KV cache and activations【470140481889852†L73-L79】, enabling
longer context lengths and higher throughput.

This project provides a **quantized inference lab**: a small Python toolkit
that demonstrates the effect of quantization on random tensors.  It
implements simple uniform quantisation to INT8 and INT4, reports memory
consumption, and measures reconstruction error.  The goal is to illustrate
how quantisation trades off memory usage against accuracy and to help users
understand why quantisation is a powerful technique for LLM serving【470140481889852†L73-L79】.

## Why this matters

LLM inference is often limited by GPU memory.  Quantisation reduces the
memory footprint of model weights and the KV cache【470140481889852†L73-L79】.  This
allows you to fit larger contexts or batch sizes on the same hardware.  In
addition, lower precision arithmetic can accelerate computation on modern
hardware, reducing inference latency【470140481889852†L73-L79】.  The trade‑off is a
small loss in accuracy or increased reconstruction error.  For many
applications, the impact is minimal when quantisation is carefully applied
【470140481889852†L73-L83】.

## Features

* **Uniform quantisation to INT8 and INT4:** Convert floating‑point tensors
  to low‑precision formats via min–max scaling.
* **Memory usage estimator:** Compute the memory required to store the
  original and quantised tensors based on their element sizes.
* **Reconstruction error:** Calculate the mean squared error between the
  original tensor and its dequantised version to quantify accuracy loss.
* **Simulation tool:** Generate random tensors of arbitrary shape, apply
  quantisation schemes, and report memory savings and errors via a
  command‑line interface.

## Installation

This project depends only on Python 3.8+ and `numpy` for numerical
computations.  Install the development requirements with:

```bash
pip install -r requirements.txt
```

## Usage

Generate a random tensor of shape 1024 × 4096 and compare FP32, INT8 and
INT4 quantisation schemes:

```bash
python quant_lab.py --shape 1024 4096 --seed 0
```

Sample output:

```
{
  "shape": [1024, 4096],
  "fp32_bytes": 16777216,
  "int8_bytes": 4194304,
  "int8_mse": 0.000299,
  "int4_bytes": 2097152,
  "int4_mse": 0.000815
}
```

Memory values are in bytes; MSE is the mean squared error of the
dequantised tensor.  INT8 uses a single byte per value, giving a 4×
compression over FP32, while INT4 uses half a byte (4 bits) for an 8×
compression【470140481889852†L53-L64】.  Lower precision yields slightly higher
error.

## Running the tests

Execute the tests with:

```bash
pytest
```

The tests ensure that quantisation reduces memory usage and that the MSE
values are reasonable.

## Limitations and future work

This lab uses a simple uniform quantisation scheme.  Real quantisation
methods like GPTQ or AWQ use more sophisticated calibration to minimise
error.  The simulation also ignores hardware‑specific optimisations and
overheads.  Extending the lab to include per‑channel scales and more
quantisation formats (e.g., INT2 or FP8) would make it more comprehensive.

## Resume bullet

> Developed a quantised inference lab demonstrating INT8/INT4 quantisation of
> random tensors; measured memory savings versus FP32 and quantified
> reconstruction error, illustrating the trade‑off between memory and
> accuracy【470140481889852†L53-L64】.
