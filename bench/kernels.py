#!/usr/bin/env python3
"""Microbenchmark W4A16 matmuls against 16-bit cuBLAS, vLLM Marlin and bitsandbytes NF4.

For each (K, N) linear shape of a model and each batch size M, times ``y = x @ W^T``:

* ``bf16 cuBLAS``: ``F.linear`` with the dense weight;
* ``qlab triton``: :func:`qlab.kernels.w4a16.w4a16_matmul` on the GPTQ layout (asym, g128);
* ``qlab triton sym``: the same kernel on symmetric weights, the format Marlin takes;
* ``vLLM marlin``: ``apply_gptq_marlin_linear`` (uint4b8, g128) from the same image;
* ``bnb nf4``: ``bitsandbytes.matmul_4bit`` (NF4, blocks of 64, fp32 absmax).

Timing replays each call in a CUDA graph (``triton.testing.do_bench_cudagraph``, median), as
decoding does, so the Python launch cost of each library does not count. A kernel that
cannot be captured is timed eagerly instead, and the ``timing`` column says which.
Effective bandwidth is the bytes a kernel must move (weights, scales, zeros, x, y) divided
by its time, also shown as a fraction of this GPU's measured device-to-device copy bandwidth.
Every kernel's output is checked against a float32 reference before it is timed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qlab.formats import QuantizedWeight, QuantSpec  # noqa: E402
from qlab.kernels.w4a16 import pack_for_kernel, w4a16_matmul  # noqa: E402
from qlab.results import write_rows  # noqa: E402

SHAPES = {
    "qwen3-0.6b": [(1024, 2048), (1024, 1024), (2048, 1024), (1024, 3072), (3072, 1024)],
    "qwen3-1.7b": [(2048, 2048), (2048, 1024), (2048, 6144), (6144, 2048)],
    "qwen3-8b": [(4096, 4096), (4096, 1024), (4096, 12288), (12288, 4096)],
}


def copy_bandwidth(nbytes: int = 256 * 2**20) -> float:
    from triton.testing import do_bench

    src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    t = do_bench(lambda: dst.copy_(src), warmup=25, rep=200, return_mode="median")
    return 2 * nbytes / (t * 1e-3)


def _check(name, y, ref):
    err = float((y.float() - ref).abs().max() / ref.abs().max())
    if err > 2e-2:
        raise AssertionError(f"{name}: relative max error {err:.4f}")
    return err


def bench_shape(k: int, n: int, ms: list[int], dtype, peak: float, skip: set[str]) -> list[dict]:
    from triton.testing import do_bench, do_bench_cudagraph

    torch.manual_seed(0)
    w = torch.randn(n, k, device="cuda") * 0.02
    asym, sym = QuantSpec(4, "group", 128, False), QuantSpec(4, "group", 128, True)
    qa, qs = QuantizedWeight.from_float(w, asym), QuantizedWeight.from_float(w, sym)
    pa = pack_for_kernel(qa.codes, qa.scale, qa.zero, asym, dtype)
    ps = pack_for_kernel(qs.codes, qs.scale, qs.zero, sym, dtype)
    w16 = w.to(dtype)
    ref_a = qa.dequantize()
    ref_s = qs.dequantize()
    kernels = {
        "bf16 cuBLAS" if dtype == torch.bfloat16 else "fp16 cuBLAS": (
            lambda x: F.linear(x, w16),
            w.float(),
            k * n * 2,
        ),
        "qlab triton": (
            lambda x: w4a16_matmul(x, *pa, 128),
            ref_a,
            k * n // 2 + (k // 128) * n * (2 + 0.5),
        ),
        "qlab triton sym": (
            lambda x: w4a16_matmul(x, *ps, 128),
            ref_s,
            k * n // 2 + (k // 128) * n * (2 + 0.5),
        ),
    }
    if "marlin" not in skip:
        from bench.marlin_ref import MarlinW4

        mar = MarlinW4(ps[0], ps[1], 128)
        kernels["vLLM marlin"] = (mar, ref_s, k * n // 2 + (k // 128) * n * 2)
    if "bnb" not in skip:
        import bitsandbytes as bnb
        import bitsandbytes.functional as bnbf

        qb, st = bnbf.quantize_4bit(w16, blocksize=64, quant_type="nf4")
        ref_b = bnbf.dequantize_4bit(qb, st).float()
        kernels["bnb nf4"] = (
            lambda x: bnb.matmul_4bit(x, qb.t(), quant_state=st),
            ref_b,
            k * n // 2 + (k * n // 64) * 4,
        )
    rows = []
    for m in ms:
        x = torch.randn(m, k, device="cuda", dtype=dtype)
        io = (m * k + m * n) * 2
        for name, (fn, ref_w, wbytes) in kernels.items():
            err = _check(name, fn(x), x.float() @ ref_w.t())
            torch.cuda.synchronize()
            try:
                t = do_bench_cudagraph(lambda fn=fn, x=x: fn(x), rep=100, return_mode="median")
                mode = "cudagraph"
            except Exception:  # noqa: BLE001 - not capturable: time it eagerly
                torch.cuda.synchronize()
                t = do_bench(lambda fn=fn, x=x: fn(x), warmup=25, rep=100, return_mode="median")
                mode = "eager"
            bw = (wbytes + io) / (t * 1e-3)
            rows.append(
                {
                    "K": k,
                    "N": n,
                    "M": m,
                    "kernel": name,
                    "us": round(t * 1e3, 2),
                    "tflops": round(2 * m * k * n / (t * 1e-3) / 1e12, 3),
                    "eff_gbps": round(bw / 1e9, 1),
                    "pct_copy_bw": round(100 * bw / peak, 1),
                    "rel_err": round(err, 5),
                    "timing": mode,
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(SHAPES))
    ap.add_argument("--m", type=int, nargs="+", default=[1, 2, 4, 8, 16, 64, 256, 1024])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    peak = copy_bandwidth()
    print(f"device copy bandwidth: {peak / 1e9:.1f} GB/s", flush=True)
    rows = []
    for model in args.models:
        for k, n in SHAPES[model]:
            for r in bench_shape(k, n, args.m, dtype, peak, set(args.skip)):
                rows.append({"model": model, **r})
                print(rows[-1], flush=True)
            torch.cuda.empty_cache()
    write_rows(
        args.out,
        rows,
        {
            "copy_bandwidth_gbps": round(peak / 1e9, 1),
            "dtype": args.dtype,
            "script": "bench/kernels.py",
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
