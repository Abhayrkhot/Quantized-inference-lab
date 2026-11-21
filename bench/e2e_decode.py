#!/usr/bin/env python3
"""End-to-end batch-1 decode speed: 16-bit model vs the same model with Triton W4A16 linears.

Greedy generation with a static KV cache, so transformers compiles the decode step
(``torch.compile``, CUDA graphs). Prefill time is measured separately with a 1-token
generation and subtracted. Every decoder linear is replaced by ``QuantLinear``
(RTN int4 g128 asym; the method does not change the speed). Embeddings and lm_head stay
16-bit in both, as in every 4-bit checkpoint format.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qlab.formats import QuantSpec  # noqa: E402
from qlab.model import load_model  # noqa: E402
from qlab.qlinear import swap_in_quant_linears  # noqa: E402
from qlab.results import write_rows  # noqa: E402


def _generate(model, ids, new):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(
        ids,
        max_new_tokens=new,
        min_new_tokens=new,
        do_sample=False,
        cache_implementation="static",
        pad_token_id=0,
    )
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def measure(model, ids, new: int, reps: int) -> dict:
    for _ in range(2):  # compile, autotune and CUDA-graph capture for prefill and decode shapes
        _generate(model, ids, new)
        _generate(model, ids, 1)
    torch.cuda.empty_cache()  # drop autotuning scratch buffers before measuring peak memory
    torch.cuda.reset_peak_memory_stats()
    decode, prefill = [], []
    for _ in range(reps):
        _, t1 = _generate(model, ids, 1)
        out, tn = _generate(model, ids, new)
        prefill.append(t1)
        decode.append((tn - t1) / (new - 1))
    decode.sort()
    prefill.sort()
    ms = decode[len(decode) // 2] * 1e3
    return {
        "decode_ms_per_token": round(ms, 3),
        "decode_tokens_per_s": round(1e3 / ms, 1),
        "prefill_ms": round(prefill[len(prefill) // 2] * 1e3, 2),
        "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "tokens": out[0, ids.shape[1] :].tolist()[:16],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for variant in ("16-bit", "int4 g128 asym (qlab triton)"):
        model, tok = load_model(args.model, args.dtype, "cuda")
        model.config.use_cache = True
        if variant != "16-bit":
            n = swap_in_quant_linears(model, QuantSpec(4, "group", 128, False))
            print(f"swapped {n} linears", flush=True)
        torch.cuda.empty_cache()
        weights_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / 2**20
        weights_mb += sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(1000, 20000, (1, args.prompt_len), generator=g).cuda()
        r = measure(model, ids, args.new, args.reps)
        rows.append(
            {
                "model": args.model,
                "weights": variant,
                "weights_mb": round(weights_mb, 1),
                "prompt_len": args.prompt_len,
                "new_tokens": args.new,
                **r,
            }
        )
        print(rows[-1], flush=True)
        del model
        torch.cuda.empty_cache()
    base, quant = rows
    quant["speedup_vs_16bit"] = round(base["decode_ms_per_token"] / quant["decode_ms_per_token"], 2)
    for r in rows:
        r["tokens"] = " ".join(map(str, r["tokens"]))
    write_rows(args.out, rows, {"script": "bench/e2e_decode.py", "args": vars(args)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
