#!/usr/bin/env python3
"""WikiText-2 perplexity of a checkpoint as served by vLLM (prompt log-probs).

Checks that an exported checkpoint is the model qlab evaluated: loaded by vLLM's own loader
and run through its kernels (GPTQ-Marlin / AWQ-Marlin), its perplexity on the same windows
must match qlab's fake-quant evaluation of the same run (within 1%).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qlab.results import read_rows, write_rows  # noqa: E402


def vllm_ppl(model_dir: str, windows, gpu_util: float, extra: dict) -> tuple[float, float]:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=model_dir,
        max_model_len=windows.shape[1] + 8,
        gpu_memory_utilization=gpu_util,
        enforce_eager=True,
        **extra,
    )
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    prompts = [TokensPrompt(prompt_token_ids=w.tolist()) for w in windows]
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    nll, count = 0.0, 0
    for w, o in zip(windows.tolist(), outs, strict=True):
        for pos in range(1, len(w)):
            nll -= o.prompt_logprobs[pos][w[pos]].logprob
            count += 1
    return math.exp(nll / count), time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--qlab-csv", help="qlab results CSV holding the same run")
    ap.add_argument("--qlab-run", help="run key in --qlab-csv to compare with")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--extra", default="{}", help="JSON kwargs for vllm.LLM")
    ap.add_argument("--windows", type=int, help="score only the first N windows (smoke tests)")
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true", help="rerun a label already in --out")
    args = ap.parse_args()
    out = Path(args.out)
    if not args.force and out.exists() and any(r["label"] == args.label for r in read_rows(out)):
        print(f"{args.label}: already in {out}, skipping (--force reruns it)", flush=True)
        return 0

    from transformers import AutoTokenizer

    from qlab.data import wikitext2

    windows = wikitext2(AutoTokenizer.from_pretrained(args.tokenizer))
    if args.windows:
        windows = windows[: args.windows]
    extra = {"max_num_seqs": args.max_num_seqs, **json.loads(args.extra)}
    ppl, secs = vllm_ppl(args.model_dir, windows, args.gpu_util, extra)
    row = {
        "label": args.label,
        "model_dir": args.model_dir,
        "vllm_wikitext2_ppl": round(ppl, 4),
        "seconds": round(secs, 1),
    }
    if args.qlab_csv and args.qlab_run:
        ref = next(r for r in read_rows(args.qlab_csv) if r["run"] == args.qlab_run)
        q = float(ref["wikitext2_ppl"])
        row |= {
            "qlab_run": args.qlab_run,
            "qlab_wikitext2_ppl": q,
            "diff_pct": round(100 * (ppl - q) / q, 3),
        }
    rows = [r for r in read_rows(out) if r["label"] != args.label] if out.exists() else []
    rows.append(row)
    write_rows(out, rows, {"script": "bench/vllm_ppl.py"})
    print(row, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
