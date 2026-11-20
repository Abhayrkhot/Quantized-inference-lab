#!/usr/bin/env python3
"""Cross-check qlab's GPTQ / AWQ / SmoothQuant against llm-compressor (runs in qil-ref).

Same model, same calibration windows (qlab.data.c4_calib) and same evaluation code
(qlab.eval.ppl); only the quantization algorithm implementation differs. The bf16 baseline
is measured again in this image, so any difference from the main image's evaluation shows.

    QIL_IMAGE=qil-ref:latest scripts/dev.sh \
        'python3 scripts/reference_llmc.py --out results/box/m2_reference.csv'
"""

from __future__ import annotations

import argparse
import sys
import time
from importlib import metadata
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qlab.data import c4_calib, c4_val, wikitext2  # noqa: E402
from qlab.eval.ppl import perplexity  # noqa: E402
from qlab.model import load_model  # noqa: E402
from qlab.results import read_rows, write_rows  # noqa: E402

# name -> (qlab-style weights label, acts label, calib seqlen, recipe factory)


def _recipes():
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

    def gptq(scheme):
        return lambda: GPTQModifier(
            targets="Linear",
            scheme=scheme,
            ignore=["lm_head"],
            dampening_frac=0.01,
            block_size=128,
            actorder=None,
        )

    return {
        "gptq_w4g128_asym": ("int4 g128 asym", "16-bit", 2048, gptq("W4A16_ASYM")),
        "gptq_w4g128_sym": ("int4 g128 sym", "16-bit", 2048, gptq("W4A16")),
        "awq_w4g128_asym": (
            "int4 g128 asym",
            "16-bit",
            512,
            lambda: AWQModifier(
                targets=["Linear"], scheme="W4A16_ASYM", ignore=["lm_head"], duo_scaling=False
            ),
        ),
        "smoothquant_w8a8": (
            "int8 channel sym",
            "int8 per-token",
            512,
            lambda: [
                SmoothQuantModifier(smoothing_strength=0.5),
                QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
            ],
        ),
    }


def _to_gpu(model):
    try:
        from llmcompressor.utils import dispatch_for_generation

        dispatch_for_generation(model)
    except Exception:  # noqa: BLE001 - fall back to a plain move
        model.to("cuda")


def save_fp8(model_name: str, out_dir: str) -> int:
    """llm-compressor FP8_DYNAMIC: per-channel FP8 weights, per-token dynamic FP8 activations."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    model, tok = load_model(model_name, "bfloat16", "cpu")
    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
    oneshot(model=model, recipe=recipe)  # data-free
    model.save_pretrained(out_dir, save_compressed=True)
    tok.save_pretrained(out_dir)
    print(f"wrote {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--save-fp8", help="write an FP8-dynamic (W8A8) checkpoint here and exit")
    args = ap.parse_args()
    if args.save_fp8:
        return save_fp8(args.model, args.save_fp8)

    from datasets import Dataset
    from llmcompressor import oneshot

    version = metadata.version("llmcompressor")
    out = Path(args.out)
    rows = list(read_rows(out)) if out.exists() else []
    done = {(r["implementation"], r["recipe"]) for r in rows}
    recipes = _recipes()
    names = ["baseline", *recipes]
    if args.only:
        names = [n for n in names if n in args.only]

    for name in names:
        impl = "qlab eval (reference image)" if name == "baseline" else f"llm-compressor {version}"
        if (impl, name) in done:
            continue
        model, tok = load_model(args.model, "bfloat16", "cpu")
        evals = {"wikitext2": wikitext2(tok), "c4": c4_val(tok)}
        t0 = time.perf_counter()
        weights, acts = "16-bit", "16-bit"
        if name != "baseline":
            weights, acts, seqlen, make = recipes[name]
            calib = c4_calib(tok, n=128, seqlen=seqlen, seed=0)
            ds = Dataset.from_dict(
                {"input_ids": calib.tolist(), "attention_mask": torch.ones_like(calib).tolist()}
            )
            oneshot(
                model=model,
                dataset=ds,
                recipe=make(),
                max_seq_length=seqlen,
                num_calibration_samples=calib.shape[0],
            )
        t_quant = time.perf_counter() - t0
        _to_gpu(model)
        model.eval()
        ppl = {
            f"{k}_ppl": round(perplexity(model, w, "cuda", args.batch_size), 4)
            for k, w in evals.items()
        }
        rows.append(
            {
                "implementation": impl,
                "recipe": name,
                "model": args.model,
                "weights": weights,
                "acts": acts,
                **ppl,
                "quant_s": round(t_quant, 1),
            }
        )
        write_rows(out, rows, {"script": "scripts/reference_llmc.py"})
        print(f"{impl} {name}: {ppl} ({t_quant:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
