#!/usr/bin/env python3
"""Regenerate the results tables in README.md and docs/*.md from the CSVs in results/.

A generated block looks like::

    <!-- qlab:begin m0_simulate -->
    ...table...
    <!-- qlab:end m0_simulate -->

Every number in those blocks comes from a results file; nothing is typed by hand.
``--check`` exits 1 if any block differs from what the results files produce (CI runs it).
Standard library only, so it runs anywhere.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ("README.md", "docs/RESULTS.md")
BLOCK = re.compile(
    r"(<!-- qlab:begin (?P<name>[\w-]+) -->\n)(?P<body>.*?)(<!-- qlab:end (?P=name) -->)", re.S
)


def _rows(root: Path, rel: str) -> list[dict[str, str]] | None:
    p = root / rel
    if not p.exists():
        return None
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def _source(root: Path, rel: str) -> str:
    meta_p = root / f"{rel}.meta.json"
    bits = [f"`{rel}`"]
    if meta_p.exists():
        env = json.loads(meta_p.read_text()).get("environment", {})
        bits.append(env.get("gpu") or "CPU")
        pk = env.get("packages", {})
        if "torch" in pk:
            bits.append(f"torch {pk['torch']}")
        bits.append(f"git {env.get('git', '?')}")
        bits.append(str(env.get("timestamp", "?"))[:10])
    return "<sub>Source: " + " · ".join(bits) + "</sub>"


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _sci(x: str) -> str:
    return f"{float(x):.2e}"


def _db(x: str) -> str:
    v = float(x)
    return "∞" if v == float("inf") else f"{v:.1f}"


def t_m0_simulate(root: Path, rel: str = "results/box/m0_simulate.csv") -> str:
    rows = _rows(root, rel)
    if rows is None:
        return "_Not run yet._"
    by = {(r["matrix"], r["format"]): r for r in rows}
    formats = [r["format"] for r in rows if r["matrix"] == "gaussian"]
    formats = [f for f in formats if f == "fp32" or f.endswith("asym")]
    body = []
    for f in formats:
        g, o = by[("gaussian", f)], by[("outliers", f)]
        body.append(
            [
                f,
                f"{float(g['bits_per_weight']):.3f}",
                _sci(g["mse"]),
                _db(g["sqnr_db"]),
                _sci(o["mse"]),
                _db(o["sqnr_db"]),
            ]
        )
    head = ["Format", "Bits/weight", "MSE (Gaussian)", "SQNR dB", "MSE (outliers)", "SQNR dB"]
    return _table(head, body) + "\n\n" + _source(root, rel)


def _ppl(x: str) -> str:
    if x in ("", None):
        return "–"
    v = float(x)
    return f"{v:.1e}" if v >= 1000 else f"{v:.2f}"


def t_m1_rtn(root: Path, rel: str = "results/box/m1_rtn.csv") -> str:
    rows = _rows(root, rel)
    if rows is None:
        return "_Not run yet._"
    base = next(r for r in rows if r["method"] == "baseline" and r["acts"] == "16-bit")
    rtn = {
        (r["granularity"], r["bits"], r["symmetric"]): r
        for r in rows
        if r["method"] == "rtn" and r["acts"] == "16-bit"
    }
    cols = [(b, sym) for b in ("8", "4", "3") for sym in ("True", "False")]
    head = ["Scales"] + [f"int{b} {'sym' if s == 'True' else 'asym'}" for b, s in cols]
    body = []
    for g, name in [
        ("tensor", "per tensor"),
        ("channel", "per channel"),
        ("g128", "groups of 128"),
        ("g64", "groups of 64"),
        ("g32", "groups of 32"),
    ]:
        cells = []
        for b, sym in cols:
            r = rtn.get((g, b, sym))
            cells.append(f"{_ppl(r['wikitext2_ppl'])} / {_ppl(r['c4_ppl'])}" if r else "–")
        body.append([name, *cells])
    note = (
        f"BF16 baseline: {_ppl(base['wikitext2_ppl'])} / {_ppl(base['c4_ppl'])}. "
        "Each cell is WikiText-2 / C4 perplexity."
    )
    return _table(head, body) + "\n\n" + note + "\n\n" + _source(root, rel)


def t_m1_w8a8(root: Path, rel: str = "results/box/m1_rtn.csv") -> str:
    rows = _rows(root, rel)
    if rows is None:
        return "_Not run yet._"
    sel = [r for r in rows if r["acts"] != "16-bit" or (r["method"] == "baseline")]
    head = ["Weights", "Activations", "WikiText-2", "C4"]
    body = [[r["weights"], r["acts"], _ppl(r["wikitext2_ppl"]), _ppl(r["c4_ppl"])] for r in sel]
    body += [
        [r["weights"], r["acts"], _ppl(r["wikitext2_ppl"]), _ppl(r["c4_ppl"])]
        for r in rows
        if r["method"] == "rtn" and r["weights"] == "int8 channel sym" and r["acts"] == "16-bit"
    ]
    return _table(head, body) + "\n\n" + _source(root, rel)


def t_m1_outliers(root: Path, rel: str = "results/box/m1_outliers_summary.csv") -> str:
    rows = _rows(root, rel)
    if rows is None:
        return "_Not run yet._"
    feeds = {
        "self_attn.q_proj": "q/k/v input (after input RMSNorm)",
        "self_attn.o_proj": "o_proj input (attention output)",
        "mlp.gate_proj": "gate/up input (after post-attention RMSNorm)",
        "mlp.down_proj": "down_proj input (SwiGLU output)",
    }
    head = [
        "Linear input",
        "Activation max/median (median over blocks)",
        "worst block",
        "Weight max/median (median over blocks)",
        "worst block",
    ]
    body = []
    for mod, label in feeds.items():
        sel = [r for r in rows if r["module"] == mod]
        if not sel:
            continue
        a = sorted(float(r["act_max_over_median"]) for r in sel)
        w = sorted(float(r["weight_max_over_median"]) for r in sel)
        body.append(
            [
                label,
                f"{a[len(a) // 2]:.0f}×",
                f"{a[-1]:.0f}×",
                f"{w[len(w) // 2]:.0f}×",
                f"{w[-1]:.0f}×",
            ]
        )
    return _table(head, body) + "\n\n" + _source(root, rel)


def t_m1_offload(root: Path, rel: str = "results/box/m1_offload.csv") -> str:
    rows = _rows(root, rel)
    if rows is None:
        return "_Not run yet._"
    head = ["Model", "Weights", "Bits/weight", "Model MB", "WikiText-2", "Eval s"]
    body = [
        [
            r["model"].split("/")[-1],
            r["weights"],
            f"{float(r['bits_per_weight']):.2f}",
            f"{float(r['model_mb']):.0f}",
            _ppl(r["wikitext2_ppl"]),
            r["eval_s"],
        ]
        for r in rows
    ]
    return _table(head, body) + "\n\n" + _source(root, rel)


def _label(r: dict[str, str]) -> str:
    """Short human label for a qlab run row."""
    params = r.get("params") or ""
    m = {
        "baseline": "BF16",
        "rtn": "RTN",
        "gptq": "GPTQ",
        "awq": "AWQ",
        "smoothquant": "SmoothQuant",
    }[r["method"]]
    extra = []
    if '"act_order": true' in params:
        extra.append("act-order")
    if '"clip": false' in params:
        extra.append("no clip search")
    if '"save": ' in params:
        extra.append("saved for export")
    if '"alpha": ' in params:
        extra.append("α=" + params.split('"alpha": ')[1].split(",")[0].split("}")[0])
    return m + (f" ({', '.join(extra)})" if extra else "")


def _find(rows, method, weights, acts="16-bit", kv="16-bit", contains=None):
    for r in rows:
        if (
            r["method"] == method
            and r["weights"] == weights
            and r["acts"] == acts
            and r.get("kv", "16-bit") == kv
            and (contains is None or contains in r["run"])
        ):
            return r
    return None


def t_m2_methods(root: Path, rel: str = "results/box/m2_methods.csv") -> str:
    rows = _rows(root, rel)
    if rows is None:
        return "_Not run yet._"
    sel = [r for r in rows if r["acts"] == "16-bit" and r.get("kv", "16-bit") == "16-bit"]
    head = ["Method", "Weights", "Bits/weight", "Model MB", "WikiText-2", "C4", "Quantize s"]
    order = {"baseline": 0, "rtn": 1, "gptq": 2, "awq": 3}
    sel.sort(
        key=lambda r: (
            -int(r["bits"]) if r["method"] != "baseline" else -99,
            r["weights"],
            order.get(r["method"], 9),
        )
    )
    body = [
        [
            _label(r),
            r["weights"],
            f"{float(r['bits_per_weight']):.2f}",
            f"{float(r['model_mb']):.0f}",
            _ppl(r["wikitext2_ppl"]),
            _ppl(r["c4_ppl"]),
            r["quant_s"],
        ]
        for r in sel
    ]
    return _table(head, body) + "\n\n" + _source(root, rel)


def t_m2_w8a8(root: Path, rel: str = "results/box/m2_methods.csv") -> str:
    rows = _rows(root, rel)
    if rows is None:
        return "_Not run yet._"
    sel = [r for r in rows if r["acts"] != "16-bit"]
    head = ["Method", "Weights", "Activations", "WikiText-2", "C4"]
    body = [
        [_label(r), r["weights"], r["acts"], _ppl(r["wikitext2_ppl"]), _ppl(r["c4_ppl"])]
        for r in sel
    ]
    return _table(head, body) + "\n\n" + _source(root, rel)


def t_m2_kv(root: Path, rel: str = "results/box/m2_methods.csv") -> str:
    rows = _rows(root, rel)
    if rows is None:
        return "_Not run yet._"
    base = _find(rows, "baseline", "16-bit")
    sel = ([base] if base else []) + [r for r in rows if r.get("kv", "16-bit") != "16-bit"]
    head = ["Weights", "KV cache", "KV bytes/token", "KV at 32k tokens", "WikiText-2", "C4"]
    body = []
    for r in sel:
        b = float(r["kv_bytes_per_token"])
        body.append(
            [
                r["weights"] if r["method"] == "baseline" else f"{_label(r)} {r['weights']}",
                r.get("kv", "16-bit"),
                f"{b:,.0f}",
                f"{b * 32768 / 2**20:,.0f} MB",
                _ppl(r["wikitext2_ppl"]),
                _ppl(r["c4_ppl"]),
            ]
        )
    return _table(head, body) + "\n\n" + _source(root, rel)


REFERENCE_PAIRS = [
    ("baseline", "BF16 (evaluation parity)", ("baseline", "16-bit", "16-bit", None)),
    (
        "gptq_w4g128_asym",
        "GPTQ int4 g128 asym",
        ("gptq", "int4 g128 asym", "16-bit", "|16-bit|16-bit|"),
    ),
    ("gptq_w4g128_sym", "GPTQ int4 g128 sym", ("gptq", "int4 g128 sym", "16-bit", None)),
    (
        "awq_w4g128_asym",
        "AWQ int4 g128 asym, no clip search",
        ("awq", "int4 g128 asym", "16-bit", "clip=False"),
    ),
    (
        "smoothquant_w8a8",
        "SmoothQuant W8A8 α=0.5",
        ("smoothquant", "int8 channel sym", "int8 per-token", "alpha=0.5"),
    ),
]


def t_m2_reference(
    root: Path,
    rel_ref: str = "results/box/m2_reference.csv",
    rel_ours: str = "results/box/m2_methods.csv",
) -> str:
    ref, ours = _rows(root, rel_ref), _rows(root, rel_ours)
    if ref is None or ours is None:
        return "_Not run yet._"
    head = [
        "Configuration",
        "qlab WikiText-2",
        "llm-compressor",
        "Δ",
        "qlab C4",
        "llm-compressor",
        "Δ",
    ]
    body = []
    for recipe, label, (method, weights, acts, contains) in REFERENCE_PAIRS:
        r = next((x for x in ref if x["recipe"] == recipe), None)
        o = _find(ours, method, weights, acts, contains=contains)
        if contains == "|16-bit|16-bit|" and o is None:
            o = _find(ours, method, weights, acts)
        if r is None or o is None:
            continue
        cells = [label]
        for d in ("wikitext2", "c4"):
            a, b = float(o[f"{d}_ppl"]), float(r[f"{d}_ppl"])
            cells += [f"{a:.3f}", f"{b:.3f}", f"{100 * (a - b) / b:+.2f}%"]
        body.append(cells)
    note = (
        "Same model, calibration windows and evaluation code; the llm-compressor runs use its "
        "own image (llm-compressor 0.15.0, transformers 5.21.0). The BF16 row is the same "
        "evaluation run in both images. The symmetric gap is a convention difference: "
        "compressed-tensors scales symmetric weights by absmax / 7.5 and uses codes -8..7, while "
        "qlab uses absmax / 7 and codes -7..7, a 7% coarser step. Where the conventions agree "
        "(asymmetric), the implementations match."
    )
    return _table(head, body) + "\n\n" + note + "\n\n" + _source(root, rel_ref)


TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande"]


def t_m2_tasks(root: Path, rel: str = "results/box/m2_tasks.csv") -> str:
    rows = _rows(root, rel)
    if rows is None:
        return "_Not run yet._"
    metric = {
        "arc_easy": "acc",
        "arc_challenge": "acc_norm",
        "hellaswag": "acc_norm",
        "piqa": "acc_norm",
        "winogrande": "acc",
    }
    head = ["Method", "Weights", "Activations"] + [f"{t} ({metric[t]})" for t in TASKS] + ["Mean"]
    body = []
    for r in rows:
        vals = [float(r[f"{t}_{metric[t]}"]) for t in TASKS if r.get(f"{t}_{metric[t]}")]
        body.append(
            [_label(r), r["weights"], r["acts"]]
            + [
                f"{100 * float(r[f'{t}_{metric[t]}']):.1f}" if r.get(f"{t}_{metric[t]}") else "–"
                for t in TASKS
            ]
            + [f"{100 * sum(vals) / len(vals):.1f}" if vals else "–"]
        )
    return _table(head, body) + "\n\n" + _source(root, rel)


def t_m2_17b(root: Path, rel: str = "results/box/m2_methods_1.7b.csv") -> str:
    rows = _rows(root, rel)
    if rows is None:
        return "_Not run yet._"
    head = [
        "Method",
        "Weights",
        "Activations",
        "KV cache",
        "Bits/weight",
        "Model MB",
        "WikiText-2",
        "Quantize s",
    ]
    body = [
        [
            _label(r),
            r["weights"],
            r["acts"],
            r.get("kv", "16-bit"),
            f"{float(r['bits_per_weight']):.2f}",
            f"{float(r['model_mb']):.0f}",
            _ppl(r["wikitext2_ppl"]),
            r["quant_s"],
        ]
        for r in rows
    ]
    return _table(head, body) + "\n\n" + _source(root, rel)


# (K, N) -> how many times that linear shape occurs in one decoder block
BLOCK_SHAPES = {
    "qwen3-0.6b": {
        (1024, 2048): 1,
        (1024, 1024): 2,
        (2048, 1024): 1,
        (1024, 3072): 2,
        (3072, 1024): 1,
    },
    "qwen3-1.7b": {(2048, 2048): 2, (2048, 1024): 2, (2048, 6144): 2, (6144, 2048): 1},
    "qwen3-8b": {(4096, 4096): 2, (4096, 1024): 2, (4096, 12288): 2, (12288, 4096): 1},
}
KERNELS = ["bf16 cuBLAS", "qlab triton", "qlab triton sym", "vLLM marlin", "bnb nf4"]


def _kernel_table(root: Path, rel: str) -> str:
    rows = _rows(root, rel)
    if rows is None:
        return "_Not run yet._"
    t = {
        (r["model"], int(r["K"]), int(r["N"]), int(r["M"]), r["kernel"]): float(r["us"])
        for r in rows
    }
    ms = sorted({int(r["M"]) for r in rows})
    head = (
        ["Model", "M (tokens)"]
        + [f"{k} µs" for k in KERNELS]
        + ["triton vs bf16", "triton sym vs marlin"]
    )
    body = []
    for model, shapes in BLOCK_SHAPES.items():
        for m in ms:
            per = {}
            for k in KERNELS:
                vals = [t.get((model, kk, nn, m, k)) for (kk, nn) in shapes]
                if all(v is not None for v in vals):
                    per[k] = sum(v * c for v, c in zip(vals, shapes.values(), strict=True))
            if "bf16 cuBLAS" not in per:
                continue
            ratio = per["bf16 cuBLAS"] / per["qlab triton"] if "qlab triton" in per else None
            vs_m = (
                per["vLLM marlin"] / per["qlab triton sym"]
                if "vLLM marlin" in per and "qlab triton sym" in per
                else None
            )
            body.append(
                [model, str(m)]
                + [f"{per[k]:.0f}" if k in per else "–" for k in KERNELS]
                + [f"{ratio:.2f}×" if ratio else "–", f"{vs_m:.2f}×" if vs_m else "–"]
            )
    note = (
        "Time for all linear layers of one decoder block (q, k, v, o, gate, up, down), summed "
        'from per-shape medians. "triton sym vs marlin" above 1× means qlab\'s kernel is '
        "faster than Marlin on the same symmetric weights."
    )
    return _table(head, body) + "\n\n" + note + "\n\n" + _source(root, rel)


def t_m3_kernels(root: Path, rel: str = "results/box/m3_kernels.csv") -> str:
    return _kernel_table(root, rel)


def t_m3_bandwidth(root: Path, rel: str = "results/box/m3_kernels.csv") -> str:
    rows = _rows(root, rel)
    if rows is None:
        return "_Not run yet._"
    sel = [r for r in rows if r["M"] == "1"]
    head = ["Model", "K × N"] + [f"{k} % of copy BW" for k in KERNELS]
    by = {(r["model"], r["K"], r["N"], r["kernel"]): r["pct_copy_bw"] for r in sel}
    body = []
    for model, shapes in BLOCK_SHAPES.items():
        for kk, nn in shapes:
            if (model, str(kk), str(nn), "bf16 cuBLAS") in by:
                body.append(
                    [model, f"{kk} × {nn}"]
                    + [by.get((model, str(kk), str(nn), k), "–") for k in KERNELS]
                )
    return _table(head, body) + "\n\n" + _source(root, rel)


def t_m3_e2e(root: Path, rel: str = "results/box/m3_e2e.csv") -> str:
    rows = _rows(root, rel)
    if rows is None:
        return "_Not run yet._"
    head = [
        "Model",
        "Linear weights",
        "Weights MB",
        "Decode tokens/s",
        "ms/token",
        "Prefill ms",
        "Peak MB",
        "Speedup",
    ]
    body = [
        [
            r["model"].split("/")[-1],
            r["weights"],
            f"{float(r['weights_mb']):.0f}",
            r["decode_tokens_per_s"],
            r["decode_ms_per_token"],
            r["prefill_ms"],
            f"{float(r['peak_mem_mb']):.0f}",
            (r.get("speedup_vs_16bit") or "1.00") + "×",
        ]
        for r in rows
    ]
    return _table(head, body) + "\n\n" + _source(root, rel)


def t_m4_parity(root: Path, rel: str = "results/box/m4_vllm_ppl.csv") -> str:
    rows = _rows(root, rel)
    if rows is None:
        return "_Not run yet._"
    head = ["Checkpoint", "vLLM WikiText-2", "qlab WikiText-2 (same run)", "Δ"]
    body = [
        [
            r["label"],
            _ppl(r["vllm_wikitext2_ppl"]),
            _ppl(r.get("qlab_wikitext2_ppl", "")),
            f"{float(r['diff_pct']):+.2f}%" if r.get("diff_pct") else "–",
        ]
        for r in rows
    ]
    return _table(head, body) + "\n\n" + _source(root, rel)


def _serve_table(root: Path, rel: str) -> str:
    rows = _rows(root, rel)
    if rows is None:
        return "_Not run yet._"
    head = [
        "Checkpoint",
        "Weights GiB",
        "KV cache tokens",
        "Concurrency",
        "TTFT p50 ms",
        "TPOT p50 ms",
        "Output tokens/s",
        "Errors",
    ]
    body = []
    for r in rows:
        if r.get("error"):
            body.append([r["label"], "–", "–", "–", "–", "–", "–", r["error"][:60]])
            continue
        ttft = r.get("ttft_p50_s")
        body.append(
            [
                r["label"],
                r.get("weights_gib", "–"),
                f"{int(float(r['kv_cache_tokens'])):,}" if r.get("kv_cache_tokens") else "–",
                r["concurrency"],
                f"{1000 * float(ttft):.0f}" if ttft else "–",
                r.get("tpot_p50_ms", "–"),
                r["tokens_per_s"],
                r["errors"],
            ]
        )
    return _table(head, body) + "\n\n" + _source(root, rel)


def t_m4_serve(root: Path, rel: str = "results/box/m4_serve.csv") -> str:
    return _serve_table(root, rel)


def t_m5_smoke(root: Path) -> str:
    rel_x, rel_b = "results/explorer/m5_smoke.csv", "results/box/m2_methods.csv"
    xr, br = _rows(root, rel_x), _rows(root, rel_b)
    if xr is None or br is None:
        return "_Not run yet._"
    b = {r["run"]: r for r in br}
    head = ["Run (Qwen3-0.6B)", "Box WikiText-2", "H200 WikiText-2", "Δ", "Box C4", "H200 C4", "Δ"]
    body = []
    for r in xr:
        if r["run"] not in b:
            continue
        cells = [_label(r) + " " + r["weights"]]
        for d in ("wikitext2", "c4"):
            x, y = float(r[f"{d}_ppl"]), float(b[r["run"]][f"{d}_ppl"])
            cells += [f"{y:.3f}", f"{x:.3f}", f"{100 * (x - y) / y:+.2f}%"]
        body.append(cells)
    return _table(head, body) + "\n\n" + _source(root, rel_x)


X = "results/explorer"
TABLES: dict[str, Callable[[Path], str]] = {
    "m4_serve_4b": partial(t_m4_serve, rel="results/box/m4_serve_4b.csv"),
    "m4_qwen3_4b": partial(t_m2_17b, rel="results/box/m4_qwen3-4b.csv"),
    "m5_smoke": t_m5_smoke,
    "m5_methods": partial(t_m2_methods, rel=f"{X}/m5_methods_8b.csv"),
    "m5_w8a8": partial(t_m2_w8a8, rel=f"{X}/m5_methods_8b.csv"),
    "m5_kv": partial(t_m2_kv, rel=f"{X}/m5_methods_8b.csv"),
    "m5_tasks": partial(t_m2_tasks, rel=f"{X}/m5_tasks_8b.csv"),
    "m5_kernels_a100": partial(t_m3_kernels, rel=f"{X}/m5_kernels_a100.csv"),
    "m5_kernels_h200": partial(t_m3_kernels, rel=f"{X}/m5_kernels_h200.csv"),
    "m5_e2e": partial(t_m3_e2e, rel=f"{X}/m5_e2e_8b_h200.csv"),
    "m5_parity": partial(t_m4_parity, rel=f"{X}/m5_vllm_ppl_8b.csv"),
    "m5_serve": partial(t_m4_serve, rel=f"{X}/m5_serve_8b_h200.csv"),
    "m3_kernels": t_m3_kernels,
    "m3_bandwidth": t_m3_bandwidth,
    "m3_e2e": t_m3_e2e,
    "m4_parity": t_m4_parity,
    "m4_serve": t_m4_serve,
    "m2_methods": t_m2_methods,
    "m2_w8a8": t_m2_w8a8,
    "m2_kv": t_m2_kv,
    "m2_reference": t_m2_reference,
    "m2_tasks": t_m2_tasks,
    "m2_17b": t_m2_17b,
    "m0_simulate": t_m0_simulate,
    "m1_rtn": t_m1_rtn,
    "m1_w8a8": t_m1_w8a8,
    "m1_outliers": t_m1_outliers,
    "m1_offload": t_m1_offload,
}


def render(text: str, root: Path = ROOT) -> str:
    def sub(m: re.Match[str]) -> str:
        name = m.group("name")
        if name not in TABLES:
            raise KeyError(f"unknown table block {name!r}")
        return f"{m.group(1)}{TABLES[name](root).rstrip()}\n{m.group(4)}"

    return BLOCK.sub(sub, text)


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if any block is stale")
    args = ap.parse_args(argv)
    stale = []
    for rel in DOCS:
        p = root / rel
        if not p.exists():
            continue
        old = p.read_text()
        new = render(old, root)
        if new != old:
            stale.append(rel)
            if not args.check:
                p.write_text(new)
    if args.check and stale:
        print("stale generated blocks in: " + ", ".join(stale), file=sys.stderr)
        print("run: python scripts/gen_readme.py", file=sys.stderr)
        return 1
    for rel in stale:
        print(f"updated {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
