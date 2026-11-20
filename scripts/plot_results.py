#!/usr/bin/env python3
"""Draw the README figures from results/ CSVs (runs in the dev container; needs matplotlib).

Static PNGs on a light surface. Series colors are a fixed categorical order (validated for
colour-vision deficiency on the adjacent pairs), marker shapes give a second encoding, and
every figure has a table with the same numbers in the README.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _rows(rel: str) -> list[dict[str, str]] | None:
    p = ROOT / rel
    if not p.exists():
        return None
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2)


def m1_rtn(out_dir: Path) -> Path | None:
    rows = _rows("results/box/m1_rtn.csv")
    if not rows:
        return None
    base = next(
        float(r["wikitext2_ppl"])
        for r in rows
        if r["method"] == "baseline" and r["acts"] == "16-bit"
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    grans = ["tensor", "channel", "g128", "g64", "g32"]
    for i, g in enumerate(grans):
        pts = sorted(
            (float(r["bits_per_weight"]), float(r["wikitext2_ppl"]))
            for r in rows
            if r["method"] == "rtn"
            and r["granularity"] == g
            and r["symmetric"] == "False"
            and r["acts"] == "16-bit"
        )
        if not pts:
            continue
        xs, ys = zip(*pts, strict=True)
        label = {"tensor": "per tensor", "channel": "per channel"}.get(g, f"groups of {g[1:]}")
        ax.plot(
            xs,
            ys,
            color=SERIES[i],
            marker=MARKERS[i],
            markersize=6,
            linewidth=2,
            label=label,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
        )
    ax.axhline(base, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(f"bf16 {base:.2f}", (8.3, base), color=INK2, fontsize=9, va="bottom", ha="right")
    ax.set_yscale("log")
    _style(
        ax,
        "Qwen3-0.6B-Base: round-to-nearest, asymmetric",
        "bits per weight (incl. scales)",
        "WikiText-2 perplexity (log)",
    )
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    path = out_dir / "m1_rtn_ppl.png"
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def m1_outliers(out_dir: Path) -> Path | None:
    rows = _rows("results/box/m1_outliers_channels.csv")
    if not rows:
        return None
    names = list(dict.fromkeys(r["linear"] for r in rows))
    pick = [n for n in names if "layers.13." in n] or names[:2]
    fig, axes = plt.subplots(1, len(pick), figsize=(4.2 * len(pick), 3.8), dpi=150, squeeze=False)
    for ax, name in zip(axes[0], pick, strict=False):
        sel = [r for r in rows if r["linear"] == name]
        act = sorted((float(r["act_absmax"]) for r in sel), reverse=True)
        w = sorted((float(r["weight_absmax"]) for r in sel), reverse=True)
        a_med, w_med = act[len(act) // 2], w[len(w) // 2]
        ax.plot(
            range(len(act)),
            [a / a_med for a in act],
            color=SERIES[0],
            linewidth=2,
            label="activation (input channel)",
        )
        ax.plot(
            range(len(w)),
            [v / w_med for v in w],
            color=SERIES[1],
            linewidth=2,
            linestyle=(0, (5, 2)),
            label="weight (input column)",
        )
        ax.set_yscale("log")
        ax.set_xscale("symlog", linthresh=10)
        _style(
            ax,
            name.replace("layers.", "block "),
            "channel rank (largest first)",
            "absmax / median channel",
        )
        ax.legend(frameon=False, labelcolor=INK, fontsize=8)
    fig.tight_layout()
    path = out_dir / "m1_outliers.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


FIGURES = [m1_rtn, m1_outliers]


def main() -> int:
    out_dir = ROOT / "results" / "box"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        p = fn(out_dir)
        print(f"{fn.__name__}: {p.relative_to(ROOT) if p else 'no data yet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
