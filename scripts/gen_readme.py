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


def t_m0_simulate(root: Path) -> str:
    rel = "results/box/m0_simulate.csv"
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


TABLES: dict[str, Callable[[Path], str]] = {
    "m0_simulate": t_m0_simulate,
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
