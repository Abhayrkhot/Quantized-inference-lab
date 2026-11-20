#!/usr/bin/env python3
"""Check that runs present in two results CSVs agree on perplexity within a tolerance.

    python3 scripts/compare_runs.py results/explorer/m5_smoke.csv results/box/m2_methods.csv

Exits 1 if any shared run differs by more than ``--tol-pct`` percent, or if no run is shared.
"""

from __future__ import annotations

import argparse
import csv
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol-pct", type=float, default=0.5)
    args = ap.parse_args()
    with open(args.a, newline="") as fa, open(args.b, newline="") as fb:
        a = {r["run"]: r for r in csv.DictReader(fa)}
        b = {r["run"]: r for r in csv.DictReader(fb)}
    shared = [k for k in a if k in b]
    if not shared:
        print("no shared runs", file=sys.stderr)
        return 1
    bad = 0
    for k in shared:
        for col in ("wikitext2_ppl", "c4_ppl"):
            if not a[k].get(col) or not b[k].get(col):
                continue
            x, y = float(a[k][col]), float(b[k][col])
            d = 100 * (x - y) / y
            ok = abs(d) <= args.tol_pct
            bad += not ok
            print(f"{'ok ' if ok else 'BAD'} {k} {col}: {x:.4f} vs {y:.4f} ({d:+.3f}%)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
