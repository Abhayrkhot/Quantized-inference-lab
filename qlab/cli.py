"""Command-line entry point: ``python -m qlab <command>``."""

from __future__ import annotations

import argparse
import sys


def _cmd_simulate(args: argparse.Namespace) -> int:
    from . import simulate
    from .results import write_rows

    rows = simulate.run(shape=tuple(args.shape), seed=args.seed)
    if args.out:
        write_rows(args.out, rows, {"command": "simulate", "shape": args.shape, "seed": args.seed})
    print(f"{'matrix':9} {'format':20} {'bpw':>7} {'bytes':>10} {'mse':>11} {'sqnr_db':>8}")
    for r in rows:
        print(
            f"{r['matrix']:9} {r['format']:20} {r['bits_per_weight']:7.3f} {r['bytes']:10d} "
            f"{r['mse']:11.3e} {r['sqnr_db']:8.2f}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qlab", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="quantize synthetic matrices (Gaussian and outliers)")
    s.add_argument("--shape", type=int, nargs=2, default=[1024, 4096], metavar=("OUT", "IN"))
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", help="write rows to this CSV (plus .meta.json)")
    s.set_defaults(func=_cmd_simulate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
