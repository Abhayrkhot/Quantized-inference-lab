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


def _cmd_run(args: argparse.Namespace) -> int:
    from .experiments import run_config

    run_config(args.config, args.out, only=args.only, device=args.device)
    return 0


def _cmd_outliers(args: argparse.Namespace) -> int:
    from .data import c4_calib
    from .model import load_model
    from .outliers import channel_rows, channel_stats, summarize
    from .results import write_rows

    model, tok = load_model(args.model, args.dtype, args.device)
    windows = c4_calib(tok, n=args.n, seqlen=args.seqlen, seed=1)
    stats = channel_stats(model, windows, args.device, batch_size=args.batch_size)
    meta = {"command": "outliers", "model": args.model, "n": args.n, "seqlen": args.seqlen}
    write_rows(f"{args.out_prefix}_summary.csv", summarize(stats), meta)
    detail = [f"layers.{i}.{m}" for i in args.detail_layers for m in args.detail_modules]
    write_rows(f"{args.out_prefix}_channels.csv", channel_rows(stats, detail), meta)
    print(f"wrote {args.out_prefix}_summary.csv and _channels.csv")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from .export import export

    out = export(args.ckpt, args.out_dir, args.tokenizer, args.format)
    print(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qlab", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="quantize synthetic matrices (Gaussian and outliers)")
    s.add_argument("--shape", type=int, nargs=2, default=[1024, 4096], metavar=("OUT", "IN"))
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", help="write rows to this CSV (plus .meta.json)")
    s.set_defaults(func=_cmd_simulate)

    r = sub.add_parser("run", help="run an experiment config (quantize + evaluate)")
    r.add_argument("config")
    r.add_argument("--out", required=True)
    r.add_argument("--only", nargs="*", help="only runs whose key contains one of these")
    r.add_argument("--device", default="cuda")
    r.set_defaults(func=_cmd_run)

    o = sub.add_parser("outliers", help="per-channel activation vs weight magnitudes")
    o.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    o.add_argument("--dtype", default="bfloat16")
    o.add_argument("--device", default="cuda")
    o.add_argument("--n", type=int, default=32, help="calibration windows")
    o.add_argument("--seqlen", type=int, default=2048)
    o.add_argument("--batch-size", type=int, default=2)
    o.add_argument("--detail-layers", type=int, nargs="*", default=[0, 13, 27])
    o.add_argument("--detail-modules", nargs="*", default=["self_attn.q_proj", "mlp.down_proj"])
    o.add_argument("--out-prefix", required=True)
    o.set_defaults(func=_cmd_outliers)

    e = sub.add_parser("export", help="write a saved run as a vLLM-loadable GPTQ/AWQ checkpoint")
    e.add_argument("ckpt", help="file written by a run with `save:`")
    e.add_argument("out_dir")
    e.add_argument("--tokenizer", required=True)
    e.add_argument("--format", default="auto", choices=["auto", "gptq", "awq"])
    e.set_defaults(func=_cmd_export)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
