"""Run a YAML experiment config: load a model once, then quantize + evaluate each run.

A config lists runs; a ``sweep`` entry expands into the cartesian product of its lists::

    name: m1_rtn
    model: Qwen/Qwen3-0.6B-Base
    dtype: bfloat16
    datasets: [wikitext2, c4]
    runs:
      - {method: baseline}
      - sweep: {method: [rtn], bits: [8, 4], granularity: [channel, g128], symmetric: [true]}
      - {method: rtn, bits: 8, granularity: channel, symmetric: true, act_bits: 8}

Results are appended to the output CSV after every run, and runs whose key is already in
the CSV are skipped, so an interrupted sweep resumes where it stopped.
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from .actquant import activation_quant
from .data import DATASETS, c4_calib
from .eval.ppl import perplexity
from .formats import QuantSpec
from .model import load_model, restore, size_report, snapshot
from .results import read_rows, write_rows
from .rtn import quantize_rtn

_RUN_KEYS = ("method", "bits", "granularity", "symmetric", "act_bits", "act_granularity")


def parse_granularity(g: str) -> tuple[str, int]:
    """``"g128"`` -> ``("group", 128)``; ``"channel"`` / ``"tensor"`` pass through."""
    if g.startswith("g") and g[1:].isdigit():
        return "group", int(g[1:])
    if g in ("channel", "tensor"):
        return g, 128
    raise ValueError(f"granularity must be tensor, channel or g<N>, got {g!r}")


@dataclass(frozen=True)
class Run:
    method: str
    bits: int | None = None
    granularity: str | None = None
    symmetric: bool = False
    act_bits: int | None = None
    act_granularity: str = "token"
    params: dict = field(default_factory=dict, hash=False, compare=False)

    @classmethod
    def from_dict(cls, d: dict) -> Run:
        d = dict(d)
        known = {k: d.pop(k) for k in list(d) if k in _RUN_KEYS}
        return cls(**known, params=d)

    def spec(self) -> QuantSpec | None:
        if self.bits is None:
            return None
        gran, gs = parse_granularity(self.granularity or "channel")
        return QuantSpec(self.bits, gran, gs, self.symmetric)

    @property
    def weights_label(self) -> str:
        s = self.spec()
        return s.label if s else "16-bit"

    @property
    def acts_label(self) -> str:
        return f"int{self.act_bits} per-{self.act_granularity}" if self.act_bits else "16-bit"

    @property
    def key(self) -> str:
        extra = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.method}|{self.weights_label}|{self.acts_label}|{extra}"


def expand_runs(entries: list[dict]) -> list[Run]:
    runs: list[Run] = []
    for e in entries:
        if "sweep" in e:
            grid = {k: v if isinstance(v, list) else [v] for k, v in e["sweep"].items()}
            for combo in itertools.product(*grid.values()):
                runs.append(Run.from_dict(dict(zip(grid, combo, strict=True))))
        else:
            runs.append(Run.from_dict(e))
    keys = [r.key for r in runs]
    dupes = {k for k in keys if keys.count(k) > 1}
    if dupes:
        raise ValueError(f"duplicate runs in config: {sorted(dupes)}")
    return runs


@dataclass
class Context:
    """What a quantization method gets besides the model and the run."""

    device: torch.device
    tokenizer: object
    seqlen: int
    batch_size: int
    offload: bool
    _calib: dict = field(default_factory=dict)

    def calib(self, n: int = 128, seed: int = 0) -> torch.Tensor:
        if (n, seed) not in self._calib:
            self._calib[(n, seed)] = c4_calib(self.tokenizer, n=n, seqlen=self.seqlen, seed=seed)
        return self._calib[(n, seed)]


Method = Callable[[nn.Module, Run, Context], dict]


def _baseline(model, run, ctx):
    return {}


def _rtn(model, run, ctx):
    return quantize_rtn(model, run.spec(), ctx.device)


METHODS: dict[str, Method] = {"baseline": _baseline, "rtn": _rtn}


def run_config(
    config_path: str | Path, out: str | Path, only: list[str] | None = None, device: str = "cuda"
) -> list[dict]:
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text())
    runs = expand_runs(cfg["runs"])
    if only:
        runs = [r for r in runs if any(o in r.key for o in only)]
    out = Path(out)
    rows: list[dict] = list(read_rows(out)) if out.exists() else []
    done = {r["run"] for r in rows}
    todo = [r for r in runs if r.key not in done]
    print(f"{len(runs)} runs in config, {len(runs) - len(todo)} already in {out}", flush=True)
    if not todo:
        return rows

    dev = torch.device(device)
    offload = bool(cfg.get("offload", False))
    seqlen, bs = int(cfg.get("seqlen", 2048)), int(cfg.get("batch_size", 4))
    model, tok = load_model(cfg["model"], cfg.get("dtype", "bfloat16"), "cpu" if offload else dev)
    base = snapshot(model)
    evals = {name: DATASETS[name](tok, seqlen=seqlen) for name in cfg["datasets"]}
    ctx = Context(dev, tok, seqlen, bs, offload)
    meta = {"config": cfg, "config_path": str(config_path)}

    for i, run in enumerate(todo, 1):
        restore(model, base)
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        info = METHODS[run.method](model, run, ctx)
        t_quant = time.perf_counter() - t0
        acts = activation_quant(model, run.act_bits, run.act_granularity) if run.act_bits else None
        t0 = time.perf_counter()
        with acts or nullcontext():
            ppl = {
                f"{n}_ppl": round(perplexity(model, w, dev, bs, offload), 4)
                for n, w in evals.items()
            }
        t_eval = time.perf_counter() - t0
        size = size_report(model, run.spec())
        row = {
            "run": run.key,
            "model": cfg["model"],
            "method": run.method,
            "weights": run.weights_label,
            "acts": run.acts_label,
            "bits": run.bits or 16,
            "granularity": run.granularity or "",
            "symmetric": run.symmetric if run.bits else "",
            "bits_per_weight": round(size["linear_bytes"] * 8 / size["linear_params"], 4),
            "linear_mb": round(size["linear_bytes"] / 2**20, 2),
            "model_mb": round(size["model_bytes"] / 2**20, 2),
            **ppl,
            "weight_rel_err": info.pop("weight_rel_err", ""),
            "quant_s": round(t_quant, 1),
            "eval_s": round(t_eval, 1),
            "params": json.dumps({**run.params, **info}, sort_keys=True)
            if run.params or info
            else "",
        }
        rows.append(row)
        write_rows(out, rows, meta)
        shown = " ".join(f"{k}={v}" for k, v in ppl.items())
        print(f"[{i}/{len(todo)}] {run.key}: {shown} ({t_quant:.0f}s + {t_eval:.0f}s)", flush=True)
    return rows
