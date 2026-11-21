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
import os
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from . import awq, gptq, smoothquant
from .actquant import activation_quant
from .data import DATASETS, c4_calib
from .eval.ppl import perplexity
from .formats import QuantSpec
from .kvquant import kv_bytes_per_token, kv_quant
from .model import decoder, load_model, restore, size_report, snapshot
from .results import read_rows, write_rows
from .rtn import quantize_rtn
from .sequential import capture, run_blocks

_RUN_KEYS = ("method", "bits", "granularity", "symmetric", "fmt", "act_bits", "act_granularity")


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
    fmt: str = "int"
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
        return QuantSpec(self.bits, gran, gs, self.symmetric, self.fmt)

    @property
    def weights_label(self) -> str:
        s = self.spec()
        return s.label if s else "16-bit"

    @property
    def acts_label(self) -> str:
        return f"int{self.act_bits} per-{self.act_granularity}" if self.act_bits else "16-bit"

    @property
    def kv_label(self) -> str:
        b = self.params.get("kv_bits")
        return f"int{b} {self.params.get('kv_mode', 'token')}" if b else "16-bit"

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
    artifacts: dict = field(default_factory=dict)
    _calib: dict = field(default_factory=dict)

    def calib(self, run: Run, n: int = 128, seqlen: int | None = None) -> torch.Tensor:
        """Calibration windows; a run can override ``calib_n`` / ``calib_seqlen``."""
        n = int(run.params.get("calib_n", n))
        seqlen = int(run.params.get("calib_seqlen", seqlen or self.seqlen))
        if (n, seqlen) not in self._calib:
            self._calib[(n, seqlen)] = c4_calib(self.tokenizer, n=n, seqlen=seqlen, seed=0)
        return self._calib[(n, seqlen)]


Method = Callable[[nn.Module, Run, Context], dict]


def _baseline(model, run, ctx):
    return {}


def _rtn(model, run, ctx):
    return quantize_rtn(model, run.spec(), ctx.device)


def _sequential(model, ctx, windows, before=None, after=None):
    inputs = capture(model, windows, ctx.device, ctx.batch_size)
    run_blocks(
        model, inputs, ctx.device, before_block=before, after_block=after, offload=ctx.offload
    )
    if not ctx.offload:
        model.to(ctx.device)


def _gptq(model, run, ctx):
    log: list = []
    hook = gptq.make_block_hook(
        run.spec(),
        ctx.device,
        ctx.artifacts,
        blocksize=int(run.params.get("blocksize", 128)),
        percdamp=float(run.params.get("damp", 0.01)),
        act_order=bool(run.params.get("act_order", False)),
        log=log,
    )
    _sequential(model, ctx, ctx.calib(run, n=128, seqlen=2048), before=hook)
    return {"gptq_loss_sum": round(sum(v for _, v in log), 2)}


def _awq(model, run, ctx):
    log: list = []
    hook = awq.make_block_hook(
        run.spec(),
        ctx.device,
        n_grid=int(run.params.get("grid", 20)),
        clip=bool(run.params.get("clip", True)),
        log=log,
        results=ctx.artifacts,
    )
    _sequential(model, ctx, ctx.calib(run, n=128, seqlen=512), after=hook)
    alphas = [a for _, a in log]
    return {"awq_alpha_mean": round(sum(alphas) / len(alphas), 3) if alphas else None}


def _smoothquant(model, run, ctx):
    hook = smoothquant.make_block_hook(run.spec(), ctx.device, float(run.params.get("alpha", 0.5)))
    _sequential(model, ctx, ctx.calib(run, n=128, seqlen=512), after=hook)
    return {}


METHODS: dict[str, Method] = {
    "baseline": _baseline,
    "rtn": _rtn,
    "gptq": _gptq,
    "awq": _awq,
    "smoothquant": _smoothquant,
}


def _tasks(
    model: nn.Module, tok, tasks: list[str], batch_size: int, limit: int | None = None
) -> dict[str, float]:
    """Zero-shot accuracy with lm-eval on the in-memory (quantized) model."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size)
    res = simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, bootstrap_iters=0, limit=limit)[
        "results"
    ]
    out = {}
    for t in tasks:
        for metric in ("acc", "acc_norm"):
            v = res.get(t, {}).get(f"{metric},none")
            if v is not None:
                out[f"{t}_{metric}"] = round(float(v), 4)
    return out


def _save(model: nn.Module, run: Run, ctx: Context, path: Path) -> None:
    """Integer weights (for export) plus every non-quantized tensor (scaled norms included)."""
    import os

    if not path.is_absolute():
        path = Path(os.environ.get("QIL_DATA", Path.home() / "qildata")) / "ckpt" / path
    path.parent.mkdir(parents=True, exist_ok=True)
    dec = decoder(model)
    prefix = next(n for n, m in model.named_modules() if m is dec)
    quantized = {f"{prefix}.{k}.weight" for k in ctx.artifacts}
    rest = {k: v.detach().cpu() for k, v in model.state_dict().items() if k not in quantized}
    torch.save(
        {
            "run": run.key,
            "spec": run.spec(),
            "layers": ctx.artifacts,
            "rest": rest,
            "config": model.config.to_dict(),
        },
        path,
    )
    print(f"saved {path}", flush=True)


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
    if cfg.get("eval_windows"):  # smoke tests only: score the first N windows
        evals = {n: w[: int(cfg["eval_windows"])] for n, w in evals.items()}
    ctx = Context(dev, tok, seqlen, bs, offload)
    meta = {"config": cfg, "config_path": str(config_path)}

    for i, run in enumerate(todo, 1):
        restore(model, base)
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        ctx.artifacts = {}
        info = METHODS[run.method](model, run, ctx)
        t_quant = time.perf_counter() - t0
        if run.params.get("save"):
            _save(model, run, ctx, Path(run.params["save"]))
        kv_bits = run.params.get("kv_bits")
        kv_mode = run.params.get("kv_mode", "token")
        t0 = time.perf_counter()
        with ExitStack() as stack:
            if run.act_bits:
                stack.enter_context(activation_quant(model, run.act_bits, run.act_granularity))
            if kv_bits:
                stack.enter_context(kv_quant(model, int(kv_bits), kv_mode))
            ppl = {
                f"{n}_ppl": round(perplexity(model, w, dev, bs, offload), 4)
                for n, w in evals.items()
            }
            if cfg.get("tasks"):
                ppl.update(
                    _tasks(
                        model,
                        tok,
                        cfg["tasks"],
                        int(cfg.get("task_batch_size", 16)),
                        cfg.get("task_limit"),
                    )
                )
        t_eval = time.perf_counter() - t0
        size = size_report(model, run.spec())
        row = {
            "run": run.key,
            "model": cfg["model"],
            "method": run.method,
            "weights": run.weights_label,
            "acts": run.acts_label,
            "kv": run.kv_label,
            "kv_bytes_per_token": round(kv_bytes_per_token(model.config, kv_bits, kv_mode), 1),
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
            "git": os.environ.get("QIL_GIT_SHA", "unknown"),
        }
        rows.append(row)
        write_rows(out, rows, meta)
        shown = " ".join(f"{k}={v}" for k, v in ppl.items())
        print(f"[{i}/{len(todo)}] {run.key}: {shown} ({t_quant:.0f}s + {t_eval:.0f}s)", flush=True)
    return rows
