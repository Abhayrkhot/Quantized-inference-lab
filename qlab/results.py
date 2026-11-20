"""Write result rows as CSV with a sidecar ``.meta.json`` describing where they came from."""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import platform
import subprocess
from collections.abc import Iterable, Mapping
from importlib import metadata
from pathlib import Path

_PACKAGES = ("torch", "triton", "transformers", "vllm", "datasets", "lm_eval", "numpy")


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg.replace("_", "-"))
    except metadata.PackageNotFoundError:
        return None


def _gpu() -> dict[str, object]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"gpu": None}
        props = torch.cuda.get_device_properties(0)
        info: dict[str, object] = {
            "gpu": props.name,
            "gpu_mem_gb": round(props.total_memory / 2**30, 2),
            "sm": f"{props.major}.{props.minor}",
            "cuda": torch.version.cuda,
        }
    except Exception:  # noqa: BLE001 - metadata must never break a run
        return {"gpu": None}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        info["driver"] = out.stdout.strip().splitlines()[0] if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, IndexError):
        info["driver"] = None
    return info


def environment() -> dict[str, object]:
    """Host, GPU, package versions, git revision and load average for a results file."""
    env: dict[str, object] = {
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "host": os.environ.get("QIL_HOST") or platform.node(),
        "python": platform.python_version(),
        "git": os.environ.get("QIL_GIT_SHA", "unknown"),
    }
    try:
        env["loadavg_1m"] = os.getloadavg()[0]
    except OSError:
        env["loadavg_1m"] = None
    env.update(_gpu())
    env["packages"] = {p: v for p in _PACKAGES if (v := _version(p))}
    return env


def write_rows(
    path: str | Path, rows: Iterable[Mapping[str, object]], extra_meta: Mapping | None = None
) -> Path:
    """Write ``rows`` to ``path`` (CSV) and ``path.meta.json``; returns the CSV path."""
    rows = list(rows)
    if not rows:
        raise ValueError("no rows to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    meta = {"environment": environment(), **(extra_meta or {})}
    Path(f"{path}.meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return path


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))
