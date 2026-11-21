#!/usr/bin/env python3
"""Serve checkpoints with vLLM one at a time and load-test each (TTFT, TPOT, throughput).

For each model: start ``vllm serve``, wait for /health, warm up, run streamed load at each
concurrency for ``--duration`` seconds (fixed prompt, ``--max-tokens`` generated with
``ignore_eos``), then stop the server and wait for its GPU memory to be freed. The KV-cache
capacity vLLM reports at startup is parsed from its log. Runs the same way in the dev
container on the box and inside the Apptainer image on Explorer.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.load_gen import measure  # noqa: E402
from qlab.results import read_rows, write_rows  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gpu_used_mb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return float(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _wait_health(url: str, proc: subprocess.Popen, timeout: float) -> None:
    import httpx

    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited with {proc.returncode}")
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError("vllm did not become healthy")


def _parse_log(text: str) -> dict:
    out = {}
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
    if m:
        out["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", text)
    if m:
        out["max_concurrency_at_len"] = float(m.group(2))
    m = re.search(r"Model loading took ([\d.]+) Gi?B", text)
    if m:
        out["weights_gib"] = float(m.group(1))
    m = re.search(r"quantization=(\w+)", text)
    if m:
        out["vllm_quantization"] = m.group(1)
    return out


def _stop(proc: subprocess.Popen) -> None:
    """Stop the server's process group; it may already have exited (failed startup)."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def _failure_reason(log_text: str, fallback: str) -> str:
    if "OutOfMemoryError" in log_text or "CUDA out of memory" in log_text:
        return "does not fit: CUDA out of memory while starting vLLM"
    if "No available memory for the cache blocks" in log_text:
        return "does not fit: no GPU memory left for the KV cache"
    return fallback


def bench_model(spec: dict, args, log_dir: Path) -> list[dict]:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    log_path = log_dir / f"vllm-{spec['label'].replace(' ', '_').replace('/', '_')}.log"
    cmd = [
        "vllm",
        "serve",
        spec["model"],
        "--port",
        str(port),
        "--served-model-name",
        "m",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_util),
        "--max-num-seqs",
        str(max(args.concurrency)),
        *spec.get("args", []),
    ]
    print(" ".join(cmd), flush=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    rows = []
    try:
        _wait_health(url, proc, args.startup_timeout)
        info = _parse_log(log_path.read_text())
        info["gpu_mem_used_mb"] = _gpu_used_mb()
        asyncio.run(measure(url, "m", 2, 10, args.max_tokens, stream=True))  # warm-up
        for c in args.concurrency:
            s = asyncio.run(measure(url, "m", c, args.duration, args.max_tokens, stream=True))
            row = {
                "label": spec["label"],
                "model": spec["model"],
                "server_args": " ".join(spec.get("args", [])),
                **info,
                **{k: v for k, v in s.items() if k != "model"},
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        _stop(proc)
        for _ in range(30):  # wait for the GPU memory to come back
            used = _gpu_used_mb()
            if used is None or used < 600:
                break
            time.sleep(2)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="JSON list of {label, model, args}")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--startup-timeout", type=float, default=900)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    specs = (
        json.loads(Path(args.models).read_text())
        if os.path.exists(args.models)
        else json.loads(args.models)
    )
    out = Path(args.out)
    log_dir = Path(os.environ.get("QIL_DATA", "/tmp")) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = list(read_rows(out)) if out.exists() else []
    done = {r["label"] for r in rows}
    for spec in specs:
        if spec["label"] in done:
            continue
        try:
            rows += bench_model(spec, args, log_dir)
        except (RuntimeError, TimeoutError) as e:
            log = log_dir / f"vllm-{spec['label'].replace(' ', '_').replace('/', '_')}.log"
            reason = _failure_reason(log.read_text() if log.exists() else "", str(e))
            rows.append({"label": spec["label"], "model": spec["model"], "error": reason})
            print(f"{spec['label']}: {e}", flush=True)
        write_rows(out, rows, {"script": "bench/serve_bench.py", "args": vars(args)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
