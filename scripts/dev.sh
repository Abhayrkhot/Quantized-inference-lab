#!/usr/bin/env bash
# Run a command in the qil-dev container on this machine (the box) with the GPU.
#   scripts/dev.sh 'pytest -q'
# Env: QIL_GPU=0 (no GPU), QIL_IMAGE (default qil-dev:latest), QIL_DATA (default ~/qildata:
# checkpoints and caches), HF cache is the host's ~/.cache/huggingface.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${QIL_IMAGE:-qil-dev:latest}
DATA=${QIL_DATA:-$HOME/qildata}
HFC=${QIL_HF:-$HOME/.cache/huggingface}
mkdir -p "$DATA/cache" "$DATA/ckpt" "$HFC"
sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
git diff --quiet HEAD -- . ':!results' 2>/dev/null || sha="$sha-dirty"   # tracked code changes only
gpu=(); [ "${QIL_GPU:-1}" = 1 ] && gpu=(--gpus all)
tty=(); [ -t 0 ] && [ -t 1 ] && tty=(-it)
exec docker run --rm "${tty[@]}" "${gpu[@]}" --ipc=host --network host \
  --user "$(id -u):$(id -g)" -e HOME=/tmp/home \
  -e HF_HOME=/hf -e HF_HUB_DISABLE_TELEMETRY=1 -e XDG_CACHE_HOME=/data/cache \
  -e TRITON_CACHE_DIR=/data/cache/triton -e TORCHINDUCTOR_CACHE_DIR=/data/cache/inductor \
  -e QIL_GIT_SHA="$sha" -e QIL_HOST="$(hostname -s)" -e QIL_DATA=/data \
  -e PYTHONPATH=/src -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD":/src -v "$DATA":/data -v "$HFC":/hf -w /src \
  "$IMAGE" bash -c 'mkdir -p "$HOME" && eval "$1"' _ "$*"
