#!/usr/bin/env bash
# M2 on the development box (inside the dev container).
#   scripts/run_m2.sh methods   Qwen3-0.6B methods sweep + saved checkpoints (M4 needs these)
#   scripts/run_m2.sh extra     zero-shot tasks, and the methods on Qwen3-1.7B (block offload)
# The llm-compressor cross-check runs separately in qil-ref (scripts/queue_box.sh m2ref).
set -euo pipefail
export PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case ${1:-methods} in
  methods)
    python3 -m pytest -q -p no:cacheprovider
    python3 -m qlab run configs/m2_methods_qwen3-0.6b.yaml --out results/box/m2_methods.csv ;;
  extra)
    python3 -m qlab run configs/m2_tasks_qwen3-0.6b.yaml --out results/box/m2_tasks.csv
    python3 -m qlab run configs/m2_methods_qwen3-1.7b.yaml --out results/box/m2_methods_1.7b.csv ;;
  *) echo "usage: run_m2.sh methods|extra"; exit 2 ;;
esac
