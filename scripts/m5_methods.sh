#!/usr/bin/env bash
# M5 step 2 (Explorer): methods on Qwen3-8B-Base, then zero-shot tasks.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -m qlab run configs/m5/methods_qwen3-8b.yaml --out results/explorer/m5_methods_8b.csv
python3 -m qlab run configs/m5/tasks_qwen3-8b.yaml --out results/explorer/m5_tasks_8b.csv
