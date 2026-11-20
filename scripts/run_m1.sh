#!/usr/bin/env bash
# M1 on the development box: RTN sweep on Qwen3-0.6B, activation/weight outlier statistics,
# and the block-offload path on Qwen3-1.7B. Runs inside the dev container.
set -euo pipefail
export PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error HF_DATASETS_DISABLE_PROGRESS_BARS=1
python3 -m pytest -q -p no:cacheprovider
python3 -m qlab run configs/m1_rtn_qwen3-0.6b.yaml --out results/box/m1_rtn.csv
python3 -m qlab outliers --model Qwen/Qwen3-0.6B-Base --out-prefix results/box/m1_outliers
python3 -m qlab run configs/m1_offload_qwen3-1.7b.yaml --out results/box/m1_offload.csv
