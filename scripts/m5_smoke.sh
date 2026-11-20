#!/usr/bin/env bash
# M5 step 1 (Explorer, inside the image): the box's Qwen3-0.6B GPTQ config, compared to the box.
set -euo pipefail
python3 -m qlab run configs/m5/smoke_qwen3-0.6b.yaml --out results/explorer/m5_smoke.csv
python3 scripts/compare_runs.py results/explorer/m5_smoke.csv results/box/m2_methods.csv --tol-pct 0.5
