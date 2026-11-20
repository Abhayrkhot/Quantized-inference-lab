#!/usr/bin/env bash
# M5 step 3 (Explorer): kernel microbenchmarks and end-to-end 8B decode on this GPU.
set -euo pipefail
tag=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | grep -o -i -E "a100|h100|h200|l40s|v100" | tr A-Z a-z || echo gpu)
python3 bench/kernels.py --out results/explorer/m5_kernels_${tag}.csv
python3 bench/e2e_decode.py --model Qwen/Qwen3-8B-Base --out results/explorer/m5_e2e_8b_${tag}.csv
python3 bench/e2e_decode.py --model Qwen/Qwen3-0.6B-Base --out results/explorer/m5_e2e_0.6b_${tag}.csv
