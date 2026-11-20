#!/usr/bin/env bash
# M3 on the development box: kernel correctness on this GPU, microbenchmarks, end-to-end decode.
set -euo pipefail
export PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error
python3 -m pytest -q -p no:cacheprovider tests/test_kernel.py tests/test_env_gpu.py
python3 bench/kernels.py --out results/box/m3_kernels.csv
python3 bench/e2e_decode.py --model Qwen/Qwen3-0.6B-Base --out results/box/m3_e2e.csv
