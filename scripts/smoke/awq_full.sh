#!/usr/bin/env bash
# Smoke test: AWQ with the full calibration set on the 4 GB GPU (see configs/smoke/m2_awq_full.yaml).
set -euo pipefail
export TRANSFORMERS_VERBOSITY=error HF_DATASETS_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -f /data/runs/awq_full.csv
python3 -m qlab run configs/smoke/m2_awq_full.yaml --out /data/runs/awq_full.csv 2>&1 | grep -E "^\[|Error|Traceback" || true
python3 -c "import torch; print('peak note: see nvidia-smi')"
