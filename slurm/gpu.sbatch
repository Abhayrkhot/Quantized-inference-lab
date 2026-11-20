#!/usr/bin/env bash
#SBATCH --job-name=qil-gpu
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/qil/logs/%x-%j.out
# Run one stage script on a GPU node, after the test suite passes on that GPU:
#   sbatch --constraint=a100@80g slurm/gpu.sbatch scripts/m5_methods.sh
#   sbatch --gres=gpu:h200:1 --partition=gpu slurm/gpu.sbatch scripts/m5_serve.sh
# QIL_GIT_SHA is passed with --export (the scratch copy has no .git).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
in_image main python3 -m pytest -q -p no:cacheprovider
in_image "${QIL_VENV:-main}" bash "$@"
