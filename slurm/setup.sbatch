#!/usr/bin/env bash
#SBATCH --job-name=qil-setup
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/qil/logs/%x-%j.out
# One-time (CPU node): venvs on top of the vLLM image, models and evaluation data in the
# scratch caches. Compute nodes reach PyPI and Hugging Face through the cluster proxy.
#   cd /scratch/$USER/qil/src && sbatch slurm/setup.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
make_venv() {  # <name> <pip args...>
  apptainer exec --bind "/scratch/$USER" "$QIL_SIF" bash -c '
    set -e
    [ -d "$QIL_ROOT/venv-$1" ] || python3 -m venv --system-site-packages "$QIL_ROOT/venv-$1"
    source "$QIL_ROOT/venv-$1/bin/activate"
    shift; eval "$@"' _ "$@"
}
FREEZE='pip freeze | grep -v " @ " > /tmp/qil-base.txt'
make_venv main "$FREEZE && pip install -c /tmp/qil-base.txt -r docker/requirements.lock && pip install --no-deps -r docker/requirements-nodeps.txt"
make_venv ref "$FREEZE && pip install -c /tmp/qil-base.txt -r docker/requirements.lock && pip install --no-deps -r docker/requirements-nodeps.txt && pip install llmcompressor==0.15.0"
in_image main python3 - <<'PY'
import torch, triton, transformers, vllm
print("main:", torch.__version__, triton.__version__, transformers.__version__, vllm.__version__)
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-8B-Base"]:
    snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt"])
    print("downloaded", m)
from transformers import AutoTokenizer
from qlab.data import c4_calib, c4_val, wikitext2
for m in ["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-8B-Base"]:
    tok = AutoTokenizer.from_pretrained(m)
    print(m, wikitext2(tok).shape, c4_val(tok).shape, c4_calib(tok).shape, c4_calib(tok, seqlen=512).shape)
PY
in_image main pip freeze > "$QIL_DATA/logs/pip-freeze-main.txt"
in_image ref python3 -c "import llmcompressor, transformers; print('ref:', llmcompressor.__version__, transformers.__version__)"
