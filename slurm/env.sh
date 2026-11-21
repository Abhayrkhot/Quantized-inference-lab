# Sourced by every job: paths, caches, and `in_image` (run a command in the vLLM image with
# qlab's venv). The image is the same vLLM release as the box's Docker image
# (vllm/vllm-openai:v0.34.0), and the venv adds the same pinned extras
# (docker/requirements.lock), so torch / triton / transformers / vLLM match the box.
export QIL_ROOT=${QIL_ROOT:-/scratch/$USER/qil}
export QIL_SIF=${QIL_SIF:-/scratch/$USER/icp/vllm.sif}
export QIL_DATA=$QIL_ROOT/data HF_HOME=$QIL_ROOT/hf HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME=$QIL_ROOT/cache TRITON_CACHE_DIR=$QIL_ROOT/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$QIL_ROOT/cache/inductor PIP_CACHE_DIR=$QIL_ROOT/pip-cache
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1  # ~/.local has an old torch
export TRANSFORMERS_VERBOSITY=error HF_DATASETS_DISABLE_PROGRESS_BARS=1
export no_proxy=localhost,127.0.0.1${no_proxy:+,$no_proxy} NO_PROXY=localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}
export QIL_HOST=$(hostname -s)
# The scratch copy is a `git archive` of one commit, recorded in GIT_SHA next to the code.
export QIL_GIT_SHA=${QIL_GIT_SHA:-$(cat "${SLURM_SUBMIT_DIR:-.}/GIT_SHA" 2>/dev/null || echo unknown)}
mkdir -p "$QIL_DATA/logs" "$HF_HOME" "$XDG_CACHE_HOME"
# The image's CUDA 13 user-mode driver needs driver >= 580; older nodes use its compat libs.
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || true)
export QIL_CUDA_COMPAT=0
if [ -n "$DRIVER" ] && [ "$DRIVER" -lt 580 ]; then QIL_CUDA_COMPAT=1; fi

# in_image <venv: main|ref> <command...>
in_image() {
  local venv=$1; shift
  apptainer exec --nv --bind "/scratch/$USER" --pwd "$PWD" "$QIL_SIF" bash -c '
    if [ "$QIL_CUDA_COMPAT" = 1 ]; then export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; fi
    source "$QIL_ROOT/venv-'"$venv"'/bin/activate"
    export PYTHONPATH=$PWD
    exec "$@"' _ "$@"
}
