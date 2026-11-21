#!/usr/bin/env bash
# M5 step 4 (Explorer, inside the image): export the 8B checkpoints from step 2, check vLLM
# serves the same model (perplexity parity), then the serving benchmark on this GPU.
# The FP8 reference checkpoint is written first with the ref venv (slurm/gpu.sbatch, QIL_VENV=ref):
#   python3 scripts/reference_llmc.py --model Qwen/Qwen3-8B-Base --save-fp8 $QIL_DATA/export/qwen3-8b-fp8-dynamic
set -euo pipefail
export VLLM_LOGGING_LEVEL=INFO
tag=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | grep -o -i -E "a100|h100|h200" | tr A-Z a-z || echo gpu)
X=$QIL_DATA/export
M=results/explorer/m5_methods_8b.csv
python3 -m qlab export "$QIL_DATA/ckpt/qwen3-8b/gptq-w4g128-sym.pt" "$X/qwen3-8b-gptq-w4g128-sym" --tokenizer Qwen/Qwen3-8B-Base
python3 -m qlab export "$QIL_DATA/ckpt/qwen3-8b/awq-w4g128-asym.pt" "$X/qwen3-8b-awq-w4g128-asym" --tokenizer Qwen/Qwen3-8B-Base
ppl() {  # <label> <model dir> <qlab run key>
  python3 bench/vllm_ppl.py --label "$1" --model-dir "$2" --tokenizer Qwen/Qwen3-8B-Base \
    --qlab-csv "$M" --qlab-run "$3" --gpu-util 0.85 --out results/explorer/m5_vllm_ppl_8b.csv
}
ppl "bf16" Qwen/Qwen3-8B-Base "baseline|16-bit|16-bit|"
ppl "qlab GPTQ int4 g128 sym" "$X/qwen3-8b-gptq-w4g128-sym" "gptq|int4 g128 sym|16-bit|save=qwen3-8b/gptq-w4g128-sym.pt"
ppl "qlab AWQ int4 g128 asym" "$X/qwen3-8b-awq-w4g128-asym" "awq|int4 g128 asym|16-bit|save=qwen3-8b/awq-w4g128-asym.pt"
sed "s#/scratch/USER/#/scratch/$USER/#g" configs/m5/serve_8b.json > "$QIL_DATA/serve_8b.json"
python3 bench/serve_bench.py --models "$QIL_DATA/serve_8b.json" --max-model-len 8192 --gpu-util 0.9 \
  --out "results/explorer/m5_serve_8b_${tag}.csv"
