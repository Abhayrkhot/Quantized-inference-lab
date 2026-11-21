#!/usr/bin/env bash
# M4 on the development box: export qlab checkpoints, check vLLM serves the same model
# (perplexity parity), then serving benchmarks; finally the Qwen3-4B laptop demo.
# Needs the FP8 reference checkpoint from the m4ref stage (scripts/queue_box.sh).
set -euo pipefail
export PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_LOGGING_LEVEL=INFO
M2=results/box/m2_methods.csv
# One 2048-token prompt at a time, prefilled in 512-token chunks: prompt log-probs over a
# 151k vocabulary would otherwise need over a GB of logits on the 4 GB card.
PPL_ARGS=(--max-num-seqs 1 --extra '{"max_num_batched_tokens": 512, "enable_chunked_prefill": true}')
python3 -m qlab export /data/ckpt/qwen3-0.6b/gptq-w4g128-sym.pt /data/export/qwen3-0.6b-gptq-w4g128-sym --tokenizer Qwen/Qwen3-0.6B-Base
python3 -m qlab export /data/ckpt/qwen3-0.6b/awq-w4g128-asym.pt /data/export/qwen3-0.6b-awq-w4g128-asym --tokenizer Qwen/Qwen3-0.6B-Base
python3 bench/vllm_ppl.py --label "bf16" --model-dir Qwen/Qwen3-0.6B-Base \
  --qlab-csv $M2 --qlab-run "baseline|16-bit|16-bit|" "${PPL_ARGS[@]}" --out results/box/m4_vllm_ppl.csv
python3 bench/vllm_ppl.py --label "qlab GPTQ int4 g128 sym" --model-dir /data/export/qwen3-0.6b-gptq-w4g128-sym \
  --qlab-csv $M2 --qlab-run "gptq|int4 g128 sym|16-bit|save=qwen3-0.6b/gptq-w4g128-sym.pt" "${PPL_ARGS[@]}" --out results/box/m4_vllm_ppl.csv
python3 bench/vllm_ppl.py --label "qlab AWQ int4 g128 asym" --model-dir /data/export/qwen3-0.6b-awq-w4g128-asym \
  --qlab-csv $M2 --qlab-run "awq|int4 g128 asym|16-bit|save=qwen3-0.6b/awq-w4g128-asym.pt" "${PPL_ARGS[@]}" --out results/box/m4_vllm_ppl.csv
python3 bench/serve_bench.py --models configs/m4_serve_box.json --out results/box/m4_serve.csv
python3 -m qlab run configs/m4_qwen3-4b.yaml --out results/box/m4_qwen3-4b.csv
python3 -m qlab export /data/ckpt/qwen3-4b/gptq-w4g128-sym.pt /data/export/qwen3-4b-gptq-w4g128-sym --tokenizer Qwen/Qwen3-4B-Base
python3 bench/serve_bench.py --models configs/m4_serve_box_4b.json --gpu-util 0.95 --max-model-len 1024 \
  --concurrency 1 8 --out results/box/m4_serve_4b.csv
