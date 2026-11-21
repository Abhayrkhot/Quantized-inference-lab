#!/usr/bin/env bash
# M5 (Explorer, ref venv): llm-compressor FP8-dynamic checkpoint of Qwen3-8B for the serving benchmark.
set -euo pipefail
python3 scripts/reference_llmc.py --model Qwen/Qwen3-8B-Base --save-fp8 "$QIL_DATA/export/qwen3-8b-fp8-dynamic"
