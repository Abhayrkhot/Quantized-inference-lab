#!/usr/bin/env bash
# Host side, on the box: wait for the remaining queue stages (M4, the llm-compressor
# cross-check) to end, collect their results from the per-commit queue clones, redraw the
# figures, regenerate the README tables, then commit and push. Nothing is committed if an
# expected results file is missing; the log says which.
#   setsid nohup scripts/finish_box.sh > ~/qildata/logs/finish.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
busy() {
  pgrep -f "bash scripts/queue_box.sh" >/dev/null ||
    docker ps -q --filter ancestor=qil-dev:latest --filter ancestor=qil-ref:latest | grep -q .
}
while busy; do sleep 60; done
echo "queues idle at $(date -Is)"
for d in "$HOME"/qil-run/*/results; do rsync -a --update "$d/" results/; done
missing=0
for f in m4_vllm_ppl m4_serve m4_qwen3-4b m4_serve_4b m2_reference; do
  [ -s "results/box/$f.csv" ] || { echo "missing results/box/$f.csv"; missing=1; }
done
[ "$(grep -c '' results/box/m2_reference.csv 2>/dev/null)" -ge 6 ] || { echo "m2_reference incomplete"; missing=1; }
QIL_GPU=0 scripts/dev.sh 'python3 scripts/plot_results.py'
python3 scripts/gen_readme.py && python3 scripts/gen_readme.py --check || exit 1
if [ "$missing" = 1 ]; then echo "not committing"; exit 1; fi
export GIT_AUTHOR_NAME="Abhay Khot" GIT_AUTHOR_EMAIL="130841999+Abhayrkhot@users.noreply.github.com"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
git add README.md results/box
git commit -q -m "M4 results and the llm-compressor cross-check on the box

vLLM perplexity parity for the exported GPTQ / AWQ checkpoints, serving benchmarks for
BF16 / GPTQ-Marlin / AWQ-Marlin / FP8 (with and without an fp8 KV cache), the Qwen3-4B
GPTQ export served on the 4 GB card, and llm-compressor GPTQ / AWQ / SmoothQuant run on the
same calibration windows and scored by the same evaluation. README tables regenerated." || exit 1
git log --format="%h %ae %ce | %s" -1
git push -q origin master && echo pushed
