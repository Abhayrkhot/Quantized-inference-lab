#!/usr/bin/env bash
# Run milestone stages one after another on the box (host side), each in its container.
#   setsid nohup scripts/queue_box.sh m2 m3 m4ref m4 m2extra m2ref > ~/qildata/logs/queue.log 2>&1 &
# Waits for any GPU job started by scripts/bg.sh to finish first, and gates timing stages on a
# quiet machine (1-minute load average below QIL_MAX_LOAD, default 3).
set -uo pipefail
cd "$(dirname "$0")/.."
LOGS=${QIL_DATA:-$HOME/qildata}/logs
mkdir -p "$LOGS"
wait_quiet() {
  for _ in $(seq 1 120); do
    awk -v m="${QIL_MAX_LOAD:-3}" '{exit !($1 < m)}' /proc/loadavg && return 0
    sleep 30
  done
  echo "load still high: $(cat /proc/loadavg); continuing" >&2
}
# Run from a clean clone at one commit, so edits synced into the working copy while the queue
# runs cannot change the code under it, and every result carries a clean git revision.
# Results are copied back to the working copy after each stage.
REPO=$(pwd)
SHA=$(git rev-parse HEAD)
RUN=${QIL_RUN_DIR:-$HOME/qil-run/${SHA:0:7}}
[ -d "$RUN/.git" ] || git clone -q "$REPO" "$RUN"
git -C "$RUN" fetch -q "$REPO" && git -C "$RUN" checkout -q --detach "$SHA"
mkdir -p "$RUN/results" && rsync -a "$REPO/results/" "$RUN/results/"
cd "$RUN"
echo "== queue at $SHA in $RUN"
for pid in "$LOGS"/*.pid; do [ -f "$pid" ] && while kill -0 "$(cat "$pid")" 2>/dev/null; do sleep 30; done; done
wait_gpu_free() {  # another qlab container (e.g. a stage of an earlier queue) still running
  while docker ps -q --filter ancestor=qil-dev:latest --filter ancestor=qil-ref:latest | grep -q .; do
    sleep 30
  done
}
LOCK=$LOGS/gpu.lock  # one stage at a time on the GPU, across queues
for stage in "$@"; do
  exec 9>"$LOCK"
  flock 9
  wait_gpu_free
  echo "== $stage $(date -Is) load $(cut -d' ' -f1 /proc/loadavg)"
  case $stage in
    m2) scripts/dev.sh 'scripts/run_m2.sh methods' ;;
    m2extra) scripts/dev.sh 'scripts/run_m2.sh extra' ;;
    m2ref) QIL_IMAGE=qil-ref:latest scripts/dev.sh \
             'export PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
              for r in baseline gptq_w4g128_asym gptq_w4g128_sym awq_w4g128_asym smoothquant_w8a8; do
                python3 scripts/reference_llmc.py --only $r --out results/box/m2_reference.csv || rc=1
              done; exit ${rc:-0}' ;;  # one process per recipe: llm-compressor keeps state between oneshot calls
    m3) wait_quiet; scripts/dev.sh scripts/run_m3.sh ;;
    m4ref) QIL_IMAGE=qil-ref:latest scripts/dev.sh \
             'TRANSFORMERS_VERBOSITY=error python3 scripts/reference_llmc.py --save-fp8 /data/export/qwen3-0.6b-fp8-dynamic' ;;
    m4) wait_quiet; scripts/dev.sh scripts/run_m4.sh ;;
    *) echo "unknown stage $stage"; exit 2 ;;
  esac
  rc=$?
  rsync -a "$RUN/results/" "$REPO/results/"
  echo "== $stage exit $rc $(date -Is)"
  flock -u 9
done
