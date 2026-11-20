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
for pid in "$LOGS"/*.pid; do [ -f "$pid" ] && while kill -0 "$(cat "$pid")" 2>/dev/null; do sleep 30; done; done
for stage in "$@"; do
  echo "== $stage $(date -Is) load $(cut -d' ' -f1 /proc/loadavg)"
  case $stage in
    m2) scripts/dev.sh 'scripts/run_m2.sh methods' ;;
    m2extra) scripts/dev.sh 'scripts/run_m2.sh extra' ;;
    m2ref) QIL_IMAGE=qil-ref:latest scripts/dev.sh \
             'PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error python3 scripts/reference_llmc.py --out results/box/m2_reference.csv' ;;
    m3) wait_quiet; scripts/dev.sh scripts/run_m3.sh ;;
    m4ref) QIL_IMAGE=qil-ref:latest scripts/dev.sh \
             'TRANSFORMERS_VERBOSITY=error python3 scripts/reference_llmc.py --save-fp8 /data/export/qwen3-0.6b-fp8-dynamic' ;;
    m4) wait_quiet; scripts/dev.sh scripts/run_m4.sh ;;
    *) echo "unknown stage $stage"; exit 2 ;;
  esac
  echo "== $stage exit $? $(date -Is)"
done
