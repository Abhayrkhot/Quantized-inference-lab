#!/usr/bin/env bash
# Start a long job in the dev container, detached from the SSH session, on the box.
#   scripts/bg.sh <name> '<cmd>'   -> log ~/qildata/logs/<name>.log, pid ~/qildata/logs/<name>.pid
#   scripts/bg.sh --wait <name>     -> block until it exits, then print the log tail and exit code
set -euo pipefail
cd "$(dirname "$0")/.."
LOGS=${QIL_DATA:-$HOME/qildata}/logs
mkdir -p "$LOGS"
if [ "$1" = --wait ]; then
  pid=$(cat "$LOGS/$2.pid")
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  tail -n "${TAIL:-25}" "$LOGS/$2.log"
  exit "$(cat "$LOGS/$2.rc" 2>/dev/null || echo 1)"
fi
name=$1; shift
rm -f "$LOGS/$name.rc"
setsid nohup bash -c 'scripts/dev.sh "$1"; echo $? > "$2"' _ "$*" "$LOGS/$name.rc" \
  > "$LOGS/$name.log" 2>&1 < /dev/null &
echo $! > "$LOGS/$name.pid"
echo "started $name (pid $!), log $LOGS/$name.log"
