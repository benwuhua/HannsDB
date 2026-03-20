#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

coord_root="$(coord_default_root)"
declare -a workers=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --coord-root)
      coord_root="$2"
      shift 2
      ;;
    --worker)
      workers+=("$2")
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p \
  "$coord_root/tasks" \
  "$coord_root/results" \
  "$coord_root/workers" \
  "$coord_root/monitor"
touch "$coord_root/events.log"

if [[ ${#workers[@]} -eq 0 ]]; then
  workers=("worker1:%1" "worker2:%2")
fi

for worker_spec in "${workers[@]}"; do
  worker_id="${worker_spec%%:*}"
  pane_id="${worker_spec#*:}"
  mkdir -p "$coord_root/workers/$worker_id"
  coord_write_state "$coord_root" "$worker_id" "$pane_id" "idle" "" "initialized"
done

printf 'coord initialized at %s\n' "$coord_root"
