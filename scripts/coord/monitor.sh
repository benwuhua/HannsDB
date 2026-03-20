#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

coord_root="$(coord_default_root)"
interval=20
once=false
heartbeat_file=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --coord-root)
      coord_root="$2"
      shift 2
      ;;
    --interval)
      interval="$2"
      shift 2
      ;;
    --once)
      once=true
      shift
      ;;
    --heartbeat-file)
      heartbeat_file="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

coord_require_dir "$coord_root/workers"
mkdir -p "$coord_root/monitor"
touch "$coord_root/events.log"

scan_once() {
  local state_file worker_id pane_id status task_id state_line snapshot_file previous
  for state_file in "$coord_root"/workers/*/state.json; do
    [[ -f "$state_file" ]] || continue
    worker_id="$(coord_extract_json_string "$state_file" "worker_id")"
    pane_id="$(coord_extract_json_string "$state_file" "pane_id")"
    status="$(coord_extract_json_string "$state_file" "status")"
    task_id="$(coord_extract_json_string "$state_file" "task_id")"
    if [[ -z "$worker_id" || -z "$pane_id" || -z "$status" ]]; then
      continue
    fi
    state_line="worker=${worker_id} pane=${pane_id} status=${status} task=${task_id}"
    snapshot_file="$coord_root/monitor/${worker_id}.last"
    previous="$(cat "$snapshot_file" 2>/dev/null || true)"
    if [[ "$state_line" != "$previous" ]]; then
      printf '%s %s\n' "$(coord_now)" "$state_line" | tee -a "$coord_root/events.log"
      printf '%s' "$state_line" >"$snapshot_file"
    fi
  done
}

if [[ "$once" == true ]]; then
  scan_once
  if [[ -n "$heartbeat_file" ]]; then
    date -u +"%Y-%m-%dT%H:%M:%SZ" >"$heartbeat_file"
  fi
  exit 0
fi

while true; do
  scan_once
  if [[ -n "$heartbeat_file" ]]; then
    date -u +"%Y-%m-%dT%H:%M:%SZ" >"$heartbeat_file"
  fi
  sleep "$interval"
done
