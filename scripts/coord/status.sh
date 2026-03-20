#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

coord_root="$(coord_default_root)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --coord-root)
      coord_root="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

coord_require_dir "$coord_root/workers"

monitor_pid_file="$coord_root/monitor/monitor.pid"
monitor_session_file="$coord_root/monitor/session"
monitor_heartbeat_file="$coord_root/monitor/heartbeat"
monitor_state="stopped"
monitor_pid=""
monitor_session=""
monitor_heartbeat=""
if [[ -f "$monitor_session_file" ]]; then
  monitor_session="$(cat "$monitor_session_file" 2>/dev/null || true)"
  if [[ -n "$monitor_session" ]] && tmux has-session -t "$monitor_session" 2>/dev/null; then
    monitor_state="running"
  fi
elif [[ -f "$monitor_pid_file" ]]; then
  monitor_pid="$(cat "$monitor_pid_file" 2>/dev/null || true)"
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    monitor_state="running"
  fi
fi
if [[ -f "$monitor_heartbeat_file" ]]; then
  monitor_heartbeat="$(cat "$monitor_heartbeat_file" 2>/dev/null || true)"
fi
printf 'monitor=%s session=%s pid=%s heartbeat=%s\n' "$monitor_state" "$monitor_session" "$monitor_pid" "$monitor_heartbeat"

watch_session_file="$coord_root/watch/session"
watch_heartbeat_file="$coord_root/watch/heartbeat"
watch_state="stopped"
watch_session=""
watch_heartbeat=""
if [[ -f "$watch_session_file" ]]; then
  watch_session="$(cat "$watch_session_file" 2>/dev/null || true)"
  if [[ -n "$watch_session" ]] && tmux has-session -t "$watch_session" 2>/dev/null; then
    watch_state="running"
  fi
fi
if [[ -f "$watch_heartbeat_file" ]]; then
  watch_heartbeat="$(cat "$watch_heartbeat_file" 2>/dev/null || true)"
fi
printf 'watch=%s session=%s heartbeat=%s\n' "$watch_state" "$watch_session" "$watch_heartbeat"

for state_file in "$coord_root"/workers/*/state.json; do
  [[ -f "$state_file" ]] || continue
  printf 'worker=%s pane=%s status=%s task=%s updated=%s note=%s\n' \
    "$(coord_extract_json_string "$state_file" "worker_id")" \
    "$(coord_extract_json_string "$state_file" "pane_id")" \
    "$(coord_extract_json_string "$state_file" "status")" \
    "$(coord_extract_json_string "$state_file" "task_id")" \
    "$(coord_extract_json_string "$state_file" "updated_at")" \
    "$(coord_extract_json_string "$state_file" "note")"
done
