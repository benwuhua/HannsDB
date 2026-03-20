#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

coord_root="$(coord_default_root)"
interval=20

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
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$coord_root/monitor"
session_file="$coord_root/monitor/session"
out_file="$coord_root/monitor/monitor.out"
heartbeat_file="$coord_root/monitor/heartbeat"
monitor_script="$SCRIPT_DIR/monitor.sh"
repo_root="$(coord_repo_root)"
session_suffix="$(printf '%s' "$coord_root" | cksum | awk '{print $1}')"
session_name="coord-monitor-${session_suffix}"

if [[ -f "$session_file" ]]; then
  existing_session="$(cat "$session_file" 2>/dev/null || true)"
  if [[ -n "$existing_session" ]] && tmux has-session -t "$existing_session" 2>/dev/null; then
    printf 'monitor already running session=%s\n' "$existing_session"
    exit 0
  fi
fi

printf '%s' "$session_name" >"$session_file"

tmux new-session -d -s "$session_name" \
  "cd '$repo_root' && bash '$monitor_script' --coord-root '$coord_root' --interval '$interval' --heartbeat-file '$heartbeat_file' >> '$out_file' 2>&1"

printf 'monitor started session=%s\n' "$session_name"
