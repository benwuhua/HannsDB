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

pid_file="$coord_root/monitor/monitor.pid"
session_file="$coord_root/monitor/session"
if [[ -f "$session_file" ]]; then
  session_name="$(cat "$session_file" 2>/dev/null || true)"
  if [[ -n "$session_name" ]] && tmux has-session -t "$session_name" 2>/dev/null; then
    tmux kill-session -t "$session_name" 2>/dev/null || true
  fi
  rm -f "$session_file"
  printf 'monitor stopped session=%s\n' "$session_name"
  exit 0
fi

if [[ ! -f "$pid_file" ]]; then
  echo "monitor not running"
  exit 0
fi

pid="$(cat "$pid_file" 2>/dev/null || true)"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  kill "$pid" 2>/dev/null || true
fi
rm -f "$pid_file"
printf 'monitor stopped pid=%s\n' "$pid"
