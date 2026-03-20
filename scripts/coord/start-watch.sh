#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

coord_root="$(coord_default_root)"
interval=20
notify_session="${COORD_NOTIFY_SESSION:-HannsDB}"

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
    --notify-session)
      notify_session="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$coord_root/watch"
session_file="$coord_root/watch/session"
out_file="$coord_root/watch/watch.out"
heartbeat_file="$coord_root/watch/heartbeat"
watch_script="$SCRIPT_DIR/watch.sh"
repo_root="$(coord_repo_root)"
session_suffix="$(printf '%s' "$coord_root" | cksum | awk '{print $1}')"
session_name="coord-watch-${session_suffix}"

if [[ -f "$session_file" ]]; then
  existing_session="$(cat "$session_file" 2>/dev/null || true)"
  if [[ -n "$existing_session" ]] && tmux has-session -t "$existing_session" 2>/dev/null; then
    printf 'watch already running session=%s\n' "$existing_session"
    exit 0
  fi
fi

printf '%s' "$session_name" >"$session_file"

tmux new-session -d -s "$session_name" \
  "cd '$repo_root' && bash '$watch_script' --coord-root '$coord_root' --interval '$interval' --notify-session '$notify_session' --heartbeat-file '$heartbeat_file' >> '$out_file' 2>&1"

printf 'watch started session=%s\n' "$session_name"
