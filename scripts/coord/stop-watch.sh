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

session_file="$coord_root/watch/session"
if [[ ! -f "$session_file" ]]; then
  echo "watch not running"
  exit 0
fi

session_name="$(cat "$session_file" 2>/dev/null || true)"
if [[ -n "$session_name" ]] && tmux has-session -t "$session_name" 2>/dev/null; then
  tmux kill-session -t "$session_name" 2>/dev/null || true
fi
rm -f "$session_file"
printf 'watch stopped session=%s\n' "$session_name"
