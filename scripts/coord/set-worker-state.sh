#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

coord_root="$(coord_default_root)"
worker_id=""
status=""
task_id=""
note=""
pane_id=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --coord-root)
      coord_root="$2"
      shift 2
      ;;
    --worker)
      worker_id="$2"
      shift 2
      ;;
    --status)
      status="$2"
      shift 2
      ;;
    --task-id)
      task_id="$2"
      shift 2
      ;;
    --note)
      note="$2"
      shift 2
      ;;
    --pane-id)
      pane_id="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$worker_id" || -z "$status" ]]; then
  echo "--worker and --status are required" >&2
  exit 2
fi

if [[ -z "$pane_id" ]]; then
  pane_id="$(coord_load_pane_id "$coord_root" "$worker_id")"
fi

coord_write_state "$coord_root" "$worker_id" "$pane_id" "$status" "$task_id" "$note"
printf 'worker=%s status=%s task=%s\n' "$worker_id" "$status" "$task_id"
