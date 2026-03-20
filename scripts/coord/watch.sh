#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

coord_root="$(coord_default_root)"
interval=20
once=false
notify_session="${COORD_NOTIFY_SESSION:-HannsDB}"
heartbeat_file=""
remind_seconds=120

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
    --notify-session)
      notify_session="$2"
      shift 2
      ;;
    --heartbeat-file)
      heartbeat_file="$2"
      shift 2
      ;;
    --remind-seconds)
      remind_seconds="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

coord_require_dir "$coord_root/workers"
mkdir -p "$coord_root/watch"
touch "$coord_root/alerts.log"

parse_utc_epoch() {
  local ts="$1"
  if date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "$ts" "+%s" >/dev/null 2>&1; then
    date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "$ts" "+%s"
    return 0
  fi
  if date -u -d "$ts" "+%s" >/dev/null 2>&1; then
    date -u -d "$ts" "+%s"
    return 0
  fi
  return 1
}

scan_once() {
  local state_file worker_id pane_id status task_id updated_at key prev line
  local now_epoch updated_epoch overdue_stamp overdue_prev_task overdue_prev_epoch
  local overdue_line
  now_epoch="$(date -u +%s)"
  for state_file in "$coord_root"/workers/*/state.json; do
    [[ -f "$state_file" ]] || continue
    worker_id="$(coord_extract_json_string "$state_file" "worker_id")"
    pane_id="$(coord_extract_json_string "$state_file" "pane_id")"
    status="$(coord_extract_json_string "$state_file" "status")"
    task_id="$(coord_extract_json_string "$state_file" "task_id")"
    updated_at="$(coord_extract_json_string "$state_file" "updated_at")"

    if [[ -z "$worker_id" || -z "$status" || -z "$task_id" ]]; then
      continue
    fi

    if [[ "$status" != "done" && "$status" != "done_with_concerns" ]]; then
      continue
    fi

    key="worker=${worker_id} pane=${pane_id} status=${status} task=${task_id} updated=${updated_at}"
    prev="$(cat "$coord_root/watch/${worker_id}.last_alert" 2>/dev/null || true)"
    if [[ "$key" == "$prev" ]]; then
      :
    else
      line="$(coord_now) ALERT ${key}"
      printf '%s\n' "$line" | tee -a "$coord_root/alerts.log"
      printf '%s' "$key" > "$coord_root/watch/${worker_id}.last_alert"

      if [[ -n "$notify_session" ]] && tmux has-session -t "$notify_session" 2>/dev/null; then
        tmux display-message -t "$notify_session" "$line" || true
      fi
    fi

    # Keep reminding if done* is not consumed promptly.
    if parse_utc_epoch "$updated_at" >/dev/null 2>&1; then
      updated_epoch="$(parse_utc_epoch "$updated_at")"
      if (( now_epoch - updated_epoch >= remind_seconds )); then
        overdue_stamp="$coord_root/watch/${worker_id}.last_overdue"
        overdue_prev_task=""
        overdue_prev_epoch=0
        if [[ -f "$overdue_stamp" ]]; then
          overdue_prev_task="$(sed -n '1p' "$overdue_stamp" 2>/dev/null || true)"
          overdue_prev_epoch="$(sed -n '2p' "$overdue_stamp" 2>/dev/null || true)"
          [[ "$overdue_prev_epoch" =~ ^[0-9]+$ ]] || overdue_prev_epoch=0
        fi
        if [[ "$overdue_prev_task" != "$task_id" || $((now_epoch - overdue_prev_epoch)) -ge $remind_seconds ]]; then
          overdue_line="$(coord_now) OVERDUE worker=${worker_id} pane=${pane_id} status=${status} task=${task_id} updated=${updated_at}"
          printf '%s\n' "$overdue_line" | tee -a "$coord_root/alerts.log"
          {
            printf '%s\n' "$task_id"
            printf '%s\n' "$now_epoch"
          } > "$overdue_stamp"
          if [[ -n "$notify_session" ]] && tmux has-session -t "$notify_session" 2>/dev/null; then
            tmux display-message -t "$notify_session" "$overdue_line" || true
          fi
        fi
      fi
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
