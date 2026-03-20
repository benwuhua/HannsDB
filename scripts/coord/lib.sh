#!/usr/bin/env bash
set -euo pipefail

coord_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$script_dir/../.." && pwd
}

coord_default_root() {
  printf '%s/.coord\n' "$(coord_repo_root)"
}

coord_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

coord_json_escape() {
  local value="${1-}"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  value="${value//$'\r'/\\r}"
  value="${value//$'\t'/\\t}"
  printf '%s' "$value"
}

coord_extract_json_string() {
  local file="$1"
  local key="$2"
  sed -n "s/.*\"$key\": \"\\([^\"]*\\)\".*/\\1/p" "$file" | head -n 1
}

coord_state_path() {
  local coord_root="$1"
  local worker_id="$2"
  printf '%s/workers/%s/state.json\n' "$coord_root" "$worker_id"
}

coord_require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "required file not found: $path" >&2
    exit 1
  fi
}

coord_require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "required directory not found: $path" >&2
    exit 1
  fi
}

coord_write_state() {
  local coord_root="$1"
  local worker_id="$2"
  local pane_id="$3"
  local status="$4"
  local task_id="$5"
  local note="$6"
  local state_path tmp_path
  state_path="$(coord_state_path "$coord_root" "$worker_id")"
  tmp_path="${state_path}.tmp.$$"
  mkdir -p "$(dirname "$state_path")"
  cat >"$tmp_path" <<EOF
{
  "worker_id": "$(coord_json_escape "$worker_id")",
  "pane_id": "$(coord_json_escape "$pane_id")",
  "status": "$(coord_json_escape "$status")",
  "task_id": "$(coord_json_escape "$task_id")",
  "updated_at": "$(coord_now)",
  "note": "$(coord_json_escape "$note")"
}
EOF
  mv "$tmp_path" "$state_path"
}

coord_load_pane_id() {
  local coord_root="$1"
  local worker_id="$2"
  local state_path
  state_path="$(coord_state_path "$coord_root" "$worker_id")"
  coord_require_file "$state_path"
  coord_extract_json_string "$state_path" "pane_id"
}

coord_load_status() {
  local coord_root="$1"
  local worker_id="$2"
  local state_path
  state_path="$(coord_state_path "$coord_root" "$worker_id")"
  coord_require_file "$state_path"
  coord_extract_json_string "$state_path" "status"
}

coord_load_task_id() {
  local coord_root="$1"
  local worker_id="$2"
  local state_path
  state_path="$(coord_state_path "$coord_root" "$worker_id")"
  coord_require_file "$state_path"
  coord_extract_json_string "$state_path" "task_id"
}

coord_load_note() {
  local coord_root="$1"
  local worker_id="$2"
  local state_path
  state_path="$(coord_state_path "$coord_root" "$worker_id")"
  coord_require_file "$state_path"
  coord_extract_json_string "$state_path" "note"
}
