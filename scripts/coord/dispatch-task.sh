#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

coord_root="$(coord_default_root)"
worker_id=""
task_id=""
pane_id=""
dry_run=false

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
    --task-id)
      task_id="$2"
      shift 2
      ;;
    --pane-id)
      pane_id="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$worker_id" || -z "$task_id" ]]; then
  echo "--worker and --task-id are required" >&2
  exit 2
fi

task_path="$coord_root/tasks/${task_id}.md"
coord_require_file "$task_path"

if [[ -z "$pane_id" ]]; then
  pane_id="$(coord_load_pane_id "$coord_root" "$worker_id")"
fi

bash "$SCRIPT_DIR/set-worker-state.sh" \
  --coord-root "$coord_root" \
  --worker "$worker_id" \
  --status assigned \
  --task-id "$task_id" \
  --note "task dispatched" \
  --pane-id "$pane_id" >/dev/null

tmp_prompt="$(mktemp "${TMPDIR:-/tmp}/coord-dispatch.${task_id}.XXXXXX")"
cat >"$tmp_prompt" <<EOF
执行 .coord/tasks/${task_id}.md
先把 .coord/workers/${worker_id}/state.json 更新为 acknowledged / working
完成后写 .coord/results/${task_id}.md
最后把 .coord/workers/${worker_id}/state.json 更新为 done 或 done_with_concerns
EOF

buffer_name="coord-${worker_id}-${task_id}"
command_block="$(cat <<EOF
# task_file=.coord/tasks/${task_id}.md
tmux load-buffer -b ${buffer_name} ${tmp_prompt}
tmux paste-buffer -b ${buffer_name} -t ${pane_id}
sleep 5
tmux send-keys -t ${pane_id} Enter
sleep 5
EOF
)"

if [[ "$dry_run" == true ]]; then
  printf '%s\n' "$command_block"
  rm -f "$tmp_prompt"
  exit 0
fi

tmux load-buffer -b "$buffer_name" "$tmp_prompt"
tmux paste-buffer -b "$buffer_name" -t "$pane_id"
sleep 5
tmux send-keys -t "$pane_id" Enter
sleep 5
rm -f "$tmp_prompt"
printf 'dispatched task=%s worker=%s pane=%s\n' "$task_id" "$worker_id" "$pane_id"
