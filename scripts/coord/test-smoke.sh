#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/hannsdb-coord-test.XXXXXX")"
cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

assert_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "expected file to exist: $path" >&2
    exit 1
  fi
}

assert_contains() {
  local path="$1"
  local expected="$2"
  if ! grep -Fq "$expected" "$path"; then
    echo "expected '$expected' in $path" >&2
    exit 1
  fi
}

assert_output_contains() {
  local output="$1"
  local expected="$2"
  if [[ "$output" != *"$expected"* ]]; then
    echo "expected output to contain '$expected'" >&2
    echo "$output" >&2
    exit 1
  fi
}

assert_output_not_contains() {
  local output="$1"
  local unexpected="$2"
  if [[ "$output" == *"$unexpected"* ]]; then
    echo "did not expect output to contain '$unexpected'" >&2
    echo "$output" >&2
    exit 1
  fi
}

cd "$ROOT_DIR"

bash scripts/coord/init.sh \
  --coord-root "$TEST_ROOT/.coord" \
  --worker worker1:%1 \
  --worker worker2:%2

assert_file "$TEST_ROOT/.coord/workers/worker1/state.json"
assert_file "$TEST_ROOT/.coord/workers/worker2/state.json"

bash scripts/coord/set-worker-state.sh \
  --coord-root "$TEST_ROOT/.coord" \
  --worker worker1 \
  --status assigned \
  --task-id T-001 \
  --note "task assigned"

assert_contains "$TEST_ROOT/.coord/workers/worker1/state.json" '"status": "assigned"'
assert_contains "$TEST_ROOT/.coord/workers/worker1/state.json" '"task_id": "T-001"'

monitor_output="$(
  bash scripts/coord/monitor.sh \
    --coord-root "$TEST_ROOT/.coord" \
    --once
)"
assert_output_contains "$monitor_output" "worker=worker1"
assert_output_contains "$monitor_output" "status=assigned"
assert_output_not_contains "$monitor_output" "pane= status="

cat >"$TEST_ROOT/.coord/tasks/T-001.md" <<'EOF'
# Task T-001

Run the task and update state/result files.
EOF

dispatch_output="$(
  bash scripts/coord/dispatch-task.sh \
    --coord-root "$TEST_ROOT/.coord" \
    --worker worker1 \
    --task-id T-001 \
    --dry-run
)"
assert_output_contains "$dispatch_output" "tmux load-buffer"
assert_output_contains "$dispatch_output" "sleep 5"
assert_output_contains "$dispatch_output" "T-001.md"

status_output="$(
  bash scripts/coord/status.sh \
    --coord-root "$TEST_ROOT/.coord"
)"
assert_output_contains "$status_output" "worker1"
assert_output_contains "$status_output" "assigned"
assert_output_contains "$status_output" "monitor=stopped"

bash scripts/coord/start-monitor.sh \
  --coord-root "$TEST_ROOT/.coord" \
  --interval 1

sleep 2

assert_file "$TEST_ROOT/.coord/monitor/session"
assert_file "$TEST_ROOT/.coord/monitor/heartbeat"

status_output="$(
  bash scripts/coord/status.sh \
    --coord-root "$TEST_ROOT/.coord"
)"
assert_output_contains "$status_output" "monitor=running"

bash scripts/coord/stop-monitor.sh \
  --coord-root "$TEST_ROOT/.coord"

sleep 1

status_output="$(
  bash scripts/coord/status.sh \
    --coord-root "$TEST_ROOT/.coord"
)"
assert_output_contains "$status_output" "monitor=stopped"

echo "coord smoke test: PASS"
