#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

resolve_vdbb_repo() {
  if [[ -n "${VDBB_REPO:-}" ]]; then
    printf '%s\n' "$VDBB_REPO"
    return 0
  fi

  local candidates=(
    "/data/work/VectorDBBench"
    "/Users/ryan/Code/vectorDB/VectorDBBench"
    "/Users/ryan/Code/VectorDBBench"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  printf '%s\n' "/data/work/VectorDBBench"
}

VDBB_REPO="$(resolve_vdbb_repo)"
LOG_PATH="${LOG_PATH:-$ROOT_DIR/logs/vectordb_bench.log}"

DB_LABEL="${DB_LABEL:-hannsdb-1536d50k-watchdog}"
TASK_LABEL="${TASK_LABEL:-$DB_LABEL}"
DB_PATH="${DB_PATH:-/tmp/${DB_LABEL}-db}"

STALL_TIMEOUT_SEC="${STALL_TIMEOUT_SEC:-300}"
POST_LOAD_TIMEOUT_SEC="${POST_LOAD_TIMEOUT_SEC:-1800}"
RESULT_DATE="${RESULT_DATE:-$(date +%Y%m%d)}"

RESULT_PATH="$VDBB_REPO/vectordb_bench/results/HannsDB/result_${RESULT_DATE}_${DB_LABEL}_hannsdb.json"
PS_SNAPSHOT="/tmp/${DB_LABEL}.ps.txt"
PARENT_SAMPLE="/tmp/${DB_LABEL}.sample.txt"
WORKER_SAMPLE="/tmp/${DB_LABEL}-worker.sample.txt"

if [[ ! -d "$VDBB_REPO" ]]; then
  echo "WATCHDOG_ERROR missing_vdbb_repo path=$VDBB_REPO" >&2
  exit 1
fi
if ! [[ "$STALL_TIMEOUT_SEC" =~ ^[0-9]+$ ]] || [[ "$STALL_TIMEOUT_SEC" -lt 1 ]]; then
  echo "WATCHDOG_ERROR invalid_stall_timeout value=$STALL_TIMEOUT_SEC" >&2
  exit 1
fi
if ! [[ "$POST_LOAD_TIMEOUT_SEC" =~ ^[0-9]+$ ]] || [[ "$POST_LOAD_TIMEOUT_SEC" -lt 1 ]]; then
  echo "WATCHDOG_ERROR invalid_post_load_timeout value=$POST_LOAD_TIMEOUT_SEC" >&2
  exit 1
fi

list_descendants() {
  local root_pid="$1"
  local queue=("$root_pid")
  local descendants=()
  local p child
  while [[ "${#queue[@]}" -gt 0 ]]; do
    p="${queue[0]}"
    queue=("${queue[@]:1}")
    while read -r child; do
      [[ -z "$child" ]] && continue
      descendants+=("$child")
      queue+=("$child")
    done < <(ps -Ao pid=,ppid= | awk -v p="$p" '$2 == p { print $1 }')
  done
  printf "%s\n" "${descendants[@]:-}"
}

graceful_stop_tree() {
  local root_pid="$1"
  local wait_sec="${2:-8}"
  local descendants=()
  local line
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    descendants+=("$line")
  done < <(list_descendants "$root_pid")

  if kill -0 "$root_pid" 2>/dev/null; then
    kill -TERM "$root_pid" 2>/dev/null || true
  fi
  for pid in "${descendants[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done

  local deadline=$((SECONDS + wait_sec))
  while [[ $SECONDS -lt $deadline ]]; do
    if ! kill -0 "$root_pid" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done

  descendants=()
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    descendants+=("$line")
  done < <(list_descendants "$root_pid")
  if kill -0 "$root_pid" 2>/dev/null; then
    kill -KILL "$root_pid" 2>/dev/null || true
  fi
  for pid in "${descendants[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
  return 0
}

capture_diagnostics() {
  local root_pid="$1"
  local py_parent_pid=""
  local py_worker_pid=""
  local fallback_worker_pid=""
  local best_worker_cpu="-1"
  local best_fallback_cpu="-1"
  local pid cmd cpu_raw cpu
  local descendants=()
  local py_descendants=()
  local line
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    descendants+=("$line")
  done < <(list_descendants "$root_pid")

  for pid in "${descendants[@]}"; do
    cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    if [[ -n "$cmd" && "$cmd" == *python* ]]; then
      py_descendants+=("$pid")
      if [[ -z "$py_parent_pid" && "$cmd" == *vectordb_bench.cli.vectordbbench* ]]; then
        py_parent_pid="$pid"
      fi
    fi
  done

  if [[ -z "$py_parent_pid" ]]; then
    for pid in "${py_descendants[@]}"; do
      py_parent_pid="$pid"
      break
    done
  fi

  # Prefer non-resource-tracker child with highest %CPU.
  for pid in "${py_descendants[@]}"; do
    [[ "$pid" == "$py_parent_pid" ]] && continue
    cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    [[ -z "$cmd" ]] && continue

    cpu_raw="$(ps -p "$pid" -o %cpu= 2>/dev/null || true)"
    cpu="$(echo "${cpu_raw:-0}" | tr -d '[:space:]')"
    [[ -z "$cpu" ]] && cpu="0"

    if [[ "$cmd" != *resource_tracker* ]]; then
      if awk -v current="$cpu" -v best="$best_worker_cpu" 'BEGIN { exit !(current > best) }'; then
        py_worker_pid="$pid"
        best_worker_cpu="$cpu"
      fi
    fi

    if awk -v current="$cpu" -v best="$best_fallback_cpu" 'BEGIN { exit !(current > best) }'; then
      fallback_worker_pid="$pid"
      best_fallback_cpu="$cpu"
    fi
  done

  if [[ -z "$py_worker_pid" ]]; then
    py_worker_pid="$fallback_worker_pid"
  fi

  echo "WATCHDOG_DIAG ps_snapshot=$PS_SNAPSHOT"
  ps -Ao pid,ppid,pgid,state,etime,%cpu,%mem,command >"$PS_SNAPSHOT" || true

  if [[ -n "$py_parent_pid" ]]; then
    cmd="$(ps -p "$py_parent_pid" -o command= 2>/dev/null || true)"
    echo "WATCHDOG_DIAG sample_parent pid=$py_parent_pid out=$PARENT_SAMPLE cmd=$cmd"
    sample "$py_parent_pid" 5 -file "$PARENT_SAMPLE" >/dev/null 2>&1 || true
  else
    echo "WATCHDOG_DIAG sample_parent skipped reason=no_python_parent"
  fi

  if [[ -n "$py_worker_pid" ]]; then
    cmd="$(ps -p "$py_worker_pid" -o command= 2>/dev/null || true)"
    cpu_raw="$(ps -p "$py_worker_pid" -o %cpu= 2>/dev/null || true)"
    cpu="$(echo "${cpu_raw:-0}" | tr -d '[:space:]')"
    echo "WATCHDOG_DIAG sample_worker pid=$py_worker_pid cpu=$cpu out=$WORKER_SAMPLE cmd=$cmd"
    sample "$py_worker_pid" 5 -file "$WORKER_SAMPLE" >/dev/null 2>&1 || true
  else
    echo "WATCHDOG_DIAG sample_worker skipped reason=no_python_worker"
  fi
}

gate_pid=""
on_interrupt() {
  echo "WATCHDOG_INTERRUPTED signal_received"
  if [[ -n "$gate_pid" ]] && kill -0 "$gate_pid" 2>/dev/null; then
    graceful_stop_tree "$gate_pid" 5
  fi
  exit 130
}
trap on_interrupt INT TERM

if [[ -f "$LOG_PATH" ]]; then
  LOG_START_LINE="$(wc -l <"$LOG_PATH" | tr -d ' ')"
else
  LOG_START_LINE="0"
fi

echo "WATCHDOG_START db_label=$DB_LABEL task_label=$TASK_LABEL db_path=$DB_PATH stall_timeout_sec=$STALL_TIMEOUT_SEC post_load_timeout_sec=$POST_LOAD_TIMEOUT_SEC result_path=$RESULT_PATH log_path=$LOG_PATH log_start_line=$LOG_START_LINE"

(
  cd "$ROOT_DIR"
  DB_LABEL="$DB_LABEL" TASK_LABEL="$TASK_LABEL" DB_PATH="$DB_PATH" \
    bash scripts/run_vdbb_hannsdb_perf1536d50k.sh
) &
gate_pid="$!"
echo "WATCHDOG_GATE_LAUNCHED pid=$gate_pid"

load_seen=0
search_seen=0
post_load_deadline=0
search_deadline=0

while true; do
  if [[ -f "$RESULT_PATH" ]]; then
    wait "$gate_pid"
    gate_rc=$?
    echo "WATCHDOG_RESULT_FOUND path=$RESULT_PATH gate_rc=$gate_rc"
    exit "$gate_rc"
  fi

  if [[ "$load_seen" -eq 0 && -f "$LOG_PATH" ]]; then
    if tail -n "+$((LOG_START_LINE + 1))" "$LOG_PATH" | grep -q "Finish loading all dataset into VectorDB"; then
      load_seen=1
      post_load_deadline=$((SECONDS + POST_LOAD_TIMEOUT_SEC))
      echo "WATCHDOG_LOAD_COMPLETE seen=1 post_load_deadline_epoch=$post_load_deadline"
    fi
  fi

  if [[ "$search_seen" -eq 0 && -f "$LOG_PATH" ]]; then
    if tail -n "+$((LOG_START_LINE + 1))" "$LOG_PATH" | grep -Eq "start serial search|start search the entire test_data to get recall and latency"; then
      search_seen=1
      search_deadline=$((SECONDS + STALL_TIMEOUT_SEC))
      echo "WATCHDOG_SEARCH_STARTED seen=1 search_deadline_epoch=$search_deadline"
    fi
  fi

  if ! kill -0 "$gate_pid" 2>/dev/null; then
    wait "$gate_pid" || gate_rc=$?
    gate_rc="${gate_rc:-0}"
    if [[ -f "$RESULT_PATH" ]]; then
      echo "WATCHDOG_GATE_EXITED_WITH_RESULT gate_rc=$gate_rc path=$RESULT_PATH"
      exit "$gate_rc"
    fi
    echo "WATCHDOG_GATE_EXITED_NO_RESULT gate_rc=$gate_rc expected_result=$RESULT_PATH"
    exit 1
  fi

  if [[ "$load_seen" -eq 1 && "$search_seen" -eq 0 && "$SECONDS" -ge "$post_load_deadline" ]]; then
    echo "WATCHDOG_POST_LOAD_TIMEOUT reached=1 timeout_sec=$POST_LOAD_TIMEOUT_SEC"
    capture_diagnostics "$gate_pid"
    graceful_stop_tree "$gate_pid" 8
    echo "WATCHDOG_EXIT status=post_load_timeout_no_search result_path=$RESULT_PATH"
    exit 125
  fi

  if [[ "$search_seen" -eq 1 && "$SECONDS" -ge "$search_deadline" ]]; then
    echo "WATCHDOG_SEARCH_TIMEOUT reached=1 timeout_sec=$STALL_TIMEOUT_SEC"
    capture_diagnostics "$gate_pid"
    graceful_stop_tree "$gate_pid" 8
    echo "WATCHDOG_EXIT status=search_timeout_no_result result_path=$RESULT_PATH"
    exit 124
  fi

  sleep 2
done
