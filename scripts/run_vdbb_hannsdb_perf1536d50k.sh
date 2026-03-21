#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VDBB_REPO="${VDBB_REPO:-/Users/ryan/Code/VectorDBBench}"
VENV_PATH="${VENV_PATH:-/Users/ryan/Code/HannsDB/.venv-hannsdb}"

DB_PATH="${DB_PATH:-/tmp/hannsdb-vdbb-1536d50k-db}"
DB_LABEL="${DB_LABEL:-hannsdb-1536d50k}"
TASK_LABEL="${TASK_LABEL:-hannsdb-1536d50k}"

K="${K:-10}"
M="${M:-16}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-64}"
EF_SEARCH="${EF_SEARCH:-32}"
NUM_CONCURRENCY="${NUM_CONCURRENCY:-1}"
CONCURRENCY_DURATION="${CONCURRENCY_DURATION:-1}"
SKIP_PY_REBUILD="${SKIP_PY_REBUILD:-0}"
PY_BINDING_FEATURES="${PY_BINDING_FEATURES:-python-binding,knowhere-backend}"
MATURIN_RELEASE="${MATURIN_RELEASE:-1}"

if [[ ! -d "$VDBB_REPO" ]]; then
  echo "VectorDBBench repo not found: $VDBB_REPO" >&2
  exit 1
fi

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "venv activate script not found: $VENV_PATH/bin/activate" >&2
  exit 1
fi

. "$VENV_PATH/bin/activate"

if [[ "$SKIP_PY_REBUILD" != "1" ]]; then
  if ! command -v maturin >/dev/null 2>&1; then
    echo "maturin not found in PATH (expected inside venv): $VENV_PATH" >&2
    exit 1
  fi

  MATURIN_ARGS=()
  if [[ "$MATURIN_RELEASE" == "1" ]]; then
    MATURIN_ARGS+=(--release)
  fi

  echo "Rebuilding hannsdb Python extension via maturin (features=$PY_BINDING_FEATURES, release=$MATURIN_RELEASE)"
  maturin develop \
    --manifest-path "$ROOT_DIR/crates/hannsdb-py/Cargo.toml" \
    --no-default-features \
    --features "$PY_BINDING_FEATURES" \
    "${MATURIN_ARGS[@]}"
fi

PYTHONPATH="$VDBB_REPO" python -m vectordb_bench.cli.vectordbbench hannsdb \
  --path "$DB_PATH" \
  --db-label "$DB_LABEL" \
  --task-label "$TASK_LABEL" \
  --case-type Performance1536D50K \
  --k "$K" \
  --m "$M" \
  --ef-construction "$EF_CONSTRUCTION" \
  --ef-search "$EF_SEARCH" \
  --skip-search-concurrent \
  --num-concurrency "$NUM_CONCURRENCY" \
  --concurrency-duration "$CONCURRENCY_DURATION"

RESULT_DATE="$(date +%Y%m%d)"
RESULT_PATH="$VDBB_REPO/vectordb_bench/results/HannsDB/result_${RESULT_DATE}_${DB_LABEL}_hannsdb.json"
echo "result file: $RESULT_PATH"
