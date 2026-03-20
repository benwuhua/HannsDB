#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VDBB_REPO="${VDBB_REPO:-/Users/ryan/Code/VectorDBBench}"
VENV_PATH="${VENV_PATH:-$ROOT_DIR/.venv-hannsdb}"

DATASET_DIR="${DATASET_DIR:-/tmp/hannsdb-custom-ds-review}"
DB_PATH="${DB_PATH:-/tmp/hannsdb-vdbb-smoke-db}"
DB_LABEL="${DB_LABEL:-hannsdb-smoke}"
TASK_LABEL="${TASK_LABEL:-hannsdb-smoke}"
CASE_NAME="${CASE_NAME:-hannsdb-smoke}"
CASE_DESC="${CASE_DESC:-HannsDB tiny smoke}"
DATASET_NAME="${DATASET_NAME:-hannsdb_smoke_ds}"

DATASET_SIZE="${DATASET_SIZE:-64}"
DATASET_DIM="${DATASET_DIM:-8}"
DATASET_FILE_COUNT="${DATASET_FILE_COUNT:-1}"
DATASET_METRIC="${DATASET_METRIC:-L2}"
K="${K:-3}"
M="${M:-16}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-32}"
EF_SEARCH="${EF_SEARCH:-16}"
CONCURRENCY_DURATION="${CONCURRENCY_DURATION:-1}"
NUM_CONCURRENCY="${NUM_CONCURRENCY:-1}"

REGENERATE=false
for arg in "$@"; do
  case "$arg" in
    --regenerate)
      REGENERATE=true
      ;;
    *)
      echo "unknown argument: $arg" >&2
      echo "usage: $0 [--regenerate]" >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$VDBB_REPO" ]]; then
  echo "VectorDBBench repo not found: $VDBB_REPO" >&2
  exit 1
fi

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "venv activate script not found: $VENV_PATH/bin/activate" >&2
  exit 1
fi

if [[ "$REGENERATE" == "true" || ! -f "$DATASET_DIR/train.parquet" || ! -f "$DATASET_DIR/test.parquet" || ! -f "$DATASET_DIR/neighbors.parquet" ]]; then
  . "$VENV_PATH/bin/activate"
  python "$ROOT_DIR/python/generate_custom_vdbb_dataset.py" \
    --output-dir "$DATASET_DIR" \
    --dimension "$DATASET_DIM" \
    --train-size "$DATASET_SIZE" \
    --test-size "$DATASET_DIM" \
    --metric "$(echo "$DATASET_METRIC" | tr '[:upper:]' '[:lower:]')" \
    --top-k "$K"
else
  echo "reusing dataset at $DATASET_DIR"
fi

. "$VENV_PATH/bin/activate"
PYTHONPATH="$VDBB_REPO" python -m vectordb_bench.cli.vectordbbench hannsdb \
  --path "$DB_PATH" \
  --db-label "$DB_LABEL" \
  --task-label "$TASK_LABEL" \
  --case-type PerformanceCustomDataset \
  --custom-case-name "$CASE_NAME" \
  --custom-case-description "$CASE_DESC" \
  --custom-dataset-name "$DATASET_NAME" \
  --custom-dataset-dir "$DATASET_DIR" \
  --custom-dataset-size "$DATASET_SIZE" \
  --custom-dataset-dim "$DATASET_DIM" \
  --custom-dataset-file-count "$DATASET_FILE_COUNT" \
  --custom-dataset-metric-type "$DATASET_METRIC" \
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
