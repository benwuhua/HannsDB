#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-/tmp/hannsdb-zvec-parity-smoke-venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "$VENV_PATH" || ! -f "$VENV_PATH/bin/activate" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi

. "$VENV_PATH/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q maturin pytest

cd "$ROOT_DIR"
cargo test -p hannsdb-core --test zvec_parity_schema -- --nocapture
cargo test -p hannsdb-core --test zvec_parity_query -- --nocapture
cargo test -p hannsdb-daemon --test http_smoke -- --nocapture

cd "$ROOT_DIR/crates/hannsdb-py"
maturin develop --features python-binding,knowhere-backend
python -m pytest \
  tests/test_typing_surface.py \
  tests/test_schema_surface.py \
  tests/test_collection_parity.py \
  tests/test_collection_facade.py \
  tests/test_query_executor.py \
  tests/test_collection_concurrency.py \
  -q

echo "benchmark notes: $ROOT_DIR/docs/vector-db-bench-notes.md"
echo "zvec parity smoke completed successfully"
