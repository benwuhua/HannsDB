#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-/tmp/hannsdb-zvec-parity-smoke-venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MATURIN_SPEC="${MATURIN_SPEC:-maturin==1.12.6}"
PYTEST_SPEC="${PYTEST_SPEC:-pytest==8.4.2}"
STAMP_PATH="$VENV_PATH/.zvec-parity-smoke-stamp"

ensure_venv() {
  local expected_python_version
  local expected_python_bin
  local recorded_python_version
  local recorded_python_bin
  local need_rebuild=false

  expected_python_bin="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
  expected_python_version="$("$PYTHON_BIN" -V 2>&1)"

  if [[ ! -d "$VENV_PATH" || ! -f "$VENV_PATH/bin/activate" || ! -f "$STAMP_PATH" ]]; then
    need_rebuild=true
  else
    recorded_python_bin="$(sed -n 's/^PYTHON_BIN=//p' "$STAMP_PATH")"
    recorded_python_version="$(sed -n 's/^PYTHON_VERSION=//p' "$STAMP_PATH")"
    if [[ "$recorded_python_bin" != "$expected_python_bin" || "$recorded_python_version" != "$expected_python_version" ]]; then
      need_rebuild=true
    fi
  fi

  if [[ "$need_rebuild" == "true" ]]; then
    rm -rf "$VENV_PATH"
    "$PYTHON_BIN" -m venv "$VENV_PATH"
    printf 'PYTHON_BIN=%s\nPYTHON_VERSION=%s\n' "$expected_python_bin" "$("$VENV_PATH/bin/python" -V 2>&1)" > "$STAMP_PATH"
    echo "created venv: $VENV_PATH"
  else
    echo "reused venv: $VENV_PATH"
  fi
  echo "venv python: $("$VENV_PATH/bin/python" -V 2>&1)"
  echo "venv python executable: $("$VENV_PATH/bin/python" -c 'import sys; print(sys.executable)')"
  echo "MATURIN_SPEC: $MATURIN_SPEC"
  echo "PYTEST_SPEC: $PYTEST_SPEC"
}

ensure_venv

. "$VENV_PATH/bin/activate"
python -m pip install -q "$MATURIN_SPEC" "$PYTEST_SPEC"

cd "$ROOT_DIR"
cargo test -p hannsdb-core --test zvec_parity_schema -- --nocapture
cargo test -p hannsdb-core --test zvec_parity_query -- --nocapture
cargo test -p hannsdb-daemon --test http_smoke -- --nocapture

cd "$ROOT_DIR/crates/hannsdb-py"
maturin develop --features python-binding,hanns-backend
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
