#!/usr/bin/env bash
# Sync HannsDB to remote x86 machine and optionally build/test
set -euo pipefail

REMOTE_HOST=root@189.1.218.159
REMOTE_DIR=/data/work/HannsDB
REMOTE_VENV_DIR=/data/work/HannsDB/.venv-hannsdb-remote
REMOTE_VDBB_REPO=/data/work/VectorDBBench
KNOWHERE_SRC=/Users/ryan/Code/vectorDB/knowhere-rs
KNOWHERE_REMOTE_DIR=/data/work/knowhere-rs
REMOTE_VDBB_BASE_DEPS="click pytz streamlit-autorefresh 'streamlit<1.44,!=1.34.0' streamlit_extras tqdm s3fs oss2 psutil polars plotly environs 'pydantic<2' scikit-learn pymilvus ujson 'hdrhistogram>=0.10.1'"

SSH_KEY=~/.ssh/agent/hannsdb-x86
SSH_PROXY_CMD="python3 /Users/ryan/Code/knowhere/scripts/remote/socks5_proxy.py --proxy-host 38.248.251.237 --proxy-port 443 --username nzyiMkgOkArt --password 19M5u8hbyF 189.1.218.159 22"
SSH_RSYNC_CMD="ssh -i $SSH_KEY -o ServerAliveInterval=30 -o ProxyCommand=\"$SSH_PROXY_CMD\""

sync_hannsdb() {
  rsync -avz --delete \
    --exclude target \
    --exclude '.git' \
    --exclude '.venv-hannsdb' \
    --exclude '.venv-hannsdb-remote' \
    -e "$SSH_RSYNC_CMD" \
    "$(dirname "$(dirname "$0")")"/ \
    "$REMOTE_HOST:$REMOTE_DIR/"
}

sync_knowhere() {
  rsync -avz --delete \
    --exclude target \
    --exclude '.git' \
    --exclude benchmark_results \
    -e "$SSH_RSYNC_CMD" \
    "$KNOWHERE_SRC"/ \
    "$REMOTE_HOST:$KNOWHERE_REMOTE_DIR/"
}

remote_ssh() {
  ssh -i "$SSH_KEY" \
    -o ServerAliveInterval=30 \
    -o ProxyCommand="$SSH_PROXY_CMD" \
    "$REMOTE_HOST" "$@"
}

bootstrap_remote_vdbb() {
  sync_hannsdb
  sync_knowhere
  remote_ssh "set -euo pipefail && \
    python3 -m venv $REMOTE_VENV_DIR && \
    $REMOTE_VENV_DIR/bin/python -m pip install --upgrade pip wheel setuptools maturin && \
    $REMOTE_VENV_DIR/bin/python -m pip install $REMOTE_VDBB_BASE_DEPS && \
    source ~/.cargo/env && cd $REMOTE_DIR && \
    VIRTUAL_ENV=$REMOTE_VENV_DIR PATH=$REMOTE_VENV_DIR/bin:\$PATH \
    $REMOTE_VENV_DIR/bin/maturin develop \
      --manifest-path $REMOTE_DIR/crates/hannsdb-py/Cargo.toml \
      --release \
      --no-default-features \
      --features python-binding,hanns-backend"
}

case "${1:-}" in
  sync-knowhere)
    sync_knowhere
    ;;
  knowhere-build)
    remote_ssh "source ~/.cargo/env && cd $KNOWHERE_REMOTE_DIR && cargo build --release 2>&1 | tail -5"
    ;;
  vdbb-bootstrap)
    bootstrap_remote_vdbb
    ;;
  vdbb-perf)
    bootstrap_remote_vdbb
    remote_ssh "cd $REMOTE_DIR && \
      VENV_PATH=$REMOTE_VENV_DIR \
      VDBB_REPO=$REMOTE_VDBB_REPO \
      SKIP_PY_REBUILD=1 \
      DB_LABEL=${DB_LABEL:-hannsdb-remote-1536d50k} \
      TASK_LABEL=${TASK_LABEL:-${DB_LABEL:-hannsdb-remote-1536d50k}} \
      DB_PATH=${DB_PATH:-/tmp/${DB_LABEL:-hannsdb-remote-1536d50k}-db} \
      K=${K:-10} M=${M:-16} EF_CONSTRUCTION=${EF_CONSTRUCTION:-64} EF_SEARCH=${EF_SEARCH:-32} \
      NUM_CONCURRENCY=${NUM_CONCURRENCY:-1} CONCURRENCY_DURATION=${CONCURRENCY_DURATION:-1} \
      bash scripts/run_vdbb_hannsdb_perf1536d50k.sh"
    ;;
  vdbb-watchdog)
    bootstrap_remote_vdbb
    remote_ssh "cd $REMOTE_DIR && \
      VENV_PATH=$REMOTE_VENV_DIR \
      VDBB_REPO=$REMOTE_VDBB_REPO \
      SKIP_PY_REBUILD=1 \
      DB_LABEL=${DB_LABEL:-hannsdb-remote-watchdog} \
      TASK_LABEL=${TASK_LABEL:-${DB_LABEL:-hannsdb-remote-watchdog}} \
      DB_PATH=${DB_PATH:-/tmp/${DB_LABEL:-hannsdb-remote-watchdog}-db} \
      STALL_TIMEOUT_SEC=${STALL_TIMEOUT_SEC:-600} POST_LOAD_TIMEOUT_SEC=${POST_LOAD_TIMEOUT_SEC:-3600} \
      K=${K:-10} M=${M:-16} EF_CONSTRUCTION=${EF_CONSTRUCTION:-64} EF_SEARCH=${EF_SEARCH:-32} \
      NUM_CONCURRENCY=${NUM_CONCURRENCY:-1} CONCURRENCY_DURATION=${CONCURRENCY_DURATION:-1} \
      bash scripts/run_vdbb_hannsdb_perf1536d50k_watchdog.sh"
    ;;
  knowhere-bench)
    remote_ssh "source ~/.cargo/env && cd $REMOTE_DIR && \
      HANNSSDB_OPT_BENCH_N=${N:-50000} \
      HANNSSDB_OPT_BENCH_DIM=${DIM:-1536} \
      HANNSSDB_OPT_BENCH_METRIC=${METRIC:-cosine} \
      cargo test -p hannsdb-core --release --features hanns-backend collection_api_optimize_benchmark_entry -- --nocapture 2>&1 | grep OPT_BENCH"
    ;;
  build)
    sync_hannsdb
    remote_ssh "source ~/.cargo/env && cd $REMOTE_DIR && cargo build -p hannsdb-core"
    ;;
  test)
    sync_hannsdb
    remote_ssh "source ~/.cargo/env && cd $REMOTE_DIR && cargo test -p hannsdb-core"
    ;;
  bench)
    sync_hannsdb
    remote_ssh "source ~/.cargo/env && cd $REMOTE_DIR && \
      HANNSSDB_OPT_BENCH_N=${N:-2000} \
      HANNSSDB_OPT_BENCH_DIM=${DIM:-256} \
      HANNSSDB_OPT_BENCH_METRIC=${METRIC:-cosine} \
      cargo test -p hannsdb-core collection_api_optimize_benchmark_entry -- --nocapture"
    ;;
  *)
    sync_hannsdb
    echo "Synced HannsDB. Usage: $0 [build|test|bench|sync-knowhere|knowhere-build|knowhere-bench|vdbb-bootstrap|vdbb-perf|vdbb-watchdog]"
    ;;
esac
