#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Synthetic benchmark controls (override via env)
N="${N:-2000}"
DIM="${DIM:-256}"
METRIC="${METRIC:-cosine}" # l2 | cosine | ip
TOPK="${TOPK:-10}"
INDEX_KIND="${INDEX_KIND:-hnsw}" # hnsw | ivf | ivf_usq | hnsw_hvq
NLIST="${NLIST:-64}"
BITS_PER_DIM="${BITS_PER_DIM:-4}"
ROTATION_SEED="${ROTATION_SEED:-42}"
RERANK_K="${RERANK_K:-64}"
HIGH_ACCURACY_SCAN="${HIGH_ACCURACY_SCAN:-false}"
HNSW_HVQ_M="${HNSW_HVQ_M:-8}"
HNSW_HVQ_M_MAX0="${HNSW_HVQ_M_MAX0:-16}"
HNSW_HVQ_EF_CONSTRUCTION="${HNSW_HVQ_EF_CONSTRUCTION:-32}"
HNSW_HVQ_EF_SEARCH="${HNSW_HVQ_EF_SEARCH:-32}"
HNSW_HVQ_NBITS="${HNSW_HVQ_NBITS:-4}"
QUERY_EF_SEARCH="${QUERY_EF_SEARCH:-}"
QUERY_NPROBE="${QUERY_NPROBE:-}"
REPEATS="${REPEATS:-3}"
FEATURES="${FEATURES:-hanns-backend}"
PROFILE="${PROFILE:-debug}" # debug | release

if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPEATS must be a positive integer, got: ${REPEATS}" >&2
  exit 2
fi

if [[ "$PROFILE" != "debug" && "$PROFILE" != "release" ]]; then
  echo "PROFILE must be debug or release, got: ${PROFILE}" >&2
  exit 2
fi

case "$INDEX_KIND" in
  hnsw|ivf|ivf_usq|hnsw_hvq) ;;
  *)
    echo "INDEX_KIND must be one of hnsw, ivf, ivf_usq, hnsw_hvq; got: ${INDEX_KIND}" >&2
    exit 2
    ;;
esac

if [[ -n "$QUERY_EF_SEARCH" ]]; then
  echo "  QUERY_PARAMS ef_search=${QUERY_EF_SEARCH} nprobe=${QUERY_NPROBE:-<none>}"
elif [[ -n "$QUERY_NPROBE" ]]; then
  echo "  QUERY_PARAMS ef_search=<none> nprobe=${QUERY_NPROBE}"
fi

if [[ "$INDEX_KIND" == "hnsw_hvq" && "$METRIC" != "ip" ]]; then
  echo "INDEX_KIND=hnsw_hvq currently requires METRIC=ip; got: ${METRIC}" >&2
  exit 2
fi

if [[ -n "$QUERY_NPROBE" && "$INDEX_KIND" != "ivf" && "$INDEX_KIND" != "ivf_usq" ]]; then
  echo "QUERY_NPROBE is only meaningful for INDEX_KIND=ivf or ivf_usq; got: ${INDEX_KIND}" >&2
  exit 2
fi

extract_ms_field() {
  local line="$1"
  local key="$2"
  local value
  value="$(printf '%s\n' "$line" | sed -n "s/.*${key}=\\([0-9][0-9]*\\).*/\\1/p")"
  if [[ -z "$value" ]]; then
    echo "failed to parse ${key} from line: ${line}" >&2
    exit 1
  fi
  printf '%s' "$value"
}

median() {
  printf '%s\n' "$@" | sort -n | awk '
    { a[NR] = $1 }
    END {
      if (NR == 0) {
        exit 1
      }
      if (NR % 2 == 1) {
        print a[(NR + 1) / 2]
      } else {
        printf "%.1f\n", (a[NR / 2] + a[NR / 2 + 1]) / 2
      }
    }
  '
}

echo "Running HannsDB optimize benchmark with synthetic data"
echo "  N=${N} DIM=${DIM} METRIC=${METRIC} TOPK=${TOPK} INDEX_KIND=${INDEX_KIND} REPEATS=${REPEATS} FEATURES=${FEATURES:-<none>} PROFILE=${PROFILE}"
case "$INDEX_KIND" in
  ivf)
    echo "  IVF_PARAMS nlist=${NLIST}"
    ;;
  ivf_usq)
    echo "  IVF_USQ_PARAMS nlist=${NLIST} bits_per_dim=${BITS_PER_DIM} rotation_seed=${ROTATION_SEED} rerank_k=${RERANK_K} high_accuracy_scan=${HIGH_ACCURACY_SCAN}"
    ;;
  hnsw_hvq)
    echo "  HNSW_HVQ_PARAMS m=${HNSW_HVQ_M} m_max0=${HNSW_HVQ_M_MAX0} ef_construction=${HNSW_HVQ_EF_CONSTRUCTION} ef_search=${HNSW_HVQ_EF_SEARCH} nbits=${HNSW_HVQ_NBITS}"
    ;;
esac

cd "$ROOT_DIR"

feature_args=()
if [[ -n "$FEATURES" ]]; then
  feature_args+=(--features "$FEATURES")
fi

profile_args=()
if [[ "$PROFILE" == "release" ]]; then
  profile_args+=(--release)
fi

declare -a create_ms_list=()
declare -a insert_ms_list=()
declare -a optimize_ms_list=()
declare -a search_ms_list=()
declare -a total_ms_list=()

for run in $(seq 1 "$REPEATS"); do
  echo "RUN_START run=${run}/${REPEATS}"
  set +e
  output="$(
    HANNSSDB_OPT_BENCH_N="$N" \
    HANNSSDB_OPT_BENCH_DIM="$DIM" \
    HANNSSDB_OPT_BENCH_METRIC="$METRIC" \
    HANNSSDB_OPT_BENCH_TOPK="$TOPK" \
    HANNSSDB_OPT_BENCH_INDEX_KIND="$INDEX_KIND" \
    HANNSSDB_OPT_BENCH_NLIST="$NLIST" \
    HANNSSDB_OPT_BENCH_BITS_PER_DIM="$BITS_PER_DIM" \
    HANNSSDB_OPT_BENCH_ROTATION_SEED="$ROTATION_SEED" \
    HANNSSDB_OPT_BENCH_RERANK_K="$RERANK_K" \
    HANNSSDB_OPT_BENCH_HIGH_ACCURACY_SCAN="$HIGH_ACCURACY_SCAN" \
    HANNSSDB_OPT_BENCH_HNSW_HVQ_M="$HNSW_HVQ_M" \
    HANNSSDB_OPT_BENCH_HNSW_HVQ_M_MAX0="$HNSW_HVQ_M_MAX0" \
    HANNSSDB_OPT_BENCH_HNSW_HVQ_EF_CONSTRUCTION="$HNSW_HVQ_EF_CONSTRUCTION" \
    HANNSSDB_OPT_BENCH_HNSW_HVQ_EF_SEARCH="$HNSW_HVQ_EF_SEARCH" \
    HANNSSDB_OPT_BENCH_HNSW_HVQ_NBITS="$HNSW_HVQ_NBITS" \
    HANNSSDB_OPT_BENCH_QUERY_EF_SEARCH="$QUERY_EF_SEARCH" \
    HANNSSDB_OPT_BENCH_QUERY_NPROBE="$QUERY_NPROBE" \
    cargo test -p hannsdb-core \
      "${profile_args[@]+${profile_args[@]}}" \
      "${feature_args[@]+${feature_args[@]}}" \
      collection_api_optimize_benchmark_entry -- --nocapture --test-threads=1 2>&1
  )"
  run_status=$?
  set -e
  printf '%s\n' "$output"
  if [[ $run_status -ne 0 ]]; then
    echo "benchmark run ${run} failed with status ${run_status}" >&2
    exit "$run_status"
  fi

  timing_line="$(printf '%s\n' "$output" | grep -m1 '^OPT_BENCH_TIMING_MS ' || true)"
  if [[ -z "$timing_line" ]]; then
    echo "missing OPT_BENCH_TIMING_MS in run ${run}" >&2
    exit 1
  fi
  echo "RUN_RAW_TIMING run=${run} ${timing_line}"

  create_ms="$(extract_ms_field "$timing_line" "create")"
  insert_ms="$(extract_ms_field "$timing_line" "insert")"
  optimize_ms="$(extract_ms_field "$timing_line" "optimize")"
  search_ms="$(extract_ms_field "$timing_line" "search")"
  total_ms="$(extract_ms_field "$timing_line" "total")"

  create_ms_list+=("$create_ms")
  insert_ms_list+=("$insert_ms")
  optimize_ms_list+=("$optimize_ms")
  search_ms_list+=("$search_ms")
  total_ms_list+=("$total_ms")

  echo "RUN_PARSED_TIMING run=${run} create_ms=${create_ms} insert_ms=${insert_ms} optimize_ms=${optimize_ms} search_ms=${search_ms} total_ms=${total_ms}"
done

median_create_ms="$(median "${create_ms_list[@]}")"
median_insert_ms="$(median "${insert_ms_list[@]}")"
median_optimize_ms="$(median "${optimize_ms_list[@]}")"
median_search_ms="$(median "${search_ms_list[@]}")"
median_total_ms="$(median "${total_ms_list[@]}")"

echo "BENCH_SUMMARY_CONFIG N=${N} DIM=${DIM} METRIC=${METRIC} TOPK=${TOPK} INDEX_KIND=${INDEX_KIND} REPEATS=${REPEATS} FEATURES=${FEATURES:-<none>} PROFILE=${PROFILE}"
echo "BENCH_SUMMARY_RAW_CREATE_MS [${create_ms_list[*]}]"
echo "BENCH_SUMMARY_RAW_INSERT_MS [${insert_ms_list[*]}]"
echo "BENCH_SUMMARY_RAW_OPTIMIZE_MS [${optimize_ms_list[*]}]"
echo "BENCH_SUMMARY_RAW_SEARCH_MS [${search_ms_list[*]}]"
echo "BENCH_SUMMARY_RAW_TOTAL_MS [${total_ms_list[*]}]"
echo "BENCH_SUMMARY_MEDIAN_MS create=${median_create_ms} insert=${median_insert_ms} optimize=${median_optimize_ms} search=${median_search_ms} total=${median_total_ms}"
