#!/usr/bin/env bash
set -euo pipefail

KN_REPO="${KN_REPO:-/Users/ryan/Code/knowhere-rs}"
KN_BENCH_NS="${KN_BENCH_NS:-5000,10000,20000}"
KN_BENCH_DIM="${KN_BENCH_DIM:-1536}"
KN_BENCH_REPEATS="${KN_BENCH_REPEATS:-3}"

if [[ ! -d "$KN_REPO" ]]; then
  echo "ERROR: KN_REPO does not exist: $KN_REPO" >&2
  exit 1
fi

if ! [[ "$KN_BENCH_REPEATS" =~ ^[0-9]+$ ]] || [[ "$KN_BENCH_REPEATS" -lt 1 ]]; then
  echo "ERROR: KN_BENCH_REPEATS must be a positive integer, got: $KN_BENCH_REPEATS" >&2
  exit 1
fi

median_from_list() {
  # stdin: one numeric value per line
  local vals=()
  local line
  while IFS= read -r line; do
    vals+=("$line")
  done < <(sort -n)
  local n="${#vals[@]}"
  if [[ "$n" -eq 0 ]]; then
    echo "nan"
    return
  fi
  if (( n % 2 == 1 )); then
    echo "${vals[$((n / 2))]}"
  else
    awk -v a="${vals[$((n / 2 - 1))]}" -v b="${vals[$((n / 2))]}" 'BEGIN { printf "%.3f\n", (a + b) / 2.0 }'
  fi
}

echo "KN_VARIANCE_CONFIG repo=$KN_REPO ns=$KN_BENCH_NS dim=$KN_BENCH_DIM repeats=$KN_BENCH_REPEATS"

OLD_IFS="$IFS"
IFS=',' read -r -a NS_ARRAY <<< "$KN_BENCH_NS"
IFS="$OLD_IFS"

for n in "${NS_ARRAY[@]}"; do
  n="$(echo "$n" | xargs)"
  if [[ -z "$n" ]]; then
    continue
  fi
  if ! [[ "$n" =~ ^[0-9]+$ ]] || [[ "$n" -lt 1 ]]; then
    echo "ERROR: invalid N value in KN_BENCH_NS: '$n'" >&2
    exit 1
  fi

  declare -a totals=()
  declare -a per_vectors=()

  for ((i = 1; i <= KN_BENCH_REPEATS; i++)); do
    echo "KN_RUN_BEGIN n=$n iter=$i"
    bench_out="$(
      cd "$KN_REPO"
      KNOWHERE_RS_HNSW_BENCH_N="$n" KNOWHERE_RS_HNSW_BENCH_DIM="$KN_BENCH_DIM" \
        cargo test --release -p knowhere-rs --lib bench_hnsw_cosine_build_hotpath_smoke \
        -- --ignored --nocapture 2>&1
    )"
    line="$(echo "$bench_out" | grep -E "hnsw_cosine_build_hotpath_smoke n=" | tail -n 1 || true)"
    if [[ -z "$line" ]]; then
      echo "ERROR: benchmark output line not found for n=$n iter=$i" >&2
      echo "$bench_out" >&2
      exit 1
    fi

    total_ms="$(echo "$line" | sed -E 's/.*total_ms=([0-9.]+).*/\1/')"
    per_vector_ms="$(echo "$line" | sed -E 's/.*per_vector_ms=([0-9.]+).*/\1/')"
    totals+=("$total_ms")
    per_vectors+=("$per_vector_ms")
    echo "KN_RUN_RAW n=$n iter=$i total_ms=$total_ms per_vector_ms=$per_vector_ms"
  done

  totals_lines="$(printf "%s\n" "${totals[@]}")"
  med="$(printf "%s\n" "$totals_lines" | median_from_list)"
  min_max_spread="$(printf "%s\n" "$totals_lines" | awk '
    NR == 1 { min = $1; max = $1 }
    { if ($1 < min) min = $1; if ($1 > max) max = $1 }
    END {
      if (NR == 0) {
        print "min=nan max=nan spread=nan"
      } else {
        printf "min=%.3f max=%.3f spread=%.3f", min, max, (max - min)
      }
    }
  ')"
  raw_csv="$(IFS=, ; echo "${totals[*]}")"
  echo "KN_SUMMARY n=$n dim=$KN_BENCH_DIM repeats=$KN_BENCH_REPEATS raw_total_ms=$raw_csv median_total_ms=$med $min_max_spread"
done
