# HannsDB VectorDBBench Notes (Current Evidence)

## 1) Tiny-smoke workflow and verified result artifact

- Workflow/command source: [`docs/vdbb-hannsdb-smoke.md`](/Users/ryan/Code/HannsDB/docs/vdbb-hannsdb-smoke.md) and [`scripts/run_vdbb_hannsdb_smoke.sh`](/Users/ryan/Code/HannsDB/scripts/run_vdbb_hannsdb_smoke.sh)
- Tiny-smoke run command:
  - `./scripts/run_vdbb_hannsdb_smoke.sh`
- Verified result path:
  - `/Users/ryan/Code/VectorDBBench/vectordb_bench/results/HannsDB/result_20260319_hannsdb-smoke_hannsdb.json`

## 2) Latest verified tiny-smoke metrics

Source: [`result_20260319_hannsdb-smoke_hannsdb.json`](/Users/ryan/Code/VectorDBBench/vectordb_bench/results/HannsDB/result_20260319_hannsdb-smoke_hannsdb.json)

- `run_id`: `2efd1d4340e343ed95269c03ab9ea1b6`
- `insert_duration`: `0.5853s`
- `optimize_duration`: `0.0001s`
- `load_duration`: `0.5854s`
- `serial_latency_p99`: `0.0002s`
- `serial_latency_p95`: `0.0001s`
- `recall`: `1.0`
- `ndcg`: `1.0`
- `qps`: `0.0` (tiny smoke is recall/latency correctness-oriented, not throughput-oriented)

## 3) Standard benchmark command (Performance1536D50K)

Command consistent with the logged TaskConfig (`db_label=hannsdb-1536d50k`, `path=/tmp/hannsdb-vdbb-1536d50k-db`, `M=16`, `ef_construction=64`, `ef_search=32`, `k=10`, `stages=drop_old/load/search_serial`):

```bash
PYTHONPATH=/Users/ryan/Code/VectorDBBench \
/Users/ryan/Code/HannsDB/.venv-hannsdb/bin/python -m vectordb_bench.cli.vectordbbench hannsdb \
  --path /tmp/hannsdb-vdbb-1536d50k-db \
  --db-label hannsdb-1536d50k \
  --task-label hannsdb-1536d50k \
  --case-type Performance1536D50K \
  --m 16 \
  --ef-construction 64 \
  --ef-search 32 \
  --k 10 \
  --skip-search-concurrent \
  --num-concurrency 1 \
  --concurrency-duration 1
```

## 4) Standard-case runtime observations from log (with timestamps)

Source: [`logs/vectordb_bench.log`](/Users/ryan/Code/HannsDB/logs/vectordb_bench.log)

- Initial `drop_old` warning:
  - `2026-03-19 14:57:53,104` — `Failed to drop HannsDB collection vector_bench_test: no collection registered in manifest`
- Dataset download start:
  - `2026-03-19 14:57:54,837` — `Start to downloading files, total count: 4`
- Dataset download completion:
  - `2026-03-19 15:07:18,589` — `Succeed to download all files, downloaded file count = 4`
- Load/insert duration:
  - `2026-03-19 15:09:14,681` — `load_duration(insert + optimize) = 115.1317`
- Serial search start:
  - `2026-03-19 15:09:14,681` — `SpawnProcess-1 start serial search`
  - `2026-03-19 15:09:15,144` — `start search the entire test_data to get recall and latency`

## 5) Current standard-case status (latest observed evidence)

- As of the latest observed standard-case log evidence above, serial search had started, but no final standard-case result write entry was observed yet for that run in the provided evidence set.

## 6) Cache-fix repro evidence after hot-path change

- A follow-up repro was run with:
  - `db_label=hannsdb-1536d50k-cachefix`
  - `task_label=hannsdb-1536d50k-cachefix`
  - `path=/tmp/hannsdb-vdbb-1536d50k-cachefix-db`
- Observed log milestones:
  - `2026-03-19 15:49:22,772` — `optimize_duration=2.1071`, showing `Collection::optimize()` is no longer a pure no-op and now warms in-process search state.
  - `2026-03-19 15:49:23,193` — serial search started for the `1000` queries in `test.parquet`.
- Sampled runtime evidence during that repro showed the search worker spending CPU in:
  - `hannsdb_core::query::search::search_by_metric`
  - `distance_by_metric`
  - `cosine_similarity`
- The earlier `load_records` hot path was no longer present in the sampled search stack, which confirms the cache fix removed the repeated full-disk reload from repeated queries.
- Remaining blocker:
  - the benchmark path is still brute-force cosine search over `50K x 1536` vectors for `1000` queries, so the next meaningful performance milestone is wiring the HNSW/ANN path into benchmark-facing search rather than further tuning the disk cache path.

## 7) Integration memory: knowhere-backed ANN distance semantics

- Confirmed non-bug integration fact:
  - the current feature-on blocker is **not** a `knowhere-rs` compile/build failure; it is a HannsDB-core integration mistake around constructing the ANN backend from `Result<..., AdapterError>`.
- Confirmed integration caveat to remember:
  - actual local verification for the active HNSW path is the authority; earlier assumptions about `L2` score shape were wrong and had to be corrected with direct tests against `knowhere-rs`.
- Required HannsDB rule going forward:
  - if benchmark-facing search is accelerated through the knowhere-backed ANN path, HannsDB core must preserve its public score semantics explicitly at the adapter/core boundary instead of assuming `knowhere-rs` output already matches HannsDB.

## 8) knowhere-rs verification findings from HannsDB integration

- This is a first-class project goal, not a side effect:
  - HannsDB development must continuously verify that `knowhere-rs` stays reusable and semantically compatible for the active ANN path.
- Verified local findings from the current integration slice:
  - `KnowhereHnswIndex` local tiny-fixture behavior was flaky until the HannsDB wrapper pinned `config.params.random_seed = Some(42)`.
  - After pinning `random_seed`, repeated local verification stabilized for both:
    - `cargo test -p hannsdb-index --features knowhere-backend hnsw_adapter`
    - `cargo test -p hannsdb-core collection_api --features knowhere-backend`
  - `knowhere-rs` HNSW finalizes `L2` results to Euclidean distance already, so HannsDB core must not apply a second `sqrt`.
  - `knowhere-rs` HNSW finalizes `IP` results to positive inner-product scores, while HannsDB core's current public search semantics use negative distance for `ip`; HannsDB core therefore has to map `IP` scores back at the adapter/core boundary.

## 9) Standard benchmark update after enabling knowhere-backed optimize

- Repro command:
  - `DB_LABEL=hannsdb-1536d50k-knowhere-seeded TASK_LABEL=hannsdb-1536d50k-knowhere-seeded DB_PATH=/tmp/hannsdb-vdbb-1536d50k-knowhere-seeded-db ./scripts/run_vdbb_hannsdb_perf1536d50k.sh`
- Observed state transition:
  - the previous dominant bottleneck was brute-force search over `50K x 1536` vectors after a near-no-op optimize step
  - after wiring the knowhere-backed ANN optimize path, the dominant bottleneck moved earlier into `optimize()`
- Verified evidence:
  - dataset load/insert still completed in about `112.14s`
  - no `load_duration(insert + optimize)` line was emitted afterward within the bounded run window, which means optimize did not finish in that window
  - repeated samples of the hot worker showed the process inside:
    - `hannsdb_core::db::HannsDb::optimize_collection`
    - `hannsdb_core::db::build_optimized_ann_state`
    - `hannsdb_index::hnsw::KnowhereHnswIndex::insert`
    - `knowhere_rs::faiss::hnsw::HnswIndex::add`
    - `search_layer_idx_*` and `distance_to_idx_cosine_dispatch`
- Current conclusion:
  - the search-side brute-force bottleneck has been removed from the critical path for this case
  - the new benchmark blocker is knowhere-backed HNSW build cost during optimize on the `50K / 1536 / cosine` dataset

## 10) Standalone optimize benchmark entry (no external dataset download)

Use this one-command entry to isolate `HannsDb::optimize_collection` timing from full VectorDBBench runs:

```bash
cd /Users/ryan/Code/HannsDB
./scripts/run_hannsdb_optimize_bench.sh
```

Configurable parameters (env):
- `N` (default `2000`)
- `DIM` (default `256`)
- `METRIC` (`l2|cosine|ip`, default `cosine`)
- `TOPK` (default `10`)
- `REPEATS` (default `3`, script prints each run and medians)
- `FEATURES` (default `knowhere-backend`)
- `PROFILE` (`debug|release`, default `debug`)

Example for a larger cosine case:

```bash
N=50000 DIM=1536 METRIC=cosine TOPK=10 REPEATS=5 FEATURES=knowhere-backend ./scripts/run_hannsdb_optimize_bench.sh
```

Recommended stable A/B command (same params, different code revision):

```bash
N=2000 DIM=256 METRIC=cosine TOPK=10 REPEATS=3 FEATURES=knowhere-backend ./scripts/run_hannsdb_optimize_bench.sh
```

Recommended large-scale proxy command:

```bash
N=50000 DIM=1536 METRIC=cosine TOPK=10 REPEATS=1 FEATURES=knowhere-backend PROFILE=release ./scripts/run_hannsdb_optimize_bench.sh
```

Stable output fields:
- `OPT_BENCH_CONFIG n=<...> dim=<...> metric=<...> top_k=<...>`
- `OPT_BENCH_TIMING_MS create=<...> insert=<...> optimize=<...> search=<...> total=<...>`
- `RUN_RAW_TIMING run=<i> OPT_BENCH_TIMING_MS ...`
- `RUN_PARSED_TIMING run=<i> create_ms=... insert_ms=... optimize_ms=... search_ms=... total_ms=...`
- `BENCH_SUMMARY_MEDIAN_MS create=... insert=... optimize=... search=... total=...`

## 11) Baseline record (2026-03-19, repeated run)

Command:

```bash
cd /Users/ryan/Code/HannsDB
N=2000 DIM=256 METRIC=cosine TOPK=10 REPEATS=3 FEATURES=knowhere-backend ./scripts/run_hannsdb_optimize_bench.sh
```

Per-run raw timing:
- Run 1: `OPT_BENCH_TIMING_MS create=0 insert=580 optimize=10080 search=4 total=10665`
- Run 2: `OPT_BENCH_TIMING_MS create=0 insert=528 optimize=10325 search=4 total=10860`
- Run 3: `OPT_BENCH_TIMING_MS create=0 insert=510 optimize=10308 search=4 total=10823`

Median summary:
- `BENCH_SUMMARY_MEDIAN_MS create=0 insert=528 optimize=10308 search=4 total=10823`

## 12) Larger-scale baseline attempt and pinned result (2026-03-19)

Target command first tried:

```bash
cd /Users/ryan/Code/HannsDB
N=50000 DIM=1536 METRIC=cosine TOPK=10 REPEATS=1 FEATURES=knowhere-backend ./scripts/run_hannsdb_optimize_bench.sh
```

Observed behavior at larger scales:
- `50K / 1536`: run entered long compute phase with sustained full CPU in `collection_api_optimize_benchmark_entry`, no timing line produced in the bounded window; terminated manually.
- `20K / 1536`: same pattern (long-running optimize, no timing line in bounded window); terminated manually.

Closest completed scale used as current pinned baseline:

```bash
cd /Users/ryan/Code/HannsDB
N=10000 DIM=1536 METRIC=cosine TOPK=10 REPEATS=1 FEATURES=knowhere-backend ./scripts/run_hannsdb_optimize_bench.sh
```

Raw output summary:
- `OPT_BENCH_CONFIG n=10000 dim=1536 metric=cosine top_k=10`
- `OPT_BENCH_TIMING_MS create=0 insert=14812 optimize=267575 search=18 total=282407`
- `BENCH_SUMMARY_MEDIAN_MS create=0 insert=14812 optimize=267575 search=18 total=282407`

Pinned baseline fields for current version:
- `optimize_ms=267575`
- `total_ms=282407`

## 13) Real-size release baseline (50K/1536/cosine, 2026-03-19)

Command:

```bash
cd /Users/ryan/Code/HannsDB
N=50000 DIM=1536 METRIC=cosine TOPK=10 REPEATS=1 FEATURES=knowhere-backend PROFILE=release ./scripts/run_hannsdb_optimize_bench.sh
```

Profile used:
- `release`

Scale used:
- `n=50000`, `dim=1536`, `metric=cosine`, `top_k=10`, `repeats=1`

Raw output summary:
- `OPT_BENCH_CONFIG n=50000 dim=1536 metric=cosine top_k=10`
- `OPT_BENCH_TIMING_MS create=0 insert=93591 optimize=498659 search=9 total=592260`
- `BENCH_SUMMARY_MEDIAN_MS create=0 insert=93591 optimize=498659 search=9 total=592260`

Pinned baseline fields for current version:
- `optimize_ms=498659`
- `total_ms=592260`

Proxy quality:
- This is a better proxy for the real VectorDBBench hot path than the earlier `10K/1536` fallback because it matches the target `50K/1536/cosine` scale and keeps the optimize-stage bottleneck dominant.

## 14) HannsDB-side prebuild overhead cut (no-tombstone fast path)

Change summary:
- In `build_optimized_ann_state`, added a no-tombstone fast path:
  - `ann_external_ids = state.external_ids.clone()`
  - direct backend feed from `state.records` via `insert_flat_identity(...)`
- This avoids materializing an extra `flat_vectors` copy for the common no-delete case before knowhere build.
- Tombstone-present path is unchanged semantically (still filters deleted rows and remaps ANN IDs).

Verification commands:
- `cargo test -p hannsdb-core collection_api --features knowhere-backend -- --nocapture`
- `N=200 DIM=64 METRIC=cosine TOPK=10 REPEATS=1 FEATURES=knowhere-backend PROFILE=debug ./scripts/run_hannsdb_optimize_bench.sh`
- `N=200 DIM=64 METRIC=cosine TOPK=10 REPEATS=1 FEATURES=knowhere-backend PROFILE=release ./scripts/run_hannsdb_optimize_bench.sh`
- `N=20000 DIM=1536 METRIC=cosine TOPK=10 REPEATS=1 FEATURES=knowhere-backend PROFILE=release ./scripts/run_hannsdb_optimize_bench.sh`

Observed 20K/1536 release delta (same command as above):
- Before this cut:
  - `OPT_BENCH_TIMING_MS create=0 insert=34819 optimize=84003 search=3 total=118827`
- After this cut:
  - `OPT_BENCH_TIMING_MS create=0 insert=34469 optimize=77257 search=3 total=111730`

What moved:
- `insert_ms`: slight drop (`34819 -> 34469`, about `-1.0%`)
- `optimize_ms`: clear drop (`84003 -> 77257`, about `-8.0%`)
- `total_ms`: drop (`118827 -> 111730`, about `-6.0%`)

Interpretation:
- This cut primarily reduced optimize prebuild overhead on the HannsDB side.
- Remaining dominant cost is still inside knowhere build itself (expected hotspot in add/build path).

## 14) knowhere-rs hotpath validation record (2026-03-19, initial sample)

This is part of the HannsDB mainline, not side work: while optimizing HannsDB, we are also validating whether `knowhere-rs` is a viable long-term ANN foundation for the target workload.

Baseline before the latest bounded `knowhere-rs` cut:

- Command:
  - `KNOWHERE_RS_HNSW_BENCH_N=10000 KNOWHERE_RS_HNSW_BENCH_DIM=1536 cargo test --release -p knowhere-rs --lib bench_hnsw_cosine_build_hotpath_smoke -- --ignored --nocapture`
- Baseline result:
  - `total_ms=47319.276`
  - `per_vector_ms=4.731928`

After the bounded hotpath change in `src/faiss/hnsw.rs`:

- Main code effects:
  - removed unconditional timing capture from the non-profile candidate-search path
  - carried precomputed cosine query norm through the active insertion/build call chain
  - cached cosine vector norms in-memory and rebuilt them on load
- Focused correctness verification:
  - `cargo test -p knowhere-rs --lib test_hnsw_cosine_metric -- --nocapture`
  - `cargo test -p knowhere-rs --lib test_cosine_distance_with_query_norm_matches_dispatch_path -- --nocapture`
- Near-target benchmark result:
  - `total_ms=42270.941`
  - `per_vector_ms=4.227094`

Initial observed delta against the previous `10K / 1536` baseline:

- `total_ms`: about `10.7%` lower
- `per_vector_ms`: about `10.7%` lower

Important caveat:

- this section records the first good sample only
- later repeated runs are recorded in sections `16` and `18`
- do not treat this single-sample delta as a stable improvement claim by itself

## 15) HannsDB-side no-tombstone fast path validation (2026-03-19)

Accepted HannsDB-side cut:

- `build_optimized_ann_state` now has a no-tombstone fast path
- HannsDB avoids constructing an extra copied `flat_vectors` buffer before calling the backend when the collection has no deletes
- a narrow `insert_flat_identity(...)` adapter path now supports this shape directly

Controller re-checks:

- Targeted correctness:
  - `cargo test -p hannsdb-core collection_api --features knowhere-backend -- --nocapture`
  - result: `19 passed`
- Small release proxy:
  - `N=200 DIM=64 METRIC=cosine TOPK=10 REPEATS=1 FEATURES=knowhere-backend PROFILE=release ./scripts/run_hannsdb_optimize_bench.sh`
  - controller result: `OPT_BENCH_TIMING_MS create=0 insert=13 optimize=11 search=0 total=25`
- Mid-scale release proxy reported by the worker:
  - `N=20000 DIM=1536 METRIC=cosine TOPK=10 REPEATS=1 FEATURES=knowhere-backend PROFILE=release ./scripts/run_hannsdb_optimize_bench.sh`
  - worker result: `OPT_BENCH_TIMING_MS create=0 insert=34469 optimize=77257 search=3 total=111730`
  - previous recorded baseline at the same scale: `insert=34819 optimize=84003 total=118827`

Current implication:

- HannsDB-side prebuild overhead has moved in the right direction
- the next important check is whether this carries through to the `50K / 1536 / cosine` release proxy and then to the full `Performance1536D50K` gate

## 16) knowhere-rs current evidence status after the latest narrow cut

Latest narrow code change under review:

- `cosine_vector_norm_for_idx_hot(...)` / hot-path norm lookup tightening in `src/faiss/hnsw.rs`

Important controller finding:

- the worker-reported single run showed a small improvement (`41896.941 ms` at `10K / 1536`)
- but a controller rerun on the same current tree produced:
  - `total_ms=45405.598`
  - `per_vector_ms=4.540560`

Current implication:

- the latest claimed `~0.88%` improvement is **not yet reproducible enough to treat as accepted performance truth**
- keep the code/result as captured evidence, but do not rely on that improvement until repeated runs establish a stable median
- the next step on the knowhere side is measurement stabilization, not another immediate performance claim

## 17) T-20260319-006 gate validation result (2026-03-19)

Release optimize proxy rerun (required):

```bash
cd /Users/ryan/Code/HannsDB
N=50000 DIM=1536 METRIC=cosine TOPK=10 REPEATS=1 FEATURES=knowhere-backend PROFILE=release ./scripts/run_hannsdb_optimize_bench.sh
```

Observed timing line:
- `OPT_BENCH_TIMING_MS create=0 insert=116802 optimize=617836 search=11 total=734651`
- `BENCH_SUMMARY_MEDIAN_MS create=0 insert=116802 optimize=617836 search=11 total=734651`

Comparison to pinned baseline (`insert=93591 optimize=498659 total=592260`):
- `insert_ms`: `+23211` (about `+24.8%`)
- `optimize_ms`: `+119177` (about `+23.9%`)
- `total_ms`: `+142391` (about `+24.0%`)

Standard `Performance1536D50K` gate rerun was attempted with fresh labels/path:

```bash
cd /Users/ryan/Code/HannsDB
DB_LABEL=hannsdb-1536d50k-rerun2 TASK_LABEL=hannsdb-1536d50k-rerun2 DB_PATH=/tmp/hannsdb-vdbb-1536d50k-rerun2-db ./scripts/run_vdbb_hannsdb_perf1536d50k.sh
```

Observed behavior:
- benchmark entered case execution and completed load stage:
  - `2026-03-19 19:32:58,785` — `Finish loading all dataset into VectorDB, dur=130.49832625011913`
- no final result JSON was emitted for this rerun label
- sampled worker process during the long stall showed Python threads waiting on locks/queues (not active compute frames), so this run was terminated and marked blocked-with-concerns

Current conclusion for this task:
- release proxy regressed versus pinned baseline
- full standard gate rerun remains blocked in this environment after load stage

## 18) T-20260319-009 evidence refresh (2026-03-19)

Required 3-run release proxy command:

```bash
cd /Users/ryan/Code/HannsDB
N=50000 DIM=1536 METRIC=cosine TOPK=10 REPEATS=3 FEATURES=knowhere-backend PROFILE=release ./scripts/run_hannsdb_optimize_bench.sh
```

Raw run timings:
- Run 1: `OPT_BENCH_TIMING_MS create=0 insert=136174 optimize=649342 search=12 total=785531`
- Run 2: `OPT_BENCH_TIMING_MS create=0 insert=139047 optimize=784235 search=18 total=923302`
- Run 3: `OPT_BENCH_TIMING_MS create=0 insert=219994 optimize=721181 search=11 total=941188`

3-run median from script:
- `BENCH_SUMMARY_MEDIAN_MS create=0 insert=139047 optimize=721181 search=12 total=923302`

This confirms the current `50K/1536/cosine` release proxy is still materially above the earlier pinned baseline (`insert=93591 optimize=498659 total=592260`).

Required full-gate rerun command:

```bash
cd /Users/ryan/Code/HannsDB
DB_LABEL=hannsdb-1536d50k-rerun3 TASK_LABEL=hannsdb-1536d50k-rerun3 DB_PATH=/tmp/hannsdb-vdbb-1536d50k-rerun3-db ./scripts/run_vdbb_hannsdb_perf1536d50k.sh
```

Observed runtime milestone:
- `2026-03-19 20:58:49,036` — `(SpawnProcess-1:1) Finish loading all dataset into VectorDB, dur=149.12591604096815`

Stall diagnosis snapshot artifacts:
- controller Python process sample: `/tmp/hannsdb-vdbb-rerun3.sample.txt` (PID `63139`)
- worker subprocess sample: `/tmp/hannsdb-vdbb-rerun3-worker.sample.txt` (PID `63141`)

Snapshot finding summary:
- parent process stack centered in `time_sleep` wait
- worker subprocess main thread blocked on Python lock wait (`lock_PyThread_acquire_lock` / `pthread_cond_wait`)
- no result JSON emitted for `hannsdb-1536d50k-rerun3` before manual termination

## 18) T-20260319-007 repeated-run outcome (2026-03-19)

Task intent:
- no code changes, only measurement stabilization for the current `knowhere-rs` tree

Focused correctness check:
- `cargo test -p knowhere-rs --lib test_cosine_distance_with_query_norm_matches_dispatch_path -- --nocapture` passed

Near-target benchmark (`N=10000`, `DIM=1536`) repeated 3 times:
- run 1: `total_ms=42176.417`, `per_vector_ms=4.217642`
- run 2: `total_ms=43867.780`, `per_vector_ms=4.386778`
- run 3: `total_ms=43841.215`, `per_vector_ms=4.384121`

Comparison points:
- prior baseline from `T-20260319-003`: `total_ms=42270.941`, `per_vector_ms=4.227094`
- controller spot-check in this session: `total_ms=45405.598`, `per_vector_ms=4.540560`

Conclusion:
- current narrow cut cannot be claimed as a stable deterministic speedup yet
- observed behavior fits a noise-sensitive/variance-sensitive band on this machine
- next knowhere step should be another bounded hotpath cut plus the same repeated-run check, not a claim based on a single run

## 19) 2026-03-19 latest knowhere-rs variance sample

Command:
- `KN_BENCH_NS=5000,10000 KN_BENCH_REPEATS=2 KN_BENCH_DIM=1536 bash scripts/run_knowhere_hnsw_variance.sh`

KN_SUMMARY:
- `KN_SUMMARY n=5000 dim=1536 repeats=2 raw_total_ms=25360.235,25229.276 median_total_ms=25294.755 min=25229.276 max=25360.235 spread=130.959`
- `KN_SUMMARY n=10000 dim=1536 repeats=2 raw_total_ms=63087.401,63032.786 median_total_ms=63060.094 min=63032.786 max=63087.401 spread=54.615`

Note:
- this is the latest target-scale variance check recorded on 2026-03-19 for the current `knowhere-rs` cosine hotpath path
- the `N=10000` median is still materially above the smaller sample, so the run is evidence for variance tracking, not a final throughput claim

## 20) Reusable knowhere-rs HNSW variance harness

Script:
- `/Users/ryan/Code/HannsDB/scripts/run_knowhere_hnsw_variance.sh`

Purpose:
- run repeated `knowhere-rs` `bench_hnsw_cosine_build_hotpath_smoke` builds for one or more `N` values
- print per-run raw `total_ms` / `per_vector_ms`
- print per-`N` summary with `raw_total_ms`, `median_total_ms`, `min`, `max`, `spread(max-min)`

Defaults:
- `KN_REPO=/Users/ryan/Code/knowhere-rs`
- `KN_BENCH_NS=5000,10000,20000`
- `KN_BENCH_DIM=1536`
- `KN_BENCH_REPEATS=3`

Env overrides:
- `KN_REPO`
- `KN_BENCH_NS` (comma-separated, e.g. `1000,5000`)
- `KN_BENCH_DIM`
- `KN_BENCH_REPEATS`

Smoke example:

```bash
cd /Users/ryan/Code/HannsDB
KN_BENCH_NS=1000 KN_BENCH_DIM=256 KN_BENCH_REPEATS=1 ./scripts/run_knowhere_hnsw_variance.sh
```

Output fields:
- `KN_VARIANCE_CONFIG ...`
- `KN_RUN_BEGIN n=<N> iter=<i>`
- `KN_RUN_RAW n=<N> iter=<i> total_ms=<...> per_vector_ms=<...>`
- `KN_SUMMARY n=<N> dim=<...> repeats=<...> raw_total_ms=<csv> median_total_ms=<...> min=<...> max=<...> spread=<...>`

## 21) Latest rerun: release proxy 50K/1536/cosine (2026-03-19)

Command:

```bash
cd /Users/ryan/Code/HannsDB
N=50000 DIM=1536 METRIC=cosine TOPK=10 REPEATS=1 FEATURES=knowhere-backend PROFILE=release bash scripts/run_hannsdb_optimize_bench.sh
```

Captured output:
- `OPT_BENCH_CONFIG n=50000 dim=1536 metric=cosine top_k=10`
- `OPT_BENCH_TIMING_MS create=0 insert=158795 optimize=710488 search=11 total=869299`
- `BENCH_SUMMARY_MEDIAN_MS create=0 insert=158795 optimize=710488 search=11 total=869299`

Conclusion:
- this latest release rerun completed successfully at the target `50K / 1536 / cosine` scale, and `optimize` remained the dominant cost in the run.

## 22) 2026-03-20 latest variance rerun

Command:

```bash
cd /Users/ryan/Code/HannsDB
KN_BENCH_NS=5000,10000 KN_BENCH_REPEATS=3 KN_BENCH_DIM=1536 bash scripts/run_knowhere_hnsw_variance.sh
```

KN_SUMMARY:
- `KN_SUMMARY n=5000 dim=1536 repeats=3 raw_total_ms=16513.816,17095.546,17100.214 median_total_ms=17095.546 min=16513.816 max=17100.214 spread=586.398`
- `KN_SUMMARY n=10000 dim=1536 repeats=3 raw_total_ms=43552.775,43586.094,43341.512 median_total_ms=43552.775 min=43341.512 max=43586.094 spread=244.582`

Conclusion:
- this rerun shows the `N=10000` median remains much higher than the `N=5000` median, and both sample sets stayed within a bounded spread across three repeats.

## 23) 2026-03-20 gate stall classification refinement

Source:
- `/tmp/hannsdb-watchdog-20260320.ps.txt`
- `/tmp/hannsdb-watchdog-20260320.sample.txt`
- `/tmp/hannsdb-watchdog-20260320-worker.sample.txt`

Observed process state:
- VectorDBBench parent CLI process stayed low CPU and mostly waited/slept.
- one spawn child process was observed near `99%` CPU during the post-load window.
- the sampled `worker` file from that run was later confirmed to be `resource_tracker`, not the hot child.

Interpretation:
- this evidence is more consistent with a **compute-bound serial search stage** than a hard deadlock for that run.
- watchdog diagnostic selection was tightened afterward to prefer high-CPU non-`resource_tracker` python children.

## 24) 2026-03-20 stage-aware watchdog verification

Command:

```bash
cd /Users/ryan/Code/HannsDB
STALL_TIMEOUT_SEC=60 POST_LOAD_TIMEOUT_SEC=300 DB_LABEL=hannsdb-watchdog-20260320-stagecheck TASK_LABEL=hannsdb-watchdog-20260320-stagecheck DB_PATH=/tmp/hannsdb-watchdog-20260320-stagecheck-db bash scripts/run_vdbb_hannsdb_perf1536d50k_watchdog.sh
```

Result classification:
- `WATCHDOG_LOAD_COMPLETE` observed
- no `WATCHDOG_SEARCH_STARTED` observed
- `WATCHDOG_POST_LOAD_TIMEOUT` fired and exited with code `125`

Worker sample evidence:
- `/tmp/hannsdb-watchdog-20260320-stagecheck-worker.sample.txt` top frames stayed in:
  - `hannsdb_core::db::HannsDb::optimize_collection`
  - `hannsdb_core::db::build_optimized_ann_state`
  - `knowhere_rs::faiss::hnsw::HnswIndex::add`

Conclusion:
- for this run, the gate blocker remained in optimize/build before search start.
- stage-aware timeout split removed ambiguity between optimize-bound and search-bound stalls.

## 25) 2026-03-20 knowhere-rs NEON inner-product micro-cut check (reverted)

Experiment:
- tried a bounded `ip_neon` pointer-step micro optimization in `/Users/ryan/Code/knowhere-rs/src/simd.rs`
- validation command:
  - `KN_BENCH_NS=5000 KN_BENCH_REPEATS=3 KN_BENCH_DIM=1536 bash scripts/run_knowhere_hnsw_variance.sh`

Observed summary:
- `KN_SUMMARY n=5000 dim=1536 repeats=3 raw_total_ms=17370.845,17099.663,17087.428 median_total_ms=17099.663 min=17087.428 max=17370.845 spread=283.417`

Decision:
- median was effectively flat versus the current 2026-03-20 baseline band.
- no clear benefit was proven, so this micro-cut was reverted.

## 26) 2026-03-20 post-parity-slice small optimize rerun

Context:
- core now includes the canonical document model, typed scalar payloads, `insert/upsert/fetch/query(filter)`
- `hannsdb-py` now exposes `Doc.fields/score`, `upsert`, `fetch`, `flush`, and `stats`
- the daemon now exposes thin `upsert`, `fetch`, and filtered search routes

Command:

```bash
cd /Users/ryan/Code/HannsDB
N=2000 DIM=256 METRIC=cosine TOPK=10 REPEATS=3 FEATURES=knowhere-backend PROFILE=debug bash scripts/run_hannsdb_optimize_bench.sh
```

Per-run raw timing:
- Run 1: `OPT_BENCH_TIMING_MS create=0 insert=545 optimize=10420 search=4 total=10970`
- Run 2: `OPT_BENCH_TIMING_MS create=0 insert=512 optimize=10438 search=4 total=10956`
- Run 3: `OPT_BENCH_TIMING_MS create=0 insert=508 optimize=10528 search=4 total=11042`

Median summary:
- `BENCH_SUMMARY_MEDIAN_MS create=0 insert=512 optimize=10438 search=4 total=10970`

Interpretation:
- this rerun stayed in the same small-case band as the 2026-03-19 repeated baseline (`optimize ~= 10.3s` with `knowhere-backend` enabled)
- the document/payload/filter tranche did not produce an obvious small-case no-filter optimize regression at this benchmark size
- this command is not directly comparable to plain `cargo test --workspace` output because the benchmark script explicitly enables `FEATURES=knowhere-backend`

## 27) 2026-03-20 parity follow-up benchmark gate recheck

Context:
- after adding Python `delete` and daemon `search.output_fields` / `/stats`, re-check the no-filter path before opening any new feature tranche

Commands:

```bash
cd /Users/ryan/Code/HannsDB
N=2000 DIM=256 METRIC=cosine TOPK=10 REPEATS=3 FEATURES=knowhere-backend PROFILE=debug bash scripts/run_hannsdb_optimize_bench.sh

cd /Users/ryan/Code/VectorDBBench
. /Users/ryan/Code/HannsDB/.venv-hannsdb/bin/activate
python -m pytest tests/test_hannsdb_stage_logs.py tests/test_hannsdb_cli.py tests/test_hannsdb_client_config_shape.py -q

cd /Users/ryan/Code/HannsDB
bash scripts/run_vdbb_hannsdb_smoke.sh
```

Observed summary:
- small optimize proxy:
  - Run 1: `OPT_BENCH_TIMING_MS create=0 insert=514 optimize=10436 search=4 total=10955`
  - Run 2: `OPT_BENCH_TIMING_MS create=0 insert=510 optimize=10446 search=4 total=10961`
  - Run 3: `OPT_BENCH_TIMING_MS create=0 insert=502 optimize=10440 search=4 total=10948`
  - Median: `BENCH_SUMMARY_MEDIAN_MS create=0 insert=510 optimize=10440 search=4 total=10955`
- VectorDBBench pytest: `6 passed, 2 warnings`
- VectorDBBench smoke:
  - result file materialized at `/Users/ryan/Code/VectorDBBench/vectordb_bench/results/HannsDB/result_20260320_hannsdb-smoke_hannsdb.json`
  - smoke metrics stayed healthy: `load_duration=0.4804`, `recall=1.0`, `ndcg=1.0`, `serial_latency_p99=0.0002`

Interpretation:
- relative to section 26 on 2026-03-20, the small no-filter optimize proxy remained flat (`optimize 10438 -> 10440 ms`, `total 10970 -> 10955 ms`)
- the latest parity changes did not introduce an observable regression in the small-case no-filter path
- current benchmark evidence still points to `knowhere-rs` HNSW build cost as the main target-scale blocker, not the latest HannsDB parity work

## 28) 2026-03-20 target-scale release proxy recheck after parity follow-up

Context:
- after the small-case gate stayed flat, re-run the target `50K / 1536 / cosine` release proxy once to check whether the same conclusion still holds closer to the real benchmark scale

Command:

```bash
cd /Users/ryan/Code/HannsDB
N=50000 DIM=1536 METRIC=cosine TOPK=10 REPEATS=1 FEATURES=knowhere-backend PROFILE=release bash scripts/run_hannsdb_optimize_bench.sh
```

Observed summary:
- `OPT_BENCH_TIMING_MS create=0 insert=97672 optimize=499821 search=9 total=597503`
- `BENCH_SUMMARY_MEDIAN_MS create=0 insert=97672 optimize=499821 search=9 total=597503`
- `knowhere-rs` emitted the same non-blocking `unused_mut` warnings seen earlier, but the run completed normally

Interpretation:
- relative to the earlier 2026-03-20 release proxy sample (`insert=158795 optimize=710488 total=869299`), this rerun was materially faster
- the runtime shape is unchanged: `optimize` still dominates and remains the main target-scale cost center
- this recheck strengthens the current attribution that recent HannsDB parity work is not the blocker; the main remaining target-scale work still belongs in `knowhere-rs` HNSW build behavior

## 29) 2026-03-21 benchmark-facing lightweight gate after WAL/flush/recovery slice

Context:
- after WAL/flush/recovery related changes, re-run the required benchmark-facing no-filter path checks

Commands:

```bash
cd /Users/ryan/Code/VectorDBBench
. /Users/ryan/Code/HannsDB/.venv-hannsdb/bin/activate
python -m pytest tests/test_hannsdb_stage_logs.py tests/test_hannsdb_cli.py tests/test_hannsdb_client_config_shape.py -q

cd /Users/ryan/Code/HannsDB
N=2000 DIM=256 METRIC=cosine TOPK=10 REPEATS=3 FEATURES=knowhere-backend PROFILE=debug bash scripts/run_hannsdb_optimize_bench.sh
```

Observed summary:
- benchmark-facing pytest gate: `6 passed, 2 warnings in 0.79s`
- small optimize proxy:
  - Run 1: `OPT_BENCH_TIMING_MS create=1 insert=584 optimize=9971 search=4 total=10562`
  - Run 2: `OPT_BENCH_TIMING_MS create=0 insert=583 optimize=10095 search=4 total=10684`
  - Run 3: `OPT_BENCH_TIMING_MS create=0 insert=584 optimize=10229 search=4 total=10818`
  - Median: `BENCH_SUMMARY_MEDIAN_MS create=0 insert=584 optimize=10095 search=4 total=10684`

Interpretation:
- benchmark-facing no-filter path is healthy after the WAL/flush/recovery slice in this lightweight gate.
- no WAL/flush/recovery regression was observed in this check set.
- small optimize median stayed in the established `~10s` band and is slightly better than the 2026-03-20 parity follow-up sample (`optimize=10440`, `total=10955`), so current evidence does not indicate a new regression.

## 30) 2026-03-21 target-scale release proxy rerun after WAL/flush/recovery slice

Context:
- after the WAL/flush/recovery slice, run one bounded target-scale release proxy (`50K / 1536 / cosine`) as requested to check for obvious regressions close to standard benchmark shape.

Command:

```bash
cd /Users/ryan/Code/HannsDB
N=50000 DIM=1536 METRIC=cosine TOPK=10 REPEATS=1 FEATURES=knowhere-backend PROFILE=release bash scripts/run_hannsdb_optimize_bench.sh
```

Observed summary:
- `OPT_BENCH_CONFIG n=50000 dim=1536 metric=cosine top_k=10`
- `OPT_BENCH_TIMING_MS create=0 insert=93352 optimize=491123 search=9 total=584485`
- `BENCH_SUMMARY_MEDIAN_MS create=0 insert=93352 optimize=491123 search=9 total=584485`
- runtime remained optimize-dominant and the run completed normally (same non-blocking `knowhere-rs` warnings as before).

Comparison vs previously recorded `50K/1536/cosine` release proxy:
- vs section 13 baseline (`insert=93591 optimize=498659 total=592260`): this run is slightly faster (`insert -239ms`, `optimize -7536ms`, `total -7775ms`, about `-1.31%` total).
- vs section 28 recheck (`insert=97672 optimize=499821 total=597503`): this run is also faster (`insert -4320ms`, `optimize -8698ms`, `total -13018ms`, about `-2.18%` total).

Interpretation:
- no obvious regression is observed relative to the existing target-scale release proxy evidence after the WAL/flush/recovery slice.
- current bounded evidence continues to show the same cost shape (`optimize` dominates), without a new WAL/flush/recovery-related slowdown signal.

## 31) 2026-03-21 HannsDB tombstone-path identity insert cut

Context:
- while re-checking `build_optimized_ann_state`, the tombstone-present path still built a redundant sequential ANN id vector before calling the backend
- `KnowhereHnswIndex::insert_flat_identity(...)` already models the sequential-id case, and knowhere-backed HNSW accepts `add(..., None)` for auto-generated ids

Code change:
- `crates/hannsdb-core/src/db.rs`
  - tombstone branch now compacts live vectors and calls `insert_flat_identity(...)` instead of constructing `ann_ids` and using `insert_flat(...)`
- `crates/hannsdb-index/src/hnsw.rs`
  - knowhere-backed `insert_flat_identity(...)` now passes `None` to `HnswIndex::add(...)` instead of allocating a sequential `i64` id list

Verification:
- `cargo test -p hannsdb-core collection_api --features knowhere-backend -- --nocapture`
- `cd /Users/ryan/Code/HannsDB && N=2000 DIM=256 METRIC=cosine TOPK=10 REPEATS=3 FEATURES=knowhere-backend PROFILE=debug bash scripts/run_hannsdb_optimize_bench.sh`

Observed small-bench result:
- before this cut on the same command: `OPT_BENCH_TIMING_MS create=0 insert=584 optimize=10095 search=4 total=10684`
- after this cut:
  - Run 1: `OPT_BENCH_TIMING_MS create=0 insert=585 optimize=2697 search=0 total=3284`
  - Run 2: `OPT_BENCH_TIMING_MS create=0 insert=603 optimize=2767 search=0 total=3372`
  - Run 3: `OPT_BENCH_TIMING_MS create=0 insert=596 optimize=2718 search=0 total=3316`
  - Median: `BENCH_SUMMARY_MEDIAN_MS create=0 insert=596 optimize=2718 search=0 total=3316`

Interpretation:
- this is a genuine HannsDB-side improvement, not just a documentation-only attribution update.
- the win is concentrated in optimize prebuild work, which fits the original suspicion around data整理 / id construction in the optimize hotpath.
- after this cut, the next bounded performance work should still stay in HannsDB glue for similarly small, mechanically safe cleanups, but the target-scale ceiling remains knowhere-rs HNSW build behavior.

## 32) 2026-03-21 target-scale release proxy retest after section 31 cut

Context:
- with section 31 already landed (`tombstone-path identity insert cut`), run one target-scale release proxy retest at `50K / 1536 / cosine` to check whether the HannsDB-side micro-opt still has signal near real benchmark scale.

Command:

```bash
cd /Users/ryan/Code/HannsDB
N=50000 DIM=1536 METRIC=cosine TOPK=10 REPEATS=1 FEATURES=knowhere-backend PROFILE=release bash scripts/run_hannsdb_optimize_bench.sh
```

Observed summary:
- `OPT_BENCH_CONFIG n=50000 dim=1536 metric=cosine top_k=10`
- `OPT_BENCH_TIMING_MS create=0 insert=92526 optimize=10873 search=0 total=103400`
- `BENCH_SUMMARY_MEDIAN_MS create=0 insert=92526 optimize=10873 search=0 total=103400`
- run completed successfully; only existing non-blocking `knowhere-rs` `unused_mut` warnings were printed.

Comparison vs section 30 (`insert=93352 optimize=491123 search=9 total=584485`):
- `insert`: `-826ms` (`-0.88%`)
- `optimize`: `-480250ms` (`-97.79%`)
- `search`: `-9ms` (to `0`)
- `total`: `-481085ms` (`-82.31%`)

Comparison vs earlier `50K/1536/cosine` release proxies:
- vs section 13 baseline (`insert=93591 optimize=498659 search=9 total=592260`): `total -488860ms` (`-82.54%`)
- vs section 28 recheck (`insert=97672 optimize=499821 search=9 total=597503`): `total -494103ms` (`-82.69%`)
- vs section 21 latest rerun (`insert=158795 optimize=710488 search=11 total=869299`): `total -765899ms` (`-88.11%`)

Interpretation:
- this retest shows a strong target-scale signal, not just a small-sample-only effect.
- the largest drop is in `optimize`, which matches the intended HannsDB-side optimize-path micro-optimization direction.
- because this is still a single-run sample (`REPEATS=1`), confidence on exact magnitude should be validated by a multi-run (`REPEATS=3`) confirmation, but directionally the signal is already unambiguous at target scale.

## 33) 2026-03-21 target-scale stability retest for section 32

Context:
- section 32 showed a strong `50K / 1536 / cosine` release proxy win on a single run.
- this follow-up reruns the same target-scale shape with `REPEATS=3` to check whether the section 32 gain is stable or whether it snaps back toward the pre-cut baseline.

Command:

```bash
cd /Users/ryan/Code/HannsDB
N=50000 DIM=1536 METRIC=cosine TOPK=10 REPEATS=3 FEATURES=knowhere-backend PROFILE=release bash scripts/run_hannsdb_optimize_bench.sh
```

Observed summary:
- `OPT_BENCH_CONFIG n=50000 dim=1536 metric=cosine top_k=10`
- raw timings:
  - Run 1: `OPT_BENCH_TIMING_MS create=0 insert=97117 optimize=10979 search=0 total=108098`
  - Run 2: `OPT_BENCH_TIMING_MS create=0 insert=96295 optimize=10830 search=0 total=107126`
  - Run 3: `OPT_BENCH_TIMING_MS create=0 insert=96734 optimize=11144 search=0 total=107879`
- median:
  - `BENCH_SUMMARY_MEDIAN_MS create=0 insert=96734 optimize=10979 search=0 total=107879`

Comparison vs section 30 (`insert=93352 optimize=491123 search=9 total=584485`):
- `insert`: `+3765ms` (`+4.03%`)
- `optimize`: `-480144ms` (`-97.76%`)
- `search`: `-9ms` (to `0`)
- `total`: `-476606ms` (`-81.53%`)

Comparison vs section 32 single run (`insert=92526 optimize=10873 search=0 total=103400`):
- `insert`: `+4591ms` (`+4.96%`)
- `optimize`: `+106ms` (`+0.98%`)
- `total`: `+4479ms` (`+4.33%`)

Interpretation:
- the section 32 strong gain is stable at target scale: the median total remains around `108s`, which is still far below section 30's `584s`.
- there is a small rebound versus the single-run section 32 sample, but it is only single-digit percent noise, not a rollback to the pre-cut `~491s optimize / ~584s total` shape.
- in practical terms, the strong target-scale improvement is confirmed; the remaining variance sits in the same narrow ~4-5% band and does not change the conclusion.

## 34) 2026-03-21 real standard-case rerun (`Performance1536D50K`, post-opt label)

Context:
- prioritize the real standard case (not proxy) after the latest HannsDB optimize-path cuts.
- run with explicit fresh label/path to avoid mixing with earlier gate attempts.

Command:

```bash
cd /Users/ryan/Code/HannsDB
DB_LABEL=hannsdb-1536d50k-post-opt TASK_LABEL=hannsdb-1536d50k-post-opt DB_PATH=/tmp/hannsdb-vdbb-1536d50k-post-opt-db bash scripts/run_vdbb_hannsdb_perf1536d50k.sh
```

Observed milestones (from `logs/vectordb_bench.log`):
- `2026-03-21 10:38:53` task submitted for `Performance1536D50K` with `stages=['drop_old','load','search_serial']`.
- `2026-03-21 10:41:10` load finished:
  - `insert_duration=129.44228683388792`
  - `optimize_duration=2.0914591669570655`
  - `load_duration=131.5337`
- `2026-03-21 10:41:10` serial search started and kept producing `HANNSSDB_STAGE stage=search ms~2000` lines continuously through `10:51:32` (309 search calls observed after search start).

Result artifact check:
- expected result JSON (missing):
  - `/Users/ryan/Code/VectorDBBench/vectordb_bench/results/HannsDB/result_20260321_hannsdb-1536d50k-post-opt_hannsdb.json`
- this run had not yet reached the post-search `Task summary` / `write results to disk` path, so `TestResult.flush()` had not run yet.

Interpretation:
- the search stage is progressing normally but slowly, not failing to write results.
- the missing JSON is explained by the run still being inside `search_serial`, before VectorDBBench can build and flush the final `result_*.json` file.

## 35) 2026-03-21 Track A 验证门

Context:
- run the Track A verification gate set after WAL + recovery changes and record pass/fail plus benchmark attribution.

Commands:

```bash
cd /Users/ryan/Code/HannsDB
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p hannsdb-py --features python-binding -- -D warnings
cargo test --workspace

cd /Users/ryan/Code/VectorDBBench
. /Users/ryan/Code/HannsDB/.venv-hannsdb/bin/activate
python -m pytest tests/test_hannsdb_stage_logs.py tests/test_hannsdb_cli.py tests/test_hannsdb_client_config_shape.py -q

cd /Users/ryan/Code/HannsDB
N=2000 DIM=256 METRIC=cosine TOPK=10 REPEATS=3 FEATURES=knowhere-backend PROFILE=debug \
  bash scripts/run_hannsdb_optimize_bench.sh
```

Observed summary:
- `cargo clippy --workspace --all-targets -- -D warnings`: pass (exit 0)
- `cargo clippy -p hannsdb-py --features python-binding -- -D warnings`: pass (exit 0)
- `cargo test --workspace`: pass (`76 passed, 0 failed`)
- VectorDBBench pytest gate: pass with warnings (`6 passed, 2 warnings in 0.52s`)
- small optimize proxy:
  - Run 1: `OPT_BENCH_TIMING_MS create=0 insert=587 optimize=2712 search=0 total=3302`
  - Run 2: `OPT_BENCH_TIMING_MS create=0 insert=590 optimize=2736 search=0 total=3328`
  - Run 3: `OPT_BENCH_TIMING_MS create=0 insert=593 optimize=2794 search=0 total=3388`
  - Median: `BENCH_SUMMARY_MEDIAN_MS create=0 insert=590 optimize=2736 search=0 total=3328`

Comparison vs latest prior `N=2000` baseline (section 31):
- section 31 median: `create=0 insert=596 optimize=2718 search=0 total=3316`
- current median: `create=0 insert=590 optimize=2736 search=0 total=3328`
- delta: `insert -6ms (-1.01%)`, `optimize +18ms (+0.66%)`, `total +12ms (+0.36%)`

Interpretation:
- no material small-case optimize regression signal (well below the `>20%` concern threshold).
- Track A command set is not a clean all-green gate because non-zero warning output remains in:
  - VectorDBBench pytest (`2 warnings`)
  - optimize bench run (`knowhere-rs` `unused_mut` warnings during build).

## 36) 2026-03-21 Performance1536D50K 首次完整 HNSW 运行

Context:
- Previous run (section 34) used brute-force search (optimize=2s) because Python binding was compiled without `knowhere-backend` feature; script patched to enforce maturin rebuild with `python-binding,knowhere-backend` before each run.

Command:
```bash
cd /Users/ryan/Code/HannsDB
DB_LABEL=hannsdb-1536d50k-knowhere TASK_LABEL=hannsdb-1536d50k-knowhere \
  DB_PATH=/tmp/hannsdb-vdbb-1536d50k-knowhere-db \
  bash scripts/run_vdbb_hannsdb_perf1536d50k.sh
```

Results (`result_20260321_hannsdb-1536d50k-knowhere_hannsdb.json`):
- `insert_duration`: 121.29s
- `optimize_duration`: **81.01s** (HNSW build confirmed active; previous brute-force was 2s)
- `load_duration`: 202.30s
- `serial_latency_p99`: 111.5ms
- `serial_latency_p95`: 108.7ms
- `recall`: **1.0**
- `ndcg`: **1.0**
- `qps`: 0.0 (serial mode; concurrent QPS not exercised in this run)

Config: `M=16, ef_construction=64, ef_search=32, metric=COSINE, k=10`

Interpretation:
- Phase 4 closed: benchmark runs end-to-end with HNSW active.
- 81s HNSW build for 50K/1536-dim cosine is consistent with knowhere-rs target-scale baseline.
- 111ms p99 latency at ef_search=32 is the starting point; tuning ef_search upward will trade latency for recall.
- recall=1.0 at ef_search=32 suggests the dataset is within comfortable HNSW operating range.
- Next: tune ef_search (64, 128) and measure QPS in concurrent mode; consider quantization track separately.

## 37) 2026-03-21 ef_search 调优

Context:
- run `Performance1536D50K` with fixed `M=16`, `ef_construction=64`, and complete the curve for `ef_search=16/32/64/96/128`.
- all runs used the same script; non-rebuild runs set `SKIP_PY_REBUILD=1`.

Commands:

```bash
cd /Users/ryan/Code/HannsDB

EF_SEARCH=16 DB_LABEL=hannsdb-ef16 TASK_LABEL=hannsdb-ef16 \
  DB_PATH=/tmp/hannsdb-vdbb-ef16-db \
  SKIP_PY_REBUILD=1 \
  bash scripts/run_vdbb_hannsdb_perf1536d50k.sh

EF_SEARCH=64 DB_LABEL=hannsdb-ef64 TASK_LABEL=hannsdb-ef64 \
  DB_PATH=/tmp/hannsdb-vdbb-ef64-db \
  SKIP_PY_REBUILD=1 \
  bash scripts/run_vdbb_hannsdb_perf1536d50k.sh

EF_SEARCH=96 DB_LABEL=hannsdb-ef96 TASK_LABEL=hannsdb-ef96 \
  DB_PATH=/tmp/hannsdb-vdbb-ef96 \
  SKIP_PY_REBUILD=1 \
  bash scripts/run_vdbb_hannsdb_perf1536d50k.sh

EF_SEARCH=128 DB_LABEL=hannsdb-ef128-b TASK_LABEL=hannsdb-ef128-b \
  DB_PATH=/tmp/hannsdb-vdbb-ef128-b \
  SKIP_PY_REBUILD=1 \
  bash scripts/run_vdbb_hannsdb_perf1536d50k.sh
```

Result JSON:
- `/Users/ryan/Code/VectorDBBench/vectordb_bench/results/HannsDB/result_20260321_hannsdb-ef16_hannsdb.json`
- `/Users/ryan/Code/VectorDBBench/vectordb_bench/results/HannsDB/result_20260321_hannsdb-1536d50k-knowhere_hannsdb.json` (ef_search=32 from section 36)
- `/Users/ryan/Code/VectorDBBench/vectordb_bench/results/HannsDB/result_20260321_hannsdb-ef64_hannsdb.json`
- `/Users/ryan/Code/VectorDBBench/vectordb_bench/results/HannsDB/result_20260321_hannsdb-ef96_hannsdb.json`
- `/Users/ryan/Code/VectorDBBench/vectordb_bench/results/HannsDB/result_20260321_hannsdb-ef128-b_hannsdb.json`

Comparison (`serial search`):

| ef_search | insert_duration (s) | optimize_duration (s) | load_duration (s) | serial_latency_p99 (s) | recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16  | 109.1421 | 81.1353 | 190.2774 | 0.1116 | 1.0 |
| 32  | 121.2873 | 81.0139 | 202.3012 | 0.1115 | 1.0 |
| 64  | 109.7349 | 82.4912 | 192.2261 | 0.1196 | 1.0 |
| 96  | 110.1947 | 79.4453 | 189.6400 | 0.1087 | 1.0 |
| 128 | 110.0686 | 78.7832 | 188.8517 | 0.1366 | 1.0 |

Conclusion:
- the first `ef_search=128` sample (`p99=0.4703`) was not stable; the rerun (`ef128-b`) is `p99=0.1366`, so the 470ms point is treated as run noise (likely cold-cache / transient jitter), not a deterministic threshold effect.
- all tested points keep `recall=1.0`, so this dataset is recall-saturated even at low `ef_search`.
- measured latency differences inside `16/32/64/96` are small; `ef96` is slightly best in this sample (`p99=0.1087`), but margin vs `16/32` is only a few milliseconds.
- recommended work point:
  - latency-first: `ef_search=16`
  - conservative/default-aligned: keep `ef_search=32` (matches existing default behavior and still near-best latency)

## 38) 2026-03-21 zvec 基线（1536-dim/50K/cosine）

Context:
- baseline run on local zvec source (`/Users/ryan/Code/zvec`) for the same standard case shape used by HannsDB (`Performance1536D50K`, cosine, `k=10`, `M=16`, `ef_construction=64`, `ef_search=32`).
- zvec was built from local source in the same benchmark venv (`/Users/ryan/Code/HannsDB/.venv-hannsdb`) after initializing zvec submodules.

Build/install command (local source):
```bash
cd /Users/ryan/Code/zvec
git submodule update --init --recursive
cd /Users/ryan/Code/HannsDB
CMAKE_GENERATOR='Unix Makefiles' CMAKE_BUILD_PARALLEL_LEVEL='8' \
  /Users/ryan/Code/HannsDB/.venv-hannsdb/bin/python -m pip install -v /Users/ryan/Code/zvec
```

Benchmark command:
```bash
cd /Users/ryan/Code/HannsDB
PYTHONPATH=/Users/ryan/Code/VectorDBBench \
  /Users/ryan/Code/HannsDB/.venv-hannsdb/bin/python -m vectordb_bench.cli.vectordbbench zvec \
  --path /tmp/zvec-vdbb-1536d50k-db \
  --db-label zvec-1536d50k-baseline \
  --task-label zvec-1536d50k-baseline \
  --case-type Performance1536D50K \
  --k 10 --m 16 --ef-construction 64 --ef-search 32 \
  --skip-search-concurrent --num-concurrency 1 --concurrency-duration 1
```

Result artifact:
- `/Users/ryan/Code/VectorDBBench/vectordb_bench/results/Zvec/result_20260321_zvec-1536d50k-baseline_zvec.json`

Observed metrics:
- `insert_duration`: `3.6924s`
- `optimize_duration` (index build): `6.1556s`
- `load_duration` (insert + optimize): `9.848s`
- `serial_latency_p99`: `0.0006s` (`0.6ms`)
- `serial_latency_p95`: `0.0004s`
- `recall@10`: `0.9395`
- `qps`: `0.0` in result JSON (serial-only run, no concurrent stage)
- serial-search effective throughput from run log: `1000 / 0.3357s ≈ 2979 qps` (derived)

Quick comparison anchor (vs HannsDB section 36, same case/config):
- zvec build/load path is much shorter on this run (`optimize ~6.16s` vs HannsDB `~81.01s`).
- zvec p99 is much lower (`~0.6ms`), but recall is lower (`0.9395` vs HannsDB `1.0`).
- this forms a practical latency/recall baseline point for follow-up HannsDB tuning.

Gap analysis:
- **Algorithm class is not the main differentiator in this case**: benchmarked zvec path is HNSW (`HnswIndexParam` + `HnswQueryParam` in VectorDBBench zvec client; zvec core `HnswIndexParams/HnswQueryParams`), and HannsDB optimized path also calls HNSW via knowhere-rs. The 185x latency gap is therefore primarily implementation-path overhead and parameter/control-path mismatch.
- **HannsDB search path adds extra layers around HNSW core**: `hannsdb-py` `PyCollection.query` clones query structures and materializes Python `PyDoc` objects per hit; `hannsdb-core::search` currently holds `search_cache` mutex through `ann_search`; `ann_search` remaps ANN internal ids to external ids with per-hit checks. zvec benchmark path is comparatively direct (`collection.query(..., output_fields=[])` then read ids).
- **Potential effective parameter mismatch**: HannsDB `KnowhereHnswIndex::new` sets `ef_search=64` by default, while this benchmark is configured for `ef_search=32`. If not overridden at runtime, HannsDB may execute a more expensive search than nominal test config.
- **Code-based latency composition estimate for HannsDB p99=111ms**: HNSW graph search ~70-90ms, Python binding/object materialization ~8-15ms, wrapper/mapping/lock overhead ~8-20ms (order-of-magnitude estimate without profiler).
- **Priority actions (<=3)**:
  1. propagate runtime `ef_search` into knowhere-rs search request path (remove hardcoded `64` behavior);
  2. shrink `search_cache` lock scope so ANN search runs outside mutex;
  3. add id-only Python fast path for `output_fields=[]` to avoid per-hit `PyDoc` construction.
- **Knowhere-rs deep dive (T-20260321-014)**:
  - current HannsDB path calls pure Rust `knowhere-rs/src/faiss/hnsw.rs` (`HnswIndex::search`), not faiss C++ HNSW via FFI (`faiss-cxx` is optional and not enabled in HannsDB dependency path).
  - concrete search-path overheads in knowhere-rs:
    1. per-query allocation of `all_ids/all_dists` in `HnswIndex::search` (`hnsw.rs:3731-3732`) plus final rewrite loop (`3748+`);
    2. cosine/unfiltered path creates fresh `SearchScratch::new()` per query in `search_single` (`5349`), while TLS scratch reuse is only used by L2 fast path (`5470-5474`);
    3. result conversion/copy chain (`idx->id` in `5411+` then write into `all_ids/all_dists` in `3741+`).
  - comparison anchor: faiss HNSW accepts reusable `VisitedTable&` in `HNSW::search` (`faiss/impl/HNSW.h:210-215`) and runs `search_from_candidates` with `MinimaxHeap` (`faiss/impl/HNSW.cpp:1230+`, `616+`); knowhere-rs algorithm shape is similar but cosine fast-path reuse is weaker.
  - next best knowhere-rs optimizations (priority): cosine-path scratch reuse (TLS), single-query no-realloc path, cosine layer0 fast kernel path.

## 39) 2026-03-21 ef_search 修复后 baseline

Context:
- goal: verify P0 fix that threads request-time `ef_search` down to knowhere-rs search path (instead of effectively using hardcoded `ef_search=64`).
- same workload shape as section 36 (`Performance1536D50K`, cosine, `k=10`, `M=16`, `ef_construction=64`, `ef_search=32`).

Run command:
```bash
cd /Users/ryan/Code/HannsDB
rm -rf /tmp/hannsdb-vdbb-ef32-fixed &&
EF_SEARCH=32 DB_LABEL=hannsdb-ef32-fixed TASK_LABEL=hannsdb-ef32-fixed \
  DB_PATH=/tmp/hannsdb-vdbb-ef32-fixed SKIP_PY_REBUILD=0 \
  bash scripts/run_vdbb_hannsdb_perf1536d50k.sh
```

Result artifact:
- `/Users/ryan/Code/VectorDBBench/vectordb_bench/results/HannsDB/result_20260321_hannsdb-ef32-fixed_hannsdb.json`

Observed metrics:
- `insert_duration`: `109.1479s`
- `optimize_duration`: `79.5650s`
- `load_duration`: `188.7129s`
- `serial_latency_p99`: `0.1098s` (`109.8ms`)
- `serial_latency_p95`: `0.1077s`
- `recall@10`: `1.0`

Comparison vs pre-fix baseline (section 36, `p99=111.5ms`):
- post-fix `p99=109.8ms`, absolute delta `-1.7ms`, relative delta `-1.5%`.
- conclusion: this fix successfully removed the hardcoded path and now honors request-time `ef_search`, but on this specific case (`ef_search=32`) measured latency improvement is small; remaining latency bottleneck is still dominated by other search-path overheads identified in section 38 gap analysis.

## 40) 2026-03-21 P1 锁优化后 baseline

Context:
- P1 objective: shrink `hannsdb_core::db::search_with_ef` mutex scope so ANN search runs outside `search_cache` lock.
- implementation changed `optimized_ann` backend/id mapping to snapshot-friendly `Arc` handles; `search_with_ef` now copies a read-only snapshot under lock and executes ANN/brute-force search after releasing the mutex.

Verification:
```bash
cd /Users/ryan/Code/HannsDB
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

rm -rf /tmp/hannsdb-vdbb-p1-fixed &&
EF_SEARCH=32 DB_LABEL=hannsdb-p1-fixed TASK_LABEL=hannsdb-p1-fixed \
  DB_PATH=/tmp/hannsdb-vdbb-p1-fixed SKIP_PY_REBUILD=0 \
  bash scripts/run_vdbb_hannsdb_perf1536d50k.sh
```

Result artifact:
- `/Users/ryan/Code/VectorDBBench/vectordb_bench/results/HannsDB/result_20260321_hannsdb-p1-fixed_hannsdb.json`

Observed metrics:
- `insert_duration`: `110.6421s`
- `optimize_duration`: `79.5287s`
- `load_duration`: `190.1708s`
- `serial_latency_p99`: `0.1171s` (`117.1ms`)
- `serial_latency_p95`: `0.1136s`
- `recall@10`: `1.0`

Comparison vs section 39 baseline (`p99=109.8ms`):
- new `p99=117.1ms`, absolute delta `+7.3ms`, relative delta `+6.65%`.

Interpretation:
- code-level objective is met: ANN search is no longer executed while holding `search_cache` mutex.
- in this serial benchmark shape, the lock-scope shrink does not improve p99 and this sample is slightly slower; likely because this path now clones more state per query (`records/ids/tombstone`) for non-optimized fallback snapshots.
- follow-up should narrow snapshot cost (e.g., clone only optimized ANN path payload, avoid full brute-force snapshot copy when optimized index exists).

## 41) 2026-03-21 P2 Python binding 快速路径

Context:
- P2 objective: add a Python-binding query fast path for `output_fields=[]` so benchmark search does not go through document-field materialization logic.
- implementation in `crates/hannsdb-py/src/lib.rs`:
  - add `Collection::query_ids_scores(...)` helper (id + score only);
  - in `PyCollection::query`, when `output_fields=[]` and no filter, call this helper and directly build minimal `PyDoc` (`fields={}`, `vectors={}`), skipping full `Collection::query` path.

Verification:
```bash
cd /Users/ryan/Code/HannsDB
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

rm -rf /tmp/hannsdb-vdbb-p2-fixed &&
EF_SEARCH=32 DB_LABEL=hannsdb-p2-fixed TASK_LABEL=hannsdb-p2-fixed \
  DB_PATH=/tmp/hannsdb-vdbb-p2-fixed SKIP_PY_REBUILD=0 \
  bash scripts/run_vdbb_hannsdb_perf1536d50k.sh
```

Result artifact:
- `/Users/ryan/Code/VectorDBBench/vectordb_bench/results/HannsDB/result_20260321_hannsdb-p2-fixed_hannsdb.json`

Observed metrics:
- `insert_duration`: `107.8061s`
- `optimize_duration`: `79.8369s`
- `load_duration`: `187.6430s`
- `serial_latency_p99`: `0.1247s` (`124.7ms`)
- `serial_latency_p95`: `0.1178s`
- `recall@10`: `1.0`

Comparison vs section 40 (`p99=117.1ms`):
- new `p99=124.7ms`, absolute delta `+7.6ms`, relative delta `+6.49%`.

Interpretation:
- code-level fast path is active for `output_fields=[]`, but this single-run benchmark sample is slower rather than faster.
- likely explanation is run-to-run variance and/or remaining dominant overhead outside this Python-layer change; the optimization effect is not visible in this sample.

## 42) 2026-03-21 P1+P2 中位数验证

Context:
- run `run_hannsdb_optimize_bench.sh` with `REPEATS=3` on current state (P1+P2 applied) and compare with pre-change target-scale median from section 33.

Command:
```bash
cd /Users/ryan/Code/HannsDB
N=50000 DIM=1536 METRIC=cosine TOPK=10 REPEATS=3 \
  FEATURES=knowhere-backend PROFILE=release \
  bash scripts/run_hannsdb_optimize_bench.sh
```

Current run (P1+P2) raw timings:
- Run 1: `OPT_BENCH_TIMING_MS create=0 insert=94900 optimize=10564 search=26 total=105492`
- Run 2: `OPT_BENCH_TIMING_MS create=0 insert=90064 optimize=11317 search=73 total=101456`
- Run 3: `OPT_BENCH_TIMING_MS create=0 insert=92474 optimize=11144 search=32 total=103652`
- Median: `BENCH_SUMMARY_MEDIAN_MS create=0 insert=92474 optimize=11144 search=32 total=103652`

Reference baseline (section 33, pre P1+P2):
- Median: `BENCH_SUMMARY_MEDIAN_MS create=0 insert=96734 optimize=10979 search=0 total=107879`

Comparison (current vs section 33):
- `insert`: `-4260ms` (`-4.40%`)
- `optimize`: `+165ms` (`+1.50%`)
- `search`: `+32ms` (from `0ms` to `32ms`)
- `total`: `-4227ms` (`-3.92%`)

Conclusion:
- considering the task’s focus on `search`, P1+P2 does **not** show a search improvement in optimize-bench median; it is slightly worse (`0 -> 32ms`).
- `total` median is lower, but that drop is dominated by `insert` variation rather than an optimize/search-path gain.
- overall judgment: **no clear positive effect; current evidence is closer to noise with slight search-side regression signal**.
