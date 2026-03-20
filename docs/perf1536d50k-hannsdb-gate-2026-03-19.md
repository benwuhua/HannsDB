# HannsDB VectorDBBench Gate Attempt: `Performance1536D50K` (2026-03-19)

## Repro command (preferred)

```bash
cd /Users/ryan/Code/HannsDB
./scripts/run_vdbb_hannsdb_perf1536d50k.sh
```

This script fixes the validated baseline parameters:
- `--case-type Performance1536D50K`
- `--k 10`
- `--m 16`
- `--ef-construction 64`
- `--ef-search 32`
- `--skip-search-concurrent`
- `--num-concurrency 1`
- `--concurrency-duration 1`

Defaults:
- venv: `/Users/ryan/Code/HannsDB/.venv-hannsdb`
- VectorDBBench repo: `/Users/ryan/Code/VectorDBBench`
- db path: `/tmp/hannsdb-vdbb-1536d50k-db`
- db/task label: `hannsdb-1536d50k`

## Equivalent raw CLI (for debugging)

```bash
cd /Users/ryan/Code/HannsDB
. .venv-hannsdb/bin/activate
PYTHONPATH=/Users/ryan/Code/VectorDBBench python -m vectordb_bench.cli.vectordbbench hannsdb \
  --path /tmp/hannsdb-vdbb-1536d50k-db \
  --db-label hannsdb-1536d50k \
  --task-label hannsdb-1536d50k \
  --case-type Performance1536D50K \
  --k 10 \
  --m 16 \
  --ef-construction 64 \
  --ef-search 32 \
  --skip-search-concurrent \
  --num-concurrency 1 \
  --concurrency-duration 1
```

## 2026-03-19 gate observation (summary)

- Start time: `2026-03-19 14:57:52 CST`
- Case resolved to built-in dataset: `OpenAI-SMALL-50K` (`dim=1536`, `size=50000`, metric `COSINE`)
- Downloaded built-in dataset files (`4/4`) to `/tmp/vectordb_bench/dataset/openai/openai_small_50k` in about `9m23s`
- Load completed (insert about `115.13s`, optimize near-zero)
- Serial search entered long-running phase; no completion/result emission in the bounded window (about `21` minutes), run was terminated
- Classification: **runtime-bound blocker**, not a hard integration crash
- Expected result path if completed:
  `/Users/ryan/Code/VectorDBBench/vectordb_bench/results/HannsDB/result_YYYYMMDD_hannsdb-1536d50k_hannsdb.json`

## T-20260319-006 rerun update (2026-03-19 evening)

Rerun command used fresh labels/path:

```bash
cd /Users/ryan/Code/HannsDB
DB_LABEL=hannsdb-1536d50k-rerun2 TASK_LABEL=hannsdb-1536d50k-rerun2 DB_PATH=/tmp/hannsdb-vdbb-1536d50k-rerun2-db ./scripts/run_vdbb_hannsdb_perf1536d50k.sh
```

Observed log milestone:
- `2026-03-19 19:32:58,785` — `(SpawnProcess-1:1) Finish loading all dataset into VectorDB, dur=130.49832625011913`

Outcome:
- no result JSON emitted for the rerun label
- run entered a long post-load stall and was terminated
- classification remains **blocked-with-concerns** for the full standard gate path on this machine/session

## T-20260319-009 rerun + snapshot update (2026-03-19 night)

Rerun command with fresh label/path:

```bash
cd /Users/ryan/Code/HannsDB
DB_LABEL=hannsdb-1536d50k-rerun3 TASK_LABEL=hannsdb-1536d50k-rerun3 DB_PATH=/tmp/hannsdb-vdbb-1536d50k-rerun3-db ./scripts/run_vdbb_hannsdb_perf1536d50k.sh
```

Observed milestone:
- `2026-03-19 20:58:49,036` — `(SpawnProcess-1:1) Finish loading all dataset into VectorDB, dur=149.12591604096815`

Post-load stall diagnosis snapshots:
- `/tmp/hannsdb-vdbb-rerun3.sample.txt` (PID `63139`, parent CLI process)
- `/tmp/hannsdb-vdbb-rerun3-worker.sample.txt` (PID `63141`, multiprocessing worker)

Snapshot summary:
- parent process remained in `time_sleep`
- worker process main thread waited on Python lock (`lock_PyThread_acquire_lock` / `pthread_cond_wait`)

Outcome:
- no result JSON emitted for `hannsdb-1536d50k-rerun3`
- run terminated after snapshot capture
- gate status remains **blocked-with-concerns**

## Watchdog wrapper (T-20260319-013)

To avoid manual post-load stall debugging, use:

```bash
cd /Users/ryan/Code/HannsDB
bash scripts/run_vdbb_hannsdb_perf1536d50k_watchdog.sh
```

Env overrides:
- `DB_LABEL` (default `hannsdb-1536d50k-watchdog`)
- `TASK_LABEL` (default same as `DB_LABEL`)
- `DB_PATH` (default `/tmp/<DB_LABEL>-db`)
- `STALL_TIMEOUT_SEC` (default `300`)
- `POST_LOAD_TIMEOUT_SEC` (default `1800`, timeout from load-complete to search-start)
- `RESULT_DATE` (default `$(date +%Y%m%d)`)

Behavior:
- launches `scripts/run_vdbb_hannsdb_perf1536d50k.sh`
- watches for:
  - result JSON: `VectorDBBench/vectordb_bench/results/HannsDB/result_<date>_<db_label>_hannsdb.json`
  - load completion marker in `logs/vectordb_bench.log` (`Finish loading all dataset into VectorDB`)
- if load is complete and no result appears for `STALL_TIMEOUT_SEC`, it auto-captures:
  - parent python sample: `/tmp/<label>.sample.txt`
  - worker subprocess sample: `/tmp/<label>-worker.sample.txt`
  - process snapshot: `/tmp/<label>.ps.txt`
  then terminates the gate process tree (graceful then force-kill).

Status lines:
- `WATCHDOG_START`
- `WATCHDOG_GATE_LAUNCHED`
- `WATCHDOG_LOAD_COMPLETE`
- `WATCHDOG_STALL_TIMEOUT`
- `WATCHDOG_DIAG ...`
- `WATCHDOG_RESULT_FOUND` or `WATCHDOG_EXIT ...`

## T-20260319-015 full watchdog timeout drill

Command:

```bash
cd /Users/ryan/Code/HannsDB
STALL_TIMEOUT_SEC=60 DB_LABEL=hannsdb-watchdog-drill TASK_LABEL=hannsdb-watchdog-drill DB_PATH=/tmp/hannsdb-watchdog-drill-db bash scripts/run_vdbb_hannsdb_perf1536d50k_watchdog.sh
```

Drill constraints:
- no manual interrupt
- let watchdog decide timeout path

Observed watchdog status lines:
- `WATCHDOG_START ... stall_timeout_sec=60 ...`
- `WATCHDOG_GATE_LAUNCHED pid=74693`
- `WATCHDOG_LOAD_COMPLETE seen=1 ...`
- `WATCHDOG_STALL_TIMEOUT reached=1 timeout_sec=60`
- `WATCHDOG_DIAG ps_snapshot=/tmp/hannsdb-watchdog-drill.ps.txt`
- `WATCHDOG_DIAG sample_parent pid=74704 out=/tmp/hannsdb-watchdog-drill.sample.txt`
- `WATCHDOG_DIAG sample_worker pid=74707 out=/tmp/hannsdb-watchdog-drill-worker.sample.txt`
- `WATCHDOG_EXIT status=stall_timeout_no_result result_path=/Users/ryan/Code/VectorDBBench/vectordb_bench/results/HannsDB/result_20260319_hannsdb-watchdog-drill_hannsdb.json`

Exit status:
- `124` (watchdog timeout path)

Artifact checks:
- `/tmp/hannsdb-watchdog-drill.sample.txt` exists (`677` lines)
- `/tmp/hannsdb-watchdog-drill-worker.sample.txt` exists (`404` lines)
- `/tmp/hannsdb-watchdog-drill.ps.txt` exists (`524` lines)

Note:
- timeout cleanup emitted Python `resource_tracker` leaked semaphore warnings after process termination; diagnostics were still produced as expected.

## 2026-03-20 watchdog gate update

Command:

```bash
cd /Users/ryan/Code/HannsDB
STALL_TIMEOUT_SEC=120 DB_LABEL=hannsdb-watchdog-20260320 TASK_LABEL=hannsdb-watchdog-20260320 DB_PATH=/tmp/hannsdb-watchdog-20260320-db bash scripts/run_vdbb_hannsdb_perf1536d50k_watchdog.sh
```

Observed watchdog events:
- `WATCHDOG_START db_label=hannsdb-watchdog-20260320 task_label=hannsdb-watchdog-20260320 db_path=/tmp/hannsdb-watchdog-20260320-db stall_timeout_sec=120 result_path=/Users/ryan/Code/VectorDBBench/vectordb_bench/results/HannsDB/result_20260320_hannsdb-watchdog-20260320_hannsdb.json`
- `WATCHDOG_GATE_LAUNCHED pid=112`
- `WATCHDOG_LOAD_COMPLETE seen=1 stall_deadline_epoch=241`
- `WATCHDOG_STALL_TIMEOUT reached=1 timeout_sec=120`
- `WATCHDOG_DIAG ps_snapshot=/tmp/hannsdb-watchdog-20260320.ps.txt`
- `WATCHDOG_DIAG sample_parent pid=122 out=/tmp/hannsdb-watchdog-20260320.sample.txt`
- `WATCHDOG_DIAG sample_worker pid=126 out=/tmp/hannsdb-watchdog-20260320-worker.sample.txt`
- `WATCHDOG_EXIT status=stall_timeout_no_result result_path=/Users/ryan/Code/VectorDBBench/vectordb_bench/results/HannsDB/result_20260320_hannsdb-watchdog-20260320_hannsdb.json`

Exit code:
- `124`

Diagnostics:
- `/tmp/hannsdb-watchdog-20260320.sample.txt` exists (`678` lines)
- `/tmp/hannsdb-watchdog-20260320-worker.sample.txt` exists (`404` lines)
- `/tmp/hannsdb-watchdog-20260320.ps.txt` exists (`663` lines)

Conclusion:
- this watchdog run completed its stall-timeout path and captured evidence, but no result JSON was emitted for the 2026-03-20 label, so the gate remains unresolved on this machine/session.

## 2026-03-20 watchdog diagnostic target fix

Issue found from the 2026-03-20 samples:
- previous worker sampling could pick a `multiprocessing.resource_tracker` process instead of the hot compute subprocess
- this made `sample_worker` less useful for root-cause analysis

Script change:
- updated `scripts/run_vdbb_hannsdb_perf1536d50k_watchdog.sh` `capture_diagnostics()` selection logic
- `sample_parent` now prefers the process containing `vectordb_bench.cli.vectordbbench`
- `sample_worker` now prefers non-`resource_tracker` python descendants and picks the highest `%CPU` candidate
- fallback remains: highest `%CPU` python descendant when no non-`resource_tracker` candidate exists
- diagnostic lines now include `cmd=` and `cpu=` for sampled parent/worker

Verification:
- `bash -n scripts/run_vdbb_hannsdb_perf1536d50k_watchdog.sh` passed

Expected impact:
- future stall captures should point to the real busy subprocess first, reducing false leads during gate debugging.

Additional observation from the 2026-03-20 snapshot:
- `/tmp/hannsdb-watchdog-20260320.ps.txt` already showed one spawn child at about `99%` CPU while no result JSON was emitted within watchdog window.
- this supports a **compute-bound search stage** interpretation more than a hard deadlock interpretation for that run.

## 2026-03-20 stagecheck (stage-aware timeout verification)

Command:

```bash
cd /Users/ryan/Code/HannsDB
STALL_TIMEOUT_SEC=60 POST_LOAD_TIMEOUT_SEC=300 DB_LABEL=hannsdb-watchdog-20260320-stagecheck TASK_LABEL=hannsdb-watchdog-20260320-stagecheck DB_PATH=/tmp/hannsdb-watchdog-20260320-stagecheck-db bash scripts/run_vdbb_hannsdb_perf1536d50k_watchdog.sh
```

Observed watchdog events:
- `WATCHDOG_START ... stall_timeout_sec=60 post_load_timeout_sec=300 ...`
- `WATCHDOG_LOAD_COMPLETE seen=1 post_load_deadline_epoch=...`
- no `WATCHDOG_SEARCH_STARTED` observed before timeout window
- `WATCHDOG_POST_LOAD_TIMEOUT reached=1 timeout_sec=300`
- `WATCHDOG_DIAG sample_worker pid=8219 cpu=100.0 ... spawn_main(... pipe_handle=41)`
- `WATCHDOG_EXIT status=post_load_timeout_no_search ...`

Exit code:
- `125`

Worker sample hotspot:
- `/tmp/hannsdb-watchdog-20260320-stagecheck-worker.sample.txt` shows the hot stack in:
  - `hannsdb_core::db::HannsDb::optimize_collection`
  - `hannsdb_core::db::build_optimized_ann_state`
  - `knowhere_rs::faiss::hnsw::HnswIndex::add`

Conclusion:
- stage-aware watchdog split is working as intended.
- this run is classified as **post-load optimize bottleneck before search start**, not search-stall.

## 2026-03-20 fixcheck

Command:

```bash
cd /Users/ryan/Code/HannsDB
STALL_TIMEOUT_SEC=60 DB_LABEL=hannsdb-watchdog-20260320-fixcheck TASK_LABEL=hannsdb-watchdog-20260320-fixcheck DB_PATH=/tmp/hannsdb-watchdog-20260320-fixcheck-db bash scripts/run_vdbb_hannsdb_perf1536d50k_watchdog.sh
```

Observed watchdog events:
- `WATCHDOG_START ... stall_timeout_sec=60 ... result_path=/Users/ryan/Code/VectorDBBench/vectordb_bench/results/HannsDB/result_20260320_hannsdb-watchdog-20260320-fixcheck_hannsdb.json`
- `WATCHDOG_GATE_LAUNCHED pid=5012`
- `WATCHDOG_LOAD_COMPLETE seen=1 stall_deadline_epoch=181`
- `WATCHDOG_STALL_TIMEOUT reached=1 timeout_sec=60`
- `WATCHDOG_DIAG ps_snapshot=/tmp/hannsdb-watchdog-20260320-fixcheck.ps.txt`
- `WATCHDOG_DIAG sample_parent pid=5022 out=/tmp/hannsdb-watchdog-20260320-fixcheck.sample.txt cmd=/opt/homebrew/Cellar/python@3.14/3.14.2/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python -m vectordb_bench.cli.vectordbbench hannsdb --path /tmp/hannsdb-watchdog-20260320-fixcheck-db --db-label hannsdb-watchdog-20260320-fixcheck --task-label hannsdb-watchdog-20260320-fixcheck --case-type Performance1536D50K --k 10 --m 16 --ef-construction 64 --ef-search 32 --skip-search-concurrent --num-concurrency 1 --concurrency-duration 1`
- `WATCHDOG_DIAG sample_worker pid=5795 cpu=100.0 out=/tmp/hannsdb-watchdog-20260320-fixcheck-worker.sample.txt cmd=/opt/homebrew/Cellar/python@3.14/3.14.2/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python -c from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=12, pipe_handle=41) --multiprocessing-fork`
- `WATCHDOG_EXIT status=stall_timeout_no_result result_path=/Users/ryan/Code/VectorDBBench/vectordb_bench/results/HannsDB/result_20260320_hannsdb-watchdog-20260320-fixcheck_hannsdb.json`

Exit code:
- `124`

Artifact checks:
- `/tmp/hannsdb-watchdog-20260320-fixcheck.sample.txt` exists
- `/tmp/hannsdb-watchdog-20260320-fixcheck-worker.sample.txt` exists
- `/tmp/hannsdb-watchdog-20260320-fixcheck.ps.txt` exists

Fixcheck conclusion:
- the diagnostic target fix **is effective**
- `sample_worker` now points at the real worker subprocess (`spawn_main`), and the `cmd=` string does **not** contain `resource_tracker`
- the run still ends in `stall_timeout_no_result`, so the gate remains unresolved, but the diagnostic sampling behavior is corrected
