# HannsDB Project Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn HannsDB from a benchmarkable prototype into a coherent local agent database with a stable `knowhere-rs` ANN path and a repeatable `VectorDBBench` story.

**Architecture:** Keep one shared Rust core as the data plane, expose Python and daemon surfaces on top, and use a staged benchmark strategy: isolated optimize benchmark first, then `VectorDBBench` smoke, then full standard-case validation.

**Tech Stack:** Rust workspace, `knowhere-rs`, PyO3, axum, Python 3.11, VectorDBBench

---

## Current state snapshot

### Completed

- [x] Workspace bootstrap
- [x] Catalog and manifest persistence
- [x] Segment file storage and tombstones
- [x] Core collection API
- [x] Feature-gated `knowhere-rs` HNSW adapter
- [x] Minimal Python binding
- [x] Minimal daemon routes
- [x] HannsDB client integration in `VectorDBBench`
- [x] Tiny benchmark smoke path
- [x] Parity smoke entrypoint for the current targeted gates
  - 2026-04-08: `scripts/run_zvec_parity_smoke.sh` now chains `zvec_parity_schema`, `zvec_parity_query`, `http_smoke`, and the current Python facade/query/concurrency smoke set.
- [x] Standalone optimize benchmark entry
- [x] Release-profile optimize benchmark proxy at real target scale (`50K / 1536 / cosine`)
- [x] First evidence-backed `knowhere-rs` HNSW hotpath improvement on near-target cosine build
- [x] Repository-local `.coord + scripts/coord` tmux collaboration control plane
- [x] Canonical document model with typed scalar payloads
- [x] Core `insert/upsert/fetch/query(filter)` path for agent-shaped data
- [x] Python `Doc.fields/score` plus `upsert/fetch/delete/flush/stats`
- [x] Thin daemon `upsert/fetch/filter-search` routes
- [x] Minimal append-only WAL record model in `hannsdb-core`
- [x] WAL mutation logging for collection/data writes
- [x] Minimal `open()` replay for WAL-owned collection lifecycles without duplicate normal reopen materialization
- [x] Minimal `flush_collection()` semantics for readable `collection/segment/tombstone/WAL` local state
- [x] Collection-level tests for pre-WAL legacy isolation and flush success/failure boundaries
- [x] Benchmark-facing revalidation after the WAL/flush/recovery slice with no observed no-filter regression
- [x] `50K/1536/cosine` release proxy rerun after the durability slice with no observed regression signal
- [x] `50K/1536/cosine` section 32 stability retest with `REPEATS=3` confirmed the strong target-scale win remains stable

### In progress

- [x] Execute Track A from `docs/superpowers/plans/2026-03-20-architecture-analysis-alignment-v1.md`:
  normalize `docs/architecture-analysis.md`, land minimal `WAL + replay` in `hannsdb-core`, then finish the remaining crash-style recovery coverage and final closeout
  - 2026-03-21: all verification gates pass (clippy clean, workspace tests clean, VectorDBBench unit tests pass, optimize bench stable); external warnings (Python 3.14 ctypes, knowhere-rs unused_mut) are upstream — HannsDB itself is clean
- [x] Stabilize and measure `knowhere-rs` HNSW build behavior at target-scale cosine workloads
  - 2026-04-13: fresh local full `Performance1536D50K` rerun completed with `insert=14.6432s`, `optimize=114.001s`, `load=128.6442s`, `p99=0.0003s`, `recall=0.9441`
  - 2026-04-13: fresh remote x86 optimize proxy at `50K / 1536 / cosine` completed with `create=0ms`, `insert=3327ms`, `optimize=18606ms`, `total=21934ms`
- [x] Close the standard `Performance1536D50K` benchmark path
  - 2026-03-21 root cause of slow search identified: Python binding was installed without `knowhere-backend`, so `optimize_collection` fell back to brute-force; script patched to force maturin rebuild with correct features before each run
  - 2026-03-21 first clean end-to-end run with knowhere HNSW: `insert=121s optimize=81s serial_latency_p99=111ms recall=1.0`; result: `result_20260321_hannsdb-1536d50k-knowhere_hannsdb.json`
  - 2026-04-13 fresh clean rerun: `result_20260413_hannsdb-p0-rerun-20260413_hannsdb.json`
  - 2026-04-13 remote hk-x86 full rerun: `result_20260413_hannsdb-hk-x86-20260413_hannsdb.json`
  - 2026-04-15 remote x86 rerun after active-snapshot-unification: `result_20260415_hannsdb-x86-active-snapshot-20260415_hannsdb.json` with `insert=43.116s optimize=79.0623s load=122.1783s p99=0.0005s recall=0.9442`; query quality stayed flat while insert/load regressed relative to the 2026-04-13 x86 rerun
- [ ] Convert current prototype durability into a real storage story
  - 2026-04-13: first explicit `storage` module slice landed (`storage::paths`, `storage::wal`, `storage::recovery`) without behavior regression; this is the new landing zone for further durability cleanup
  - 2026-04-13: segment read-path helpers are now starting to move under `storage::segment_io`, shrinking `db.rs` while keeping `wal_recovery` and `collection_api` green
  - 2026-04-13: primary-key registry persistence helpers are now starting to move under `storage::primary_keys`, keeping string-PK mode conversion/storage logic out of `db.rs`
  - 2026-04-13: row/tombstone live-view helpers also moved under `storage::segment_io`; `db.rs` is now materially less responsible for raw storage/file-layout mechanics than at the start of the day
  - 2026-04-13: document/schema/value validation helpers were moved out of `db.rs`, and query hit ordering/projection now reuses shared executor helpers instead of duplicating sort/project logic
  - 2026-04-13: `CollectionHandle` shed duplicated derived state (`root`, `name`) and several placeholder/wrapper layers, so the remaining work is increasingly about durable semantics instead of code-organization debt
- [ ] Multi-segment management
  - 2026-04-13: no longer truly "not started" — rollover rules, segment-set layout, multi-segment reads, and segment-aware reopen/search coverage are already green; remaining work is broader operational hardening and story completion
  - 2026-04-14: active write routing after rollover is now also segment-aware for `insert` / `insert_documents` / `upsert_documents`; the next remaining storage-story work is less about basic correctness and more about deeper forward-store / runtime orchestration maturity
  - 2026-04-14: active-segment mutation authority is now explicitly split: `SegmentWriter` owns append/rollover/sealing mechanics, `VersionSet` / `SegmentManager` own topology, and `db.rs` keeps WAL / mutation policy / ANN invalidation / compaction triggering
  - 2026-04-14: `flush_collection()` is now segment-aware after rollover as well; active-segment Arrow snapshot materialization no longer mis-targets the root-level legacy path, and Arrow-only reopen works for the active segment after multi-segment flush
  - 2026-04-14: persisted-read authority is now less implicit: segment loads start from `segment.json.storage_format`, so `jsonl` segments prefer JSONL but can fall back to Arrow snapshots when JSONL is absent, while `arrow` segments prefer Arrow. This closes a stale/corrupt-sidecar class of reopen bugs without widening into a full recovery rewrite
- [ ] Compaction/rebuild workflow
  - 2026-04-13: no longer truly "not started" — compaction merge behavior, tombstone filtering, reopen coverage, and daemon/admin hooks exist; remaining work is turning the implemented path into a more complete production workflow story

### Not started

- [ ] Richer agent-oriented data model beyond the current v1 single-primary-vector slice
- [ ] Full crash-style recovery semantics and durable flush guarantees

## Phase 1: Freeze the current source of truth

**Purpose:** Make project status obvious and reduce confusion between “planned”, “implemented”, and “being benchmarked”.

**Files:**
- Create: `docs/hannsdb-project-design.md`
- Create: `docs/hannsdb-project-plan.md`
- Modify as needed: `docs/vector-db-bench-notes.md`

- [x] Write a top-level design document that reflects the real current architecture.
- [x] Write a top-level plan document that marks completed work separately from upcoming work.
- [x] Keep benchmark notes as the detailed evidence log, not the only place where current status exists.

**Exit criteria:**
- A new reader can understand the project from `docs/hannsdb-project-design.md` and `docs/hannsdb-project-plan.md` alone.

## Phase 2: Stabilize the ANN integration boundary

**Purpose:** Make the `knowhere-rs` path trustworthy enough to build the rest of HannsDB on top of it.

**Files:**
- Modify: `crates/hannsdb-index/src/adapter.rs`
- Modify: `crates/hannsdb-index/src/hnsw.rs`
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-index/tests/hnsw_adapter.rs`
- Modify: `crates/hannsdb-core/tests/collection_api.rs`
- External verify path: `/Users/ryan/Code/knowhere-rs/src/faiss/hnsw.rs`

- [x] Lock down score semantics for `l2` and `ip`.
- [x] Stabilize tiny-fixture HNSW behavior with deterministic seeding.
- [x] Remove avoidable HannsDB-side copy/flatten overhead before `knowhere-rs`.
- [x] Add focused tests for semantic compatibility.
- [x] Measure current target-scale build time with the latest `knowhere-rs` cosine optimizations.
- 2026-03-19 latest variance sample is recorded in `docs/vector-db-bench-notes.md` under "2026-03-19 latest knowhere-rs variance sample".
- [x] Decide whether the next performance step belongs in HannsDB glue code or inside `knowhere-rs`. Next cut should prioritize the `knowhere-rs` HNSW build hotpath; HannsDB work should stay limited to adapter-boundary overhead trims and score-semantic preservation.

**Exit criteria:**
- Targeted feature-on tests pass consistently.
- Benchmark notes clearly state what HannsDB must adapt at the adapter/core boundary.
- Current `knowhere-rs` performance evidence is recorded for target-like workloads.

## Phase 3: Close the optimize benchmark loop

**Purpose:** Make performance work repeatable without needing the full benchmark harness on every change.

**Files:**
- Modify: `crates/hannsdb-core/tests/collection_api.rs`
- Modify: `scripts/run_hannsdb_optimize_bench.sh`
- Modify: `docs/vector-db-bench-notes.md`

- [x] Add a synthetic optimize benchmark test entry.
- [x] Add a shell entrypoint that can run repeated measurements and report medians.
- [x] Record a verified small baseline (`N=2000`, `DIM=256`, `cosine`).
- [x] Record the first larger-scale baseline that is meaningfully close to `50K / 1536 / cosine`.
- [x] Use this entry as the default controller check before escalating to full `VectorDBBench`.

**Exit criteria:**
- One command produces stable benchmark fields for create/insert/optimize/search/total.
- Current median baseline is documented.

## Phase 4: Close the standard benchmark path

**Purpose:** Get HannsDB to reliably run the meaningful `VectorDBBench` path, not just the tiny smoke.

**Files:**
- Modify: `scripts/run_vdbb_hannsdb_perf1536d50k.sh`
- Modify: `docs/vector-db-bench-notes.md`
- External repo: `/Users/ryan/Code/VectorDBBench`

- [x] Wire HannsDB into `VectorDBBench` with a local-path embedded client.
- [x] Pass tiny smoke and custom dataset flows.
- [x] Finish `Performance1536D50K` with current code and record a complete result.
  - 2026-04-13 current result: `vectordb_bench/results/HannsDB/result_20260413_hannsdb-p0-rerun-20260413_hannsdb.json`
- [x] Decide whether v1 benchmark success requires only “run completes” or also a minimum latency/throughput target.
  - v1 success = run completes with HNSW active (optimize >30s); latency/QPS improvement is future work
- [x] Keep a reproducible command and result artifact path for the latest standard-case run.
  - command: `DB_LABEL=hannsdb-p0-rerun-20260413 TASK_LABEL=hannsdb-p0-rerun-20260413 DB_PATH=/tmp/hannsdb-p0-rerun-20260413-db bash scripts/run_vdbb_hannsdb_perf1536d50k.sh`
  - result: `vectordb_bench/results/HannsDB/result_20260413_hannsdb-p0-rerun-20260413_hannsdb.json`
  - remote hk-x86 result: `/data/work/VectorDBBench/vectordb_bench/results/HannsDB/result_20260413_hannsdb-hk-x86-20260413_hannsdb.json`

**Exit criteria:**
- A full standard-case run completes end-to-end.
- The latest command, logs, and result file are recorded in docs.

## Phase 5: Finish the storage story

**Purpose:** Move HannsDB from “benchmarkable prototype” to “credible local database”.

**Files:**
- Create: `crates/hannsdb-core/src/storage/mod.rs`
- Create: `crates/hannsdb-core/src/storage/wal.rs`
- Create: `crates/hannsdb-core/src/storage/recovery.rs`
- Create: `crates/hannsdb-core/tests/recovery.rs`
- Modify: `crates/hannsdb-core/src/db.rs`

- [x] Add WAL for collection and data mutations.
- [x] Add replay on open for WAL-owned collection lifecycles.
- [x] Define what `flush_collection()` actually guarantees.
- [x] Prove reopen/recovery behavior with crash-style tests.
  - 2026-03-21: 4 crash-style scenarios pass (missing records.bin, missing segment.json, truncated WAL tail, missing tombstones after delete); WAL tail truncation now gracefully skipped on reopen (not InvalidData)

**Exit criteria:**
- HannsDB can reopen safely after interrupted writes in the supported v1 paths.

## Phase 6: Evolve from single-segment prototype to manageable local DB

**Purpose:** Prepare for realistic collection growth and lifecycle management.

**Files:**
- Modify: `crates/hannsdb-core/src/segment/*`
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-daemon/src/routes.rs`
- Add tests under `crates/hannsdb-core/tests/`

- [x] Introduce explicit segment rollover rules.
- [x] Add compaction/rebuild plan for tombstoned data.
- [x] Clarify when ANN state is rebuilt versus incrementally updated.
  - 2026-04-13: persisted ANN completeness/`ann_ready` contracts are now explicit and tested:
    - optimize marks ANN ready
    - subsequent writes invalidate persisted ANN state
    - reopen preserves completeness when a persisted ANN blob still exists
    - `collection_info` and daemon routes now agree on both the `ready` and `stale-after-write` states
- [x] Expose minimal admin controls through the daemon if needed.

**Exit criteria:**
- The code no longer assumes “one collection, one forever-growing segment” as the only lifecycle model.

## Phase 7: Expand from benchmark shape to agent-data shape

**Purpose:** Make HannsDB more than a benchmark adapter.

**Files:**
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-py/src/lib.rs`
- Potentially add new model modules under `crates/hannsdb-core/src/`

- [x] Define the v1 canonical record model for agent use.
- [x] Decide how payload/text/session fields are stored relative to vectors.
- [x] Add retrieval APIs that are useful beyond pure top-k vector search.
- [x] Keep ANN index as derived acceleration state, not the only truth.
- [x] Execute the detailed parity slice plan in `docs/superpowers/plans/2026-03-20-hannsdb-zvec-parity-v1.md`.
- 2026-03-20 update: the first parity tranche is now implemented in core/Python/daemon for one-primary-vector collections with typed scalar payloads, `upsert`, `fetch`, correct filtered query, daemon `search.output_fields`, and an explicit daemon `/stats` alias.
- 2026-04-08 update: the Python parity slice now includes `hannsdb.typing` public wrappers, the schema/metadata bridge preserves `QuantizeType`, and the Python concurrency gate covers single-vector `insert`, `upsert`, `fetch`, and `query` on the current supported surface.

**Exit criteria:**
- HannsDB has a clear agent-facing data model, not only benchmark-facing inserts and queries.

## Phase 8: Harden the daemon only after the core deserves it

**Purpose:** Avoid overbuilding the service surface before the storage and ANN layers stabilize.

**Files:**
- Modify: `crates/hannsdb-daemon/src/api.rs`
- Modify: `crates/hannsdb-daemon/src/routes.rs`
- Modify: `crates/hannsdb-daemon/tests/http_smoke.rs`

- [x] Add the thin collection/search/admin baseline.
- [x] Add only the missing routes that the now-stable core genuinely needs.
- [ ] Avoid adding background task complexity before WAL/recovery and segment lifecycle are defined.

**Exit criteria:**
- Daemon stays thin, coherent, and downstream of core decisions.

## Recommended near-term execution order

1. Convert the current durability/runtime prototype into a cleaner long-term storage story.
2. Clarify ANN rebuild vs incremental-update rules under the now multi-segment lifecycle.
3. Continue expanding public/product surface to match the already-landed engine/runtime capability.
4. Use the fresh full benchmark + remote x86 proxy checkpoints as the new baseline for the next performance cut.

## What this project is doing right now

If you only want the short answer:

- HannsDB already has a working local core, Python binding, daemon, and benchmark smoke path.
- The current main blocker is no longer “can the standard benchmark run at all”; the benchmark path is now executable again.
- The main remaining gap is turning the current durability/runtime prototype into a cleaner long-term storage story, while continuing to close product-surface and performance gaps against zvec.
- The controller now uses the standalone release optimize benchmark as the default check before escalating to full `VectorDBBench`.
- In parallel, the project is validating and improving `knowhere-rs` as a long-term ANN foundation for this workload.

This file should be treated as the current top-level execution plan.
