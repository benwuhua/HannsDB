# HannsDB Architecture Analysis Alignment V1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Treat [`docs/architecture-analysis.md`](/Users/ryan/Code/HannsDB/docs/architecture-analysis.md) as the target architecture baseline, decompose it into executable tracks, and implement the first missing architecture slice: durability foundation (`WAL + recovery`) without regressing the current benchmark path.

**Architecture:** The architecture-analysis document spans multiple independent subsystems. Do not attempt to "implement the whole doc" in one patch. Keep the existing v1 bets intact: one shared Rust core, ANN as derived state, single segment per collection, one primary vector field, and thin Python/daemon surfaces. Execute the target architecture in tracks; start with durability because it is the smallest missing core invariant that improves the storage model without forcing multi-segment or daemon complexity.

**Tech Stack:** Rust workspace, `serde`, `serde_json`, append-only file I/O, PyO3, axum, `cargo test`, `cargo clippy`, VectorDBBench, `knowhere-rs`

## Status update (2026-03-20)

Completed in this repo state:

- target-architecture baseline docs were normalized before execution
- `crates/hannsdb-core/src/wal.rs` exists with append-only JSONL WAL records
- collection/data mutating public methods append WAL records
- `HannsDb::open()` performs minimal replay for WAL-owned collection lifecycles when on-disk state is missing/incomplete
- repeated normal reopen does not duplicate already-materialized rows
- `flush_collection()` now validates the minimal v1 flush boundary:
  readable `collection.json`, `segment.json`, `tombstones.json`, and `wal.jsonl`
- collection-level tests now prove:
  - flush succeeds after `create+insert` and `create+insert+delete`
  - flush fails if `wal.jsonl`, `segment.json`, or `tombstones.json` is missing
  - WAL replay does not mutate a pre-WAL legacy collection under partial-storage replay
- wal-recovery tests now also prove:
  - drop-ending WAL replay clears manifest entry and collection dir
  - WAL-owned collection replay restores missing `payloads.jsonl`
  - stale partial files are discarded and latest live view is reconstructed correctly
- `cargo test -p hannsdb-core wal_recovery -- --nocapture`
- `cargo test -p hannsdb-core collection_api -- --nocapture`
- `cargo test -p hannsdb-core`
- `cargo test --workspace`
- `cargo clippy -p hannsdb-core --all-targets -- -D warnings`

Still remaining in Track A:

- re-run the benchmark-facing verification gates and record attribution

---

## Scope split

The target architecture document covers multiple independent subsystems. Execute them in this order instead of trying to land all of them together:

1. **Track A: Durability foundation**
   - WAL records for collection mutations
   - replay on `HannsDb::open()`
   - tighten `flush_collection()` semantics
2. **Track B: Segment lifecycle**
   - explicit segment abstraction instead of "single forever-growing files"
   - compaction hooks
   - rebuild/cleanup semantics for tombstoned rows
3. **Track C: Retrieval/index expansion**
   - scalar secondary indexes
   - multi-vector collection model
   - incremental ANN maintenance only if benchmark path survives
4. **Track D: Service hardening**
   - daemon background jobs
   - auth/TLS/rate-limits
   - production concerns, not benchmark-path blockers

This plan only executes **Track A**. The other tracks need separate plans after Track A is stable.

## Chunk 1: Normalize the target architecture into an executable baseline

### Task 1: Reconcile the architecture-analysis doc with current code before using it as a build target

**Files:**
- Modify: `docs/architecture-analysis.md`
- Modify: `docs/hannsdb-project-plan.md`
- Reference: `docs/hannsdb-project-design.md`

- [ ] Write down which statements in `docs/architecture-analysis.md` are target decisions versus current-state descriptions.
- [ ] Correct any implementation-facts that are currently wrong or ambiguous:
  - storage path should match the actual `collections/<name>/...` layout until/unless a storage migration is explicitly planned
  - `HnswBackend` trait signatures should match the actual adapter API
  - `flush_collection()` should be described as a limited boundary until WAL exists
  - brute-force search should be described as "collect + sort + truncate" unless code is changed
- [ ] Add a short note in the top-level project plan that `docs/architecture-analysis.md` is now the target architecture baseline and that Track A is the active execution path.
- [ ] Run: `rg -n "target architecture baseline|Track A|WAL|recovery" docs`
Expected: the new execution path is visible in repo docs.

## Chunk 2: WAL record model and storage

### Task 2: Add a focused WAL module to `hannsdb-core`

**Files:**
- Create: `crates/hannsdb-core/src/wal.rs`
- Modify: `crates/hannsdb-core/src/lib.rs`
- Test: `crates/hannsdb-core/tests/wal_recovery.rs`

- [ ] Write the failing tests first for WAL record round-trip:
  - create collection intent
  - insert raw vectors intent
  - insert documents intent
  - upsert documents intent
  - delete ids intent
- [ ] Run: `cargo test -p hannsdb-core wal_recovery -- --nocapture`
Expected: fail because no WAL module or tests exist yet.
- [ ] Add a minimal append-only WAL file format under the DB root:
  - default path: `<root>/wal.jsonl`
  - one JSON record per operation
  - include enough payload to replay without consulting volatile state
- [ ] Keep the WAL schema narrow and explicit. Do not add checksums, segment rollover, compaction, or versioned migrations in this first slice.
- [ ] Run: `cargo test -p hannsdb-core wal_recovery -- --nocapture`
Expected: WAL record round-trip tests pass.

### Task 3: Define replay-safe operation payloads

**Files:**
- Modify: `crates/hannsdb-core/src/wal.rs`
- Modify: `crates/hannsdb-core/src/document.rs`
- Test: `crates/hannsdb-core/tests/wal_recovery.rs`

- [ ] Write failing tests for replay payload completeness:
  - collection creation record includes schema
  - raw insert record includes ids and flattened vectors
  - document insert/upsert includes full typed fields and vector
  - delete record includes ids
- [ ] Run: `cargo test -p hannsdb-core wal_recovery -- --nocapture`
Expected: fail because current record payloads are incomplete.
- [ ] Make the WAL payloads self-contained enough to reconstruct state from an empty data directory plus `wal.jsonl`.
- [ ] Reject partially-specified records in deserialization.
- [ ] Run: `cargo test -p hannsdb-core wal_recovery -- --nocapture`
Expected: pass.

## Chunk 3: Wire WAL writes into collection mutations

### Task 4: Log collection creation and drop intents

**Files:**
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-core/tests/collection_api.rs`
- Test: `crates/hannsdb-core/tests/wal_recovery.rs`

- [ ] Write failing tests that prove creating a collection appends a WAL record and that replay can restore the manifest and collection metadata.
- [ ] Run: `cargo test -p hannsdb-core wal_recovery -- --nocapture`
Expected: fail because create/drop do not write WAL yet.
- [ ] Append WAL records inside:
  - `create_collection_with_schema`
  - `drop_collection`
- [ ] Keep write ordering simple and explicit:
  - write WAL record
  - apply storage mutation
  - never silently skip WAL for mutating operations
- [ ] Run: `cargo test -p hannsdb-core wal_recovery -- --nocapture`
Expected: pass.

### Task 5: Log insert/upsert/delete intents

**Files:**
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-core/tests/document_api.rs`
- Test: `crates/hannsdb-core/tests/wal_recovery.rs`

- [ ] Write failing tests for replay after:
  - `insert(ids, vectors)`
  - `insert_documents`
  - `upsert_documents`
  - `delete`
- [ ] Run: `cargo test -p hannsdb-core wal_recovery -- --nocapture`
Expected: fail because mutating data paths do not write WAL records.
- [ ] Append WAL records from all mutating data paths.
- [ ] Preserve the current append-only row semantics:
  - inserts append
  - upserts tombstone old rows and append new rows
  - deletes only tombstone
- [ ] Do not change the benchmark-facing no-filter search path while wiring WAL.
- [ ] Run: `cargo test -p hannsdb-core wal_recovery -- --nocapture`
Expected: pass.

## Chunk 4: Replay on open and recovery behavior

### Task 6: Replay WAL during `HannsDb::open()`

**Files:**
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-core/tests/collection_api.rs`
- Test: `crates/hannsdb-core/tests/wal_recovery.rs`

- [ ] Write failing tests for these recovery cases:
  - open on empty root with no WAL still works
  - open after WAL-only create can rebuild manifest + collection files
  - open after WAL-only create + insert can rebuild readable collection data
  - open after WAL-only upsert/delete can rebuild latest live view
- [ ] Run: `cargo test -p hannsdb-core wal_recovery -- --nocapture`
Expected: fail because `open()` only validates existing files and does not replay.
- [ ] Add replay on open:
  - load or create manifest root
  - if WAL exists, replay into missing or incomplete storage state
  - avoid duplicating already-materialized rows on normal reopen
- [ ] Keep replay idempotent for repeated `open()` calls on an already-materialized database.
- [ ] Run: `cargo test -p hannsdb-core wal_recovery -- --nocapture`
Expected: pass.

### Task 7: Tighten and document `flush_collection()` semantics

**Files:**
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-core/tests/collection_api.rs`
- Modify: `docs/architecture-analysis.md`
- Modify: `docs/hannsdb-project-design.md`

- [ ] Write failing tests that show `flush_collection()` is meaningful after WAL exists.
- [ ] Run: `cargo test -p hannsdb-core collection_api -- --nocapture`
Expected: fail because flush still only checks collection metadata readability.
- [ ] Define minimal v1 flush semantics:
  - collection metadata readable
  - WAL append visible on disk
  - segment/tombstone files readable
- [ ] Do not claim full durability stronger than the implementation can provide.
- [ ] Run: `cargo test -p hannsdb-core collection_api -- --nocapture`
Expected: pass.

## Chunk 5: Surface compatibility and benchmark gates

### Task 8: Keep Python and daemon surfaces behavior-stable while core durability changes

**Files:**
- Modify only if needed: `crates/hannsdb-py/src/lib.rs`
- Modify only if needed: `crates/hannsdb-daemon/src/routes.rs`
- Test: `crates/hannsdb-daemon/tests/http_smoke.rs`

- [ ] Run the existing wrapper and daemon tests before changing any surface code.
- [ ] Run: `cargo test --workspace`
Expected: pass before WAL work begins.
- [ ] Only touch the Python or daemon surface if the core API shape actually changes.
- [ ] Keep benchmark-facing no-filter query behavior identical:
  - no filter → current search path
  - filter → current document query path
- [ ] Run: `cargo test --workspace`
Expected: pass after WAL integration.

### Task 9: Re-run the benchmark-facing gates and record attribution

**Files:**
- Modify: `docs/vector-db-bench-notes.md`
- Modify if needed: `docs/hannsdb-project-plan.md`

- [ ] Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: pass.
- [ ] Run: `cargo clippy -p hannsdb-py --features python-binding -- -D warnings`
Expected: pass.
- [ ] Run: `cd /Users/ryan/Code/VectorDBBench && . /Users/ryan/Code/HannsDB/.venv-hannsdb/bin/activate && python -m pytest tests/test_hannsdb_stage_logs.py tests/test_hannsdb_cli.py tests/test_hannsdb_client_config_shape.py -q`
Expected: pass.
- [ ] Run: `N=2000 DIM=256 METRIC=cosine TOPK=10 REPEATS=3 FEATURES=knowhere-backend PROFILE=debug bash scripts/run_hannsdb_optimize_bench.sh`
Expected: remain in the current small-case band with no obvious no-filter regression.
- [ ] Record whether any regression points to:
  - WAL glue in HannsDB
  - or the existing `knowhere-rs` optimize/search behavior

## Final verification checklist

- [ ] Run: `cargo test --workspace`
Expected: pass.
- [ ] Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: pass.
- [ ] Run: `cargo clippy -p hannsdb-py --features python-binding -- -D warnings`
Expected: pass.
- [ ] Run: `cd /Users/ryan/Code/VectorDBBench && . /Users/ryan/Code/HannsDB/.venv-hannsdb/bin/activate && python -m pytest tests/test_hannsdb_stage_logs.py tests/test_hannsdb_cli.py tests/test_hannsdb_client_config_shape.py -q`
Expected: pass.

## Sequencing note

Execute this plan before any of the other architecture-analysis gaps.

Do **not** start:
- multi-segment management
- scalar secondary indexes
- multi-vector retrieval/indexing
- daemon background jobs

until Track A (`WAL + recovery`) is implemented and the benchmark-facing no-filter path is still healthy.
