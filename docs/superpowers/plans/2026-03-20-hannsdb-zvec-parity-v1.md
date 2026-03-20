# HannsDB zvec Parity V1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the highest-value `zvec` functionality gap for HannsDB by adding typed scalar fields, `upsert`, `fetch`, filterable `query`, and basic `stats`/`flush` semantics for local agent data, while preserving the existing `VectorDBBench` path and continuously verifying `knowhere-rs`.

**Architecture:** Keep HannsDB as a `dual-surface` system with one shared Rust core. Extend the core from append-only `id + primary vector` storage into a canonical document model with typed scalar payload persisted beside the current vector files. Keep ANN as derived acceleration state over one primary vector field in v1; when a scalar filter is present, prefer a correct filtered brute-force path over an incorrect ANN post-filter shortcut. Expose the same contract through PyO3 and the thin daemon without turning the daemon into the source of truth.

**Tech Stack:** Rust workspace, `serde`, `serde_json`, PyO3, axum, `cargo test`, `pytest`, VectorDBBench, `knowhere-rs`

---

## Chunk 1: Scope and file map

### Intended v1 parity slice

- Add typed scalar fields with a narrow agent-oriented value set: `string`, `int64`, `float64`, `bool`.
- Keep exactly one searchable vector field in v1. Reject multi-vector indexing explicitly instead of pretending to support it.
- Add `upsert`, `fetch`, `delete`, `stats`, and `flush` on the embedded and Python surfaces.
- Make `query(..., filter=...)` work with a deliberately small filter subset: field comparisons with `==`, `!=`, `>`, `>=`, `<`, `<=`, joined by `and`.
- Keep `VectorDBBench` compatibility: the no-filter single-vector path must stay stable and must continue to use the existing optimize/search flow.
- Treat `knowhere-rs` verification as part of the feature work: every tranche must re-check whether observed regressions belong to HannsDB glue or the ANN backend.

### Explicit non-goals for this plan

- WAL and crash recovery
- true multi-vector ANN
- scalar secondary indexes
- `create_index`, `drop_index`
- `add_column`, `drop_column`, `alter_column`
- full SQL-like filter grammar with `or`, nested parentheses, or functions

### File map

**Files:**
- Create: `crates/hannsdb-core/src/document.rs`
- Create: `crates/hannsdb-core/src/query/filter.rs`
- Create: `crates/hannsdb-core/src/segment/payloads.rs`
- Create: `crates/hannsdb-core/tests/document_api.rs`
- Modify: `crates/hannsdb-core/src/catalog/collection.rs`
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-core/src/lib.rs`
- Modify: `crates/hannsdb-core/src/query/mod.rs`
- Modify: `crates/hannsdb-core/src/segment/mod.rs`
- Modify: `crates/hannsdb-core/tests/collection_api.rs`
- Modify: `crates/hannsdb-py/src/lib.rs`
- Modify: `crates/hannsdb-daemon/src/api.rs`
- Modify: `crates/hannsdb-daemon/src/routes.rs`
- Modify: `crates/hannsdb-daemon/tests/http_smoke.rs`
- Modify: `docs/hannsdb-project-plan.md`
- Modify: `docs/hannsdb-project-design.md`
- Modify: `docs/vector-db-bench-notes.md`

### Acceptance bar

- `Doc.fields` round-trips through core persistence and Python binding.
- `Collection.upsert()` replaces existing ids by tombstoning the old row and appending the new row.
- `Collection.fetch()` returns stored fields and the primary vector.
- `Collection.query(filter=...)` returns correct filtered results even before any scalar index exists.
- Existing benchmark-oriented flows still pass with no filter and one vector field.

## Chunk 2: Core data model and persistence

### Task 1: Introduce a canonical document model in `hannsdb-core`

**Files:**
- Create: `crates/hannsdb-core/src/document.rs`
- Modify: `crates/hannsdb-core/src/lib.rs`
- Modify: `crates/hannsdb-core/src/catalog/collection.rs`
- Test: `crates/hannsdb-core/tests/document_api.rs`

- [ ] Write failing tests for typed `FieldValue`, `Document`, and collection schema metadata round-trip.
- [ ] Run: `cargo test -p hannsdb-core document_api -- --nocapture`
Expected: fail because the document model and schema metadata do not exist yet.
- [ ] Add a focused document module with `FieldValue`, `Document`, and a minimal collection schema that names one primary vector field.
- [ ] Extend `CollectionMetadata` so collection files record the primary vector name and declared scalar field schemas instead of only `dimension` and `metric`.
- [ ] Run: `cargo test -p hannsdb-core document_api -- --nocapture`
Expected: pass.

### Task 2: Persist scalar payload beside the existing vector files

**Files:**
- Create: `crates/hannsdb-core/src/segment/payloads.rs`
- Modify: `crates/hannsdb-core/src/segment/mod.rs`
- Modify: `crates/hannsdb-core/src/db.rs`
- Test: `crates/hannsdb-core/tests/document_api.rs`

- [ ] Write failing tests for row-aligned payload persistence across reopen.
- [ ] Run: `cargo test -p hannsdb-core document_api -- --nocapture`
Expected: fail because payload storage is not loaded or saved.
- [ ] Add a simple append-only `payloads.jsonl` segment file where each live row stores scalar fields for the same row index used by `ids.bin` and `records.bin`.
- [ ] Keep the current `records.bin` as the v1 source of truth for the primary vector to avoid destabilizing the benchmark path.
- [ ] Make reopen load payload rows alongside ids, vectors, and tombstones.
- [ ] Run: `cargo test -p hannsdb-core document_api -- --nocapture`
Expected: pass.

### Task 3: Add `insert`, `upsert`, `fetch`, `delete`, `stats`, and `flush` semantics to the core

**Files:**
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-core/tests/collection_api.rs`
- Test: `crates/hannsdb-core/tests/document_api.rs`

- [ ] Write failing tests for:
  - duplicate insert rejection
  - `upsert` replacing an existing id
  - `fetch` returning stored fields and vectors
  - `delete` hiding fetched and queried rows
  - `flush_collection` and `get_collection_info` staying coherent after mixed writes
- [ ] Run: `cargo test -p hannsdb-core collection_api -- --nocapture`
Expected: fail because these document-oriented paths are missing.
- [ ] Refactor the core API so `insert` and `upsert` operate on canonical documents instead of raw `ids + flattened vectors`.
- [ ] Implement `upsert` by tombstoning the old row and appending a new row. Do not attempt in-place mutation.
- [ ] Add a `fetch_documents` path that reconstructs full documents from row-aligned ids, payloads, vectors, and tombstones.
- [ ] Keep `flush_collection()` as a durability boundary over existing file writes and document its still-limited guarantee until WAL exists.
- [ ] Run: `cargo test -p hannsdb-core collection_api -- --nocapture`
Expected: pass.
- [ ] Run: `cargo test -p hannsdb-core document_api -- --nocapture`
Expected: pass.

## Chunk 3: Query and filter behavior

### Task 4: Add a minimal scalar filter parser and evaluator

**Files:**
- Create: `crates/hannsdb-core/src/query/filter.rs`
- Modify: `crates/hannsdb-core/src/query/mod.rs`
- Modify: `crates/hannsdb-core/src/db.rs`
- Test: `crates/hannsdb-core/tests/document_api.rs`

- [ ] Write failing tests for `role == "user"`, `turn >= 3`, and `session_id == "s1" and turn >= 2`.
- [ ] Run: `cargo test -p hannsdb-core document_api -- --nocapture`
Expected: fail because filter parsing and evaluation do not exist.
- [ ] Implement a small AST and parser for comparison predicates joined by `and`.
- [ ] Keep literal support limited to strings, integers, floats, and booleans. Reject unsupported grammar with `InvalidInput`.
- [ ] Run: `cargo test -p hannsdb-core document_api -- --nocapture`
Expected: pass.

### Task 5: Make filtered query correct before making it fast

**Files:**
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-core/tests/collection_api.rs`
- Test: `crates/hannsdb-core/tests/document_api.rs`

- [ ] Write failing tests that prove filtered query returns the correct top-k even when the nearest unfiltered neighbors should be excluded.
- [ ] Run: `cargo test -p hannsdb-core document_api -- --nocapture`
Expected: fail because the current query path ignores fields and filters.
- [ ] Route `query(filter=...)` through a filtered brute-force path over live rows in v1.
- [ ] Keep the ANN optimized path for the existing no-filter benchmark path only.
- [ ] Make query results return `score`, requested fields, and ids. Do not return vectors from `query` in v1; `fetch` is the full-record path.
- [ ] Run: `cargo test -p hannsdb-core collection_api -- --nocapture`
Expected: pass.
- [ ] Run: `cargo test -p hannsdb-core document_api -- --nocapture`
Expected: pass.

## Chunk 4: Python parity slice

### Task 6: Expand the PyO3 surface to a minimal `zvec`-shaped document API

**Files:**
- Modify: `crates/hannsdb-py/src/lib.rs`
- Test: `crates/hannsdb-py/src/lib.rs`
- External verify: `/Users/ryan/Code/VectorDBBench/tests/test_hannsdb_stage_logs.py`

- [ ] Write failing binding tests for:
  - `Doc.fields` round-trip
  - `Collection.upsert`
  - `Collection.fetch`
  - `Collection.query(..., filter=...)`
  - `Collection.flush`
  - `Collection.stats`
- [ ] Run: `cargo test -p hannsdb-py -- --nocapture`
Expected: fail because the binding still only exposes benchmark-minimal methods.
- [ ] Replace stringly `Vec<(String, String)>` field handling with native typed field conversion at the Python boundary.
- [ ] Add `score` to returned query documents so the Python `Doc` shape is useful beyond benchmark-only ids.
- [ ] Keep the public Python API explicitly constrained to one searchable vector field. Reject multi-vector indexing with a clear error.
- [ ] Run: `cargo test -p hannsdb-py -- --nocapture`
Expected: pass.

### Task 7: Re-verify the existing `VectorDBBench` integration after Python API growth

**Files:**
- Modify only if needed: `/Users/ryan/Code/VectorDBBench/vectordb_bench/backend/clients/hannsdb/hannsdb.py`
- Test: `/Users/ryan/Code/VectorDBBench/tests/test_hannsdb_cli.py`
- Test: `/Users/ryan/Code/VectorDBBench/tests/test_hannsdb_client_config_shape.py`
- Test: `/Users/ryan/Code/VectorDBBench/tests/test_hannsdb_stage_logs.py`

- [ ] Run the existing HannsDB client tests before any adapter change.
- [ ] Run: `cd /Users/ryan/Code/VectorDBBench && . /Users/ryan/Code/HannsDB/.venv-hannsdb/bin/activate && python -m pytest tests/test_hannsdb_stage_logs.py tests/test_hannsdb_cli.py tests/test_hannsdb_client_config_shape.py -q`
Expected: pass before and after the Python API expansion.
- [ ] Only change the VectorDBBench adapter if the binding change forces a compatibility shim.
- [ ] Record whether any new regression points back to HannsDB surface changes or to `knowhere-rs` optimize/search behavior.

## Chunk 5: Daemon and docs

### Task 8: Add a thin daemon parity slice without changing the source of truth

**Files:**
- Modify: `crates/hannsdb-daemon/src/api.rs`
- Modify: `crates/hannsdb-daemon/src/routes.rs`
- Modify: `crates/hannsdb-daemon/tests/http_smoke.rs`

- [ ] Write failing HTTP smoke tests for:
  - fetch by id
  - upsert by id
  - search with scalar filter
  - collection stats
- [ ] Run: `cargo test -p hannsdb-daemon http_smoke -- --nocapture`
Expected: fail because those routes do not exist.
- [ ] Add only the routes that map directly to the new core capabilities. Do not invent background jobs or asynchronous maintenance APIs here.
- [ ] Keep route naming consistent with the current thin surface, even if that means using explicit `/fetch` or `/upsert` endpoints instead of trying to emulate a full remote database API.
- [ ] Run: `cargo test -p hannsdb-daemon http_smoke -- --nocapture`
Expected: pass.

### Task 9: Update project docs and keep the benchmark gate live

**Files:**
- Modify: `docs/hannsdb-project-plan.md`
- Modify: `docs/hannsdb-project-design.md`
- Modify: `docs/vector-db-bench-notes.md`

- [ ] Add the new document model, filter behavior, and one-primary-vector rule to the design doc.
- [ ] Mark this parity slice as the active Phase 7 execution path in the top-level plan.
- [ ] Re-run the small optimize benchmark and record whether document/filter work changed the no-filter path.
- [ ] Run: `bash scripts/run_hannsdb_optimize_bench.sh 2000 256 cosine 3`
Expected: produce a comparable small baseline with no obvious regression.
- [ ] If the no-filter benchmark regresses materially, record whether the regression is in HannsDB document reconstruction or in `knowhere-rs` build/search behavior before making more feature changes.

## Final verification checklist

- [ ] Run: `cargo test --workspace`
Expected: pass.
- [ ] Run: `cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings`
Expected: pass.
- [ ] Run: `cd /Users/ryan/Code/VectorDBBench && . /Users/ryan/Code/HannsDB/.venv-hannsdb/bin/activate && python -m pytest tests/test_hannsdb_stage_logs.py tests/test_hannsdb_cli.py tests/test_hannsdb_client_config_shape.py -q`
Expected: pass.
- [ ] Run one HannsDB smoke command and confirm the result path still materializes.
- [ ] Record any evidence that a regression belongs to `knowhere-rs`, not just HannsDB, in `docs/vector-db-bench-notes.md`.

## Sequencing note

Implement this plan in the written order:

1. core document model and persistence
2. correct filter semantics
3. Python API
4. VectorDBBench compatibility check
5. daemon parity slice
6. docs and benchmark evidence

Do not start DDL or WAL work until this parity slice is complete and the no-filter benchmark path is still healthy.
