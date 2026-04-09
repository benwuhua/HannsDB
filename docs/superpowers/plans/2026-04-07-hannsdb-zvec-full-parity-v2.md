# HannsDB zvec Full Parity V2 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining code-level gap between the current HannsDB codebase and the current `zvec` codebase, including schema model, index model, planner/query model, Python surface, and quality/perf infrastructure, while preserving HannsDB's existing benchmark path during migration.

**Architecture:** Do not try to clone `zvec` line-for-line. Keep HannsDB Rust-first, but evolve it from a single-primary-vector document store with one ANN acceleration path into a field-oriented engine with per-field index descriptors, collection-level runtime handles, a typed query planner/executor, and a productized Python facade. Treat algorithm parity and product-surface parity as separate workstreams that meet behind a stable internal query/index contract.

**Tech Stack:** Rust workspace, `serde`, `serde_json`, PyO3, axum, `numpy`, `pytest`, `cargo test`, `VectorDBBench`, `knowhere-rs` and/or new ANN backends added under `crates/hannsdb-index`

---

## Chunk 1: Current gap and parity target

### Current code comparison

- HannsDB core is still centered on one `HannsDb` object with a coarse `search_cache: Mutex<HashMap<...>>`, one cached ANN state per collection, and a direct `search_with_ef()` / `query_documents()` path in one file.
  Evidence: `crates/hannsdb-core/src/db.rs`
- HannsDB schema is still effectively one primary vector plus a narrow scalar field set.
  Evidence: `crates/hannsdb-core/src/document.rs`
- HannsDB ANN integration is still one backend trait and one primary index flow.
  Evidence: `crates/hannsdb-index/src/adapter.rs`, `crates/hannsdb-index/src/hnsw.rs`
- HannsDB Python is still a thin PyO3 wrapper around the Rust core, with direct method calls and no higher-level query executor or extension layer.
  Evidence: `crates/hannsdb-py/src/lib.rs`
- HannsDB daemon is still a thin HTTP wrapper around `Arc<Mutex<HannsDb>>`.
  Evidence: `crates/hannsdb-daemon/src/routes.rs`
- zvec has a much richer native runtime: `CollectionImpl` owns schema locks, write locks, a `SegmentManager`, a `VersionManager`, and a SQL engine.
  Evidence: `/Users/ryan/Code/vectorDB/zvec/src/db/collection.cc`
- zvec schema already supports multiple vector fields, richer scalar types, sparse vectors, and per-field index params.
  Evidence: `/Users/ryan/Code/vectorDB/zvec/python/zvec/model/schema/collection_schema.py`, `/Users/ryan/Code/vectorDB/zvec/python/zvec/model/schema/field_schema.py`
- zvec Python already has a higher-level query layer with `QueryContext`, `QueryExecutorFactory`, concurrent fan-out, and reranker integration.
  Evidence: `/Users/ryan/Code/vectorDB/zvec/python/zvec/executor/query_executor.py`
- zvec exposes DDL for `CreateIndex`, `DropIndex`, `Optimize`, column changes, and `GroupByQuery`, not just insert/fetch/query.
  Evidence: `/Users/ryan/Code/vectorDB/zvec/src/binding/python/model/python_collection.cc`, `/Users/ryan/Code/vectorDB/zvec/src/db/collection.cc`
- zvec quality coverage is much broader: core math/metric tests, SQL planner tests, crash recovery tests, Python concurrency/recall/rerank tests, and HNSW-RabitQ coverage.
  Evidence: `/Users/ryan/Code/vectorDB/zvec/tests`, `/Users/ryan/Code/vectorDB/zvec/python/tests`

### What “full parity” means in this repo

- Match the `zvec` capability envelope, not its C++ implementation details.
- HannsDB must support:
  - multi-vector collections
  - dense and sparse vector field metadata
  - per-field vector/scalar index DDL
  - a typed query context that can express multiple vector recalls plus filter/output controls
  - a planner/executor layer that can merge, rerank, and optionally group results
  - a Python package shape closer to `zvec`, not only a raw extension module
  - materially broader recovery/concurrency/recall/perf test coverage
- HannsDB does not need to copy every `zvec` algorithm immediately, but it does need a stable internal API that makes `Flat`, `HNSW`, `IVF`, and at least one quantized path pluggable.

### Decision gates before execution

- Decision Gate A: parity scope
  Choose whether “full parity” includes zvec-only product extensions such as embedding-function integrations in the first pass. Recommended: no. First reach engine/query/Python parity, then add extension integrations.
- Decision Gate B: quantized ANN path
  Choose whether HNSW-RabitQ-like parity is implemented as:
  - a Rust-native implementation under `crates/hannsdb-index`, or
  - an external backend adapter behind a stable HannsDB trait.
  Recommended: external backend adapter first, Rust-native later if the performance gap justifies it.
- Decision Gate C: service surface
  Decide whether the daemon needs to mimic the full Python surface. Recommended: no. Keep daemon thin and focused on core capabilities; parity target is primarily embedded engine + Python package.

## Chunk 2: File map and responsibilities

### Core runtime and catalog

**Files:**
- Create: `crates/hannsdb-core/src/catalog/index.rs`
- Create: `crates/hannsdb-core/src/query/ast.rs`
- Create: `crates/hannsdb-core/src/query/planner.rs`
- Create: `crates/hannsdb-core/src/query/executor.rs`
- Create: `crates/hannsdb-core/src/segment/manager.rs`
- Create: `crates/hannsdb-core/src/segment/version_set.rs`
- Modify: `crates/hannsdb-core/src/catalog/collection.rs`
- Modify: `crates/hannsdb-core/src/catalog/mod.rs`
- Modify: `crates/hannsdb-core/src/catalog/version.rs`
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-core/src/document.rs`
- Modify: `crates/hannsdb-core/src/lib.rs`
- Modify: `crates/hannsdb-core/src/query/filter.rs`
- Modify: `crates/hannsdb-core/src/query/mod.rs`
- Modify: `crates/hannsdb-core/src/segment/mod.rs`

### Index abstraction

**Files:**
- Create: `crates/hannsdb-index/src/descriptor.rs`
- Create: `crates/hannsdb-index/src/factory.rs`
- Create: `crates/hannsdb-index/src/flat.rs`
- Create: `crates/hannsdb-index/src/ivf.rs`
- Create: `crates/hannsdb-index/src/scalar.rs`
- Modify: `crates/hannsdb-index/src/adapter.rs`
- Modify: `crates/hannsdb-index/src/hnsw.rs`
- Modify: `crates/hannsdb-index/src/lib.rs`

### Python product surface

**Files:**
- Create: `crates/hannsdb-py/python/hannsdb/__init__.py`
- Create: `crates/hannsdb-py/python/hannsdb/model/__init__.py`
- Create: `crates/hannsdb-py/python/hannsdb/model/collection.py`
- Create: `crates/hannsdb-py/python/hannsdb/model/doc.py`
- Create: `crates/hannsdb-py/python/hannsdb/model/schema/__init__.py`
- Create: `crates/hannsdb-py/python/hannsdb/model/schema/collection_schema.py`
- Create: `crates/hannsdb-py/python/hannsdb/model/schema/field_schema.py`
- Create: `crates/hannsdb-py/python/hannsdb/model/param/__init__.py`
- Create: `crates/hannsdb-py/python/hannsdb/model/param/vector_query.py`
- Create: `crates/hannsdb-py/python/hannsdb/executor/__init__.py`
- Create: `crates/hannsdb-py/python/hannsdb/executor/query_executor.py`
- Create: `crates/hannsdb-py/python/hannsdb/extension/__init__.py`
- Modify: `crates/hannsdb-py/Cargo.toml`
- Modify: `crates/hannsdb-py/src/lib.rs`

### Daemon, tests, and perf gates

**Files:**
- Modify: `crates/hannsdb-daemon/src/api.rs`
- Modify: `crates/hannsdb-daemon/src/routes.rs`
- Modify: `crates/hannsdb-daemon/tests/http_smoke.rs`
- Create: `crates/hannsdb-core/tests/zvec_parity_schema.rs`
- Create: `crates/hannsdb-core/tests/zvec_parity_query.rs`
- Create: `crates/hannsdb-core/tests/zvec_parity_recovery.rs`
- Create: `crates/hannsdb-index/tests/ivf_adapter.rs`
- Create: `crates/hannsdb-index/tests/index_registry.rs`
- Create: `crates/hannsdb-py/tests/test_schema_surface.py`
- Create: `crates/hannsdb-py/tests/test_query_executor.py`
- Create: `crates/hannsdb-py/tests/test_collection_parity.py`
- Create: `scripts/run_zvec_parity_smoke.sh`
- Modify: `docs/vector-db-bench-notes.md`
- Modify: `docs/hannsdb-project-plan.md`

## Chunk 3: Foundation tasks

### Task 1: Freeze the parity acceptance matrix in tests before refactoring

**Files:**
- Create: `crates/hannsdb-core/tests/zvec_parity_schema.rs`
- Create: `crates/hannsdb-core/tests/zvec_parity_query.rs`
- Create: `crates/hannsdb-py/tests/test_schema_surface.py`
- Create: `crates/hannsdb-py/tests/test_query_executor.py`

- [ ] Write failing Rust tests for:
  - multiple vector fields in one collection schema
  - per-field index metadata round-trip
  - nullable and array scalar field metadata
  - query-by-id and multi-vector query request shapes
- [ ] Run: `cargo test -p hannsdb-core zvec_parity_schema -- --nocapture`
Expected: FAIL because current schema and query model are single-vector.
- [ ] Write failing Python tests for:
  - `CollectionSchema(vectors=[...])`
  - `VectorSchema(..., index_param=IVFIndexParam(...))`
  - `QueryContext(queries=[...])`
  - `QueryExecutorFactory.create(...)`
- [ ] Run: `cd crates/hannsdb-py && maturin develop --features python-binding,knowhere-backend && python -m pytest tests/test_schema_surface.py tests/test_query_executor.py -q`
Expected: FAIL because the pure-Python facade and richer param objects do not exist.
- [ ] Commit checkpoint:
```bash
git add crates/hannsdb-core/tests/zvec_parity_schema.rs crates/hannsdb-core/tests/zvec_parity_query.rs crates/hannsdb-py/tests/test_schema_surface.py crates/hannsdb-py/tests/test_query_executor.py
git commit -m "test: add zvec parity acceptance harness"
```

### Task 2: Generalize schema and metadata from one primary vector to field-oriented schema

**Files:**
- Modify: `crates/hannsdb-core/src/document.rs`
- Modify: `crates/hannsdb-core/src/catalog/collection.rs`
- Modify: `crates/hannsdb-core/src/lib.rs`
- Modify: `crates/hannsdb-py/src/lib.rs`
- Test: `crates/hannsdb-core/tests/zvec_parity_schema.rs`
- Test: `crates/hannsdb-py/tests/test_schema_surface.py`

- [ ] Write the failing migration test for loading an old single-vector collection metadata file into the new schema model.
- [ ] Run: `cargo test -p hannsdb-core zvec_parity_schema -- --nocapture`
Expected: FAIL because backward-compatible schema migration does not exist.
- [ ] Replace the current one-primary-vector schema model with explicit field registries.

```rust
pub enum ScalarDataType { Int32, Int64, UInt32, UInt64, Float32, Float64, String, Bool, ArrayInt64, ArrayFloat64, ArrayString, ArrayBool }

pub enum VectorDataType { DenseFp16, DenseFp32, DenseFp64, DenseInt8, SparseFp16, SparseFp32 }

pub struct ScalarFieldSchema {
    pub name: String,
    pub data_type: ScalarDataType,
    pub nullable: bool,
    pub index: Option<ScalarIndexDescriptor>,
}

pub struct VectorFieldSchema {
    pub name: String,
    pub data_type: VectorDataType,
    pub dimension: Option<usize>,
    pub index: Option<VectorIndexDescriptor>,
}
```

- [ ] Keep one backward-compatible helper that maps the current benchmark schema into the new representation without changing existing callers all at once.
- [ ] Update the PyO3 bridge so Python schema objects map onto the richer Rust schema rather than synthesizing a single primary vector internally.
- [ ] Run: `cargo test -p hannsdb-core zvec_parity_schema -- --nocapture`
Expected: PASS.
- [ ] Run: `cd crates/hannsdb-py && maturin develop --features python-binding,knowhere-backend && python -m pytest tests/test_schema_surface.py -q`
Expected: PASS.
- [ ] Commit checkpoint:
```bash
git add crates/hannsdb-core/src/document.rs crates/hannsdb-core/src/catalog/collection.rs crates/hannsdb-core/src/lib.rs crates/hannsdb-py/src/lib.rs
git commit -m "feat: generalize collection schema to field registry"
```

### Task 3: Replace coarse collection state with collection handles, segment manager, and version set

**Files:**
- Create: `crates/hannsdb-core/src/segment/manager.rs`
- Create: `crates/hannsdb-core/src/segment/version_set.rs`
- Modify: `crates/hannsdb-core/src/catalog/version.rs`
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-core/src/segment/mod.rs`
- Test: `crates/hannsdb-core/tests/zvec_parity_recovery.rs`
- Test: `crates/hannsdb-core/tests/wal_recovery.rs`

- [ ] Write failing tests for:
  - concurrent read + optimize against the same collection
  - reopen after optimize using versioned segment metadata
  - collection-local cache invalidation without a single process-wide mutex bottleneck
- [ ] Run: `cargo test -p hannsdb-core zvec_parity_recovery wal_recovery -- --nocapture`
Expected: FAIL because current runtime is still centered on one `HannsDb` cache map and ad hoc segment loading.
- [ ] Introduce a collection handle layer so `HannsDb` owns collection descriptors and each opened collection owns its own read/write state, version set, and index registry references.
- [ ] Split segment concerns:
  - `manager.rs` owns active + immutable segment discovery
  - `version_set.rs` owns versioned metadata and recovery checkpoints
  - `db.rs` stops directly stitching every segment concern inline
- [ ] Keep the old benchmark path working by building compatibility wrappers over the new handle APIs before deleting the legacy helpers.
- [ ] Run: `cargo test -p hannsdb-core zvec_parity_recovery wal_recovery -- --nocapture`
Expected: PASS.
- [ ] Run: `cargo test -p hannsdb-core collection_api lifecycle -- --nocapture`
Expected: PASS.
- [ ] Commit checkpoint:
```bash
git add crates/hannsdb-core/src/segment/manager.rs crates/hannsdb-core/src/segment/version_set.rs crates/hannsdb-core/src/catalog/version.rs crates/hannsdb-core/src/db.rs crates/hannsdb-core/src/segment/mod.rs
git commit -m "refactor: add collection handles and versioned segments"
```

## Chunk 4: Index and planner tasks

### Task 4: Introduce per-field index descriptors and explicit index DDL

**Files:**
- Create: `crates/hannsdb-core/src/catalog/index.rs`
- Create: `crates/hannsdb-index/src/descriptor.rs`
- Create: `crates/hannsdb-index/src/factory.rs`
- Create: `crates/hannsdb-index/src/flat.rs`
- Create: `crates/hannsdb-index/src/ivf.rs`
- Create: `crates/hannsdb-index/src/scalar.rs`
- Modify: `crates/hannsdb-index/src/adapter.rs`
- Modify: `crates/hannsdb-index/src/hnsw.rs`
- Modify: `crates/hannsdb-index/src/lib.rs`
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-py/src/lib.rs`
- Modify: `crates/hannsdb-daemon/src/api.rs`
- Modify: `crates/hannsdb-daemon/src/routes.rs`
- Test: `crates/hannsdb-index/tests/index_registry.rs`
- Test: `crates/hannsdb-index/tests/ivf_adapter.rs`

- [ ] Write failing tests for:
  - `create_index(field, HNSW)`
  - `create_index(field, IVF)`
  - `drop_index(field)`
  - scalar filter index registration
  - persisted index descriptor recovery
- [ ] Run: `cargo test -p hannsdb-index index_registry ivf_adapter -- --nocapture`
Expected: FAIL because the current index crate only exposes one HNSW-oriented adapter.
- [ ] Add index descriptors that decouple schema metadata from concrete backend instances.

```rust
pub enum VectorIndexKind { Flat, Hnsw, Ivf, QuantizedHnsw }
pub enum ScalarIndexKind { Inverted }

pub struct VectorIndexDescriptor {
    pub field_name: String,
    pub kind: VectorIndexKind,
    pub metric: String,
    pub params: serde_json::Value,
}
```

- [ ] Add a factory layer so `db.rs` asks for an index by descriptor instead of constructing `KnowhereHnswIndex` directly.
- [ ] Implement `Flat` and `IVF` first. Leave the quantized path behind Decision Gate B, but keep the descriptor stable now.
- [ ] Add Python and daemon `create_index` / `drop_index` entry points only after the Rust index registry passes.
- [ ] Run: `cargo test -p hannsdb-index index_registry ivf_adapter -- --nocapture`
Expected: PASS.
- [ ] Run: `cargo test -p hannsdb-core collection_api -- --nocapture`
Expected: PASS.
- [ ] Commit checkpoint:
```bash
git add crates/hannsdb-core/src/catalog/index.rs crates/hannsdb-index/src/descriptor.rs crates/hannsdb-index/src/factory.rs crates/hannsdb-index/src/flat.rs crates/hannsdb-index/src/ivf.rs crates/hannsdb-index/src/scalar.rs crates/hannsdb-index/src/adapter.rs crates/hannsdb-index/src/hnsw.rs crates/hannsdb-index/src/lib.rs crates/hannsdb-core/src/db.rs crates/hannsdb-py/src/lib.rs crates/hannsdb-daemon/src/api.rs crates/hannsdb-daemon/src/routes.rs
git commit -m "feat: add per-field index descriptors and DDL"
```

### Task 5: Add a typed query AST, planner, and executor

**Files:**
- Create: `crates/hannsdb-core/src/query/ast.rs`
- Create: `crates/hannsdb-core/src/query/planner.rs`
- Create: `crates/hannsdb-core/src/query/executor.rs`
- Modify: `crates/hannsdb-core/src/query/filter.rs`
- Modify: `crates/hannsdb-core/src/query/mod.rs`
- Modify: `crates/hannsdb-core/src/db.rs`
- Test: `crates/hannsdb-core/tests/zvec_parity_query.rs`

- [ ] Write failing tests for:
  - multiple vector recalls in one query
  - query by explicit vector and by document id
  - filter pushdown into scalar index when available
  - merge/rerank over multiple vector fields
  - group-by rejection path first, then group-by execution later
- [ ] Run: `cargo test -p hannsdb-core zvec_parity_query -- --nocapture`
Expected: FAIL because current query flow is still `search_with_ef` plus a narrow filter path.
- [ ] Introduce a typed query contract.

```rust
pub struct QueryContext {
    pub top_k: usize,
    pub filter: Option<FilterExpr>,
    pub output_fields: Option<Vec<String>>,
    pub include_vector: bool,
    pub vector_queries: Vec<VectorQuery>,
    pub reranker: Option<RerankSpec>,
}

pub enum PlanNode {
    ForwardScan,
    ScalarRecall,
    VectorRecall { field: String, index: VectorIndexKind },
    MergeTopK,
    Rerank,
    GroupBy,
}
```

- [ ] Keep the current benchmark path as one planner fast path: one dense vector query, no filter, one ANN index, no rerank.
- [ ] Implement execution in layers:
  - filter-only scan
  - single-vector recall
  - multi-vector recall + merge
  - rerank hook
  - group-by last
- [ ] Run: `cargo test -p hannsdb-core zvec_parity_query -- --nocapture`
Expected: PASS.
- [ ] Run: `cargo test -p hannsdb-core bench_suite -- --nocapture`
Expected: PASS with no benchmark-helper regressions.
- [ ] Commit checkpoint:
```bash
git add crates/hannsdb-core/src/query/ast.rs crates/hannsdb-core/src/query/planner.rs crates/hannsdb-core/src/query/executor.rs crates/hannsdb-core/src/query/filter.rs crates/hannsdb-core/src/query/mod.rs crates/hannsdb-core/src/db.rs
git commit -m "feat: add query planner and executor"
```

## Chunk 5: Python, daemon, and quality tasks

### Task 6: Build a zvec-shaped pure-Python facade on top of the extension module

**Files:**
- Create: `crates/hannsdb-py/python/hannsdb/__init__.py`
- Create: `crates/hannsdb-py/python/hannsdb/model/__init__.py`
- Create: `crates/hannsdb-py/python/hannsdb/model/collection.py`
- Create: `crates/hannsdb-py/python/hannsdb/model/doc.py`
- Create: `crates/hannsdb-py/python/hannsdb/model/schema/__init__.py`
- Create: `crates/hannsdb-py/python/hannsdb/model/schema/collection_schema.py`
- Create: `crates/hannsdb-py/python/hannsdb/model/schema/field_schema.py`
- Create: `crates/hannsdb-py/python/hannsdb/model/param/__init__.py`
- Create: `crates/hannsdb-py/python/hannsdb/model/param/vector_query.py`
- Create: `crates/hannsdb-py/python/hannsdb/executor/__init__.py`
- Create: `crates/hannsdb-py/python/hannsdb/executor/query_executor.py`
- Create: `crates/hannsdb-py/python/hannsdb/extension/__init__.py`
- Modify: `crates/hannsdb-py/Cargo.toml`
- Modify: `crates/hannsdb-py/src/lib.rs`
- Test: `crates/hannsdb-py/tests/test_collection_parity.py`
- Test: `crates/hannsdb-py/tests/test_query_executor.py`

- [ ] Write failing Python tests that mirror the current zvec ergonomics:
  - schema objects live in pure Python
  - collection query is routed through a `QueryExecutorFactory`
  - numpy conversion happens in Python before hitting the extension
  - multi-query merge/rerank hooks can run without changing the native ABI for every UX tweak
- [ ] Run: `cd crates/hannsdb-py && maturin develop --features python-binding,knowhere-backend && python -m pytest tests/test_collection_parity.py tests/test_query_executor.py -q`
Expected: FAIL because the current package is extension-only.
- [ ] Teach maturin to ship a Python source tree alongside the extension module.
- [ ] Keep the Rust extension focused on:
  - collection lifecycle
  - typed schema/value marshalling
  - query execution entry points
  - index DDL
- [ ] Move high-churn UX code into pure Python:
  - `Collection`
  - `CollectionSchema`
  - `VectorSchema`
  - `QueryContext`
  - `QueryExecutorFactory`
  - rerank/extension hooks
- [ ] Run: `cd crates/hannsdb-py && maturin develop --features python-binding,knowhere-backend && python -m pytest tests/test_collection_parity.py tests/test_query_executor.py -q`
Expected: PASS.
- [ ] Commit checkpoint:
```bash
git add crates/hannsdb-py/Cargo.toml crates/hannsdb-py/src/lib.rs crates/hannsdb-py/python/hannsdb crates/hannsdb-py/tests/test_collection_parity.py crates/hannsdb-py/tests/test_query_executor.py
git commit -m "feat: add productized Python facade"
```

### Task 7: Keep the daemon thin, but expose the new core capabilities cleanly

**Files:**
- Modify: `crates/hannsdb-daemon/src/api.rs`
- Modify: `crates/hannsdb-daemon/src/routes.rs`
- Modify: `crates/hannsdb-daemon/tests/http_smoke.rs`

- [ ] Write failing HTTP smoke tests for:
  - `create_index`
  - `drop_index`
  - multi-vector query request body
  - fetch with output field selection
- [ ] Run: `cargo test -p hannsdb-daemon http_smoke -- --nocapture`
Expected: FAIL because the daemon only exposes a thin single-query shape today.
- [ ] Add request/response types that map directly onto the new query and index descriptors. Do not recreate planner logic in the daemon.
- [ ] Keep daemon internals as a transport wrapper over collection handles rather than a new state machine.
- [ ] Run: `cargo test -p hannsdb-daemon http_smoke -- --nocapture`
Expected: PASS.
- [ ] Commit checkpoint:
```bash
git add crates/hannsdb-daemon/src/api.rs crates/hannsdb-daemon/src/routes.rs crates/hannsdb-daemon/tests/http_smoke.rs
git commit -m "feat: expand daemon surface for parity"
```

### Task 8: Build the missing quality/perf gate that zvec already has

**Files:**
- Modify: `docs/vector-db-bench-notes.md`
- Modify: `docs/hannsdb-project-plan.md`
- Create: `scripts/run_zvec_parity_smoke.sh`
- Create: `crates/hannsdb-core/tests/zvec_parity_recovery.rs`
- Create: `crates/hannsdb-py/tests/test_collection_parity.py`

- [ ] Add recovery tests for reopen after index build, reopen after optimize, and crash-like partial metadata repair.
- [ ] Add query correctness tests for:
  - mixed scalar/vector filters
  - multi-vector merge
  - rerank hooks
  - recall sanity against brute force on small synthetic corpora
- [ ] Add Python concurrency tests that mimic `zvec`'s `test_collection_concurrency.py` shape.
- [ ] Add a single scripted parity smoke command:
```bash
bash scripts/run_zvec_parity_smoke.sh
```
Expected: builds the Python package, runs targeted Rust tests, runs targeted Python tests, and prints benchmark notes location.
- [ ] Re-run HannsDB's existing benchmark-relevant tests plus a `VectorDBBench` smoke slice.
- [ ] Run:
```bash
cargo test --workspace
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cd crates/hannsdb-py && maturin develop --features python-binding,knowhere-backend
cd /Users/ryan/Code/VectorDBBench && python -m pytest tests/test_hannsdb_stage_logs.py tests/test_hannsdb_cli.py tests/test_hannsdb_client_config_shape.py -q
```
Expected: PASS.
- [ ] Commit checkpoint:
```bash
git add docs/vector-db-bench-notes.md docs/hannsdb-project-plan.md scripts/run_zvec_parity_smoke.sh crates/hannsdb-core/tests/zvec_parity_recovery.rs crates/hannsdb-py/tests/test_collection_parity.py
git commit -m "test: add parity verification and perf gates"
```

## Recommended execution order

1. Task 1: parity acceptance harness
2. Task 2: schema generalization
3. Task 3: collection handles + versioned segments
4. Task 4: per-field index registry and DDL
5. Task 5: query planner and executor
6. Task 6: pure-Python facade
7. Task 7: daemon surface
8. Task 8: quality/perf gates

## Sequencing rules

- Do not start IVF or quantized index work until the field-oriented schema and index descriptor contract are stable.
- Do not start multi-query Python UX before the Rust query planner contract exists.
- Do not widen the daemon first; daemon work must trail the core and Python API.
- Keep a compatibility adapter for the existing single-vector benchmark path until the new planner path matches or beats today's QPS/recall envelope.
- Treat HNSW-RabitQ parity as optional for the first merge if Decision Gate B is unresolved, but do not let that block the descriptor/planner refactor.

## Final acceptance bar

- HannsDB can define and persist multi-vector schemas with richer scalar types.
- HannsDB can create and drop per-field vector and scalar indexes.
- HannsDB can execute multi-vector typed queries through a planner/executor contract.
- HannsDB ships a Python facade with `CollectionSchema`, `VectorSchema`, `QueryContext`, and `QueryExecutorFactory`.
- HannsDB has recovery/concurrency/recall coverage closer to `zvec`'s current test envelope.
- Existing HannsDB benchmark integrations still work during and after the migration.

