# HannsDB zvec Remaining Parity V3 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining substantive parity gap between HannsDB and the current local `zvec` codebase, with priority on engine-native capabilities that are still missing after the V2 parity work already merged into `main`.

**Architecture:** Do not re-open the already-landed V2 surface work. Keep the current Rust-first HannsDB structure, but finish the migration from a primary-vector-special-cased engine into a genuinely field-uniform collection engine. Prioritize core-native behavior first: uniform document storage, native mutation semantics, richer query planning, and per-field indexed runtime. Treat extension ecosystem parity and service/runtime polish as later workstreams layered on top of a correct core.

**Tech Stack:** Rust workspace, `serde`, `serde_json`, PyO3, axum, `pytest`, `cargo test`, `numpy`, `knowhere-rs`, existing `hannsdb-index` abstraction, existing parity smoke scripts

---

## Chunk 1: Remaining gap snapshot and file map

### Current state after V2

- Multi-vector schema metadata, typed query, `include_vector`, `query_by_id_field_name`, `delete_by_filter`, daemon delete-by-filter transport, and the pure-Python collection facade are already merged.
- Secondary vector fields can participate in typed recall, and indexed secondary fields can take the single-vector fast path.
- The current parity gap is no longer “missing top-level API names”. It is mostly:
  - field-uniform document/storage semantics
  - richer scalar/vector type coverage
  - native schema mutation and native update semantics
  - richer query planner/executor behavior
  - per-field persisted ANN/runtime parity
  - runtime/service depth
  - extension ecosystem parity

### Remaining non-goals for the first execution batch

- Do not build a full SQL engine clone in the first batch.
- Do not chase embedding/reranker cloud integrations before the core-native query and mutation gaps are closed.
- Do not rewrite daemon into a separate source of truth; it remains a thin transport over core.

### File map for remaining parity work

**Core data model and mutation**
- Modify: `crates/hannsdb-core/src/document.rs`
- Modify: `crates/hannsdb-core/src/catalog/collection.rs`
- Create: `crates/hannsdb-core/src/catalog/schema_mutation.rs`
- Modify: `crates/hannsdb-core/src/catalog/mod.rs`
- Modify: `crates/hannsdb-core/src/db.rs`
- Create: `crates/hannsdb-core/src/segment/field_store.rs`
- Create: `crates/hannsdb-core/src/segment/sparse.rs`
- Modify: `crates/hannsdb-core/src/segment/mod.rs`
- Modify: `crates/hannsdb-core/src/segment/vectors.rs`

**Query and execution**
- Modify: `crates/hannsdb-core/src/query/ast.rs`
- Modify: `crates/hannsdb-core/src/query/filter.rs`
- Create: `crates/hannsdb-core/src/query/order_by.rs`
- Create: `crates/hannsdb-core/src/query/rerank.rs`
- Modify: `crates/hannsdb-core/src/query/planner.rs`
- Modify: `crates/hannsdb-core/src/query/executor.rs`
- Modify: `crates/hannsdb-core/src/query/mod.rs`

**Index/runtime**
- Create: `crates/hannsdb-core/src/segment/index_runtime.rs`
- Modify: `crates/hannsdb-core/src/segment/manager.rs`
- Modify: `crates/hannsdb-core/src/segment/version_set.rs`
- Modify: `crates/hannsdb-index/src/adapter.rs`
- Modify: `crates/hannsdb-index/src/descriptor.rs`
- Modify: `crates/hannsdb-index/src/factory.rs`
- Modify: `crates/hannsdb-index/src/hnsw.rs`
- Modify: `crates/hannsdb-index/src/ivf.rs`
- Create: `crates/hannsdb-index/src/sparse.rs`

**Python and daemon**
- Modify: `crates/hannsdb-py/src/lib.rs`
- Create: `crates/hannsdb-py/src/query_bridge.rs`
- Modify: `crates/hannsdb-py/python/hannsdb/model/collection.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/doc.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/schema/field_schema.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/vector_query.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/index_params.py`
- Modify: `crates/hannsdb-py/python/hannsdb/executor/query_executor.py`
- Modify: `crates/hannsdb-daemon/src/api.rs`
- Modify: `crates/hannsdb-daemon/src/routes.rs`
- Create: `crates/hannsdb-daemon/src/routes_mutation.rs`
- Create: `crates/hannsdb-daemon/src/routes_search.rs`
- Modify: `crates/hannsdb-daemon/tests/http_smoke.rs`

**Tests and gates**
- Modify: `crates/hannsdb-core/tests/zvec_parity_schema.rs`
- Create: `crates/hannsdb-core/tests/zvec_parity_mutation.rs`
- Create: `crates/hannsdb-core/tests/zvec_parity_query_planning.rs`
- Create: `crates/hannsdb-core/tests/zvec_parity_rerank.rs`
- Create: `crates/hannsdb-core/tests/zvec_parity_index_runtime.rs`
- Modify: `crates/hannsdb-core/tests/collection_api.rs`
- Modify: `crates/hannsdb-core/tests/wal_recovery.rs`
- Create: `crates/hannsdb-core/tests/zvec_parity_types.rs`
- Create: `crates/hannsdb-core/tests/zvec_parity_sparse.rs`
- Create: `crates/hannsdb-py/tests/test_collection_mutation_surface.py`
- Create: `crates/hannsdb-py/tests/test_query_order_surface.py`
- Create: `crates/hannsdb-py/tests/test_query_rerank_surface.py`
- Create: `crates/hannsdb-py/tests/test_sparse_surface.py`
- Modify: `crates/hannsdb-py/tests/test_collection_facade.py`
- Modify: `crates/hannsdb-py/tests/test_collection_parity.py`
- Modify: `crates/hannsdb-py/tests/test_query_executor.py`
- Modify: `crates/hannsdb-py/tests/test_schema_surface.py`
- Modify: `crates/hannsdb-py/tests/test_typing_surface.py`
- Create: `crates/hannsdb-daemon/tests/http_remaining_parity.rs`
- Modify: `scripts/run_zvec_parity_smoke.sh`

## Chunk 2: P0 engine parity foundation

### Task 1: Freeze the remaining parity gaps in tests before changing the core

**Files:**
- Create: `crates/hannsdb-core/tests/zvec_parity_mutation.rs`
- Modify: `crates/hannsdb-core/tests/zvec_parity_schema.rs`
- Modify: `crates/hannsdb-core/tests/wal_recovery.rs`
- Modify: `crates/hannsdb-py/tests/test_schema_surface.py`
- Create: `crates/hannsdb-py/tests/test_collection_mutation_surface.py`

- [ ] Add failing Rust tests for:
  - field-uniform document fetch/update without primary-vector special-casing
  - core-native partial update behavior expectations
  - native column DDL and schema-mutation recovery expectations
- [ ] Run: `cargo test -p hannsdb-core --test zvec_parity_schema --test zvec_parity_mutation --test wal_recovery -- --nocapture`
Expected: FAIL in the P0 storage/mutation gaps, with existing V2 tests staying green.
- [ ] Add failing Python tests for:
  - `Collection.update(...)` behaving like a real native update contract, not a facade-only fetch/merge/upsert shim
  - `add_column/drop_column/alter_column` becoming true capabilities instead of `NotImplementedError`
- [ ] Run: `uv run --with pytest pytest crates/hannsdb-py/tests/test_schema_surface.py crates/hannsdb-py/tests/test_collection_mutation_surface.py -q`
Expected: FAIL on the P0 facade tests for native update and column mutation.
- [ ] Commit checkpoint:
```bash
git add crates/hannsdb-core/tests/zvec_parity_mutation.rs crates/hannsdb-core/tests/zvec_parity_schema.rs crates/hannsdb-core/tests/wal_recovery.rs crates/hannsdb-py/tests/test_schema_surface.py crates/hannsdb-py/tests/test_collection_mutation_surface.py
git commit -m "test: lock remaining zvec parity gaps"
```

### Task 2: Replace the `vector + vectors` dual-track document model with a field-uniform vector store

**Files:**
- Modify: `crates/hannsdb-core/src/document.rs`
- Create: `crates/hannsdb-core/src/segment/field_store.rs`
- Modify: `crates/hannsdb-core/src/segment/mod.rs`
- Modify: `crates/hannsdb-core/src/segment/vectors.rs`
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-core/tests/document_api.rs`
- Modify: `crates/hannsdb-core/tests/multi_vector_document.rs`
- Modify: `crates/hannsdb-core/tests/zvec_parity_query.rs`
- Modify: `crates/hannsdb-py/python/hannsdb/model/doc.py`
- Modify: `crates/hannsdb-py/src/lib.rs`
- Test: `crates/hannsdb-core/tests/zvec_parity_schema.rs`
- Test: `crates/hannsdb-core/tests/collection_api.rs`
- Test: `crates/hannsdb-core/tests/wal_recovery.rs`
- Test: `crates/hannsdb-py/tests/test_collection_facade.py`

- [ ] Run the Task 1 red suites that cover field-uniform storage behavior and confirm the primary/secondary vector parity cases are still failing before implementation.
- [ ] Run: `cargo test -p hannsdb-core --test zvec_parity_schema --test zvec_parity_query --test collection_api --test document_api --test multi_vector_document --test wal_recovery -- --nocapture`
Expected: FAIL because `Document` still stores `vector` separately from `vectors` and persisted reopen behavior is not field-uniform yet.
- [ ] Change `Document` so vectors are stored and transported as one field-keyed map, with `primary_vector` only remaining in schema/catalog, not as a separate runtime payload field.
- [ ] Move row-aligned vector persistence behind `segment/field_store.rs` so `db.rs` stops stitching primary and secondary vector storage separately.
- [ ] Update PyO3 and pure-Python `Doc` conversion so they do not need to synthesize primary-vector special cases.
- [ ] Run: `cargo test -p hannsdb-core --test zvec_parity_schema --test zvec_parity_query --test collection_api --test document_api --test multi_vector_document --test wal_recovery -- --nocapture`
Expected: PASS.
- [ ] Run: `uv run --with pytest pytest crates/hannsdb-py/tests/test_collection_facade.py -q`
Expected: PASS.
- [ ] Commit checkpoint:
```bash
git add crates/hannsdb-core/src/document.rs crates/hannsdb-core/src/segment/field_store.rs crates/hannsdb-core/src/segment/mod.rs crates/hannsdb-core/src/segment/vectors.rs crates/hannsdb-core/src/db.rs crates/hannsdb-core/tests/zvec_parity_schema.rs crates/hannsdb-core/tests/zvec_parity_query.rs crates/hannsdb-core/tests/collection_api.rs crates/hannsdb-core/tests/document_api.rs crates/hannsdb-core/tests/multi_vector_document.rs crates/hannsdb-core/tests/wal_recovery.rs crates/hannsdb-py/python/hannsdb/model/doc.py crates/hannsdb-py/src/lib.rs crates/hannsdb-py/tests/test_collection_facade.py
git commit -m "refactor: make document vector storage field-uniform"
```

### Task 3: Make update and schema mutation native core capabilities

**Files:**
- Create: `crates/hannsdb-core/src/catalog/schema_mutation.rs`
- Modify: `crates/hannsdb-core/src/catalog/mod.rs`
- Modify: `crates/hannsdb-core/src/catalog/collection.rs`
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-core/tests/collection_api.rs`
- Modify: `crates/hannsdb-core/tests/zvec_parity_mutation.rs`
- Modify: `crates/hannsdb-core/tests/wal_recovery.rs`
- Modify: `crates/hannsdb-py/src/lib.rs`
- Modify: `crates/hannsdb-py/python/hannsdb/model/collection.py`
- Modify: `crates/hannsdb-py/tests/test_collection_facade.py`
- Modify: `crates/hannsdb-py/tests/test_collection_parity.py`
- Modify: `crates/hannsdb-py/tests/test_collection_mutation_surface.py`
- Modify: `crates/hannsdb-py/tests/test_schema_surface.py`

- [ ] Run the Task 1 red suites that cover native update and schema mutation, and confirm those mutation-specific cases are still failing before implementation.
- [ ] Run: `cargo test -p hannsdb-core --test collection_api --test zvec_parity_mutation --test wal_recovery -- --nocapture`
Expected: FAIL because these are still missing or facade-only.
- [ ] Move `update` semantics into core so Python no longer needs to emulate it with `fetch -> merge -> upsert`.
- [ ] Implement schema mutation in a focused `catalog/schema_mutation.rs` module rather than growing `db.rs` further.
- [ ] Expose the new core methods through PyO3 and delete the current Python-only `NotImplementedError` stubs.
- [ ] Run: `cargo test -p hannsdb-core --test collection_api --test zvec_parity_mutation --test wal_recovery -- --nocapture`
Expected: PASS.
- [ ] Run: `uv run --with pytest pytest crates/hannsdb-py/tests/test_collection_mutation_surface.py crates/hannsdb-py/tests/test_collection_facade.py crates/hannsdb-py/tests/test_collection_parity.py crates/hannsdb-py/tests/test_schema_surface.py -q`
Expected: PASS.
- [ ] Commit checkpoint:
```bash
git add crates/hannsdb-core/src/catalog/schema_mutation.rs crates/hannsdb-core/src/catalog/mod.rs crates/hannsdb-core/src/catalog/collection.rs crates/hannsdb-core/src/db.rs crates/hannsdb-core/tests/collection_api.rs crates/hannsdb-core/tests/zvec_parity_mutation.rs crates/hannsdb-core/tests/wal_recovery.rs crates/hannsdb-py/src/lib.rs crates/hannsdb-py/python/hannsdb/model/collection.py crates/hannsdb-py/tests/test_collection_mutation_surface.py crates/hannsdb-py/tests/test_collection_facade.py crates/hannsdb-py/tests/test_collection_parity.py crates/hannsdb-py/tests/test_schema_surface.py
git commit -m "feat: add native update and schema mutation"
```

### Chunk 2 verification gate

- [ ] Run: `cargo test -p hannsdb-core --test zvec_parity_schema --test zvec_parity_mutation --test collection_api --test wal_recovery -- --nocapture`
Expected: PASS.
- [ ] Run: `uv run --with pytest pytest crates/hannsdb-py/tests/test_schema_surface.py crates/hannsdb-py/tests/test_collection_mutation_surface.py crates/hannsdb-py/tests/test_collection_facade.py crates/hannsdb-py/tests/test_collection_parity.py -q`
Expected: PASS.

## Chunk 3: P1 query planner and runtime parity

### Task 4: Expand filter/query planning beyond the current narrow typed path

**Files:**
- Modify: `crates/hannsdb-core/src/query/ast.rs`
- Modify: `crates/hannsdb-core/src/query/filter.rs`
- Create: `crates/hannsdb-core/src/query/order_by.rs`
- Modify: `crates/hannsdb-core/src/query/planner.rs`
- Modify: `crates/hannsdb-core/src/query/executor.rs`
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-core/src/query/mod.rs`
- Create: `crates/hannsdb-core/tests/zvec_parity_query_planning.rs`
- Modify: `crates/hannsdb-py/src/lib.rs`
- Create: `crates/hannsdb-py/src/query_bridge.rs`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/vector_query.py`
- Create: `crates/hannsdb-py/tests/test_query_order_surface.py`

- [ ] Add failing Rust tests for:
  - `or` and parenthesized filter expressions
  - order-by on scalar fields after vector recall
  - filter-only scans that still support projection/order semantics
  - typed queries that combine more than one vector field plus richer filtering
- [ ] Run: `cargo test -p hannsdb-core --test zvec_parity_query_planning -- --nocapture`
Expected: FAIL because current filter grammar and planner are intentionally narrow.
- [ ] Add failing Python tests for typed `order_by` transport and projection behavior through `QueryContext`.
- [ ] Run: `uv run --with pytest pytest crates/hannsdb-py/tests/test_query_order_surface.py -q`
Expected: FAIL because Python typed-query transport does not yet carry `order_by`.
- [ ] Split ordering logic into `query/order_by.rs` and keep `planner.rs` focused on plan selection.
- [ ] Apply ordering and projection in `query/executor.rs` and only keep plan selection in `planner.rs`.
- [ ] Extract typed query transport into `crates/hannsdb-py/src/query_bridge.rs`, keep `lib.rs` to module wiring and thin delegation, and extend that bridge with the minimum `order_by` shape needed for the parity tests.
- [ ] Extend `FilterExpr`/AST to represent the minimum richer grammar needed for the parity tests instead of bolting more cases into the current string-split parser.
- [ ] Keep this batch intentionally below “full SQL engine clone”; if a requirement depends on parser/analyzer complexity comparable to zvec SQL, mark it explicitly out of scope for V3 instead of partially implementing it.
- [ ] Run: `cargo test -p hannsdb-core --test zvec_parity_query_planning -- --nocapture`
Expected: PASS for the targeted richer grammar/planner cases.
- [ ] Run: `uv run --with pytest pytest crates/hannsdb-py/tests/test_query_order_surface.py -q`
Expected: PASS.
- [ ] Commit checkpoint:
```bash
git add crates/hannsdb-core/src/query/ast.rs crates/hannsdb-core/src/query/filter.rs crates/hannsdb-core/src/query/order_by.rs crates/hannsdb-core/src/query/planner.rs crates/hannsdb-core/src/query/executor.rs crates/hannsdb-core/src/db.rs crates/hannsdb-core/src/query/mod.rs crates/hannsdb-core/tests/zvec_parity_query_planning.rs crates/hannsdb-py/src/lib.rs crates/hannsdb-py/src/query_bridge.rs crates/hannsdb-py/python/hannsdb/model/param/vector_query.py crates/hannsdb-py/tests/test_query_order_surface.py
git commit -m "feat: expand typed filter and order planning"
```

### Task 5: Move reranking into core and finish the multi-vector typed execution story

**Files:**
- Modify: `crates/hannsdb-core/src/query/ast.rs`
- Create: `crates/hannsdb-core/src/query/rerank.rs`
- Modify: `crates/hannsdb-core/src/query/mod.rs`
- Modify: `crates/hannsdb-core/src/query/planner.rs`
- Modify: `crates/hannsdb-core/src/query/executor.rs`
- Create: `crates/hannsdb-core/tests/zvec_parity_rerank.rs`
- Modify: `crates/hannsdb-py/src/lib.rs`
- Modify: `crates/hannsdb-py/src/query_bridge.rs`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/vector_query.py`
- Modify: `crates/hannsdb-py/python/hannsdb/executor/query_executor.py`
- Modify: `crates/hannsdb-py/tests/test_query_executor.py`
- Create: `crates/hannsdb-py/tests/test_query_rerank_surface.py`

- [ ] Write failing tests for:
  - `query_by_id + reranker`
  - `group_by + reranker`
  - mixed multi-vector weighted rerank on a real collection
  - Python reranker paths no longer raising facade-only `NotImplementedError`
  - existing `QueryExecutor` reranker integration continuing to work after the transport/contract move
- [ ] Run: `cargo test -p hannsdb-core --test zvec_parity_rerank -- --nocapture`
Expected: FAIL because core typed planner still rejects `reranker`.
- [ ] Run: `uv run --with pytest pytest crates/hannsdb-py/tests/test_query_rerank_surface.py -q`
Expected: FAIL because reranker correctness is still owned by Python and the core-native surface does not exist yet.
- [ ] Introduce a focused `query/rerank.rs` module with a core-native rerank contract and a built-in weighted/RRF path.
- [ ] Extend `query/ast.rs` and the Python query param model so rerank requests have a structured serializable spec instead of relying on the current placeholder model string or opaque Python-only objects.
- [ ] Preserve existing custom/object-based Python reranker compatibility through a thin adapter while moving built-in weighted/RRF correctness into core-native behavior.
- [ ] Simplify the Python executor so it stops owning correctness for multi-vector rerank orchestration; transport should now flow through `query_bridge.rs` and thin `lib.rs` wiring.
- [ ] Run: `cargo test -p hannsdb-core --test zvec_parity_rerank -- --nocapture`
Expected: PASS.
- [ ] Run: `uv run --with pytest pytest crates/hannsdb-py/tests/test_query_rerank_surface.py crates/hannsdb-py/tests/test_query_executor.py -q`
Expected: PASS.
- [ ] Commit checkpoint:
```bash
git add crates/hannsdb-core/src/query/ast.rs crates/hannsdb-core/src/query/rerank.rs crates/hannsdb-core/src/query/mod.rs crates/hannsdb-core/src/query/planner.rs crates/hannsdb-core/src/query/executor.rs crates/hannsdb-core/tests/zvec_parity_rerank.rs crates/hannsdb-py/src/lib.rs crates/hannsdb-py/src/query_bridge.rs crates/hannsdb-py/python/hannsdb/model/param/vector_query.py crates/hannsdb-py/python/hannsdb/executor/query_executor.py crates/hannsdb-py/tests/test_query_rerank_surface.py crates/hannsdb-py/tests/test_query_executor.py
git commit -m "feat: add core-native typed reranking"
```

### Task 6: Finish per-field indexed runtime parity instead of relying on lazy in-memory fallbacks

**Files:**
- Modify: `crates/hannsdb-core/src/db.rs`
- Create: `crates/hannsdb-core/src/segment/index_runtime.rs`
- Modify: `crates/hannsdb-core/src/segment/mod.rs`
- Modify: `crates/hannsdb-core/src/segment/manager.rs`
- Modify: `crates/hannsdb-core/src/segment/version_set.rs`
- Modify: `crates/hannsdb-index/src/descriptor.rs`
- Modify: `crates/hannsdb-index/src/factory.rs`
- Modify: `crates/hannsdb-index/src/hnsw.rs`
- Modify: `crates/hannsdb-index/src/ivf.rs`
- Create: `crates/hannsdb-core/tests/zvec_parity_index_runtime.rs`
- Modify: `crates/hannsdb-core/tests/collection_api.rs`

- [ ] Write failing tests for:
  - persisted secondary-field ANN reuse across reopen
  - optimize/flush keeping per-field descriptors and cached runtime aligned
  - per-field ANN invalidation after update/delete/schema mutation
- [ ] Run: `cargo test -p hannsdb-core --test zvec_parity_index_runtime --test collection_api -- --nocapture`
Expected: FAIL because secondary fast paths still depend on lazy runtime assembly and do not have full persisted parity.
- [ ] Extract per-field runtime/index lifecycle bookkeeping into `segment/index_runtime.rs` before modifying reopen/flush/invalidation behavior in `db.rs`.
- [ ] Persist and reload per-field descriptor/runtime state uniformly for primary and secondary indexed fields.
- [ ] Align optimize/flush with the same runtime state contract instead of letting lifecycle code diverge across operations.
- [ ] Invalidate per-field ANN state after update/delete/schema mutation through the extracted runtime helper instead of duplicating invalidation rules in `db.rs`.
- [ ] Run: `cargo test -p hannsdb-core --test zvec_parity_index_runtime --test collection_api -- --nocapture`
Expected: PASS.
- [ ] Commit checkpoint:
```bash
git add crates/hannsdb-core/src/db.rs crates/hannsdb-core/src/segment/index_runtime.rs crates/hannsdb-core/src/segment/mod.rs crates/hannsdb-core/src/segment/manager.rs crates/hannsdb-core/src/segment/version_set.rs crates/hannsdb-index/src/descriptor.rs crates/hannsdb-index/src/factory.rs crates/hannsdb-index/src/hnsw.rs crates/hannsdb-index/src/ivf.rs crates/hannsdb-core/tests/zvec_parity_index_runtime.rs crates/hannsdb-core/tests/collection_api.rs
git commit -m "feat: unify per-field indexed runtime"
```

### Chunk 3 verification gate

- [ ] Run: `cargo test -p hannsdb-core --test zvec_parity_query_planning --test zvec_parity_rerank --test zvec_parity_index_runtime --test collection_api -- --nocapture`
Expected: PASS.
- [ ] Run: `uv run --with pytest pytest crates/hannsdb-py/tests/test_query_order_surface.py crates/hannsdb-py/tests/test_query_rerank_surface.py crates/hannsdb-py/tests/test_query_executor.py -q`
Expected: PASS.

## Chunk 4: P2 type-system, sparse, and service parity

### Task 7: Expand the type system to the next useful zvec-compatible subset

**Files:**
- Modify: `crates/hannsdb-core/src/document.rs`
- Modify: `crates/hannsdb-core/src/catalog/collection.rs`
- Modify: `crates/hannsdb-py/src/lib.rs`
- Modify: `crates/hannsdb-py/python/hannsdb/model/collection.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/schema/field_schema.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/index_params.py`
- Create: `crates/hannsdb-core/tests/zvec_parity_types.rs`
- Modify: `crates/hannsdb-py/tests/test_schema_surface.py`
- Modify: `crates/hannsdb-py/tests/test_typing_surface.py`

- [ ] Add failing tests for:
  - only the still-missing post-V2 scalar types: `INT32`, `UINT32`, `UINT64`, `FLOAT`
  - array scalar value storage and fetch for the newly added scalar types
  - only the still-missing post-V2 vector metadata types: `VECTOR_FP16`, `VECTOR_FP64`, `VECTOR_INT8`
- [ ] Run: `uv run --with pytest pytest crates/hannsdb-py/tests/test_schema_surface.py crates/hannsdb-py/tests/test_typing_surface.py -q`
Expected: FAIL because the Python surface still mirrors the narrower current Rust enum and data-type mapping.
- [ ] Run: `cargo test -p hannsdb-core --test zvec_parity_types -- --nocapture`
Expected: FAIL because the current type system is still `String/Int64/Float64/Bool/VectorFp32`.
- [ ] Implement the smallest internally coherent type expansion first; do not add sparse vector values in this task.
- [ ] Keep Python `DataType` and schema wrappers exactly aligned with the new Rust enum values.
- [ ] Run: `cargo test -p hannsdb-core --test zvec_parity_types -- --nocapture`
Expected: PASS.
- [ ] Run: `uv run --with pytest pytest crates/hannsdb-py/tests/test_schema_surface.py crates/hannsdb-py/tests/test_typing_surface.py -q`
Expected: PASS.
- [ ] Commit checkpoint:
```bash
git add crates/hannsdb-core/src/document.rs crates/hannsdb-core/src/catalog/collection.rs crates/hannsdb-py/src/lib.rs crates/hannsdb-py/python/hannsdb/model/collection.py crates/hannsdb-py/python/hannsdb/model/schema/field_schema.py crates/hannsdb-py/python/hannsdb/model/param/index_params.py crates/hannsdb-core/tests/zvec_parity_types.rs crates/hannsdb-py/tests/test_schema_surface.py crates/hannsdb-py/tests/test_typing_surface.py
git commit -m "feat: expand core and python type system"
```

### Task 8: Add sparse vector field support as a dedicated workstream

**Files:**
- Create: `crates/hannsdb-index/src/sparse.rs`
- Modify: `crates/hannsdb-index/src/adapter.rs`
- Modify: `crates/hannsdb-index/src/descriptor.rs`
- Modify: `crates/hannsdb-index/src/factory.rs`
- Modify: `crates/hannsdb-index/src/lib.rs`
- Modify: `crates/hannsdb-core/src/document.rs`
- Modify: `crates/hannsdb-core/src/db.rs`
- Create: `crates/hannsdb-core/src/segment/sparse.rs`
- Modify: `crates/hannsdb-core/src/segment/field_store.rs`
- Modify: `crates/hannsdb-core/src/segment/mod.rs`
- Modify: `crates/hannsdb-core/src/segment/vectors.rs`
- Modify: `crates/hannsdb-core/src/query/ast.rs`
- Modify: `crates/hannsdb-core/src/query/planner.rs`
- Modify: `crates/hannsdb-core/src/query/executor.rs`
- Modify: `crates/hannsdb-py/src/lib.rs`
- Modify: `crates/hannsdb-py/python/hannsdb/model/collection.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/doc.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/vector_query.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/schema/field_schema.py`
- Create: `crates/hannsdb-core/tests/zvec_parity_sparse.rs`
- Create: `crates/hannsdb-py/tests/test_sparse_surface.py`

- [ ] Write failing tests for sparse vector schema, sparse doc storage, sparse fetch round-trip, and sparse recall requests.
- [ ] Add one backend-selection smoke that proves the sparse path is registered and instantiable from core/Python, not just present as a dead module.
- [ ] Run: `uv run --with pytest pytest crates/hannsdb-py/tests/test_sparse_surface.py -q`
Expected: FAIL because Python schema/doc/query transport is still dense-only.
- [ ] Run: `cargo test -p hannsdb-core --test zvec_parity_sparse -- --nocapture`
Expected: FAIL because sparse vectors do not exist in the runtime.
- [ ] Implement sparse vectors as a separate path from dense vectors. Do not overload the current dense row codec with ad hoc sentinel encodings.
- [ ] Register and export the sparse backend through `hannsdb-index/src/factory.rs` and `hannsdb-index/src/lib.rs` so the new path is actually reachable.
- [ ] Run: `cargo test -p hannsdb-core --test zvec_parity_sparse -- --nocapture`
Expected: PASS.
- [ ] Run: `uv run --with pytest pytest crates/hannsdb-py/tests/test_sparse_surface.py -q`
Expected: PASS.
- [ ] Commit checkpoint:
```bash
git add crates/hannsdb-index/src/sparse.rs crates/hannsdb-index/src/adapter.rs crates/hannsdb-index/src/descriptor.rs crates/hannsdb-index/src/factory.rs crates/hannsdb-index/src/lib.rs crates/hannsdb-core/src/document.rs crates/hannsdb-core/src/db.rs crates/hannsdb-core/src/segment/sparse.rs crates/hannsdb-core/src/segment/field_store.rs crates/hannsdb-core/src/segment/mod.rs crates/hannsdb-core/src/segment/vectors.rs crates/hannsdb-core/src/query/ast.rs crates/hannsdb-core/src/query/planner.rs crates/hannsdb-core/src/query/executor.rs crates/hannsdb-py/src/lib.rs crates/hannsdb-py/python/hannsdb/model/collection.py crates/hannsdb-py/python/hannsdb/model/doc.py crates/hannsdb-py/python/hannsdb/model/param/vector_query.py crates/hannsdb-py/python/hannsdb/model/schema/field_schema.py crates/hannsdb-core/tests/zvec_parity_sparse.rs crates/hannsdb-py/tests/test_sparse_surface.py
git commit -m "feat: add sparse vector field support"
```

### Task 9: Lift daemon/runtime depth only after the core gaps are closed

**Files:**
- Modify: `crates/hannsdb-daemon/src/api.rs`
- Modify: `crates/hannsdb-daemon/src/routes.rs`
- Create: `crates/hannsdb-daemon/src/routes_mutation.rs`
- Create: `crates/hannsdb-daemon/src/routes_search.rs`
- Modify: `crates/hannsdb-daemon/tests/http_smoke.rs`
- Create: `crates/hannsdb-daemon/tests/http_remaining_parity.rs`
- Modify: `scripts/run_zvec_parity_smoke.sh`

- [ ] Add failing daemon tests for mutation routes:
  - `POST /collections/:collection/records/update` delegating to native update semantics
  - `POST /collections/:collection/schema/columns` for `add_column`
  - `PATCH /collections/:collection/schema/columns/:field_name` for `alter_column`
  - `DELETE /collections/:collection/schema/columns/:field_name` for `drop_column`
- [ ] Add failing daemon tests for typed/search payload parity:
  - collection creation/fetch/search payloads carrying the newly added scalar and vector type variants from Task 7
  - sparse-vector schema/create/fetch/search payloads added in Task 8
  - `POST /collections/:collection/search` typed payloads carrying the richer `order_by` / `reranker` contracts added in earlier tasks
- [ ] Run: `cargo test -p hannsdb-daemon --test http_remaining_parity -- --nocapture`
Expected: FAIL until daemon transport is aligned.
- [ ] Split mutation and richer typed-search handlers into `routes_mutation.rs` and `routes_search.rs` instead of continuing to grow the existing top-level route file and smoke suite.
- [ ] Run: `cargo test -p hannsdb-daemon --test http_remaining_parity -- --nocapture`
Expected: PASS.
- [ ] Run: `cargo test -p hannsdb-daemon --test http_smoke -- --nocapture`
Expected: PASS.
- [ ] Run: `bash scripts/run_zvec_parity_smoke.sh`
Expected: PASS.
- [ ] Commit checkpoint:
```bash
git add crates/hannsdb-daemon/src/api.rs crates/hannsdb-daemon/src/routes.rs crates/hannsdb-daemon/src/routes_mutation.rs crates/hannsdb-daemon/src/routes_search.rs crates/hannsdb-daemon/tests/http_smoke.rs crates/hannsdb-daemon/tests/http_remaining_parity.rs scripts/run_zvec_parity_smoke.sh
git commit -m "feat: align daemon transport with remaining core parity"
```

## Final verification checklist

- [ ] Run: `cargo test -p hannsdb-core --test zvec_parity_schema --test zvec_parity_mutation --test zvec_parity_query_planning --test zvec_parity_rerank --test zvec_parity_index_runtime --test zvec_parity_types --test zvec_parity_sparse --test collection_api --test wal_recovery -- --nocapture`
Expected: PASS.
- [ ] Run: `cargo test -p hannsdb-daemon --test http_smoke -- --nocapture`
Expected: PASS.
- [ ] Run: `cargo test -p hannsdb-daemon --test http_remaining_parity -- --nocapture`
Expected: PASS.
- [ ] Run: `cargo test -p hannsdb-py --features python-binding --lib -- --nocapture`
Expected: PASS.
- [ ] Run: `uv run --with pytest pytest crates/hannsdb-py/tests/test_schema_surface.py crates/hannsdb-py/tests/test_typing_surface.py crates/hannsdb-py/tests/test_collection_mutation_surface.py crates/hannsdb-py/tests/test_query_order_surface.py crates/hannsdb-py/tests/test_query_rerank_surface.py crates/hannsdb-py/tests/test_sparse_surface.py crates/hannsdb-py/tests/test_collection_parity.py crates/hannsdb-py/tests/test_collection_facade.py crates/hannsdb-py/tests/test_query_executor.py -q`
Expected: PASS.
- [ ] Run: `bash scripts/run_zvec_parity_smoke.sh`
Expected: PASS.
- [ ] Run: `git diff --check`
Expected: no output.

## Sequencing note

Implement this plan in order:

1. lock the remaining parity gaps in tests
2. make the document model field-uniform
3. move update and schema mutation into core
4. deepen the typed planner/executor
5. unify per-field indexed runtime
6. expand types
7. add sparse vectors
8. align daemon/runtime transport

Do not start embedding-function or hosted-reranker parity until this plan is complete. Those are product extensions, not core engine blockers.
