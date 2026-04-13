# HannsDB zvec String PK Query Slice Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship one end-to-end parity slice that lets the Python API accept real string primary keys, use them through `query_by_id`, expose `order_by`, and keep existing numeric-ID behavior working.

**Architecture:** Keep HannsDB's existing numeric internal row/document paths for ANN and segment logic. Add a small persisted collection-level PK registry that maps public string keys to internal numeric IDs, and route Python insert/fetch/delete/query paths through that registry instead of forcing every public ID to parse as `i64`. Treat `order_by` as a surface-alignment task: core already supports it, so the work is to transport and validate it cleanly through PyO3 and the pure-Python facade.

**Tech Stack:** Rust workspace, `serde`, `serde_json`, PyO3, `maturin`, `pytest`, existing `hannsdb-core` query planner/executor, existing pure-Python collection facade

---

## Chunk 1: Scope lock and file map

### Current state to preserve

- Core already has `order_by` in `crates/hannsdb-core/src/query/ast.rs` and validates/applies it in `query/planner.rs`, `query/executor.rs`, and `db.rs`.
- Pure-Python `Doc` already stores `id` as `str`, so the user-facing model is ready for opaque string IDs.
- `QueryContext.query_by_id` already accepts `str` values in Python, but the PyO3 bridge still converts them with `parse_doc_id(...)`, so only numeric strings work.
- `Collection.query_context(...)` already routes directly into core when no reranker is used; this slice should not redesign reranking.

### Explicit non-goals for this slice

- Do not redesign WAL crash recovery for string PKs.
- Do not change ANN/search internals to use string IDs directly.
- Do not broaden into schema-mutation parity, quantization, or runtime/storage rearchitecture.
- Do not unlock every blocked query combination; only expose the minimum `order_by` surface needed for sorted query parity.

### File map

**Core PK registry**
- Create: `crates/hannsdb-core/src/pk.rs`
- Modify: `crates/hannsdb-core/src/lib.rs`
- Modify: `crates/hannsdb-core/src/catalog/collection.rs`
- Modify: `crates/hannsdb-core/src/catalog/mod.rs`
- Modify: `crates/hannsdb-core/src/db.rs`

**PyO3 and Python surface**
- Create: `crates/hannsdb-py/src/query_bridge.rs`
- Modify: `crates/hannsdb-py/src/lib.rs`
- Modify: `crates/hannsdb-py/python/hannsdb/model/collection.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/vector_query.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/__init__.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/__init__.py`
- Modify: `crates/hannsdb-py/python/hannsdb/__init__.py`

**Tests and verification**
- Create: `crates/hannsdb-core/tests/zvec_parity_string_pk.rs`
- Modify: `crates/hannsdb-py/tests/test_collection_facade.py`
- Modify: `crates/hannsdb-py/tests/test_query_executor.py`
- Create: `crates/hannsdb-py/tests/test_query_order_surface.py`
- Create: `scripts/smoke_string_pk_order_by.py`

---

## Chunk 2: Red tests and persisted PK registry

### Task 1: Lock the slice boundary in failing tests

**Files:**
- Create: `crates/hannsdb-core/tests/zvec_parity_string_pk.rs`
- Modify: `crates/hannsdb-py/tests/test_collection_facade.py`
- Modify: `crates/hannsdb-py/tests/test_query_executor.py`
- Create: `crates/hannsdb-py/tests/test_query_order_surface.py`

- [ ] **Step 1: Add failing Rust tests for collection-level string PK behavior**

```rust
#[test]
fn zvec_parity_string_pk_reopens_and_query_by_id_resolves_alphanumeric_key() {
    // create collection
    // insert docs with public keys like "user-a" / "user-b"
    // reopen DB
    // resolve query_by_id through the string-PK path
    // assert returned docs still surface the original public keys
}
```

- [ ] **Step 2: Run the Rust red test**

Run: `cargo test -p hannsdb-core --test zvec_parity_string_pk -- --nocapture`  
Expected: FAIL because there is no persisted PK registry or string-key lookup path yet.

- [ ] **Step 3: Add failing Python tests for the public slice**

```python
def test_real_collection_query_by_id_accepts_alphanumeric_string_pk(tmp_path):
    ...

def test_query_context_accepts_order_by_object():
    ...

def test_real_collection_query_orders_by_scalar_field(tmp_path):
    ...
```

- [ ] **Step 4: Build the Python extension and run the Python red tests**

Run:

```bash
cd crates/hannsdb-py && \
maturin develop --features python-binding,hanns-backend && \
python -m pytest tests/test_collection_facade.py tests/test_query_executor.py tests/test_query_order_surface.py -q
```

Expected: FAIL with the current `doc id must parse to i64` path and missing `order_by` surface.

- [ ] **Step 5: Commit the red boundary**

```bash
git add crates/hannsdb-core/tests/zvec_parity_string_pk.rs crates/hannsdb-py/tests/test_collection_facade.py crates/hannsdb-py/tests/test_query_executor.py crates/hannsdb-py/tests/test_query_order_surface.py
git commit -m "test: lock string pk query slice boundaries"
```

### Task 2: Add a persisted collection-level PK registry in core

**Files:**
- Create: `crates/hannsdb-core/src/pk.rs`
- Modify: `crates/hannsdb-core/src/lib.rs`
- Modify: `crates/hannsdb-core/src/catalog/collection.rs`
- Modify: `crates/hannsdb-core/src/catalog/mod.rs`
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-core/tests/zvec_parity_string_pk.rs`

- [ ] **Step 1: Define the core PK model in a focused module**

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrimaryKeyMode {
    Numeric,
    String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrimaryKeyRegistry {
    pub mode: PrimaryKeyMode,
    pub next_internal_id: i64,
    pub key_to_id: BTreeMap<String, i64>,
    pub id_to_key: BTreeMap<i64, String>,
}
```

- [ ] **Step 2: Persist PK mode in collection metadata with a backward-compatible default**

Run: `cargo test -p hannsdb-core --test zvec_parity_string_pk -- --nocapture`  
Expected: still FAIL, but now compile against a real `PrimaryKeyMode` and registry type.

- [ ] **Step 3: Add collection-level helpers in `db.rs`**

Implement helpers that keep ANN/runtime numeric internally:

- `load_primary_key_registry(...)`
- `save_primary_key_registry(...)`
- `assign_internal_ids_for_public_keys(...)`
- `resolve_public_keys_to_internal_ids(...)`
- `display_key_for_internal_id(...)`

Keep the rule explicit:

- default collection mode is `numeric`
- first non-numeric public key upgrades the collection to `string`
- mixed numeric/non-numeric collection behavior is rejected unless explicitly normalized by the chosen mode

- [ ] **Step 4: Route the string-PK core API through the registry**

Add narrow core entry points instead of rewriting every legacy API:

- `insert_documents_with_public_keys(...)`
- `fetch_documents_by_public_keys(...)`
- `delete_by_public_keys(...)`
- `resolve_query_ids_by_public_keys(...)`

These should delegate to existing numeric internals after registry resolution.

- [ ] **Step 5: Run the focused core suite**

Run: `cargo test -p hannsdb-core --test zvec_parity_string_pk -- --nocapture`  
Expected: PASS.

- [ ] **Step 6: Commit the registry foundation**

```bash
git add crates/hannsdb-core/src/pk.rs crates/hannsdb-core/src/lib.rs crates/hannsdb-core/src/catalog/collection.rs crates/hannsdb-core/src/catalog/mod.rs crates/hannsdb-core/src/db.rs crates/hannsdb-core/tests/zvec_parity_string_pk.rs
git commit -m "feat: add persisted collection primary key registry"
```

### Chunk 2 verification gate

- [ ] Run: `cargo test -p hannsdb-core --test zvec_parity_string_pk -- --nocapture`
Expected: PASS.

---

## Chunk 3: PyO3 bridge and Python surface

### Task 3: Replace PyO3 `i64` coercion with PK-aware bridge logic

**Files:**
- Create: `crates/hannsdb-py/src/query_bridge.rs`
- Modify: `crates/hannsdb-py/src/lib.rs`
- Modify: `crates/hannsdb-py/python/hannsdb/model/collection.py`
- Modify: `crates/hannsdb-py/tests/test_collection_facade.py`
- Modify: `crates/hannsdb-py/tests/test_query_executor.py`

- [ ] **Step 1: Move query/doc transport helpers out of `lib.rs` into `query_bridge.rs`**

Start by extracting the currently tangled functions:

- `parse_query_ids(...)`
- `py_query_context_to_core(...)`
- `core_document_from_doc(...)`
- `doc_from_core_document(...)`

This keeps the new PK behavior isolated instead of growing `lib.rs` further.

- [ ] **Step 2: Remove `parse_doc_id(...)` from the public document/query path**

Replace the current coercion with a bridge that:

- keeps numeric collections on the existing fast path
- resolves string public IDs through the new core PK registry
- returns original public keys on fetch/query results

- [ ] **Step 3: Route all public ID-bearing collection methods through the same bridge**

Apply the bridge to:

- `insert(...)`
- `upsert(...)`
- `update(...)`
- `fetch(...)`
- `delete(...)`
- `query_context(...)`

Do not leave any path doing ad hoc `str -> i64` parsing.

- [ ] **Step 4: Run the focused Python suite**

Run:

```bash
cd crates/hannsdb-py && \
maturin develop --features python-binding,hanns-backend && \
python -m pytest tests/test_collection_facade.py tests/test_query_executor.py -q
```

Expected: PASS for alphanumeric string-PK insert/fetch/query-by-id while existing numeric-ID tests stay green.

- [ ] **Step 5: Commit the PK-aware bridge**

```bash
git add crates/hannsdb-py/src/query_bridge.rs crates/hannsdb-py/src/lib.rs crates/hannsdb-py/python/hannsdb/model/collection.py crates/hannsdb-py/tests/test_collection_facade.py crates/hannsdb-py/tests/test_query_executor.py
git commit -m "feat: bridge python collection ids through pk registry"
```

### Task 4: Expose `order_by` through the Python query surface

**Files:**
- Modify: `crates/hannsdb-py/src/query_bridge.rs`
- Modify: `crates/hannsdb-py/src/lib.rs`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/vector_query.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/collection.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/__init__.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/__init__.py`
- Modify: `crates/hannsdb-py/python/hannsdb/__init__.py`
- Create: `crates/hannsdb-py/tests/test_query_order_surface.py`

- [ ] **Step 1: Add a minimal public order-by type**

```python
@dataclass
class QueryOrderBy:
    field_name: str
    descending: bool = False
```

Wire it into:

- `QueryContext(order_by=...)`
- `Collection.query(..., order_by=...)`

- [ ] **Step 2: Encode `order_by` in the PyO3 bridge**

Map the Python object into the already-existing core `OrderBy` struct and keep core validation authoritative.

- [ ] **Step 3: Cover the minimum supported surface**

Add tests for:

- ascending scalar sort
- descending scalar sort
- legacy `Collection.query(..., order_by=...)` kwargs path
- vector-field rejection bubbling back from core as a stable public error

- [ ] **Step 4: Run the order-by suite**

Run:

```bash
cd crates/hannsdb-py && \
maturin develop --features python-binding,hanns-backend && \
python -m pytest tests/test_query_order_surface.py tests/test_collection_facade.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the surface alignment**

```bash
git add crates/hannsdb-py/src/query_bridge.rs crates/hannsdb-py/src/lib.rs crates/hannsdb-py/python/hannsdb/model/param/vector_query.py crates/hannsdb-py/python/hannsdb/model/collection.py crates/hannsdb-py/python/hannsdb/model/__init__.py crates/hannsdb-py/python/hannsdb/model/param/__init__.py crates/hannsdb-py/python/hannsdb/__init__.py crates/hannsdb-py/tests/test_query_order_surface.py crates/hannsdb-py/tests/test_collection_facade.py
git commit -m "feat: expose python order_by query surface"
```

### Chunk 3 verification gate

- [ ] Run:

```bash
cd crates/hannsdb-py && \
maturin develop --features python-binding,hanns-backend && \
python -m pytest tests/test_collection_facade.py tests/test_query_executor.py tests/test_query_order_surface.py -q
```

Expected: PASS.

---

## Chunk 4: End-to-end verification

### Task 5: Add a lightweight smoke script and final gates

**Files:**
- Create: `scripts/smoke_string_pk_order_by.py`

- [ ] **Step 1: Write a tiny smoke script**

The script should:

1. create a collection with one scalar sort field and one dense vector
2. insert documents with IDs like `"user-a"`, `"user-b"`, `"user-c"`
3. query by `query_by_id=["user-b"]`
4. run a second query with `order_by=QueryOrderBy(field_name="rank", descending=False)`
5. reopen the collection and repeat the string-PK query

- [ ] **Step 2: Run the smoke script**

Run:

```bash
cd crates/hannsdb-py && \
maturin develop --features python-binding,hanns-backend && \
python ../scripts/smoke_string_pk_order_by.py
```

Expected: exit 0, with no assertion failures.

- [ ] **Step 3: Run the full final verification set**

Run:

```bash
cargo test -p hannsdb-core --test zvec_parity_string_pk -- --nocapture
cd crates/hannsdb-py && maturin develop --features python-binding,hanns-backend
cd crates/hannsdb-py && python -m pytest tests/test_collection_facade.py tests/test_query_executor.py tests/test_query_order_surface.py -q
cd crates/hannsdb-py && python ../scripts/smoke_string_pk_order_by.py
```

Expected:

- core string-PK suite PASS
- Python facade/query/order-by suites PASS
- smoke script PASS

- [ ] **Step 4: Commit the final verification harness**

```bash
git add scripts/smoke_string_pk_order_by.py
git commit -m "test: add string pk order by smoke coverage"
```

---

## Execution notes

- Keep the first green milestone narrow: string PKs must work cleanly for Python-driven insert/fetch/delete/query paths, but crash-recovery parity is intentionally deferred.
- Do not widen this slice into daemon transport until the local Python path is stable.
- If `PrimaryKeyMode` inference becomes ambiguous during implementation, prefer an explicit collection-level rule over silent coercion.
- If the PyO3 bridge starts growing again, split a second focused file rather than re-expanding `lib.rs`.

Plan complete and saved to `docs/superpowers/plans/2026-04-12-hannsdb-zvec-string-pk-query-slice.md`. Ready to execute?
