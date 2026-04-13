# HannsDB Query Surface Parity Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the Python/public query surface with core-supported query combinations so built-in rerankers work through the facade with `query_by_id`, `group_by`, and `order_by`.

**Architecture:** Keep the existing `QueryContext` and `Collection.query(...)` public API. Change the Python executor from a blanket Python-only rerank path to a hybrid dispatcher: built-in rerankers use the full core-native query path, while custom Python rerankers keep the existing fan-out behavior. Let core remain authoritative for validation, ordering, grouping, and built-in rerank semantics.

**Tech Stack:** Pure Python facade/executor, PyO3-backed native collection bridge, existing HannsDB core query planner/executor, `pytest`, lightweight smoke script

---

## Chunk 1: Scope lock and failing parity tests

### Current state to preserve

- `QueryContext` already exposes `group_by`, `reranker`, `query_by_id`, and `order_by`.
- `Collection.query(...)` already accepts legacy kwargs for the same concepts.
- `py_query_context_to_core(...)` already maps these fields into core.
- The current blocker is mainly `crates/hannsdb-py/python/hannsdb/executor/query_executor.py`, which still rejects `query_by_id + reranker` and `group_by + reranker`.

### File map

**Executor and facade**
- Modify: `crates/hannsdb-py/python/hannsdb/executor/query_executor.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/collection.py` only if kwargs plumbing needs cleanup

**Tests**
- Modify: `crates/hannsdb-py/tests/test_query_executor.py`
- Modify: `crates/hannsdb-py/tests/test_collection_facade.py`
- Modify: `crates/hannsdb-py/tests/test_query_order_surface.py` if combined order/rerank coverage belongs there
- Modify: `scripts/smoke_string_pk_order_by.py` or add a sibling smoke script if needed

### Task 1: Convert current facade blockers into red parity tests

**Files:**
- Modify: `crates/hannsdb-py/tests/test_query_executor.py`
- Modify: `crates/hannsdb-py/tests/test_collection_facade.py`

- [ ] **Step 1: Replace the built-in reranker rejection tests with positive parity tests**

Target the tests that currently assert:

- `query_by_id is not supported by the Python facade yet`
- `group_by is not supported by the Python facade yet`

Turn them into positive tests for built-in rerankers.

- [ ] **Step 2: Add executor-level red tests for built-in reranker combinations**

Add tests for:

- built-in `RrfReRanker` + `query_by_id`
- built-in `RrfReRanker` + `group_by`
- built-in `RrfReRanker` + `order_by`

- [ ] **Step 3: Add collection-facade red tests for legacy kwargs parity**

Add kwargs-path equivalents for:

- `query_by_id + reranker`
- `group_by + reranker`
- `order_by + reranker`

- [ ] **Step 4: Run the red suite**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
python -m pytest tests/test_query_executor.py -k 'reranker and (query_by_id or group_by or order)' -q && \
python -m pytest tests/test_collection_facade.py -k 'reranker and (query_by_id or group_by or order)' -q
```

Expected: FAIL because the executor still blocks built-in reranker combinations.

---

## Chunk 2: Hybrid executor dispatch

### Task 2: Route built-in rerankers through core-native query execution

**Files:**
- Modify: `crates/hannsdb-py/python/hannsdb/executor/query_executor.py`

- [ ] **Step 1: Identify the built-in reranker classes explicitly**

Import and use:

- `RrfReRanker`
- `WeightedReRanker`

Do not infer by attribute shape.

- [ ] **Step 2: Add a dedicated built-in reranker dispatch branch**

Before the Python fan-out path, add:

- if `context.reranker` is built-in, call `collection.query_context(context)` directly
- return the result untouched

This lets core handle:

- recall-source combination
- reranking
- ordering
- grouping
- validation

- [ ] **Step 3: Remove obsolete facade-level blockers for built-in rerankers**

Delete or narrow the current guards that reject:

- `query_by_id`
- `group_by`

They should no longer fire for the built-in reranker path.

- [ ] **Step 4: Preserve the existing custom reranker fan-out path**

Keep:

- custom Python rerankers
- duplicate field-label stability
- concurrency behavior
- current reranker result merging contract

Do not regress existing custom reranker tests while fixing built-in parity.

- [ ] **Step 5: Run the focused suite**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
python -m pytest tests/test_query_executor.py -k 'supports_builtin_rrf_reranker or query_by_id_with_reranker or group_by_with_reranker or order' -q
```

Expected: PASS.

---

## Chunk 3: Legacy kwargs and public contract cleanup

### Task 3: Keep `Collection.query(...)` kwargs behavior aligned with `QueryContext`

**Files:**
- Modify: `crates/hannsdb-py/python/hannsdb/model/collection.py` only if dispatch/plumbing differs
- Modify: `crates/hannsdb-py/tests/test_collection_facade.py`

- [ ] **Step 1: Check kwargs-path construction**

Verify `_build_query_context(...)` forwards:

- `query_by_id`
- `group_by`
- `reranker`
- `order_by`

Only patch this file if the plumbing is still asymmetric.

- [ ] **Step 2: Add positive kwargs-path parity coverage**

Make sure `Collection.query(...kwargs...)` supports:

- built-in reranker + `query_by_id`
- built-in reranker + `group_by`
- built-in reranker + `order_by`

- [ ] **Step 3: Remove or rewrite stale facade-error assertions**

Any test that still expects facade-only `NotImplementedError` for a built-in reranker shape that core supports should be rewritten to the positive contract.

- [ ] **Step 4: Run the facade suite**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
python -m pytest tests/test_collection_facade.py -k 'builtin_rrf_reranker or weighted_reranker or query_by_id_with_reranker or group_by_with_reranker or order' -q
```

Expected: PASS.

---

## Chunk 4: Final verification and smoke coverage

### Task 4: Add or extend smoke coverage for combined query shapes

**Files:**
- Modify: `scripts/smoke_string_pk_order_by.py` or create a sibling script if the existing smoke becomes too overloaded

- [ ] **Step 1: Add one combined public query flow**

The smoke flow should exercise at least one shape that used to be blocked by the Python executor, for example:

- string PK + built-in reranker + `query_by_id`
or
- built-in reranker + `group_by`

- [ ] **Step 2: Run the smoke flow**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
python ../../scripts/smoke_string_pk_order_by.py
```

Expected: exit 0.

- [ ] **Step 3: Run the final verification set**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
python -m pytest tests/test_query_executor.py -k 'reranker or order_by_object or alphanumeric_query_by_id or multi_query_and_query_by_id' -q && \
python -m pytest tests/test_collection_facade.py -k 'reranker or routes_query_by_id_and_output_fields or alphanumeric_query_by_id' -q && \
python -m pytest tests/test_query_order_surface.py -q && \
python ../../scripts/smoke_string_pk_order_by.py
```

Expected:

- built-in reranker combination tests PASS
- existing simple reranker tests stay green
- existing non-reranker query-by-id/order-by tests stay green
- smoke PASS

---

## Execution notes

- Do not expand this plan into schema-mutation or type/index-param parity.
- Keep core unchanged unless a genuine bridge bug is discovered; this project should stay Python-executor-first.
- Prefer deleting obsolete executor restrictions over layering new flags on top of them.
- If a custom Python reranker combination is still impossible to emulate faithfully, keep that limitation explicit rather than pretending it is parity-complete.

Plan complete and saved to `docs/superpowers/plans/2026-04-12-hannsdb-query-surface-parity.md`. Ready to execute?
