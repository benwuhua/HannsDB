# HannsDB Schema Mutation Rename+Widening Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing widening-only `field_schema` migration lane so `alter_column(...)` can perform **rename + widening-only migration** in one honest, replay-safe operation.

**Architecture:** Reuse the current widening-only migration path and expand it with explicit rename+widening migration instructions. Keep the public contract unchanged, keep rename-only and same-name widening green, reject indexed-column migration and all broader migration shapes, and require replay-safe/idempotent behavior after WAL append.

**Tech Stack:** Pure Python facade, PyO3 binding, Rust core DDL/WAL/recovery path, `pytest`, `cargo test`, `cargo check`

---

## File Structure

### Public/Python surface
- Modify: `crates/hannsdb-py/python/hannsdb/model/collection.py`
- Modify: `crates/hannsdb-py/tests/test_schema_mutation_surface.py`
- Modify: `crates/hannsdb-py/tests/test_collection_facade.py`

### PyO3/native bridge
- Modify: `crates/hannsdb-py/src/lib.rs`

### Core / WAL / recovery
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-core/src/wal.rs`
- Modify: `crates/hannsdb-core/src/storage/recovery.rs`
- Modify only if needed: `crates/hannsdb-core/src/forward_store/schema.rs`
- Modify only if needed: `crates/hannsdb-core/src/segment/arrow_io.rs`

### Tests
- Modify: `crates/hannsdb-core/tests/collection_api.rs`
- Modify: `crates/hannsdb-core/tests/wal_recovery.rs`

---

## Chunk 1: Lock the rename+widening contract in tests

### Task 1: Python/facade red tests

**Files:**
- Modify: `crates/hannsdb-py/tests/test_schema_mutation_surface.py`
- Modify: `crates/hannsdb-py/tests/test_collection_facade.py`

- [ ] **Step 1: Add positive tests for supported rename+widening cases**

Add real collection tests for:
- `score:int32 -> total_score:int64`
- `count:uint32 -> doc_count:uint64`
- `ratio:float -> ratio64:float64`

Each test should verify:
1. schema name changed
2. schema type changed
3. fetched row no longer exposes the old key
4. fetched row exposes the new key with the widened value

- [ ] **Step 2: Add negative tests for unsupported combined shapes**

Add tests for explicit rejection of:
- rename + non-widening type migration
- rename + narrowing
- rename + nullable change
- rename + array change
- rename + vector migration
- mismatch between `new_name` and `field_schema.name`

Expected: explicit `NotImplementedError` or `ValueError`.

- [ ] **Step 3: Add indexed-column rejection test for rename+widening**

Create a scalar index on the source field, then attempt rename+widening, and assert explicit rejection.

- [ ] **Step 4: Keep old paths green**

Do not remove current positive tests for:
- rename-only
- same-name widening-only

Add one test that proves they still work independently after this lane.

- [ ] **Step 5: Run the Python red suite**

Run:
```bash
source .venv-hannsdb/bin/activate && python -m pytest   crates/hannsdb-py/tests/test_schema_mutation_surface.py   crates/hannsdb-py/tests/test_collection_facade.py -q
```

Expected: FAIL on the new rename+widening tests.

### Task 2: Core/WAL red tests

**Files:**
- Modify: `crates/hannsdb-core/tests/collection_api.rs`
- Modify: `crates/hannsdb-core/tests/wal_recovery.rs`

- [ ] **Step 1: Add core positive tests for supported rename+widening**

Add tests that prove:
- schema field name changed
- schema field type changed
- old row key removed
- new row key present
- value widened correctly

Cover all three supported type pairs.

- [ ] **Step 2: Add core negative tests**

Add tests for:
- indexed-column rename+widening rejection
- unsupported combined migration shapes
- `new_name` / `field_schema.name` mismatch rejection

- [ ] **Step 3: Add WAL/recovery red test**

Add a reopen/replay test showing rename+widening survives reboot and converges to the final state.

- [ ] **Step 4: Run the core red suite**

Run:
```bash
cargo test -p hannsdb-core --test collection_api --test wal_recovery -- --nocapture
```

Expected: FAIL on the new rename+widening behavior.

---

## Chunk 2: Facade and bridge classification

### Task 3: Teach the Python facade to classify rename+widening requests

**Files:**
- Modify: `crates/hannsdb-py/python/hannsdb/model/collection.py`

- [ ] **Step 1: Extend `_normalize_alter_column_input(...)`**

Update the helper so it can classify:
- rename-only
- same-name widening-only (existing lane)
- rename + widening-only
- unsupported migration

For rename+widening, enforce:
- `field_schema.name` is canonical target name
- `new_name` may be omitted/empty
- if `new_name` is provided, it must equal `field_schema.name`

- [ ] **Step 2: Keep explicit rejection boundaries in facade code**

Facade must still reject before core call:
- indexed-column migration when that state is observable through the facade
- unsupported source/target pairs
- nullable/array/vector changes
- name mismatch between `new_name` and `field_schema.name`

- [ ] **Step 3: Re-run Python tests**

Run:
```bash
source .venv-hannsdb/bin/activate && python -m pytest   crates/hannsdb-py/tests/test_schema_mutation_surface.py   crates/hannsdb-py/tests/test_collection_facade.py -q
```

Expected: still FAIL until bridge/core support lands.

### Task 4: Extend the PyO3 bridge to emit rename+widening instructions

**Files:**
- Modify: `crates/hannsdb-py/src/lib.rs`

- [ ] **Step 1: Extend the migration classifier**

The bridge should now classify exactly these supported cases:
- rename-only
- same-name widening-only
- rename + widen-int32-to-int64
- rename + widen-uint32-to-uint64
- rename + widen-float-to-float64

Everything else must still reject explicitly.

- [ ] **Step 2: Keep `field_schema.name` as canonical rename target**

Do not invent a second rename source of truth.
If `new_name` is present and different, fail explicitly.

- [ ] **Step 3: Pass only the tiny migration instruction to core**

Do not generalize the bridge into a broad migration descriptor system.

- [ ] **Step 4: Re-run Python tests**

Run the same Python suite as above.

Expected: still FAIL until core execution/WAL support is added.

---

## Chunk 3: Core execution and WAL / replay support

### Task 5: Extend the tiny migration instruction set

**Files:**
- Modify: `crates/hannsdb-core/src/wal.rs`
- Modify: `crates/hannsdb-core/src/db.rs`

- [ ] **Step 1: Extend `AlterColumnMigration` with rename+widening variants**

Add explicit variants for the three rename+widening cases.
Do not replace the enum with a generic migration framework.

- [ ] **Step 2: Keep the existing same-name widening variants intact**

This lane must not regress the already-implemented widening-only behavior.

- [ ] **Step 3: Extend `WalRecord::AlterColumn` usage**

Ensure the WAL record can carry:
- old name
- final new name
- target field schema
- tiny migration instruction

without ambiguity.

### Task 6: Implement rename+widening execution in one shared helper

**Files:**
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify only if needed: `crates/hannsdb-core/src/forward_store/schema.rs`
- Modify only if needed: `crates/hannsdb-core/src/segment/arrow_io.rs`

- [ ] **Step 1: Extend the live migration helper**

The helper must, for supported rename+widening:
- move field values from `old_name` to `new_name`
- widen values to the target type
- ensure the old key no longer remains in migrated live rows

- [ ] **Step 2: Keep replay-safe / idempotent semantics**

If replay happens after partial persistence, the helper must converge to the same final result without duplicating old/new keys.

- [ ] **Step 3: Preserve rename-only and same-name widening behavior**

Do not fork separate special cases if the same helper can keep the semantics aligned.

- [ ] **Step 4: Enforce indexed-column rejection in core**

If the source field has a scalar index descriptor, fail explicitly before doing any rename+widening work.

- [ ] **Step 5: Run focused core tests**

Run:
```bash
cargo test -p hannsdb-core --test collection_api -- --nocapture
```

Expected: PASS on the new rename+widening tests.

### Task 7: Preserve replay semantics

**Files:**
- Modify: `crates/hannsdb-core/src/storage/recovery.rs`
- Modify: `crates/hannsdb-core/tests/wal_recovery.rs`

- [ ] **Step 1: Replay through the same internal helper**

Ensure rename+widening replay does not fork from live execution.

- [ ] **Step 2: Run WAL/recovery tests**

Run:
```bash
cargo test -p hannsdb-core --test wal_recovery -- --nocapture
```

Expected: PASS.

---

## Chunk 4: Final verification and bounded cleanup

### Task 8: Final verification

- [ ] **Step 1: Run Python verification**

```bash
source .venv-hannsdb/bin/activate && python -m pytest   crates/hannsdb-py/tests/test_schema_mutation_surface.py   crates/hannsdb-py/tests/test_collection_facade.py -q
```

Expected: PASS.

- [ ] **Step 2: Run core verification**

```bash
cargo test -p hannsdb-core --test collection_api --test wal_recovery -- --nocapture
```

Expected: PASS.

- [ ] **Step 3: Run build/type verification**

```bash
cargo check -p hannsdb-core --features hanns-backend
cargo check -p hannsdb-py --features python-binding,hanns-backend
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add   crates/hannsdb-py/python/hannsdb/model/collection.py   crates/hannsdb-py/src/lib.rs   crates/hannsdb-core/src/db.rs   crates/hannsdb-core/src/wal.rs   crates/hannsdb-core/src/storage/recovery.rs   crates/hannsdb-core/src/forward_store/schema.rs   crates/hannsdb-core/src/segment/arrow_io.rs   crates/hannsdb-py/tests/test_schema_mutation_surface.py   crates/hannsdb-py/tests/test_collection_facade.py   crates/hannsdb-core/tests/collection_api.rs   crates/hannsdb-core/tests/wal_recovery.rs

git commit
```

Use a Lore-protocol message explaining why rename+widening was the smallest honest extension of the existing widening-only lane.

---

## Review Checklist For Execution
- Keep rename target canonical: `field_schema.name`
- If `new_name` is provided, it must equal `field_schema.name`
- Preserve rename-only and same-name widening behavior
- Reject indexed-column migration explicitly
- Keep replay semantics on the same helper as live execution
- Do not broaden into general migration-engine work
- Do not run benchmark locally or remotely for this lane

Plan complete and saved to `docs/superpowers/plans/2026-04-15-hannsdb-schema-mutation-rename-widening.md`. Ready to execute?
