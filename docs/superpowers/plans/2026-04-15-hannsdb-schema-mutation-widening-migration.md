# HannsDB Schema Mutation Widening Migration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `alter_column(..., field_schema=...)` into a real, honest widening-only migration path for `int32 -> int64`, `uint32 -> uint64`, and `float -> float64`, while preserving explicit failure for every migration shape outside that subset.

**Architecture:** Keep this slice bounded to Python/public surface, the PyO3 bridge, and a minimal core/WAL/recovery migration instruction. The facade and bridge should classify rename-only vs widening-migration requests; core should execute the tiny supported subset through one shared helper and replay the same instruction safely from WAL.

**Tech Stack:** Pure Python facade, PyO3 binding, Rust core DDL/WAL/recovery path, `pytest`, `cargo test`, `cargo check`

---

## File Structure

### Public/Python surface
- Modify: `crates/hannsdb-py/python/hannsdb/model/collection.py`
- Modify only if needed: `crates/hannsdb-py/python/hannsdb/__init__.py`
- Modify only if needed: `crates/hannsdb-py/python/hannsdb/model/__init__.py`
- Modify only if needed: `crates/hannsdb-py/python/hannsdb/model/param/__init__.py`

### PyO3/native bridge
- Modify: `crates/hannsdb-py/src/lib.rs`

### Core / WAL / recovery
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-core/src/wal.rs`
- Modify: `crates/hannsdb-core/src/storage/recovery.rs`
- Modify only if needed: `crates/hannsdb-core/src/document.rs`

### Tests
- Modify: `crates/hannsdb-py/tests/test_schema_mutation_surface.py`
- Modify: `crates/hannsdb-py/tests/test_collection_facade.py`
- Modify: `crates/hannsdb-core/tests/collection_api.rs`
- Modify: `crates/hannsdb-core/tests/wal_recovery.rs`

---

## Chunk 1: Lock the widening-only contract in tests

### Task 1: Python/facade red tests for widening-only migration

**Files:**
- Modify: `crates/hannsdb-py/tests/test_schema_mutation_surface.py`
- Modify: `crates/hannsdb-py/tests/test_collection_facade.py`

- [ ] **Step 1: Add positive tests for the supported widening subset**

Add tests for real collection behavior covering:
- `int32 -> int64`
- `uint32 -> uint64`
- `float -> float64`

Each test should:
1. create a collection with the source field type already in schema
2. insert at least one row with that field populated
3. call `alter_column(..., field_schema=FieldSchema(...target_type...))`
4. fetch the row after migration
5. assert the field value is preserved semantically and now materializes under the widened type contract

- [ ] **Step 2: Add negative tests for unsupported migration shapes**

Add facade-level tests for explicit failures on:
- `string -> int64`
- `int64 -> string`
- `float64 -> int64`
- any narrowing
- rename + type migration in one call
- nullable change with same type
- array change with same type
- vector-field migration

Expected: explicit `NotImplementedError` or `ValueError` depending on the case.

- [ ] **Step 3: Add negative tests for indexed-column migration**

Add a test that:
1. creates a scalar index descriptor on the source field
2. attempts a supported widening migration on that indexed field
3. asserts explicit rejection

This locks the spec’s “indexed-column behavior is rejected in this lane” rule.

- [ ] **Step 4: Keep rename-only behavior green**

Do not remove the current rename-only positive tests. If helpful, add one explicit test showing:
- rename-only still succeeds
- migration-only succeeds
- rename + migration together is rejected

- [ ] **Step 5: Run the Python red suite**

Run:
```bash
source .venv-hannsdb/bin/activate && python -m pytest   crates/hannsdb-py/tests/test_schema_mutation_surface.py   crates/hannsdb-py/tests/test_collection_facade.py -q
```

Expected: FAIL on the new widening-migration tests.

### Task 2: Core/WAL red tests for widening migration

**Files:**
- Modify: `crates/hannsdb-core/tests/collection_api.rs`
- Modify: `crates/hannsdb-core/tests/wal_recovery.rs`

- [ ] **Step 1: Add core tests for supported widening cases**

Add tests that prove existing rows are really migrated for:
- `int32 -> int64`
- `uint32 -> uint64`
- `float -> float64`

Each test should verify both:
- schema metadata changed
- fetched existing row values changed consistently

- [ ] **Step 2: Add core tests for rejected unsupported cases**

Add tests for:
- string/numeric cross-class migration
- narrowing migration
- indexed-column migration rejection
- rename + migration in one step rejection

- [ ] **Step 3: Add WAL/recovery red test**

Add a test proving:
- widening migration is appended to WAL before durable mutation semantics depend on it
- reopen/replay preserves the migrated field type/value

- [ ] **Step 4: Run the core red suite**

Run:
```bash
cargo test -p hannsdb-core --test collection_api --test wal_recovery -- --nocapture
```

Expected: FAIL on the new widening-migration behavior.

---

## Chunk 2: Facade and bridge classification

### Task 3: Teach the Python facade to distinguish rename-only vs widening migration

**Files:**
- Modify: `crates/hannsdb-py/python/hannsdb/model/collection.py`

- [ ] **Step 1: Extend `_normalize_alter_column_input(...)`**

Update it so it can classify these cases:
- rename-only (current supported path)
- widening-migration request with same name + supported source/target pair
- unsupported migration request

The helper should return a normalized small instruction or equivalent tuple rather than only `old_name/new_name`.

- [ ] **Step 2: Enforce the lane’s explicit boundary in facade code**

Reject in facade before core call:
- `field_schema` without a supported widening pair
- rename + migration together
- nullable/array changes
- vector migration
- unsupported source/target types

- [ ] **Step 3: Keep the public error split honest**

Use:
- `NotImplementedError` for valid-but-unsupported migration shapes
- `ValueError` for malformed input / impossible request shape

- [ ] **Step 4: Re-run focused Python tests**

Run:
```bash
source .venv-hannsdb/bin/activate && python -m pytest   crates/hannsdb-py/tests/test_schema_mutation_surface.py   crates/hannsdb-py/tests/test_collection_facade.py -q
```

Expected: still FAIL until bridge/core execution lands.

### Task 4: Extend the PyO3 bridge for widening-only migration

**Files:**
- Modify: `crates/hannsdb-py/src/lib.rs`

- [ ] **Step 1: Evolve the native `alter_column(...)` signature**

Move from the current rename-only native call toward a canonical richer signature that can carry:
- `old_name`
- `new_name`
- `field_schema`
- `option`

Keep compatibility in Python if practical; otherwise let the Python facade own compatibility and make the native signature canonical.

- [ ] **Step 2: Introduce a tiny native-side classification helper**

In the bridge, classify exactly these cases:
- rename-only
- widen-int32-to-int64
- widen-uint32-to-uint64
- widen-float-to-float64

Everything else must be rejected explicitly.

- [ ] **Step 3: Pass only the minimal migration instruction to core**

Do not pass a generalized migration schema object if a tiny internal enum/instruction is enough.

- [ ] **Step 4: Re-run Python tests**

Run:
```bash
source .venv-hannsdb/bin/activate && python -m pytest   crates/hannsdb-py/tests/test_schema_mutation_surface.py   crates/hannsdb-py/tests/test_collection_facade.py -q
```

Expected: still FAIL until core execution and WAL support land.

---

## Chunk 3: Core migration instruction + execution

### Task 5: Add a minimal `AlterColumnMigration` instruction in core

**Files:**
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-core/src/wal.rs`

- [ ] **Step 1: Introduce a tiny internal migration representation**

Keep it fixed to the lane’s subset, e.g. something equivalent to:
- rename-only
- widen-int32-to-int64
- widen-uint32-to-uint64
- widen-float-to-float64

Do **not** build a general migration engine.

- [ ] **Step 2: Add core validation for indexed-column rejection**

Before executing a widening migration, inspect scalar index descriptors for the field.
If an index exists on that field, reject the migration explicitly.

- [ ] **Step 3: Implement actual value migration for existing rows**

For supported widening migrations:
- load/iterate live rows
- widen the field values into the target type
- update stored rows/documents through one internal helper
- update schema metadata

The implementation should ensure fetched rows after migration reflect the widened type/value semantics.

- [ ] **Step 4: Preserve rename-only behavior**

Keep existing rename-only path working, but do not mix it with migration in this lane.

- [ ] **Step 5: Run focused core tests**

Run:
```bash
cargo test -p hannsdb-core --test collection_api -- --nocapture
```

Expected: PASS on the new widening tests.

### Task 6: Make WAL/recovery preserve migration semantics

**Files:**
- Modify: `crates/hannsdb-core/src/wal.rs`
- Modify: `crates/hannsdb-core/src/storage/recovery.rs`
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-core/tests/wal_recovery.rs`

- [ ] **Step 1: Extend `WalRecord::AlterColumn`**

Carry the minimal widening instruction rather than only old/new names when a migration is involved.

- [ ] **Step 2: Enforce the spec’s WAL-first ordering**

Document in code/comments and implement so that:
- migration WAL record is appended before durable data rewrite depends on it
- replay uses the same helper as live execution

- [ ] **Step 3: Replay through the same migration helper**

Do not fork live vs replay semantics.

- [ ] **Step 4: Run WAL/recovery verification**

Run:
```bash
cargo test -p hannsdb-core --test wal_recovery -- --nocapture
```

Expected: PASS.

---

## Chunk 4: Final verification and bounded cleanup

### Task 7: Run final verification for the lane

**Files:**
- No new file scope beyond the lane files above

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
git add   crates/hannsdb-py/python/hannsdb/model/collection.py   crates/hannsdb-py/src/lib.rs   crates/hannsdb-core/src/db.rs   crates/hannsdb-core/src/wal.rs   crates/hannsdb-core/src/storage/recovery.rs   crates/hannsdb-py/tests/test_schema_mutation_surface.py   crates/hannsdb-py/tests/test_collection_facade.py   crates/hannsdb-core/tests/collection_api.rs   crates/hannsdb-core/tests/wal_recovery.rs

git commit
```

Use a Lore-protocol commit message describing why widening-only migration was chosen as the smallest honest `field_schema` step.

---

## Review Checklist For Execution
- Keep the widening subset fixed to exactly three cases.
- Reject indexed-column migration explicitly.
- Reject rename + migration together.
- Do not widen into nullable/array/vector migration.
- Keep replay semantics on the same helper as live execution.
- Do not run benchmark locally or remotely for this lane.

Plan complete and saved to `docs/superpowers/plans/2026-04-15-hannsdb-schema-mutation-widening-migration.md`. Ready to execute?
