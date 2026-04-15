# HannsDB Schema Mutation Constant-Expression Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an honest schema-mutation public contract that supports richer `add_column(...)` / `alter_column(...)` shapes and one real execution subset: constant-expression backfill for `add_column(...)`.

**Architecture:** Keep this slice bounded to Python/public surface, PyO3 bridge, and the smallest core/WAL changes needed to backfill a new scalar column from a constant literal. The facade and bridge should parse and validate the narrow literal grammar; core should receive a normalized constant backfill instruction rather than a general expression language.

**Tech Stack:** Pure Python facade, PyO3 binding, Rust core DDL/WAL/recovery path, `pytest`, `cargo test`, `cargo check`

---

## File Structure

### Public/Python surface
- Modify: `crates/hannsdb-py/python/hannsdb/model/collection.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/add_column_option.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/alter_column_option.py`
- Modify: `crates/hannsdb-py/python/hannsdb/__init__.py` only if exports need adjustment
- Modify: `crates/hannsdb-py/python/hannsdb/model/__init__.py` only if exports need adjustment
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/__init__.py` only if exports need adjustment

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
- Optional create: `scripts/smoke_schema_mutation_constant_expression.py`

---

## Chunk 1: Lock the contract with failing tests

### Task 1: Public-shape and literal-grammar red tests

**Files:**
- Modify: `crates/hannsdb-py/tests/test_schema_mutation_surface.py`
- Modify: `crates/hannsdb-py/tests/test_collection_facade.py`

- [ ] **Step 1: Add failing tests for richer `add_column(...)` shape**

Add tests for:
- `collection.add_column(FieldSchema(...), expression='', option=AddColumnOption(...))`
- `collection.add_column(FieldSchema(...), expression='"hello"', option=AddColumnOption(...))`
- `collection.add_column(FieldSchema(...), expression='""', option=AddColumnOption(...))`

- [ ] **Step 2: Add failing tests for the supported constant subset**

Cover at least:
- string constant to string column
- integer constant to int64/int32 column
- float constant to float/float64 column
- `true` / `false` to bool column
- `null` to nullable column

- [ ] **Step 3: Add failing tests for explicit unsupported shapes**

Add negative tests for:
- `expression='name'`
- `expression='a + b'`
- `expression='foo()'`
- `expression='+1'`
- `expression='1e6'`
- `expression='"unterminated'`
- `expression='null'` on non-nullable column

Expected: explicit `NotImplementedError` or `ValueError` depending on the case.

- [ ] **Step 4: Add failing tests for `alter_column(...)` richer shape boundary**

Add tests for:
- rename-only path still works
- `field_schema=FieldSchema(...)` migration request still fails explicitly
- `new_name=None` still fails where rename is required

- [ ] **Step 5: Run the Python red suite**

Run:
```bash
source .venv-hannsdb/bin/activate && python -m pytest   crates/hannsdb-py/tests/test_schema_mutation_surface.py   crates/hannsdb-py/tests/test_collection_facade.py -q
```

Expected: FAIL on the new constant-expression tests.

### Task 2: Core/WAL red tests for backfill and replay

**Files:**
- Modify: `crates/hannsdb-core/tests/collection_api.rs`
- Modify: `crates/hannsdb-core/tests/wal_recovery.rs`

- [ ] **Step 1: Add failing core test for constant backfill on existing rows**

Add a test proving:
- create collection
- insert existing rows
- add scalar column with constant expression
- fetch old rows
- new field exists with constant value

- [ ] **Step 2: Add failing core test for nullable `null` constant**

Add a test proving:
- nullable column accepts `null`
- fetched rows contain the field with null semantics expected by current document model

- [ ] **Step 3: Add failing WAL replay test**

Add a WAL/recovery test proving:
- `AddColumn` with constant-expression backfill survives reopen/replay
- old rows still have the backfilled value after reopen

- [ ] **Step 4: Run the core red tests**

Run:
```bash
cargo test -p hannsdb-core --test collection_api --test wal_recovery -- --nocapture
```

Expected: FAIL on the new add-column constant-expression behavior.

---

## Chunk 2: Public surface and bridge normalization

### Task 3: Normalize richer schema-mutation inputs in the Python facade

**Files:**
- Modify: `crates/hannsdb-py/python/hannsdb/model/collection.py`
- Modify only if needed: param export files listed above

- [ ] **Step 1: Introduce one explicit constant-expression normalization helper**

Add a helper in `collection.py` that:
- distinguishes legacy `expression=""` from explicit empty-string literal `expression='""'`
- trims surrounding whitespace before classification
- recognizes only the approved grammar
- returns a small normalized form (type tag + value), or raises explicit error

- [ ] **Step 2: Keep unsupported expression shapes explicit**

The helper must reject:
- field references
- computed expressions
- scientific notation
- leading `+`
- unsupported escapes/embedded quotes (unless you decide to fully support them identically in bridge + tests)

- [ ] **Step 3: Preserve existing rename-only `alter_column(...)` path**

Keep current successful rename behavior; only broaden the contract shape and keep migration-style requests explicitly rejected.

- [ ] **Step 4: Re-run the focused Python tests**

Run:
```bash
source .venv-hannsdb/bin/activate && python -m pytest   crates/hannsdb-py/tests/test_schema_mutation_surface.py   crates/hannsdb-py/tests/test_collection_facade.py -q
```

Expected: still FAIL until the bridge/core path can execute the normalized constant backfill.

### Task 4: Extend the PyO3 bridge to carry the constant subset honestly

**Files:**
- Modify: `crates/hannsdb-py/src/lib.rs`

- [ ] **Step 1: Extend the native `add_column(...)` signature**

Evolve the current simplified binding:
- from `(field_name, data_type, nullable, array)`
- to a richer shape that can receive `FieldSchema`, `expression`, and `option`

Keep compatibility in Python if possible; otherwise let the Python facade own compatibility and make the native signature canonical.

- [ ] **Step 2: Parse only the normalized constant subset**

The bridge should accept the narrow grammar and translate it into a concrete constant-value/backfill request.
It must **not** become a general expression evaluator.

- [ ] **Step 3: Keep explicit error mapping**

Use:
- `PyValueError` for type/nullability/grammar mismatch
- `PyValueError` or `PyNotImplementedError`-style mapping for unsupported expression forms (follow current project conventions)

- [ ] **Step 4: Re-run the Python suite**

Run:
```bash
source .venv-hannsdb/bin/activate && python -m pytest   crates/hannsdb-py/tests/test_schema_mutation_surface.py   crates/hannsdb-py/tests/test_collection_facade.py -q
```

Expected: still FAIL until core add-column execution understands the backfill request.

---

## Chunk 3: Core add-column constant backfill + replay support

### Task 5: Add the minimal core execution path for constant backfill

**Files:**
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify only if needed: `crates/hannsdb-core/src/document.rs`

- [ ] **Step 1: Introduce a minimal backfill instruction type**

Add a tiny internal representation for add-column backfill, e.g. “no backfill” vs “constant scalar value”.
Do **not** introduce a general expression AST.

- [ ] **Step 2: Extend `add_column` internal flow to apply constant backfill to existing rows**

Current behavior leaves old rows without the new field. Change it only for the constant-backfill case:
- existing rows get the constant scalar value
- no-expression legacy path keeps old behavior

- [ ] **Step 3: Validate destination-type compatibility centrally**

The core path should reject impossible writes even if facade/bridge validation missed something.

- [ ] **Step 4: Run focused core tests**

Run:
```bash
cargo test -p hannsdb-core --test collection_api -- --nocapture
```

Expected: PASS on the new backfill tests.

### Task 6: Make WAL/recovery preserve the new semantics

**Files:**
- Modify: `crates/hannsdb-core/src/wal.rs`
- Modify: `crates/hannsdb-core/src/storage/recovery.rs`
- Modify: `crates/hannsdb-core/src/db.rs`
- Modify: `crates/hannsdb-core/tests/wal_recovery.rs`

- [ ] **Step 1: Extend `WalRecord::AddColumn` to carry the minimal backfill instruction**

Keep it narrow: legacy none + constant scalar only.

- [ ] **Step 2: Replay through the same internal add-column path**

Ensure replay uses the same add-column internal helper so backfill semantics do not fork.

- [ ] **Step 3: Run WAL/recovery verification**

Run:
```bash
cargo test -p hannsdb-core --test wal_recovery -- --nocapture
```

Expected: PASS.

---

## Chunk 4: Final verification and smoke

### Task 7: Run full required verification for this lane

**Files:**
- No new file scope beyond the lane files above
- Optional create: `scripts/smoke_schema_mutation_constant_expression.py`

- [ ] **Step 1: Add or update a lightweight smoke flow if useful**

The smoke should prove:
- create collection
- insert rows
- `add_column(..., expression='"hello"')`
- fetch/query rows
- verify field values

- [ ] **Step 2: Run the Python verification set**

Run:
```bash
source .venv-hannsdb/bin/activate && python -m pytest   crates/hannsdb-py/tests/test_schema_mutation_surface.py   crates/hannsdb-py/tests/test_collection_facade.py -q
```

Expected: PASS.

- [ ] **Step 3: Run the core verification set**

Run:
```bash
cargo test -p hannsdb-core --test collection_api --test wal_recovery -- --nocapture
```

Expected: PASS.

- [ ] **Step 4: Run build/type verification**

Run:
```bash
cargo check -p hannsdb-py --features python-binding,hanns-backend
cargo check -p hannsdb-core --features hanns-backend
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add   crates/hannsdb-py/python/hannsdb/model/collection.py   crates/hannsdb-py/src/lib.rs   crates/hannsdb-core/src/db.rs   crates/hannsdb-core/src/wal.rs   crates/hannsdb-core/src/storage/recovery.rs   crates/hannsdb-py/tests/test_schema_mutation_surface.py   crates/hannsdb-py/tests/test_collection_facade.py   crates/hannsdb-core/tests/collection_api.rs   crates/hannsdb-core/tests/wal_recovery.rs

git commit
```

Use a Lore-protocol message describing why constant-expression backfill was chosen as the smallest honest parity step.

---

## Review Checklist For Plan Execution
- Keep `expression=""` as the legacy no-expression form.
- Use explicit quoted strings like `expression='""'` and `expression='"hello"'` for string constants.
- Reject scientific notation, leading `+`, field references, and computed expressions.
- Do not widen into field-schema migration or vector-column mutation.
- Do not run benchmark locally or remotely for this lane.

Plan complete and saved to `docs/superpowers/plans/2026-04-15-hannsdb-schema-mutation-constant-expression.md`. Ready to execute?
