# HannsDB Schema Mutation Surface Parity Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align HannsDB's Python/public schema-mutation surface with zvec-style `FieldSchema + expression + option` contracts while preserving today's real scalar DDL behavior and rejecting unsupported richer semantics explicitly.

**Architecture:** Keep core DDL semantics unchanged for this slice: scalar `add_column`, `drop_column`, and rename-only `alter_column` remain the only real execution paths. Expand the Python facade and PyO3 bridge to accept richer schema-mutation signatures, normalize legacy simplified calls for compatibility, expose `AddColumnOption` and `AlterColumnOption` as public parameter objects, and make unsupported semantics such as non-empty expressions, vector-column DDL, and `field_schema`-driven migrations fail clearly.

**Tech Stack:** Pure Python facade, PyO3-backed native binding, existing HannsDB core scalar DDL APIs, `pytest`, lightweight smoke script

---

## Chunk 1: Scope lock and red contract tests

### Current state to preserve

- Core already supports scalar `add_column`, `drop_column`, and rename-only `alter_column`.
- Core DDL tests already exist in `crates/hannsdb-core/tests/collection_api.rs`; this slice should not broaden into core migration work.
- The Python facade currently blocks all three column-mutation methods before delegating to core.
- The native binding in `crates/hannsdb-py/src/lib.rs` only supports simplified scalar signatures:
  - `add_column(field_name, data_type, nullable, array)`
  - `drop_column(field_name)`
  - `alter_column(field_name, new_name)`

### File map

**Python/public surface**
- Create: `crates/hannsdb-py/python/hannsdb/model/param/add_column_option.py`
- Create: `crates/hannsdb-py/python/hannsdb/model/param/alter_column_option.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/collection.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/__init__.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/__init__.py`
- Modify: `crates/hannsdb-py/python/hannsdb/__init__.py`

**PyO3 binding**
- Modify: `crates/hannsdb-py/src/lib.rs`

**Tests and verification**
- Create: `crates/hannsdb-py/tests/test_schema_mutation_surface.py`
- Modify: `crates/hannsdb-py/tests/test_collection_facade.py`
- Create: `scripts/smoke_schema_mutation_surface.py`

### Task 1: Replace the current facade blockers with red parity tests

**Files:**
- Create: `crates/hannsdb-py/tests/test_schema_mutation_surface.py`
- Modify: `crates/hannsdb-py/tests/test_collection_facade.py`

- [ ] **Step 1: Add red tests for new public imports and constructors**

Add tests that assert these objects are importable and constructible:

```python
def test_add_column_option_is_public():
    opt = hannsdb.AddColumnOption(concurrency=2)
    assert opt.concurrency == 2

def test_alter_column_option_is_public():
    opt = hannsdb.AlterColumnOption(concurrency=3)
    assert opt.concurrency == 3
```

- [ ] **Step 2: Add red tests for the canonical schema-mutation contract**

Add tests for:

- `collection.add_column(FieldSchema(...))`
- `collection.add_column(FieldSchema(...), expression="", option=AddColumnOption(...))`
- `collection.alter_column("old", new_name="new", option=AlterColumnOption(...))`

These should initially fail because the facade still raises `NotImplementedError`.

- [ ] **Step 3: Add red tests for explicit unsupported semantics**

Add tests that expect clear failures for:

- non-empty `expression`
- vector-style add-column input
- `alter_column(..., field_schema=FieldSchema(...))` migration requests

- [ ] **Step 4: Rewrite the current blanket blocker test in `test_collection_facade.py`**

Replace:

- `test_collection_column_mutation_surfaces_raise_before_core_delegation`

with contract-focused tests that distinguish:

- supported scalar delegation
- explicit unsupported-shape rejection

- [ ] **Step 5: Run the red suite**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
python -m pytest tests/test_schema_mutation_surface.py tests/test_collection_facade.py -k 'column or mutation or schema_mutation' -q
```

Expected: FAIL because the public option types and richer DDL signatures do not exist yet.

---

## Chunk 2: Public option types and export plumbing

### Task 2: Add public `AddColumnOption` and `AlterColumnOption` wrappers

**Files:**
- Create: `crates/hannsdb-py/python/hannsdb/model/param/add_column_option.py`
- Create: `crates/hannsdb-py/python/hannsdb/model/param/alter_column_option.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/__init__.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/__init__.py`
- Modify: `crates/hannsdb-py/python/hannsdb/__init__.py`
- Modify: `crates/hannsdb-py/tests/test_schema_mutation_surface.py`

- [ ] **Step 1: Add minimal pure-Python wrapper types**

Follow the `CollectionOption` / `OptimizeOption` pattern:

```python
@dataclass(frozen=True)
class AddColumnOption:
    concurrency: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.concurrency, int):
            raise TypeError("concurrency must be an int")
        if self.concurrency < 0:
            raise ValueError("concurrency must be >= 0")

    def _get_native(self):
        return _native_module.AddColumnOption(self.concurrency)
```

Mirror the same shape for `AlterColumnOption`.

- [ ] **Step 2: Export the new option types consistently**

Update:

- `hannsdb.model.param`
- `hannsdb.model`
- top-level `hannsdb`

so callers can import the new types from the same places they import other public params.

- [ ] **Step 3: Run the option-surface tests**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
python -m pytest tests/test_schema_mutation_surface.py -k 'option or public' -q
```

Expected: still FAIL until the native binding exports matching `_native` option classes.

---

## Chunk 3: PyO3 option classes and richer native DDL signatures

### Task 3: Extend the native Python binding to accept the richer contract shape

**Files:**
- Modify: `crates/hannsdb-py/src/lib.rs`
- Modify: `crates/hannsdb-py/tests/test_schema_mutation_surface.py`

- [ ] **Step 1: Add native option pyclasses**

In `lib.rs`, add:

- `PyAddColumnOption`
- `PyAlterColumnOption`

with a minimal stable surface:

```rust
#[pyclass(name = "AddColumnOption", module = "hannsdb")]
struct PyAddColumnOption { inner: AddColumnOption }

#[pymethods]
impl PyAddColumnOption {
    #[new]
    #[pyo3(signature = (concurrency=0))]
    fn new(concurrency: usize) -> Self { ... }

    #[getter]
    fn concurrency(&self) -> usize { ... }
}
```

Mirror the same for alter-column.

- [ ] **Step 2: Export the native option classes**

Update the module initialization section so `_native_module.AddColumnOption` and `_native_module.AlterColumnOption` exist before the pure-Python wrappers call `_get_native()`.

- [ ] **Step 3: Replace simplified DDL-only signatures with richer normalization-aware signatures**

Change the native collection methods to accept the richer contract shape:

- `add_column(field_schema, expression="", option=None)`
- `alter_column(field_name, new_name=None, field_schema=None, option=None)`

The binding should:

- normalize `FieldSchema` input
- reject `vector_fp32`
- reject non-empty `expression`
- reject `field_schema` migration requests
- ignore or validate option objects without inventing unsupported behavior

- [ ] **Step 4: Keep direct simplified compatibility where practical**

If direct native compatibility matters, preserve normalization for old argument forms rather than forcing an immediate breaking change. If that is too awkward in PyO3, keep compatibility in the pure-Python facade and make the native signature canonical.

- [ ] **Step 5: Build the extension and run focused tests**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
uvx maturin develop --uv --features python-binding,hanns-backend && \
python -m pytest tests/test_schema_mutation_surface.py -k 'option or add_column or alter_column' -q
```

Expected: PASS for public option construction and direct supported/rejected native-bridge semantics.

---

## Chunk 4: Python facade normalization and legacy compatibility

### Task 4: Make `Collection` use the richer schema-mutation contract

**Files:**
- Modify: `crates/hannsdb-py/python/hannsdb/model/collection.py`
- Modify: `crates/hannsdb-py/tests/test_collection_facade.py`
- Modify: `crates/hannsdb-py/tests/test_schema_mutation_surface.py`

- [ ] **Step 1: Normalize `add_column` inputs in the facade**

Replace the current simplified method:

```python
def add_column(self, field_name, data_type="string", nullable=False, array=False):
    return self._core.add_column(field_name, data_type, nullable, array)
```

with normalization that supports both:

- canonical form: `FieldSchema`, `expression`, `option`
- legacy simplified form: `field_name`, `data_type`, `nullable`, `array`

Use `_coerce_field_schema(...)` when possible so schema objects and schema-like inputs normalize consistently.

- [ ] **Step 2: Normalize `alter_column` inputs in the facade**

Support:

- canonical rename call: `alter_column("old", new_name="new", option=...)`
- canonical richer call with `field_schema=...` that fails explicitly
- legacy simplified call: `alter_column("old", "new")`

- [ ] **Step 3: Keep `drop_column` as a thin pass-through**

Do not add fake option or schema behavior to `drop_column`.

- [ ] **Step 4: Refresh schema metadata after successful mutation**

Ensure the facade continues to reflect updated schema state after:

- add-column success
- rename success

If the current native collection object already refreshes internally, verify it. If not, update the facade to rebuild or refresh schema after mutation.

- [ ] **Step 5: Run the facade suite**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
python -m pytest tests/test_collection_facade.py -k 'column or mutation' -q && \
python -m pytest tests/test_schema_mutation_surface.py -q
```

Expected: PASS.

---

## Chunk 5: End-to-end smoke and verification

### Task 5: Add a focused smoke flow for the supported schema-mutation path

**Files:**
- Create: `scripts/smoke_schema_mutation_surface.py`

- [ ] **Step 1: Add a smoke script that exercises the supported canonical contract**

The script should:

1. create a small collection with one scalar field and one vector field
2. call `add_column(FieldSchema(...), expression="", option=AddColumnOption(...))`
3. insert or query enough data to confirm the new field exists and defaults to `None` for old rows
4. call `alter_column("old_name", new_name="new_name", option=AlterColumnOption(...))`
5. reopen the collection and verify the schema still reflects the rename

- [ ] **Step 2: Add one explicit unsupported-shape check to the smoke script**

Assert that one of these raises the expected error:

- `expression="score * 2"`
- `field_schema=FieldSchema(...)` on `alter_column`

This keeps the public contract honest.

- [ ] **Step 3: Run the smoke flow**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
python ../../scripts/smoke_schema_mutation_surface.py
```

Expected: exit 0.

- [ ] **Step 4: Run the final verification set**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
uvx maturin develop --uv --features python-binding,hanns-backend && \
python -m pytest tests/test_schema_mutation_surface.py -q && \
python -m pytest tests/test_collection_facade.py -k 'column or mutation' -q && \
python ../../scripts/smoke_schema_mutation_surface.py
```

Expected:

- public option types PASS
- supported scalar schema-mutation contract PASS
- legacy simplified compatibility PASS
- unsupported richer semantics fail explicitly and are asserted in tests/smoke

---

## Execution notes

- Keep this plan out of core migration work. If implementation starts requiring true expression backfill or schema migration in core, stop and split a new spec.
- Prefer explicit `NotImplementedError` or `ValueError` with stable wording over silently ignoring unsupported fields like `expression` or `field_schema`.
- Do not regress the existing simplified call forms unless the repository already intends a breaking API change.
- If direct `_native` compatibility and pure-Python facade compatibility conflict, prioritize the documented public `hannsdb.Collection` contract and keep the break localized and explicit.

Plan complete and saved to `docs/superpowers/plans/2026-04-12-hannsdb-schema-mutation-surface-parity.md`. Ready to execute?
