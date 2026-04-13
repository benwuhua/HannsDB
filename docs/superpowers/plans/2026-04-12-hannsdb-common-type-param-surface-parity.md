# HannsDB Common Type And Param Surface Parity Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand HannsDB's Python/public common type and parameter surface with additional bridgeable `DataType` values, `FlatIndexParam`, and `IVFQueryParam`, while keeping advanced param families explicitly out of scope.

**Architecture:** Keep this slice at the Python/public surface and binding-bridge layer. Add only the common contracts HannsDB can support honestly: extend `typing.DataType` to values already recognized by the Rust bridge, add pure-Python/common native wrappers for `FlatIndexParam` and `IVFQueryParam`, and update `VectorSchema` and `VectorQuery` normalization so these new objects flow through existing create/query paths without implying new ANN algorithms.

**Tech Stack:** Pure Python wrappers, existing `hannsdb.typing` enums, PyO3 binding in `crates/hannsdb-py/src/lib.rs`, existing HannsDB core-backed create/query paths, `pytest`, lightweight smoke script

---

## Chunk 1: Scope lock and red public-surface tests

### Current state to preserve

- `hannsdb.typing.DataType` currently exposes only `String`, `Int64`, `Float64`, `Bool`, and `VectorFp32`.
- `hannsdb.model.param.index_params` currently exposes only `HnswIndexParam`, `IVFIndexParam`, and `HnswQueryParam`.
- `VectorSchema` currently accepts `HnswIndexParam` / `IVFIndexParam`.
- `VectorQuery.param` is still effectively unchecked `Any`, but the native bridge only has a clean path for `HnswQueryParam`.

### Explicit non-goals for this slice

- Do not add `InvertIndexParam`.
- Do not add `HnswRabitqIndexParam` or `HnswRabitqQueryParam`.
- Do not add sparse/array/vector-int family `DataType` values.
- Do not implement new ANN algorithms or query semantics beyond honest contract mapping.

### File map

**Typing and param surface**
- Modify: `crates/hannsdb-py/python/hannsdb/typing/data_type.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/index_params.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/__init__.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/__init__.py`
- Modify: `crates/hannsdb-py/python/hannsdb/__init__.py`

**Schema/query normalization**
- Modify: `crates/hannsdb-py/python/hannsdb/model/schema/field_schema.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/vector_query.py`

**PyO3 binding**
- Modify: `crates/hannsdb-py/src/lib.rs`

**Tests and verification**
- Modify: `crates/hannsdb-py/tests/test_typing_surface.py`
- Create: `crates/hannsdb-py/tests/test_common_param_surface.py`
- Create: `scripts/smoke_common_type_param_surface.py`

### Task 1: Add red tests that lock the common-surface gap

**Files:**
- Modify: `crates/hannsdb-py/tests/test_typing_surface.py`
- Create: `crates/hannsdb-py/tests/test_common_param_surface.py`

- [ ] **Step 1: Add red tests for the new common `DataType` members**

Add tests for:

```python
def test_data_type_exposes_common_scalar_members():
    assert str(hannsdb.DataType.Int32) == "int32"
    assert str(hannsdb.DataType.UInt32) == "uint32"
    assert str(hannsdb.DataType.UInt64) == "uint64"
    assert str(hannsdb.DataType.Float) == "float"
```

- [ ] **Step 2: Add red tests for `FlatIndexParam`**

Add tests that assert:

- `hannsdb.FlatIndexParam` is importable from top-level and `model.param`
- it is a pure-Python wrapper
- it validates `metric_type`
- `VectorSchema(..., index_param=hannsdb.FlatIndexParam(...))` is accepted

- [ ] **Step 3: Add red tests for `IVFQueryParam`**

Add tests that assert:

- `hannsdb.IVFQueryParam` is importable from top-level and `model.param`
- it validates `nprobe`
- `VectorQuery(..., param=hannsdb.IVFQueryParam(...))` is accepted

- [ ] **Step 4: Add red tests for explicit non-goals**

Add tests that verify advanced types are still absent, for example:

- `not hasattr(hannsdb, "InvertIndexParam")`
- `not hasattr(hannsdb, "HnswRabitqIndexParam")`
- `not hasattr(hannsdb.DataType, "VectorFp16")`

- [ ] **Step 5: Run the red suite**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
python -m pytest tests/test_typing_surface.py tests/test_common_param_surface.py -q
```

Expected: FAIL because the new `DataType` members and common param wrappers do not exist yet.

---

## Chunk 2: Pure-Python common param wrappers and exports

### Task 2: Add `FlatIndexParam` and `IVFQueryParam` to the public Python surface

**Files:**
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/index_params.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/__init__.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/__init__.py`
- Modify: `crates/hannsdb-py/python/hannsdb/__init__.py`
- Modify: `crates/hannsdb-py/tests/test_common_param_surface.py`

- [ ] **Step 1: Add a `FlatIndexParam` dataclass**

Follow the existing wrapper style in `index_params.py`:

```python
@dataclass(frozen=True)
class FlatIndexParam:
    metric_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metric_type",
            _validate_metric_type("metric_type", self.metric_type),
        )

    def _get_native(self):
        return _native_module.FlatIndexParam(metric_type=self.metric_type)
```

Keep the contract narrow unless quantize support is already verified and intentional.

- [ ] **Step 2: Add an `IVFQueryParam` dataclass**

Use a minimal surface:

```python
@dataclass(frozen=True)
class IVFQueryParam:
    nprobe: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "nprobe", _require_int("nprobe", self.nprobe))

    def _get_native(self):
        return _native_module.IVFQueryParam(nprobe=self.nprobe)
```

- [ ] **Step 3: Export the new param wrappers**

Update:

- `hannsdb.model.param`
- `hannsdb.model`
- top-level `hannsdb`

so the new wrappers are available through the same public routes as the existing params.

- [ ] **Step 4: Run the wrapper/export tests**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
python -m pytest tests/test_common_param_surface.py -k 'FlatIndexParam or IVFQueryParam or absent' -q
```

Expected: still FAIL until the native binding exposes matching `_native` classes.

---

## Chunk 3: Expand `DataType` and native binding support

### Task 3: Add the common `DataType` values and native param classes

**Files:**
- Modify: `crates/hannsdb-py/python/hannsdb/typing/data_type.py`
- Modify: `crates/hannsdb-py/src/lib.rs`
- Modify: `crates/hannsdb-py/tests/test_typing_surface.py`
- Modify: `crates/hannsdb-py/tests/test_common_param_surface.py`

- [ ] **Step 1: Add the common scalar `DataType` members**

In `typing/data_type.py`, add:

- `Int32`
- `UInt32`
- `UInt64`
- `Float`

Do not add advanced vector/sparse/array values in this slice.

- [ ] **Step 2: Confirm the Rust binding already recognizes the same values**

Update `lib.rs` only where needed so the native `DataType` pyclass exposes the same common values:

- `Int32`
- `UInt32`
- `UInt64`
- `Float`

If the parser or getters are inconsistent, patch them in this same chunk.

- [ ] **Step 3: Add native `PyFlatIndexParam` and `PyIVFQueryParam` classes**

In `lib.rs`, add:

- `PyFlatIndexParam`
- `PyIVFQueryParam`

with minimal stable surfaces:

```rust
#[pyclass(name = "FlatIndexParam", module = "hannsdb")]
struct PyFlatIndexParam { inner: FlatIndexParam }

#[pyclass(name = "IVFQueryParam", module = "hannsdb")]
struct PyIVFQueryParam { inner: IvfQueryParam }
```

Expose only the fields this slice actually supports.

- [ ] **Step 4: Export the native classes**

Update the module initialization section so `_native_module.FlatIndexParam` and `_native_module.IVFQueryParam` exist before the pure-Python wrappers call `_get_native()`.

- [ ] **Step 5: Run the focused suite**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
uvx maturin develop --uv --features python-binding,hanns-backend && \
python -m pytest tests/test_typing_surface.py tests/test_common_param_surface.py -k 'DataType or FlatIndexParam or IVFQueryParam' -q
```

Expected: PASS for the newly added common `DataType` members and native param wrapper bridges.

---

## Chunk 4: Schema and query normalization

### Task 4: Accept the new common params in `VectorSchema` and `VectorQuery`

**Files:**
- Modify: `crates/hannsdb-py/python/hannsdb/model/schema/field_schema.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/vector_query.py`
- Modify: `crates/hannsdb-py/src/lib.rs`
- Modify: `crates/hannsdb-py/tests/test_common_param_surface.py`

- [ ] **Step 1: Update `VectorSchema` normalization**

Make `VectorSchema(..., index_param=...)` accept:

- `FlatIndexParam`
- `HnswIndexParam`
- `IVFIndexParam`

Reject unsupported param types explicitly instead of letting them drift into generic `Any` paths.

- [ ] **Step 2: Update `VectorQuery` normalization**

Make `VectorQuery(..., param=...)` accept:

- `HnswQueryParam`
- `IVFQueryParam`

Reject unsupported query param objects explicitly.

- [ ] **Step 3: Update the native bridge extraction logic**

Where `lib.rs` currently branches on `PyHnswIndexParam` / `PyIVFIndexParam` or `PyHnswQueryParam`, extend the checks to include the new common classes. Keep error wording clear:

- `index_param must be FlatIndexParam, HnswIndexParam, or IVFIndexParam`
- `query param must be HnswQueryParam or IVFQueryParam`

- [ ] **Step 4: Run the normalization suite**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
python -m pytest tests/test_common_param_surface.py -k 'VectorSchema or VectorQuery or native' -q
```

Expected: PASS.

---

## Chunk 5: End-to-end smoke and final verification

### Task 5: Add a smoke flow for the common-surface slice

**Files:**
- Create: `scripts/smoke_common_type_param_surface.py`

- [ ] **Step 1: Add a smoke script that exercises the new common surface**

The script should:

1. create a collection whose schema uses one of the newly exposed scalar `DataType` values
2. create a vector field with `FlatIndexParam`
3. insert a small dataset
4. run a query with `IVFQueryParam`
5. reopen the collection and verify the schema still reflects the common-surface types cleanly

If a specific query-time effect cannot be asserted meaningfully, assert at least that the parameter object is accepted and the query path executes.

- [ ] **Step 2: Assert the non-goals remain absent**

In the smoke script or focused tests, keep one explicit check that advanced contracts are still absent:

- `InvertIndexParam`
- `HnswRabitq*`
- advanced vector/sparse/array `DataType` values

- [ ] **Step 3: Run the smoke flow**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
python ../../scripts/smoke_common_type_param_surface.py
```

Expected: exit 0.

- [ ] **Step 4: Run the final verification set**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
uvx maturin develop --uv --features python-binding,hanns-backend && \
python -m pytest tests/test_typing_surface.py -q && \
python -m pytest tests/test_common_param_surface.py -q && \
python ../../scripts/smoke_common_type_param_surface.py
```

Expected:

- common `DataType` values PASS
- `FlatIndexParam` PASS
- `IVFQueryParam` PASS
- schema/query normalization PASS
- advanced param families remain absent

---

## Execution notes

- Keep this slice honest. If `FlatIndexParam` or `IVFQueryParam` start requiring new ANN behavior, stop and split a new design.
- Prefer explicit unsupported-type errors over permissive `Any` acceptance in `VectorSchema` and `VectorQuery`.
- Do not add advanced `DataType` values just because zvec has them; this slice is only for common values already supportable by HannsDB.
- If native and pure-Python surfaces diverge, prioritize the documented public `hannsdb` contract and localize any necessary bridge cleanup.

Plan complete and saved to `docs/superpowers/plans/2026-04-12-hannsdb-common-type-param-surface-parity.md`. Ready to execute?
