# HannsDB IvfUsq Surface Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose HannsDB's first honest Hanns-native quantized ANN public surface by adding `IvfUsqIndexParam` and `IvfUsqQueryParam`, wiring them through the Python and PyO3 layers, and preserving explicit `ivf_usq` identity without silently degrading to plain IVF.

**Architecture:** Keep this slice at the public API, binding, and schema/catalog-contract layer. Add distinct `IvfUsq*` wrapper and native types, extend `VectorSchema` and `VectorQuery` to accept them, preserve `kind = "ivf_usq"` through serialization, and make unsupported backend execution fail explicitly rather than being coerced into existing plain-`ivf` behavior.

**Tech Stack:** Pure Python param wrappers, PyO3 binding in `crates/hannsdb-py/src/lib.rs`, HannsDB catalog/schema serialization, existing Python facade normalization, `pytest`, lightweight smoke scripts

---

## Chunk 1: Scope lock and red public-surface tests

### Current state to preserve

- `hannsdb` publicly exposes `FlatIndexParam`, `HnswIndexParam`, `IVFIndexParam`, `HnswQueryParam`, and `IVFQueryParam`.
- `VectorSchema.index_param` accepts only plain `flat`, `hnsw`, and `ivf` parameter families.
- `VectorQuery.param` accepts only `HnswQueryParam` and `IVFQueryParam`.
- Catalog serialization currently distinguishes `flat`, `hnsw`, and `ivf`, but not `ivf_usq`.

### Explicit non-goals for this slice

- Do not add `HnswUsqIndexParam` or `HnswUsqQueryParam`.
- Do not add `HnswRabitqIndexParam` or `HnswRabitqQueryParam`.
- Do not add `InvertIndexParam`.
- Do not silently map `IvfUsq*` to plain `IVF*`.
- Do not implement a brand-new ANN backend if HannsDB cannot already support the route.

### File map

**Pure Python surface**
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/index_params.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/__init__.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/__init__.py`
- Modify: `crates/hannsdb-py/python/hannsdb/__init__.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/schema/field_schema.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/vector_query.py`

**PyO3/native bridge**
- Modify: `crates/hannsdb-py/src/lib.rs`

**Tests and verification**
- Create: `crates/hannsdb-py/tests/test_ivf_usq_surface.py`
- Modify: `crates/hannsdb-py/tests/test_common_param_surface.py`
- Create: `scripts/smoke_ivf_usq_surface.py`

### Task 1: Add red tests that lock the `IvfUsq` contract

**Files:**
- Create: `crates/hannsdb-py/tests/test_ivf_usq_surface.py`
- Modify: `crates/hannsdb-py/tests/test_common_param_surface.py`

- [ ] **Step 1: Add red tests for public exports**

Add tests asserting:

```python
import hannsdb

def test_ivf_usq_public_exports_exist():
    assert hasattr(hannsdb, "IvfUsqIndexParam")
    assert hasattr(hannsdb, "IvfUsqQueryParam")
```

- [ ] **Step 2: Add red tests for constructor validation**

Add tests that verify:

- `metric_type` is validated
- `nlist`, `bits_per_dim`, `rotation_seed`, `rerank_k`, and `nprobe` require real ints
- `use_high_accuracy_scan` requires a real bool

- [ ] **Step 3: Add red tests for `VectorSchema` and `VectorQuery` acceptance**

Add tests asserting:

```python
schema = hannsdb.VectorSchema(
    "embedding",
    hannsdb.DataType.VectorFp32,
    dimension=32,
    index_param=hannsdb.IvfUsqIndexParam(
        metric_type=hannsdb.MetricType.L2,
        nlist=64,
        bits_per_dim=4,
        rotation_seed=42,
        rerank_k=64,
        use_high_accuracy_scan=False,
    ),
)

query = hannsdb.VectorQuery(
    "embedding",
    [0.1] * 32,
    param=hannsdb.IvfUsqQueryParam(nprobe=4),
)
```

- [ ] **Step 4: Add red tests for explicit non-fallback behavior**

Add tests that assert one of the following:

- catalog/schema serialization preserves `ivf_usq`
- unsupported runtime paths raise an explicit error mentioning `ivf_usq`

The test should fail if the implementation silently behaves like plain IVF.

- [ ] **Step 5: Run the red suite**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
python -m pytest tests/test_ivf_usq_surface.py tests/test_common_param_surface.py -q
```

Expected: FAIL because `IvfUsq*` types and contract wiring do not exist yet.

---

## Chunk 2: Pure-Python wrappers and exports

### Task 2: Add `IvfUsqIndexParam` and `IvfUsqQueryParam`

**Files:**
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/index_params.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/__init__.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/__init__.py`
- Modify: `crates/hannsdb-py/python/hannsdb/__init__.py`
- Modify: `crates/hannsdb-py/tests/test_ivf_usq_surface.py`

- [ ] **Step 1: Add the `IvfUsqIndexParam` dataclass**

Implement a narrow wrapper in `index_params.py`:

```python
@dataclass(frozen=True)
class IvfUsqIndexParam:
    metric_type: str | None = None
    nlist: int = 1024
    bits_per_dim: int = 4
    rotation_seed: int = 42
    rerank_k: int = 64
    use_high_accuracy_scan: bool = False
```

Validate fields using the same helper style already used by the other param wrappers.

- [ ] **Step 2: Add the `IvfUsqQueryParam` dataclass**

Implement a narrow wrapper:

```python
@dataclass(frozen=True)
class IvfUsqQueryParam:
    nprobe: int = 1
```

with strict integer validation and `_get_native()`.

- [ ] **Step 3: Export the wrappers**

Update:

- `hannsdb.model.param`
- `hannsdb.model`
- top-level `hannsdb`

so both wrappers are available everywhere the existing param families are exported.

- [ ] **Step 4: Run the wrapper-only tests**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
python -m pytest tests/test_ivf_usq_surface.py -k 'export or validation' -q
```

Expected: wrapper validation passes, but native-bridge tests still fail until `_native` exposes matching classes.

---

## Chunk 3: Native binding and distinct `ivf_usq` identity

### Task 3: Add PyO3 classes and internal enum branches

**Files:**
- Modify: `crates/hannsdb-py/src/lib.rs`
- Modify: `crates/hannsdb-py/tests/test_ivf_usq_surface.py`

- [ ] **Step 1: Add Rust structs for `IvfUsqIndexParam` and `IvfUsqQueryParam`**

Introduce distinct internal structs in `lib.rs`, for example:

```rust
pub struct IvfUsqIndexParam {
    pub metric_type: Option<MetricType>,
    pub nlist: usize,
    pub bits_per_dim: usize,
    pub rotation_seed: usize,
    pub rerank_k: usize,
    pub use_high_accuracy_scan: bool,
}

pub struct IvfUsqQueryParam {
    pub nprobe: usize,
}
```

- [ ] **Step 2: Extend internal enums so `ivf_usq` is distinct**

Add explicit enum branches rather than reusing plain IVF:

```rust
pub enum IndexParam {
    Flat(FlatIndexParam),
    Hnsw(HnswIndexParam),
    Ivf(IvfIndexParam),
    IvfUsq(IvfUsqIndexParam),
}
```

and do the same for query param handling if needed.

- [ ] **Step 3: Add PyO3 classes**

Expose:

- `PyIvfUsqIndexParam`
- `PyIvfUsqQueryParam`

with only the public fields from the spec.

- [ ] **Step 4: Export the native classes**

Wire both pyclasses into the module initialization so `_native.IvfUsqIndexParam` and `_native.IvfUsqQueryParam` exist.

- [ ] **Step 5: Make catalog serialization preserve `kind = \"ivf_usq\"`**

Update the serialization helpers so `vector_index_catalog_json(...)` and any related schema/index metadata paths preserve:

```json
{ "kind": "ivf_usq", ... }
```

This step must not serialize `IvfUsqIndexParam` as plain `ivf`.

- [ ] **Step 6: Run native-bridge tests**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
uvx maturin develop --uv --features python-binding,hanns-backend && \
python -m pytest tests/test_ivf_usq_surface.py -k 'native or serialize or export' -q
```

Expected: PASS for native creation and serialized `ivf_usq` identity tests.

---

## Chunk 4: Python facade integration and explicit runtime contract

### Task 4: Accept `IvfUsq*` in `VectorSchema` and `VectorQuery`

**Files:**
- Modify: `crates/hannsdb-py/python/hannsdb/model/schema/field_schema.py`
- Modify: `crates/hannsdb-py/python/hannsdb/model/param/vector_query.py`
- Modify: `crates/hannsdb-py/src/lib.rs`
- Modify: `crates/hannsdb-py/tests/test_ivf_usq_surface.py`
- Modify: `crates/hannsdb-py/tests/test_common_param_surface.py`

- [ ] **Step 1: Extend `VectorSchema` normalization**

Update `_normalize_index_param(...)` so it accepts:

- `FlatIndexParam`
- `HnswIndexParam`
- `IVFIndexParam`
- `IvfUsqIndexParam`

and rejects everything else with a precise type error.

- [ ] **Step 2: Extend `VectorQuery` normalization**

Update the query param validator so it accepts:

- `HnswQueryParam`
- `IVFQueryParam`
- `IvfUsqQueryParam`

- [ ] **Step 3: Extend the native conversion points**

Update `PyVectorSchema::new`, `PyVectorQuery::new`, and any helper extraction points so they recognize `PyIvfUsqIndexParam` / `PyIvfUsqQueryParam`.

- [ ] **Step 4: Add the explicit runtime contract**

At the first truthful execution boundary, make one of these behaviors explicit:

1. If HannsDB already supports `ivf_usq`, route to the real path.
2. If it does not, raise an explicit error that mentions `ivf_usq` and does not silently downgrade to plain IVF.

Prefer the earliest stable boundary that avoids partial work, such as schema application, index creation, or query dispatch.

- [ ] **Step 5: Run the focused facade suite**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
python -m pytest tests/test_ivf_usq_surface.py tests/test_common_param_surface.py -q
```

Expected: PASS, including the explicit unsupported-runtime contract if backend execution is not yet wired.

---

## Chunk 5: Smoke verification and regression closeout

### Task 5: Add smoke coverage and verify no plain-IVF regressions

**Files:**
- Create: `scripts/smoke_ivf_usq_surface.py`
- Modify: `crates/hannsdb-py/tests/test_common_param_surface.py`

- [ ] **Step 1: Add a smoke script for the `IvfUsq` surface**

Create a script that:

- imports `IvfUsqIndexParam` and `IvfUsqQueryParam`
- builds a `CollectionSchema` using `VectorSchema(..., index_param=IvfUsqIndexParam(...))`
- constructs a matching `VectorQuery(..., param=IvfUsqQueryParam(...))`
- reopens or rehydrates schema metadata if supported
- asserts preserved `ivf_usq` identity
- asserts the explicit runtime error if full backend support is not present

- [ ] **Step 2: Add regression assertions for plain IVF**

Extend tests so existing `IVFIndexParam` / `IVFQueryParam` behavior still works and is still serialized as plain `ivf`, not `ivf_usq`.

- [ ] **Step 3: Run the final verification set**

Run:

```bash
cd crates/hannsdb-py && \
source .venv/bin/activate && \
uvx maturin develop --uv --features python-binding,hanns-backend && \
python -m pytest tests/test_ivf_usq_surface.py tests/test_common_param_surface.py -q && \
python ../../scripts/smoke_ivf_usq_surface.py
```

Expected:

- `pytest` PASS
- smoke script exits `0`
- if backend execution is unsupported, the script should verify the expected explicit error rather than treating it as failure

- [ ] **Step 4: Commit**

```bash
git add \
  crates/hannsdb-py/python/hannsdb/model/param/index_params.py \
  crates/hannsdb-py/python/hannsdb/model/param/__init__.py \
  crates/hannsdb-py/python/hannsdb/model/__init__.py \
  crates/hannsdb-py/python/hannsdb/__init__.py \
  crates/hannsdb-py/python/hannsdb/model/schema/field_schema.py \
  crates/hannsdb-py/python/hannsdb/model/param/vector_query.py \
  crates/hannsdb-py/src/lib.rs \
  crates/hannsdb-py/tests/test_ivf_usq_surface.py \
  crates/hannsdb-py/tests/test_common_param_surface.py \
  scripts/smoke_ivf_usq_surface.py
git commit -m "feat: add ivf usq public surface"
```

---

## Execution Notes

- Use TDD strictly: red test, minimal implementation, green verification.
- Do not silently coerce `IvfUsq*` into `IVF*`.
- Prefer explicit `NotImplementedError` / `ValueError` / runtime errors mentioning `ivf_usq` over dishonest fallback.
- Keep this slice limited to `IvfUsq`; `HnswUsq` is a separate follow-up project.
