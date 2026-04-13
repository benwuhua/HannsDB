# HannsDB Common Type And Param Surface Parity Design

**Date:** 2026-04-12

**Goal:** Define the next executable Python/public parity slice by expanding HannsDB's most common type and parameter surfaces to better match zvec without exposing advanced or unsupported contracts prematurely.

## Context

Recent parity work has already improved the public Python layer in important ways:

- string primary keys are usable publicly
- `query_by_id`, `order_by`, and built-in reranker combinations are exposed
- schema mutation now has a richer public contract with explicit unsupported-shape errors

That changes the remaining public gap again.

The next blocker is not a single missing feature.  
It is that HannsDB's common Python type and parameter surface is still much narrower than zvec's.

Today HannsDB's public surface includes:

- `DataType` with only:
  - `String`
  - `Int64`
  - `Float64`
  - `Bool`
  - `VectorFp32`
- vector index params:
  - `HnswIndexParam`
  - `IVFIndexParam`
- vector query params:
  - `HnswQueryParam`

Compared with zvec, the public gap is obvious even before advanced parity:

- no `FlatIndexParam`
- no `IVFQueryParam`
- `DataType` is missing several common scalar values HannsDB already uses internally
- `VectorSchema` and `VectorQuery` public contracts are narrower than the engine-facing code paths imply

At the same time, zvec exposes many advanced types and params that HannsDB still cannot honestly support yet, such as:

- `InvertIndexParam`
- `HnswRabitqIndexParam`
- `HnswRabitqQueryParam`
- extended vector and sparse data-type families
- array type families

So the right next step is not "full param parity."  
The right next step is a narrower common-surface slice.

## Problem Statement

HannsDB currently undersells its practical Python/public surface in two ways:

1. **Common scalar and vector type options are artificially narrow.**
   Public `DataType` does not reflect several values already recognized in the Rust binding and schema bridge.

2. **Common vector index/query param shapes are incomplete.**
   Users cannot express an explicit flat vector index configuration or an IVF query-time parameter object through the same public style that zvec supports.

If we skip directly to advanced params, this slice will explode in scope and start promising contracts that core or the binding layer cannot support cleanly.  
If we do nothing, HannsDB keeps looking less mature than it actually is in normal usage.

## Scope

This design covers one focused Python/public parity project:

1. expand public `DataType` to the common values HannsDB can already support honestly
2. add public `FlatIndexParam`
3. add public `IVFQueryParam`
4. extend `VectorSchema.index_param` to accept the common vector index param family
5. extend `VectorQuery.param` to accept the common query param family
6. add focused public-surface tests and smoke verification

## Out of Scope

This design does **not** include:

- `InvertIndexParam`
- `HnswRabitqIndexParam`
- `HnswRabitqQueryParam`
- full `IndexParam` / `QueryParam` / `VectorIndexParam` base-class parity
- sparse vector type parity
- array data-type parity
- additional ANN algorithms
- runtime/storage changes
- schema-mutation work

Those remain separate workstreams.

## Success Criteria

This parity slice is complete only if all of the following are true:

1. `DataType` publicly exposes the common scalar values already bridgeable through HannsDB.
2. `FlatIndexParam` is importable from the public package and usable in `VectorSchema`.
3. `IVFQueryParam` is importable from the public package and usable in `VectorQuery`.
4. `VectorSchema.index_param` accepts `FlatIndexParam`, `HnswIndexParam`, and `IVFIndexParam`.
5. `VectorQuery.param` accepts `HnswQueryParam` and `IVFQueryParam`.
6. Unsupported advanced params remain absent rather than being exposed as fake parity.

## Candidate Approaches

### Approach 1: Common Surface First

Expand only the highest-value, most honestly supportable public surface:

- common scalar `DataType`
- `FlatIndexParam`
- `IVFQueryParam`

Pros:

- closes the most visible day-to-day parity gaps
- stays within current binding and engine reality
- produces a clean, testable slice

Cons:

- advanced zvec param parity remains open

### Approach 2: Full Param Family Push

Attempt to expose most of zvec's parameter families in one wave, including advanced quantized/index/query objects.

Pros:

- looks closer to zvec on paper

Cons:

- much larger scope
- high risk of dishonest or partial contracts
- likely to collide with unsupported core behavior

### Approach 3: DataType-Only Cleanup

Only expand `DataType`, leave param families unchanged.

Pros:

- smallest change set

Cons:

- weak user value
- leaves the most visible param-surface gap unresolved

## Recommendation

Use **Approach 1: Common Surface First**.

This is the right trade-off because:

- it closes the most common zvec-comparison gap without overpromising
- the Rust binding already knows about more scalar data types than the public `typing.DataType` currently exposes
- `FlatIndexParam` and `IVFQueryParam` can be represented honestly as public contract additions
- it avoids expanding into advanced quantized, sparse, or inverted-index parity before HannsDB is ready

The key design principle is:

**Expose the common contracts HannsDB can support honestly. Leave advanced parity absent until it is real.**

## Architecture

### 1. Public `DataType` Expansion

This slice should expand `hannsdb.typing.DataType` only to values HannsDB can already bridge cleanly through the binding and schema layer.

Recommended additions:

- `Int32`
- `UInt32`
- `UInt64`
- `Float`

Keep existing values:

- `String`
- `Int64`
- `Float64`
- `Bool`
- `VectorFp32`

Do **not** add advanced values that the public contract cannot yet support honestly, including:

- `VectorFp16`
- `VectorFp64`
- `VectorInt8`
- sparse vector data types
- array data types

This keeps `DataType` aligned with current practical support rather than with aspirational zvec breadth.

### 2. `FlatIndexParam`

Add a public pure-Python `FlatIndexParam` wrapper following the existing param-wrapper style.

The contract should be narrow:

- accept `metric_type`
- optionally accept `quantize_type` only if HannsDB can already map that value honestly through the existing flat/no-extra-index path

If quantization on flat is not truly supported through this public route today, keep the shape minimal and reject unsupported options explicitly.

This object should be importable from:

- `hannsdb`
- `hannsdb.model.param`

`VectorSchema` should accept `FlatIndexParam` alongside existing `HnswIndexParam` and `IVFIndexParam`.

### 3. `IVFQueryParam`

Add a public pure-Python `IVFQueryParam` wrapper for query-time IVF controls.

The public contract should be minimal and honest.  
If `nprobe` is the only stable field, expose only `nprobe`.

This object should be importable from:

- `hannsdb`
- `hannsdb.model.param`

`VectorQuery.param` should accept `IVFQueryParam` alongside `HnswQueryParam`.

### 4. Bridge Semantics

This slice does not invent new ANN behavior.

Instead:

- `VectorSchema.index_param` normalization should recognize:
  - `FlatIndexParam`
  - `HnswIndexParam`
  - `IVFIndexParam`
- `VectorQuery.param` bridge logic should recognize:
  - `HnswQueryParam`
  - `IVFQueryParam`

If the core path ultimately ignores some aspect of a newly added public param, that behavior must be explicit and intentional, not accidental.

### 5. Compatibility Model

This slice has two compatibility goals.

#### Existing HannsDB callers

Existing code using:

- `HnswIndexParam`
- `IVFIndexParam`
- `HnswQueryParam`
- current `DataType` values

must continue to work unchanged.

#### New common-surface callers

New callers should be able to write:

- `VectorSchema(..., index_param=FlatIndexParam(...))`
- `VectorQuery(..., param=IVFQueryParam(...))`
- `FieldSchema(..., data_type=DataType.UInt32)` and the other newly exposed common scalar types

### 6. Error Handling

This slice should prefer explicit validation failures over soft coercion.

Examples:

- unsupported `metric_type`
- invalid `nprobe`
- unsupported advanced `DataType` values not included in this slice
- unsupported advanced param objects passed into `VectorSchema` or `VectorQuery`

Errors do not need to match zvec text exactly. They do need to explain whether the issue is:

- invalid value
- unsupported param type
- unsupported advanced contract

## Data Flow

The intended public flow is:

1. user constructs common param/type objects in pure Python
2. schema/query wrappers validate and normalize them
3. `_get_native()` or bridge helpers translate them into the native binding objects
4. existing core-backed create/query/index code paths execute without requiring new ANN algorithms

Unsupported advanced objects should fail before reaching ambiguous native behavior.

## Testing Strategy

Testing should cover three layers.

### Typing Surface

- `DataType` exports the new common values
- new values are accepted by `FieldSchema` and existing schema wrappers

### Param Surface

- `FlatIndexParam` public import, construction, validation, and native bridge
- `IVFQueryParam` public import, construction, validation, and native bridge
- `VectorSchema` accepts `FlatIndexParam`
- `VectorQuery` accepts `IVFQueryParam`

### End-to-End Smoke

- create collection with `FlatIndexParam`
- query with `IVFQueryParam`
- verify common `DataType` values survive schema declaration / reopen

## Risks

### Risk 1: `DataType` expands beyond real support

Mitigation:

- only add values already supported by the Rust bridge
- keep advanced vector/sparse/array types out of this slice

### Risk 2: `FlatIndexParam` implies new ANN behavior

Mitigation:

- document it as contract alignment, not algorithm expansion
- map only to existing flat-capable behavior

### Risk 3: Query-param surface becomes inconsistent

Mitigation:

- keep `IVFQueryParam` minimal
- accept only the parameter family HannsDB can actually bridge today

## Implementation Notes

The main files likely involved are:

- `crates/hannsdb-py/python/hannsdb/typing/data_type.py`
- `crates/hannsdb-py/python/hannsdb/model/param/index_params.py`
- `crates/hannsdb-py/python/hannsdb/model/param/__init__.py`
- `crates/hannsdb-py/python/hannsdb/model/__init__.py`
- `crates/hannsdb-py/python/hannsdb/__init__.py`
- `crates/hannsdb-py/python/hannsdb/model/param/vector_query.py`
- `crates/hannsdb-py/python/hannsdb/model/schema/field_schema.py`
- `crates/hannsdb-py/src/lib.rs`
- `crates/hannsdb-py/tests/test_typing_surface.py`
- a new focused param-surface test file

This should remain a Python/public surface and binding-bridge project. If implementation starts requiring new ANN runtime behavior or advanced quantized/index families, that has escaped this spec and should become a separate slice.
