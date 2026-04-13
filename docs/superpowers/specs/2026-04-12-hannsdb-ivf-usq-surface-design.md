# HannsDB IvfUsq Surface Design

**Date:** 2026-04-12

**Goal:** Define an executable Hanns-native quantized ANN public-surface slice by exposing `IvfUsq` as an honest Python/public contract, rather than forcing zvec's `HnswRabitq` naming onto different underlying semantics.

## Context

Recent parity work has already tightened HannsDB's Python/public layer in several places:

- string primary keys are public and persistent
- `query_by_id`, `order_by`, and built-in reranker combinations are exposed
- schema mutation has richer public contracts
- common type and param surfaces now include `FlatIndexParam` and `IVFQueryParam`

That changes the remaining gap again.

The next meaningful gap is no longer "basic API breadth."  
It is that HannsDB still does not expose its stronger quantized ANN direction as a coherent public surface.

Code comparison shows an important distinction:

- zvec exposes a complete `HnswRabitqIndexParam` / `HnswRabitqQueryParam` product surface
- Hanns has broader underlying USQ-based quantization machinery, including:
  - a reusable `UsqQuantizer`
  - `IvfUsq`
  - HNSW + USQ/HVQ paths
  - PCA + USQ variants

So the right next move is not to mimic zvec's name first.  
The right next move is to expose one honest Hanns-native quantized ANN surface end to end.

`IvfUsq` is the best first slice because its parameter shape is already relatively stable and concrete:

- index-time: `nlist`, `bits_per_dim`, `rotation_seed`, `rerank_k`, `use_high_accuracy_scan`
- query-time: `nprobe`

That makes it a better first public quantized path than `HnswUsq`, whose current naming and implementation are more fragmented.

## Problem Statement

HannsDB currently undersells its quantized ANN capability in a specific way:

1. **The engine family is richer than the public Python contract.**
   Hanns has real USQ-based ANN code, but HannsDB does not yet present that as a first-class public index/query family.

2. **The next parity step risks becoming dishonest if it copies zvec naming too early.**
   Exposing `HnswRabitq*` directly over non-equivalent Hanns implementations would create a fake compatibility layer.

3. **Users cannot currently select a Hanns-native quantized IVF path explicitly.**
   That leaves a meaningful gap between real engine capability and public product surface.

## Scope

This design covers one focused public-surface project:

1. add `IvfUsqIndexParam`
2. add `IvfUsqQueryParam`
3. extend `VectorSchema.index_param` to accept `IvfUsqIndexParam`
4. extend `VectorQuery.param` to accept `IvfUsqQueryParam`
5. add PyO3/native bridge support for the new parameter family
6. make catalog/schema serialization distinguish `ivf` vs `ivf_usq`
7. add focused public tests and smoke verification

## Out of Scope

This design does **not** include:

- `HnswUsqIndexParam`
- `HnswUsqQueryParam`
- `HnswRabitqIndexParam`
- `HnswRabitqQueryParam`
- `InvertIndexParam`
- generalized quantized ANN abstraction layers
- backend/runtime implementation of a brand-new ANN algorithm
- storage/runtime refactors
- performance tuning work
- compatibility aliases to zvec naming

Those remain separate workstreams.

## Success Criteria

This slice is complete only if all of the following are true:

1. `IvfUsqIndexParam` is publicly importable from `hannsdb`.
2. `IvfUsqQueryParam` is publicly importable from `hannsdb`.
3. `VectorSchema.index_param` accepts `IvfUsqIndexParam`.
4. `VectorQuery.param` accepts `IvfUsqQueryParam`.
5. Public serialization/catalog output distinguishes `ivf_usq` from plain `ivf`.
6. If the current HannsDB backend cannot execute `ivf_usq` truthfully yet, the failure mode is explicit and stable rather than silently degrading to plain IVF.

## Candidate Approaches

### Approach 1: Extend Existing `IVFIndexParam`

Add USQ-specific fields directly onto existing `IVFIndexParam` and `IVFQueryParam`.

Pros:

- smallest API surface change
- fewer new classes

Cons:

- mixes plain IVF and quantized IVF into one type
- weakens semantic clarity
- makes future evolution harder

### Approach 2: Add Explicit `IvfUsq*` Types

Expose `IvfUsqIndexParam` and `IvfUsqQueryParam` as a distinct Hanns-native family.

Pros:

- most honest contract
- aligns with Hanns engine naming
- cleanly separates plain IVF from quantized IVF
- creates a usable template for later `HnswUsq*`

Cons:

- not name-compatible with zvec
- adds two more public classes

### Approach 3: Placeholder Surface Only

Expose `IvfUsq*` names now but reject most use at runtime.

Pros:

- fastest to expose names

Cons:

- low user value
- creates fake parity
- not worth doing

## Recommendation

Use **Approach 2: Add Explicit `IvfUsq*` Types**.

This is the right choice because:

- it is honest about Hanns-native semantics
- it exposes real capability instead of building a compatibility facade first
- it avoids collapsing plain IVF and quantized IVF into one overloaded public type
- it creates a clean next-step template for future quantized ANN slices

The key principle is:

**Expose the stronger Hanns-native surface truthfully before adding any compatibility alias layer.**

## Architecture

### 1. Public Parameter Types

Add two new public pure-Python wrappers:

- `IvfUsqIndexParam`
- `IvfUsqQueryParam`

They should be exported from:

- `hannsdb`
- `hannsdb.model`
- `hannsdb.model.param`

The public shape should stay narrow and stable.

### 2. `IvfUsqIndexParam` Contract

Expose these fields:

- `metric_type`
- `nlist`
- `bits_per_dim`
- `rotation_seed`
- `rerank_k`
- `use_high_accuracy_scan`

Validation should be strict in Python before crossing into native code:

- integer fields must be real ints
- `metric_type` must be one of the supported metrics
- booleans must be real bools
- no extra zvec-style quantizer/training knobs in this slice

This object should remain a pure wrapper, matching the style already used by:

- `FlatIndexParam`
- `HnswIndexParam`
- `IVFIndexParam`

### 3. `IvfUsqQueryParam` Contract

Expose only:

- `nprobe`

That keeps query-time shape aligned with the already-stable IVF public pattern while still distinguishing the quantized family explicitly.

### 4. `VectorSchema` Integration

`VectorSchema.index_param` should accept:

- `FlatIndexParam`
- `HnswIndexParam`
- `IVFIndexParam`
- `IvfUsqIndexParam`

Type errors should be raised at the Python layer for unsupported objects.

### 5. `VectorQuery` Integration

`VectorQuery.param` should accept:

- `HnswQueryParam`
- `IVFQueryParam`
- `IvfUsqQueryParam`

This remains a public-contract extension only.  
It should not silently map `IvfUsqQueryParam` into plain `IVFQueryParam` behavior unless the backend path is explicitly documented as identical.

### 6. Native Binding

The PyO3 layer should gain:

- internal Rust structs for `IvfUsqIndexParam` and `IvfUsqQueryParam`
- pyclasses for both
- enum wiring so internal schema/query representations can carry them distinctly

This is also where public honesty matters most:

- if HannsDB can serialize and preserve `ivf_usq` metadata but cannot execute it yet, that is acceptable for this slice only if runtime paths fail explicitly
- it must not silently serialize `IvfUsqIndexParam` as plain `ivf`

### 7. Catalog And Schema Serialization

Current catalog/index serialization already distinguishes index kinds like `flat`, `hnsw`, and `ivf`.

This slice should add:

- `kind = "ivf_usq"`

and preserve relevant parameter fields in serialized form.

That ensures reopen and schema inspection do not erase the selected quantized family.

### 8. Backend Execution Contract

This slice is about public-surface truthfulness, not backend reinvention.

So there are only two acceptable behaviors:

1. **Real support path exists**
   `IvfUsqIndexParam` and `IvfUsqQueryParam` run through a true `ivf_usq` backend path.

2. **Real support path does not exist yet**
   collection creation, schema application, or query execution fails with a clear and stable error such as `NotImplementedError` / `ValueError` / runtime error with explicit `ivf_usq` wording.

What is not acceptable:

- silently downcasting `IvfUsqIndexParam` to plain IVF
- silently ignoring quantized fields
- pretending schema reopen succeeded while losing `ivf_usq` identity

## Testing Strategy

Add focused tests in the HannsDB Python package for:

- public exports
- constructor validation
- `_get_native()` bridging
- `VectorSchema` acceptance
- `VectorQuery` acceptance
- serialized schema/catalog identity for `ivf_usq`
- explicit unsupported-runtime failure, if backend execution is not wired yet

Add a smoke script covering:

- schema creation with `IvfUsqIndexParam`
- reopen/schema inspection
- query-context creation with `IvfUsqQueryParam`
- expected error contract if execution remains unsupported

## Risks

### Risk 1: Fake parity by fallback

The biggest risk is exposing `IvfUsq*` but internally treating it as plain IVF.

Mitigation:

- keep a distinct enum branch and serialized kind
- require explicit failure if runtime support is missing

### Risk 2: Overdesigning the parameter shape

It would be easy to add more quantization knobs because Hanns internals already have more machinery.

Mitigation:

- keep the public contract limited to the stable, obviously useful fields already visible in `IvfUsqConfig`

### Risk 3: Scope bleed into `HnswUsq`

Once quantized ANN work starts, it is tempting to expose multiple families in one wave.

Mitigation:

- lock this slice to `IvfUsq` only
- treat `HnswUsq` as a separate follow-up design

## Open Questions

1. Does HannsDB currently have a backend execution path that can preserve and execute `ivf_usq` end to end, or only enough schema/catalog machinery to expose the public contract first?
2. Should backend-missing behavior fail at collection creation time, optimize/build time, or first query time?
3. Is `rerank_k` already meaningful in HannsDB's current vector execution layer, or does it need to be treated as preserved-but-not-executed metadata in this slice?

These questions affect implementation detail, but not the public-design direction.
