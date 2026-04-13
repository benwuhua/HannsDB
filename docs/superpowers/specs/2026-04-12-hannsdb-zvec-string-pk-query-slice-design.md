# HannsDB zvec String PK Query Slice Design

**Date:** 2026-04-12

**Goal:** Define a single executable parity slice that closes one high-value user-facing gap against zvec by adding string primary-key support and exposing the missing Python query surface needed to use it end to end.

## Context

Recent parity work already moved HannsDB beyond the "missing basic features" stage:

- core supports `query_by_id`
- core supports `group_by`
- core has richer filter grammar than the earlier gap analysis assumed
- core supports sparse vectors and reranker primitives
- basic schema mutation exists

The remaining gap is no longer "whether the engine can do anything," but whether a user can access mature capabilities through a coherent public surface.

The most leverageable gap for the next executable slice is:

1. Python/public API still treats document IDs as effectively `i64`
2. Python facade still hides part of the query surface that core already models
3. parity work is missing a single end-to-end slice that starts from external data modeling and ends in verification

This makes `string PK + query_by_id/order_by facade exposure + verification` the right next slice.

## Problem Statement

Compared with zvec, HannsDB still lacks a clean public path for common application data models where:

- document IDs are strings from an external system
- clients need to fetch rows by those IDs
- clients need stable ordering in the same public query surface

Today the core and facade are misaligned:

- parts of the engine can already represent the query semantics
- the Python layer still narrows IDs and query options
- tests verify individual internals, but not this product-level workflow

Without this slice, HannsDB remains hard to position as zvec-parity-capable even if core internals have advanced.

## Scope

This design covers only one vertical slice:

1. string primary-key support through the public collection API
2. `query_by_id` support for string IDs in the Python-facing path
3. `order_by` exposure in the Python-facing query path
4. parity and regression tests for the slice
5. minimal verification commands proving the slice works end to end

## Out of Scope

This design does **not** include:

- storage/runtime rearchitecture
- Arrow/Parquet adoption
- quantization
- full schema-mutation parity with zvec
- full query-combination parity
- broad daemon protocol redesign
- sparse/vector-family productization

Those remain later workstreams and should not block this slice.

## Success Criteria

The slice is complete only if all of the following are true:

1. A Python client can create and use string document IDs without coercing them to `i64`.
2. `query_by_id` works for string IDs through the public API.
3. `order_by` is exposed through the Python query surface and works for the supported minimal cases.
4. Existing numeric-ID behavior remains compatible.
5. Core tests and Python facade tests cover the new path.
6. A small verification flow proves the slice works without depending on future runtime work.

## Recommended Approach

Use a compatibility-preserving vertical slice:

- keep existing internal numeric fast paths where they are already assumed
- introduce a stable external PK layer that can represent strings
- route `query_by_id` through that external PK abstraction
- expose only the minimal `order_by` surface that core can support cleanly today
- gate scope tightly to avoid expanding into query-planner redesign

This is preferable to either:

- a facade-only patch, which would paper over the PK model mismatch
- a runtime-first rewrite, which would be too large for a single executable plan

## Architecture

### 1. External PK Abstraction

Introduce an explicit external document-key model at the API boundary.

The key requirement is to stop treating public document IDs as "really `i64` with nicer syntax."  
The public API should accept string keys as first-class values.

The implementation should preserve two invariants:

1. external keys are stable and user-facing
2. internal row/document identifiers remain free to use existing numeric/runtime-friendly representations where appropriate

This separation gives HannsDB a path to zvec-style PK semantics without forcing a storage rewrite in the same slice.

### 2. Collection Write/Read Path

Collection insert, update, and lookup paths need a single shared rule for PK handling.

The slice should define:

- how string PK values enter the collection
- where they are normalized and validated
- how they map into existing internal row/doc references
- how `query_by_id` resolves them back to the correct document

The design should avoid per-call ad hoc coercion in the Python binding.  
Instead, PK conversion should happen in one small, testable abstraction layer.

### 3. Python Query Surface

The Python query API should expose only the minimal additions required for this slice:

- `query_by_id` accepting string IDs
- `order_by` exposed in `QueryContext` or the equivalent public query builder

This should stay intentionally narrow:

- no attempt to unlock every blocked query combination
- no broad redesign of query planning
- no speculative API expansion beyond what tests will cover

The purpose is to align public surface with already-existing core semantics, not to create a new query DSL.

### 4. Compatibility Model

This slice must not break existing integer-ID users.

Compatibility requirements:

- integer IDs continue to work
- mixed validation failures are explicit and deterministic
- existing parity tests that depend on numeric IDs continue to pass or are updated only where the public contract changes

If a fully mixed PK mode is too ambiguous, the collection-level rule should be made explicit rather than silently guessing.

## Data Flow

The intended happy path is:

1. user defines or inserts documents with string PK values
2. Python facade validates and forwards PKs without forcing `i64`
3. core stores/resolves the external PK through a dedicated mapping layer
4. `query_by_id` resolves the same PK and returns the expected row
5. result ordering can be controlled through the minimal supported `order_by` surface

This data flow is the actual product slice to verify.

## Error Handling

The slice should prefer explicit failures over permissive coercion.

Examples of behavior that should be defined and tested:

- invalid PK type supplied for the collection mode
- duplicate string PK on insert
- `query_by_id` with missing string PK
- unsupported `order_by` shape or field type
- ambiguous mixed-PK usage if that mode is disallowed

Public error messages do not need to be perfect in this slice, but they must be stable and intention-revealing.

## Testing Strategy

Testing needs to prove the slice at three layers.

### Core

- string PK insert and lookup
- duplicate/missing PK behavior
- backward compatibility for numeric IDs
- `query_by_id` correctness on string PK collections

### Python Facade

- collection creation or insert path with string PK
- `query_by_id` by string PK
- `order_by` exposure and basic sorted results
- compatibility checks for old integer-ID paths

### End-to-End Verification

- a small Python-driven flow that creates data, inserts rows, queries by string PK, and runs a sorted query
- commands must be lightweight and runnable in the current repo workflow

## Risks

### Risk 1: PK semantics leak into too many runtime assumptions

Mitigation:

- isolate public PK handling behind one abstraction
- avoid storage refactors in this slice

### Risk 2: `order_by` surface expands beyond what core reliably supports

Mitigation:

- expose only the subset already represented cleanly in core
- reject unsupported combinations explicitly

### Risk 3: Numeric compatibility regresses

Mitigation:

- keep old-path regression tests in the same implementation wave
- do not remove numeric parsing until string and numeric paths are both proven

## File Planning Constraints

The implementation plan should keep responsibilities explicit and small:

- core PK abstraction and resolution logic should live in one focused area
- Python API typing/validation should stay separate from query execution plumbing
- query-surface exposure should be split from PK storage logic
- tests should mirror the same decomposition

The resulting implementation plan should avoid one giant "parity" task and instead break this slice into bite-sized TDD tasks with exact commands.

## Plan Boundary

The implementation plan that follows from this design should cover only:

1. PK model introduction
2. core resolution path
3. Python facade exposure
4. parity/regression tests
5. lightweight verification

Anything beyond that belongs in a later plan.
