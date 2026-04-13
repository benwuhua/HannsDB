# HannsDB Query Surface Parity Design

**Date:** 2026-04-12

**Goal:** Define the next executable parity slice after string-PK support by aligning the Python/public query surface with query capabilities that HannsDB core already supports.

## Context

The string-PK slice closed one major product gap:

- Python/public API now accepts string document IDs
- `query_by_id` works through the public collection surface
- `order_by` is exposed publicly
- focused core/Python/smoke coverage exists

That work changed the nature of the remaining Python/public gap.

The next blocker is no longer "missing primitive query fields on `QueryContext`."  
The blocker is that the Python executor still applies extra facade-level restrictions and reranking rules that are narrower than core behavior.

Today:

- core already supports `query_by_id`
- core already supports `group_by`
- core already supports `order_by`
- core already supports built-in rerankers
- the Python executor still rejects some combinations, especially around reranker fan-out

This means the product gap is now concentrated in one place:

1. **Python query execution semantics are stricter than core semantics**
2. **Built-in rerankers do not benefit from core's native combined-query path**
3. **Public contract is fragmented between "works in core" and "blocked in facade"**

## Problem Statement

Compared with zvec, HannsDB still feels incomplete at the Python/public layer because the same query shape can:

- be representable in core
- be accepted by the Python object model
- still be rejected by the Python executor with `NotImplementedError`

The clearest examples today are:

- `query_by_id + reranker`
- `group_by + reranker`

These are not fundamentally missing engine features. They are mostly executor/facade policy choices.

As long as those choices stay in place, HannsDB will continue to undersell its actual query capability and will require users to learn implementation artifacts instead of product semantics.

## Scope

This design covers one focused parity project:

1. align Python query execution with core-supported query combinations
2. make built-in rerankers use the strongest available execution path
3. preserve existing public `QueryContext` / `Collection.query(...)` entry points
4. add regression coverage for combined query shapes
5. add a small smoke path for the combined public workflow

## Out of Scope

This design does **not** include:

- storage/runtime redesign
- ANN/search algorithm changes
- new reranker algorithms
- schema-mutation parity
- index/data-type parameter parity
- daemon/protocol redesign

Those remain separate workstreams.

## Success Criteria

This parity slice is complete only if all of the following are true:

1. Built-in rerankers can be used through the public Python query API with core-supported combinations.
2. `query_by_id + reranker` no longer fails just because the Python executor blocks it.
3. `group_by + reranker` no longer fails just because the Python executor blocks it.
4. `Collection.query(...)` kwargs and `QueryContext(...)` behave consistently for the supported shapes.
5. Existing simple reranker and non-reranker paths remain compatible.
6. Focused Python tests and a smoke flow prove the combined public behavior.

## Candidate Approaches

### Approach 1: Python-Only Parity

Keep all reranking in the Python executor and teach it to reproduce core combination semantics for:

- `query_by_id`
- `group_by`
- `order_by`

Pros:

- one execution model for built-in and custom rerankers
- no need to distinguish built-in vs custom reranker at dispatch time

Cons:

- duplicates more query semantics in Python
- higher risk of drift from core behavior
- requires Python reimplementation of final ordering/grouping semantics

### Approach 2: Hybrid Core-First Dispatch

Use core-native execution for built-in rerankers when the context is fully representable in core, and keep Python fan-out only for custom rerankers.

Pros:

- leverages the engine's existing combined-query semantics
- removes facade-only blockers with less duplicated logic
- keeps Python custom reranker support intact

Cons:

- execution model differs between built-in and custom rerankers
- contract must be explicit so behavior is understandable

### Approach 3: Surface Cleanup Only

Expose everything publicly, but continue to reject hard combinations in the executor with cleaner errors.

Pros:

- smallest change set
- lowest implementation risk

Cons:

- does not close the real parity gap
- keeps core/facade mismatch alive

## Recommendation

Use **Approach 2: Hybrid Core-First Dispatch**.

This is the best trade-off for the current codebase because:

- core already knows how to combine reranker, `query_by_id`, `group_by`, and `order_by`
- the Python executor is the actual bottleneck
- built-in rerankers are the parity target that matters most for zvec-style public API expectations

The design should keep custom rerankers working on the current Python fan-out path, but should stop forcing built-in rerankers through a narrower Python-only pipeline when core already has the right semantics.

## Architecture

### 1. Public Contract

The public Python API should continue to use the existing entry points:

- `QueryContext(...)`
- `Collection.query(query_context=...)`
- `Collection.query(...kwargs...)`

The change is not a new API surface.  
The change is that executor dispatch becomes capability-aware rather than facade-limited.

### 2. Executor Dispatch Model

The executor should distinguish two cases:

#### Built-in rerankers

When the reranker is a built-in HannsDB reranker already representable in core:

- pass the full query context to `collection.query_context(...)`
- let core perform recall, reranking, ordering, and grouping with native semantics

This is the primary parity path.

#### Custom rerankers

When the reranker is a user-defined Python reranker:

- retain the current Python fan-out model
- keep the contract explicit for shapes the Python fan-out path still cannot faithfully emulate

This preserves extensibility without forcing the parity-critical built-in path through weaker semantics.

### 3. Combination Semantics

The executor should stop applying Python-only `NotImplementedError` guards for combinations that are already supported by core on the built-in reranker path.

The key combined shapes to support are:

- vector queries + built-in reranker + `query_by_id`
- vector queries + built-in reranker + `group_by`
- vector queries + built-in reranker + `order_by`
- legacy kwargs path for the same combinations

The core engine remains authoritative for validation and unsupported-shape errors.

### 4. Error Handling

Public error behavior should become simpler:

- if core supports the shape, the query should run
- if core rejects the shape, the public API should surface that error
- the Python executor should only raise facade-level `NotImplementedError` where the limitation is truly Python-specific

That keeps the product contract honest.

### 5. Testing Strategy

Testing needs to cover three layers.

#### Query Executor

- built-in reranker + `query_by_id`
- built-in reranker + `group_by`
- built-in reranker + `order_by`
- compatibility for existing custom reranker tests

#### Collection Facade

- `QueryContext(...)` path for combined shapes
- legacy kwargs path for the same shapes
- stable projection/output-field behavior

#### Smoke

- a single Python flow that exercises a built-in reranker with at least one previously blocked combination

## Risks

### Risk 1: Built-in and custom reranker paths diverge too much

Mitigation:

- keep the dispatch rule explicit and narrow
- document that built-in rerankers use core-native semantics

### Risk 2: Existing tests encode facade bugs as expected behavior

Mitigation:

- convert tests that currently assert `NotImplementedError` into positive parity tests when the engine supports the shape
- keep custom-reranker tests separate from built-in parity tests

### Risk 3: Legacy kwargs path drifts from `QueryContext`

Mitigation:

- pair every new `QueryContext` parity test with a kwargs-path equivalent where the surface is meant to match

## Plan Boundary

The implementation plan that follows from this design should cover only:

1. built-in reranker dispatch alignment
2. facade/query executor restriction cleanup
3. combined-shape regression tests
4. a lightweight smoke path
