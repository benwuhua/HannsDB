# Delete By Filter Design

**Date:** 2026-04-09

**Goal:** Add a core-backed `delete_by_filter` path for HannsDB and expose it through `hannsdb-py`, while keeping the scope limited to current filter syntax and existing delete/WAL behavior.

## Context

HannsDB already has:

- segment-aware read/query paths
- segment-aware `delete(ids)` semantics
- a reusable filter parser for query-time filtering
- Python/native collection surfaces that currently expose `delete_by_filter(...)` as explicit unsupported behavior

That makes `delete_by_filter` the next smallest core-backed parity slice. The foundational risk from before was that delete semantics were still flat-layout-only; that has already been fixed by making `delete(ids)` operate on the segment-aware runtime view.

## Chosen Scope

This slice will implement only:

1. `hannsdb-core` support for `delete_by_filter(collection, filter) -> usize`
2. `hannsdb-py` support for native and pure-Python `Collection.delete_by_filter(...)`
3. Regression tests in core and Python

This slice will **not** implement:

- daemon transport / HTTP routes
- column DDL
- new filter syntax
- new WAL record types
- mutation planner refactors

## API Shape

### Core

Add:

- `HannsDb::delete_by_filter(&mut self, collection: &str, filter: &str) -> io::Result<usize>`

Semantics:

- parse the filter with the existing `parse_filter(...)`
- resolve matching documents against the segment-aware latest-live view
- collect matching external ids
- delegate actual mutation to the existing segment-aware `delete(ids)` path
- return the number of newly deleted live documents

### Python

Replace the current `NotImplementedError` surface with real delegation:

- native `_native.Collection.delete_by_filter(filter: str) -> int`
- pure `hannsdb.Collection.delete_by_filter(filter: str) -> int`

Semantics match core:

- invalid filter expressions still surface as errors from core
- return value is the count of newly deleted live documents

## Execution Model

Implementation should stay intentionally simple:

1. Reuse the existing filter parser and matching logic.
2. Evaluate the filter only on the segment-aware latest-live view: at most one row per external id may be considered, and a newer tombstoned row must still shadow any older matching row.
3. Reuse the existing `delete(ids)` write path for actual tombstone mutation and WAL appends.

This means `delete_by_filter` does **not** need its own mutation primitive or WAL record. The operation can be modeled as:

- find current matching ids in the latest-live view, with newest-row-wins semantics
- call `delete(ids)` once with that deduplicated id set

## Correctness Requirements

- Only latest-live documents may be considered for deletion.
- Documents already shadowed by newer versions must not be re-deleted via older rows.
- Documents already tombstoned must not contribute to the return count.
- A newer already-tombstoned row must still shadow an older row that would otherwise match the filter.
- Multi-segment layouts must behave the same as segment-aware latest-live filtering semantics.
- Legacy single-segment collections must keep working.
- Invalid filters must still fail instead of silently deleting nothing.
- A valid filter with no matches must return `0`.
- Repeating the same `delete_by_filter` after the first delete must return `0`.
- WAL behavior stays unchanged: one `delete_by_filter` call delegates to one `delete(ids)` call and therefore emits the same existing `Delete { ids }` record shape, with no new WAL variant.

## Non-Goals / Constraints

- Do not add daemon support in this slice.
- Do not change filter grammar beyond what `parse_filter(...)` already supports.
- Do not add `delete_by_filter` to typed query/planner infrastructure.
- Do not optimize for large filtered deletes yet; scan-based selection is acceptable for this slice.

## Validation

### Core tests

Add regression coverage for:

- single-segment `delete_by_filter`
- multi-segment `delete_by_filter` respecting latest-live shadowing
- multi-segment `delete_by_filter` where a newer already-tombstoned row shadows an older matching row
- valid filter with no matches returning `0`
- repeated `delete_by_filter` returning `0` on the second call
- invalid filter errors
- reopen / WAL-replay behavior remaining consistent with the existing `delete(ids)` contract

### Python tests

Add regression coverage for:

- real collection `delete_by_filter(...)` positive behavior, including integer return count
- native collection `delete_by_filter(...)` positive behavior, including integer return count
- supported calls no longer raising `NotImplementedError`
- invalid filter propagation

## Risks

- The operation is still scan-based and can be expensive on large collections; acceptable for this slice.
- Because it reuses `delete(ids)`, WAL behavior remains stable but not specially compact for filter deletes; acceptable for this slice.
