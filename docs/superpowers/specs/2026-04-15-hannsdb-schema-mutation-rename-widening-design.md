# HannsDB Schema Mutation Rename+Widening Design

**Date:** 2026-04-15

## Goal
Advance HannsDB's schema-mutation depth by extending the just-landed widening-only `field_schema` migration lane into a second honest subset:

> **rename + widening-only migration**

This means a single `alter_column(...)` call may now change both:
- the field name
- the field type

but only for the already-approved widening subset:
- `int32 -> int64`
- `uint32 -> uint64`
- `float -> float64`

## Why This Slice
The current widening-only lane already proves HannsDB can:
- classify supported `field_schema` migration requests honestly
- migrate stored values for a tiny supported subset
- persist and replay that migration through WAL / recovery

The next smallest real step is **not** a broader migration engine.
It is simply allowing the same tiny widening subset to happen together with a rename.

## Scope
This design covers one bounded vertical slice:
1. allow `rename + widening-only migration` in one `alter_column(...)`
2. keep rename-only behavior green
3. keep same-name widening behavior green
4. preserve WAL/recovery semantics for the combined rename+migration instruction
5. keep indexed-column migration explicitly rejected

## Out of Scope
This design does **not** include:
- any non-widening type conversion
- string/number conversion
- nullable / array flag changes
- vector-field migration
- indexed-column migration rebuild/invalidation flows
- a generalized migration descriptor or migration engine
- benchmark work

## Success Criteria
The slice is complete only if all of the following are true:
1. `alter_column(..., field_schema=...)` supports:
   - `score:int32 -> total_score:int64`
   - `count:uint32 -> doc_count:uint64`
   - `ratio:float -> ratio64:float64`
2. Existing live rows reflect both the **new name** and the **widened type** after migration.
3. Rename-only behavior still works unchanged.
4. Same-name widening-only behavior still works unchanged.
5. Indexed-column rename+migration remains explicitly rejected.
6. WAL/replay preserves the combined rename+widening result after reopen.
7. Unsupported shapes still fail explicitly and honestly.

## Public Contract
The public contract remains unchanged:

```python
collection.alter_column(
    old_name,
    new_name=None,
    field_schema=None,
    option=AlterColumnOption(),
)
```

This lane only changes which subset of requests is treated as executable.

### Rename target rule
For a supported rename+widening request:
- `field_schema.name` is the canonical rename target
- `new_name` may be omitted/empty for convenience
- if `new_name` is provided, it must exactly equal `field_schema.name`
- any mismatch between `new_name` and `field_schema.name` is an explicit error

## Supported Subset
A request is supported in this lane only when all of the following are true:
1. `old_name` is an existing scalar field
2. `field_schema` describes a scalar field
3. `field_schema.name != old_name`
4. `new_name` must either be omitted/empty or exactly equal to `field_schema.name`
5. source/target type pair is exactly one of:
   - `int32 -> int64`
   - `uint32 -> uint64`
   - `float -> float64`
5. nullable/array flags are unchanged
6. the source field does **not** already have a scalar index descriptor

## Explicit Unsupported Shapes
The following must continue to fail explicitly:
- rename + non-widening migration
- rename + narrowing
- rename + nullable change
- rename + array change
- rename + vector migration
- rename + indexed-column migration
- any broader schema migration outside this tiny subset

## Error Strategy
- **Shape is valid but not supported in this lane** → `NotImplementedError`
- **Malformed request / missing field / name conflict** → keep existing `ValueError` / core error mapping
- **Migration/replay failure** → keep current core/native error mapping

## Recommended Approach
Use **Option A: extend the existing tiny `AlterColumnMigration` instruction set**.

### Why this is preferred
This approach is the smallest honest extension of the widening-only lane:
- no new abstraction leap
- no general migration descriptor
- no hidden two-step semantics in user-facing logic
- WAL/replay stays explicit and auditable

## Architecture

### 1. Python facade
Files likely involved:
- `crates/hannsdb-py/python/hannsdb/model/collection.py`
- facade tests

Responsibilities:
- distinguish four categories cleanly:
  - rename-only
  - same-name widening-only
  - rename + widening-only
  - unsupported migration
- keep explicit rejections for all out-of-scope shapes

### 2. PyO3/native bridge
Files likely involved:
- `crates/hannsdb-py/src/lib.rs`

Responsibilities:
- classify rename+widening requests into the tiny supported subset
- pass a minimal instruction to core
- avoid broadening into a generic migration object

### 3. Core
Files likely involved:
- `crates/hannsdb-core/src/db.rs`
- `crates/hannsdb-core/src/wal.rs`
- `crates/hannsdb-core/src/storage/recovery.rs`

Responsibilities:
- extend the existing migration instruction enum with rename+widening variants
- update both schema metadata and live row field key/value together
- replay the same instruction through the same internal helper

## Execution Semantics
For supported rename+widening migration:
1. validate request shape and index-state constraints
2. append WAL record first (same WAL-first principle as the widening-only lane)
3. run one shared internal helper that is responsible for **both** key rename and value widening together
4. update schema metadata to `new_name + widened type` through that same logical path
5. replay uses the same helper after reopen

### Replay-safety rule
The combined rename+widen helper must be treated as **replay-safe / idempotent** for the supported subset.

Required rule:
- if a crash happens after WAL append but after only part of the row rewrite is persisted, replay must safely converge the collection to the final migrated state rather than duplicating or corrupting the field
- the implementation may achieve this through idempotent rewrite logic or an equivalent small-scope atomicity guarantee, but the contract for this slice is that partial-apply state must not leave replay semantics ambiguous

## Indexed-Column Rule
This lane keeps the same strict rule as widening-only migration:

- if the source field already has a scalar index descriptor, reject the rename+widening migration explicitly
- do not rebuild, invalidate, or remap indexes in this slice

That keeps the lane focused on schema/value migration only.

## Candidate Approaches Considered

### Option A — extend the current `AlterColumnMigration` enum with rename+widening variants (**recommended**)
**Pros**
- smallest delta from the current implementation
- clean WAL/replay semantics
- no hidden multi-step behavior

**Cons**
- enum grows a bit

### Option B — perform rename and widening as two internal sub-steps under one public call
**Pros**
- may reuse more existing rename code

**Cons**
- easier to create subtle ordering / replay ambiguity
- user-visible semantics become less explicit internally

### Option C — introduce a generalized migration descriptor
**Pros**
- more future-flexible

**Cons**
- too large a jump for this lane
- starts to pull the system toward a general migration engine

## Recommendation
Choose **Option A**.

It is the most conservative and honest extension of the current widening-only lane.

## Testing Strategy

### Python/facade
- rename+widening positive tests for all three supported pairs
- rename-only regression test remains green
- same-name widening regression test remains green
- unsupported rename+migration shapes fail explicitly
- indexed-column rename+migration fails explicitly

### Core
- live rows reflect both new key and widened value
- old key no longer appears after migration
- unsupported pairs still fail

### WAL / recovery
- rename+widening survives reopen/replay
- replay uses the same internal helper as live execution

## Risks

### Risk 1: rename-only and rename+migration paths diverge subtly
**Mitigation:** keep them adjacent in classification and test both explicitly.

### Risk 2: row values widen but field key rename is only partial
**Mitigation:** require tests that assert old key is gone and new key exists after migration.

### Risk 3: replay semantics fork from live execution
**Mitigation:** encode rename+widening as one explicit WAL instruction and replay through the same helper.

## Bottom Line
The next honest schema-mutation step should be:

> **rename + widening-only migration for the same tiny supported type subset, while continuing to reject indexed columns and every broader migration shape explicitly**

This is the smallest trustworthy extension of the widening-only lane without turning HannsDB into a general migration engine.
