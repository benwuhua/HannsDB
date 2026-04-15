# HannsDB Schema Mutation Widening-Migration Design

**Date:** 2026-04-15

## Goal
Advance HannsDB's schema-mutation parity by turning `alter_column(..., field_schema=...)` from a contract-only surface into a **small, real migration capability**.

This slice keeps the richer zvec-like public contract, but only makes one narrow subset executable:
- `int32 -> int64`
- `uint32 -> uint64`
- `float -> float64`

Everything else remains explicitly rejected.

## Why This Slice
The latest public/query/schema work already landed:
- honest `InvertIndexParam`
- custom reranker + `query_by_id` / `group_by`
- constant-expression backfill for `add_column(...)`

That means the remaining schema-mutation gap is now more focused:
- richer contract shape already exists
- the next honest step is not more surface, but a **real migration subset** behind `field_schema`

## Scope
This design covers one bounded vertical slice:
1. keep the richer public `alter_column(...)` contract
2. add real execution for a widening-only migration subset
3. preserve rename-only behavior
4. persist the migration semantics through WAL / recovery
5. add tests proving both the supported subset and the explicit unsupported boundary

## Out of Scope
This design does **not** include:
- string/number conversion
- any narrowing conversion
- mixed rename + type migration in one step
- nullable / array flag migration
- vector-column migration
- a general migration engine
- benchmark work

## Success Criteria
The slice is complete only if all of the following are true:
1. `alter_column(..., field_schema=...)` executes for the widening-only subset.
2. Existing rows are migrated to the widened type, not just schema metadata rewritten.
3. Rename-only behavior stays correct.
4. Unsupported migrations fail explicitly and clearly.
5. WAL / recovery preserves the widened result after reopen.
6. Tests prove both the supported subset and rejected boundary.

## Public Contract
The public contract remains:

```python
collection.alter_column(
    old_name,
    new_name=None,
    field_schema=None,
    option=AlterColumnOption(),
)
```

## Supported Migration Subset
This lane makes only the following type migrations real:
- `int32 -> int64`
- `uint32 -> uint64`
- `float -> float64`

Additional constraints:
- scalar fields only
- column name must remain unchanged for the migration path in this lane
- one migration at a time

## Explicit Unsupported Shapes
The following must fail explicitly:
- `string -> int64`
- `int64 -> string`
- `float64 -> int64`
- any narrowing conversion
- rename + type migration in the same operation
- `nullable` / `array` changes
- vector-field migration
- any `field_schema` change outside the widening subset

## Error Strategy
- **Shape is valid but not in the supported subset** → `NotImplementedError`
- **Malformed input / illegal schema request** → `ValueError`
- **Core migration failure / replay failure** → keep the current core/native error mapping

## Recommended Approach
Use a **minimal core migration instruction**.

### Why this approach
The widening subset should be represented explicitly in core rather than simulated in Python.
That gives the cleanest semantics for:
- data rewrite behavior
- WAL persistence
- replay correctness
- future extension to other migration subsets

## Architecture

### 1. Python/public surface
Files likely involved:
- `crates/hannsdb-py/python/hannsdb/model/collection.py`
- Python mutation tests

Responsibilities:
- normalize richer `alter_column(...)` inputs
- distinguish rename-only vs widening-migration requests
- reject unsupported shapes before they pretend to work

### 2. PyO3/native bridge
Files likely involved:
- `crates/hannsdb-py/src/lib.rs`

Responsibilities:
- accept richer `field_schema` migration requests
- classify them into:
  - rename-only
  - supported widening migration
  - unsupported migration
- pass only the minimal supported instruction to core

### 3. Core
Files likely involved:
- `crates/hannsdb-core/src/db.rs`
- `crates/hannsdb-core/src/wal.rs`
- `crates/hannsdb-core/src/storage/recovery.rs`

Responsibilities:
- introduce a minimal `AlterColumnMigration` instruction
- perform actual row-value widening for existing documents
- keep schema metadata and stored row values in sync
- replay the same migration semantics from WAL

## Minimal Internal Instruction
The recommended internal model is something like:
- rename-only
- widen-int32-to-int64
- widen-uint32-to-uint64
- widen-float-to-float64

This lane should **not** introduce a general field-conversion framework.

## Execution Semantics
For a supported widening migration:
1. validate the requested source/target field types
2. validate extra lane constraints (especially scalar-index state; see below)
3. append a WAL record carrying the minimal migration instruction
4. run the migration through one shared internal helper that rewrites live rows and updates schema metadata together
5. on replay, run the same internal migration path

### Crash / WAL ordering rule
This lane should treat the widening migration as a **WAL-first** operation, not a best-effort in-place rewrite followed by WAL append.

Required rule:
- if the migration starts, its WAL record must be appended **before** the durable data rewrite begins
- replay must be safe to apply through the same helper if a crash happens after WAL append but before the on-disk rewrite fully finishes

In other words, this slice must prefer replay-safe ordering over trying to look atomic without a real transaction layer.

## Rename Rule
Rename-only remains supported exactly as before.

Rename + type migration together is explicitly out of scope for this lane.
That keeps the migration semantics small and auditable.

## Indexed-Column Rule
This lane should be explicit about scalar indexes on the migrating column.

Recommended rule for this slice:
- if the target scalar column already has a scalar index descriptor / built scalar index, **reject the widening migration explicitly**
- do **not** silently rebuild, coerce, or invalidate scalar indexes in this same lane

Why:
- rebuilding/invalidation semantics would widen this slice into a broader migration + indexing project
- explicit rejection is more honest than pretending existing scalar indexes remain valid across type migration

A later dedicated lane can decide whether indexed-column widening should rebuild indexes, invalidate descriptors, or follow some other policy.

## Candidate Approaches Considered

### Option A — minimal core migration instruction (**recommended**)
**Pros**
- clean WAL / replay semantics
- true execution capability, not facade emulation
- future extension path remains open

**Cons**
- touches core and WAL, not just facade

### Option B — emulate migration in Python using old primitives
**Pros**
- superficially smaller core diff

**Cons**
- poor WAL/recovery story
- easy to become dishonest / fragile
- not suitable for a trusted data-plane contract

### Option C — keep contract-only surface
**Pros**
- smallest change

**Cons**
- does not satisfy the requirement to add a real migration subset

## Recommendation
Choose **Option A**.

It is the smallest approach that gives HannsDB a real, replay-safe migration capability without pretending to have a full migration engine.

## Testing Strategy

### Python/facade
- supported widening migrations succeed
- unsupported migrations fail explicitly
- rename-only still works

### Core
- existing rows are widened correctly
- fetched/query-visible values use the widened type
- unsupported migrations error cleanly

### WAL / recovery
- widening migration survives reopen/replay
- replay uses the same migration path, not a forked interpretation

## Risks

### Risk 1: Scope expands into a general migration engine
**Mitigation:** keep the internal instruction enum tiny and fixed to three widening cases.

### Risk 2: Schema metadata changes but row values do not
**Mitigation:** require tests that verify fetched old rows after migration.

### Risk 3: Replay path diverges from live execution path
**Mitigation:** WAL should store the same narrow migration instruction and replay through the same internal helper.

## Bottom Line
The next honest schema-mutation lane should be:

> **real widening-only `field_schema` migration for `alter_column(...)`, with strict explicit failure for every migration shape outside that tiny supported subset**

This is the smallest trustworthy step from “contract shape only” toward genuine schema-mutation depth.
