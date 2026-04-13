# HannsDB Schema Mutation Surface Parity Design

**Date:** 2026-04-12

**Goal:** Define the next executable Python/public parity slice by aligning HannsDB's schema-mutation surface with the zvec-style contract shape without expanding into a storage or migration-engine project.

## Context

Recent parity work changed the shape of the remaining gap:

- string PK is now usable through the public Python API
- `query_by_id`, `order_by`, and built-in reranker combinations are now exposed through the public query surface
- core already supports basic scalar column DDL:
  - `add_column`
  - `drop_column`
  - `alter_column` rename
- core tests already cover:
  - add-column schema extension
  - duplicate-column rejection
  - drop-column removal
  - rename success and conflict cases
  - WAL replay for add/drop/rename

The next blocker is no longer missing primitive DDL operations.

The blocker is that HannsDB's Python/public schema-mutation contract is still much thinner than zvec's:

- `Collection.add_column(...)` currently accepts only `field_name + data_type + nullable + array`
- `Collection.alter_column(...)` currently accepts only `field_name + new_name`
- there is no public `AddColumnOption`
- there is no public `AlterColumnOption`
- there is no public `expression` hook on add-column
- there is no public `field_schema` hook on alter-column

That means HannsDB looks less capable than it really is, even where the underlying engine already has enough primitive DDL support to justify a richer public contract.

## Problem Statement

Compared with zvec, HannsDB's schema-mutation API currently has two problems:

1. **The public contract is too narrow.**
   Users cannot write zvec-style code that passes a `FieldSchema`, expression string, or option object through the HannsDB facade.

2. **The contract shape and true engine capability are conflated.**
   Some richer zvec-style calls are not supported because HannsDB lacks the engine behavior today.
   Other calls are absent simply because the facade never exposed them.

If HannsDB keeps the current simplified signatures, every later DDL enhancement will require another public API break or another compatibility layer. The parity-friendly move is to align the contract shape now and make unsupported semantics explicit.

## Scope

This design covers one focused Python/public parity project:

1. expose zvec-shaped schema-mutation entry points in the HannsDB Python facade
2. add public option types for schema mutation
3. accept `FieldSchema` objects as the public DDL input type
4. preserve current core-backed behavior for supported scalar DDL operations
5. make unsupported richer DDL semantics fail explicitly and consistently
6. add regression and smoke coverage for the public contract

## Out of Scope

This design does **not** include:

- expression evaluation or backfill execution in Python
- core support for computed add-column population
- core support for full schema migration via `alter_column(field_schema=...)`
- vector-column DDL
- storage/runtime redesign
- WAL format redesign
- query/index/type parity outside schema mutation

Those remain separate workstreams.

## Success Criteria

This parity slice is complete only if all of the following are true:

1. Python users can call `Collection.add_column(field_schema=..., expression=..., option=...)`.
2. Python users can call `Collection.alter_column(old_name=..., new_name=..., field_schema=..., option=...)`.
3. `AddColumnOption` and `AlterColumnOption` are importable from the public HannsDB package and expose their current supported properties.
4. Supported scalar DDL behavior continues to work through the richer public API shape.
5. Unsupported richer semantics fail with stable, intention-revealing errors instead of silently ignoring inputs.
6. Legacy simplified call forms continue to work or are transparently normalized through the richer contract.

## Candidate Approaches

### Approach 1: Compatibility Facade First

Align the public Python API to the zvec-style contract now, but only execute the semantics HannsDB actually supports today.

Behavior:

- `add_column(field_schema, expression="", option=...)`
  - supports scalar `FieldSchema`
  - supports empty expression only
- `alter_column(old_name, new_name=None, field_schema=None, option=...)`
  - supports rename
  - exposes `field_schema` in the public contract
  - rejects unsupported migration semantics explicitly

Pros:

- closes the public parity gap without pretending the engine does more than it does
- avoids future API churn when core gains richer DDL support
- keeps implementation contained to facade, PyO3 bridge, and tests

Cons:

- some zvec-style calls will still fail at runtime
- contract and engine capability remain intentionally asymmetric for now

### Approach 2: Python-Layer Semantic Emulation

Expose the richer contract and also emulate missing behavior in Python, such as expression backfill or partial schema migration, by composing existing primitives.

Pros:

- looks closer to zvec behavior immediately

Cons:

- turns the Python facade into a partial execution engine
- creates correctness and performance risk
- duplicates logic that should live in core if it exists at all

### Approach 3: Core-First Parity

Delay public contract alignment until core can support expression backfill and richer alter-column semantics directly.

Pros:

- cleanest semantic story

Cons:

- leaves the current public API gap open
- expands a Python/public parity task into a core migration project

## Recommendation

Use **Approach 1: Compatibility Facade First**.

This is the right trade-off for the current repo because:

- the active parity track is focused on Python/public API maturity
- core already has enough scalar DDL behavior to justify the richer contract shape
- unsupported richer semantics can be represented honestly with explicit errors
- this preserves a stable forward path when core later adds expression backfill or schema migration support

The key design principle is:

**Align the public contract now. Do not fake engine behavior.**

## Architecture

### 1. Public API Shape

The Python facade should move from simplified scalar-argument DDL methods to zvec-shaped methods:

- `Collection.add_column(field_schema, expression="", option=AddColumnOption())`
- `Collection.drop_column(field_name)`
- `Collection.alter_column(old_name, new_name=None, field_schema=None, option=AlterColumnOption())`

The facade may continue to normalize older simplified call forms internally for compatibility, but the canonical documented contract should be the schema-object form.

### 2. Public Schema Input

`FieldSchema` should become the canonical scalar DDL payload type.

This keeps the API consistent with:

- collection schema declaration
- zvec-style DDL shape
- future richer field-level behavior

`VectorSchema` is not part of this parity slice. If a caller attempts vector-column DDL through this path, the facade should reject it explicitly.

### 3. Mutation Option Types

HannsDB should expose lightweight `AddColumnOption` and `AlterColumnOption` public types.

For this slice, they only need to represent the currently supported surface, which is expected to be minimal. If the only stable property is `concurrency`, expose just that. If the current implementation ignores the option value, that should be acceptable as long as:

- the object exists publicly
- the contract is stable
- unsupported option effects are not silently misrepresented

These option types should be importable from:

- `hannsdb`
- `hannsdb.model.param`

The export pattern should match existing public param objects.

### 4. Supported Semantics

The richer public contract should map to current engine behavior as follows.

#### `add_column`

Supported:

- scalar `FieldSchema`
- `expression == ""`
- nullable/array flags that current core already understands

Rejected explicitly:

- non-empty `expression`
- vector-field add-column
- unsupported data types

#### `alter_column`

Supported:

- rename-only operations
- no-op normalization where `new_name` is omitted and no richer mutation is requested

Rejected explicitly:

- `field_schema` supplied for an actual schema migration
- option-driven behavior that core does not support
- ambiguous or contradictory rename/schema requests

`drop_column` remains simple and should not change semantically.

### 5. Compatibility Model

There are two compatibility targets in this slice.

#### Existing HannsDB callers

Current simplified forms should keep working where practical, for example:

- `add_column("session_id", "string", False, False)`
- `alter_column("old_name", "new_name")`

The facade can normalize these calls to the richer contract shape internally.

#### New parity-style callers

New callers should be able to write:

- `add_column(FieldSchema(...), expression="", option=AddColumnOption(...))`
- `alter_column("old_name", new_name="new_name", field_schema=None, option=AlterColumnOption(...))`

This allows new usage to converge on the future-facing contract without breaking existing code immediately.

### 6. Error Handling

This slice should prefer explicit contract errors over best-effort coercion.

Examples that should produce clear failures:

- `add_column` called with a non-`FieldSchema` object that cannot be normalized
- `add_column(..., expression="score * 2")` before engine support exists
- `add_column` with vector schema or `vector_fp32`
- `alter_column(..., field_schema=FieldSchema(...))` when schema migration is not supported
- malformed option objects

Errors do not need to exactly mirror zvec wording. They do need to make the limit obvious.

## Data Flow

The intended public flow for supported operations is:

1. caller constructs `FieldSchema` and optional mutation option object
2. Python facade normalizes legacy or canonical inputs
3. PyO3 bridge validates supported vs unsupported mutation shapes
4. supported scalar DDL calls are translated to current core primitives
5. collection metadata is refreshed and reflected back through the facade

Unsupported shapes should fail before pretending to execute.

## Testing Strategy

Testing should cover three layers.

### Python Public Surface

- public import and construction of `AddColumnOption`
- public import and construction of `AlterColumnOption`
- `Collection.add_column(FieldSchema(...))`
- `Collection.alter_column(..., new_name=..., option=...)`
- legacy simplified call normalization

### Python Facade Behavior

- scalar add-column still updates collection schema
- rename still updates collection schema
- explicit failures for non-empty expression
- explicit failures for vector-field add-column
- explicit failures for `field_schema` migration requests

### End-to-End Smoke

- create collection
- add scalar column via `FieldSchema`
- rename scalar column via richer signature
- reopen and verify schema reflects the mutation path

## Risks

### Risk 1: Public contract grows faster than engine support

Mitigation:

- fail unsupported semantics explicitly
- document supported vs rejected shapes in tests and spec

### Risk 2: Legacy callers regress during signature expansion

Mitigation:

- keep compatibility normalization in the facade
- preserve focused legacy tests alongside parity tests

### Risk 3: Option types become misleading stubs

Mitigation:

- expose only stable properties
- do not imply unsupported execution semantics
- make bridge-level validation reject unsupported effects clearly

## Implementation Notes

The main files likely involved are:

- `crates/hannsdb-py/python/hannsdb/model/collection.py`
- `crates/hannsdb-py/python/hannsdb/model/param/__init__.py`
- `crates/hannsdb-py/python/hannsdb/__init__.py`
- `crates/hannsdb-py/src/lib.rs`
- `crates/hannsdb-py/tests/test_collection_facade.py`
- a new focused Python surface test file for schema-mutation contract coverage

This should stay a facade/PyO3/test project. If the implementation starts requiring expression execution or true schema migration logic in core, the scope has escaped this spec and should be split into a separate design.
