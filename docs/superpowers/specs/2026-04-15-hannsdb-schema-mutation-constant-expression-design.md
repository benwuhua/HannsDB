# HannsDB Schema Mutation Constant-Expression Design

**Date:** 2026-04-15

## Goal
Advance HannsDB's schema-mutation parity against zvec by adding a more zvec-like public contract for schema mutation while preserving honest boundaries. This slice should expose richer `add_column(...)` / `alter_column(...)` shapes publicly, but only make one small execution subset real: **constant-expression backfill for `add_column(...)`**.

## Why This Slice
Recent parity work has already narrowed several query/public-surface gaps. The remaining `query/schema/PK` gap is now more concentrated in schema-mutation depth.

The user explicitly chose these boundaries for the next lane:
- keep **honest parity**
- prefer **strict explicit errors** over silent downgrade
- do **contract shape + a small real execution subset**
- make the real execution subset be **`add_column(..., expression=...)`**
- keep that subset at **constant expressions only**

So this lane is not “full schema mutation parity.” It is a bounded step toward it.

## Scope
This design covers one focused vertical slice:
1. public contract shape for:
   - `add_column(field_schema, expression="", option=AddColumnOption())`
   - `alter_column(old_name, new_name=None, field_schema=None, option=AlterColumnOption())`
2. real execution support for a **constant-expression subset** of `add_column(..., expression=...)`
3. explicit error boundaries for unsupported schema-mutation shapes
4. tests and smoke coverage for the new contract and constant-backfill behavior

## Out of Scope
This design does **not** include:
- general expression parsing or execution
- references to existing fields in expressions
- computed expressions (`a + b`, function calls, conditionals, etc.)
- `alter_column(..., field_schema=...)` migrations
- vector-column add/alter mutation
- storage/runtime changes
- benchmark work (local or remote)

## Success Criteria
The slice is complete only if all of the following are true:
1. The public Python contract accepts the richer zvec-like mutation shapes.
2. `add_column(..., expression=...)` supports **constant expressions only**.
3. Supported constants are validated against the destination field type.
4. Unsupported expression shapes fail explicitly and clearly.
5. `alter_column(..., field_schema=...)` remains exposed as shape but fails explicitly rather than pretending to work.
6. Existing supported scalar DDL behavior remains compatible.
7. Tests prove both the honest supported subset and the explicit unsupported boundary.

## Recommended Approach
Use a **shape-first, constant-subset execution** approach:
- make the public API look more like zvec where that is honest
- keep only the smallest execution subset real
- reject everything else explicitly

This is the right trade-off because it creates genuine progress without dragging HannsDB core into a half-built expression engine or fake migration story.

## Public Contract

### `add_column(...)`
Target public shape:

```python
collection.add_column(
    field_schema,
    expression="",
    option=AddColumnOption(),
)
```

### Empty-string vs omitted-expression rule
`expression=""` remains the **legacy no-expression / default** form in this lane. It does **not** mean “backfill with empty string.”

To avoid ambiguity, the supported constant-expression subset requires explicit literal syntax for strings. So:
- `expression=""` => no expression provided / existing legacy behavior
- `expression='""'` => explicit empty-string constant
- `expression='"hello"'` => explicit string constant

### `alter_column(...)`
Target public shape:

```python
collection.alter_column(
    old_name,
    new_name=None,
    field_schema=None,
    option=AlterColumnOption(),
)
```

The second shape is primarily contract alignment in this lane; real execution stays limited to rename-only behavior plus explicit failure for migration-style requests.

## Supported Execution Subset
This lane should make **only one real semantic expansion** executable:

### Constant-expression backfill for `add_column(...)`
Supported constants:
- string literal
- integer literal
- floating-point literal
- boolean literal (`true` / `false`)
- `null` only when the destination field is nullable

### Literal syntax rules
The supported subset should use one simple, explicit grammar shared by facade and bridge:
- **string literals**: double-quoted only, e.g. `"hello"`, `""`
- **integers**: base-10 signed/unsigned decimal forms accepted only when they fit the destination type
- **floats**: simple decimal floating literals only (for example `1.0`, `0.25`, `-3.5`); scientific notation is out of scope for this lane
- **signed numbers**: leading `-` is allowed; leading `+` is not part of the supported grammar in this lane
- **booleans**: lowercase `true` / `false` only
- **null**: lowercase `null` only
- surrounding whitespace may be trimmed before classification
- **string escapes/embedded quotes**: out of scope for this lane unless the implementation can support them identically in facade and bridge; otherwise reject them explicitly

Anything outside this narrow grammar is unsupported in this lane and should fail explicitly rather than being heuristically parsed.

### Type rules
- string columns accept only string constants
- integer/float columns accept numeric constants compatible with the destination type
- bool columns accept only boolean constants
- nullable columns may accept `null`
- non-nullable columns reject `null`

## Explicit Unsupported Shapes
The following must fail clearly rather than degrade silently:
- references to existing fields
- arithmetic or computed expressions
- function-style expressions
- conditional expressions
- vector-typed add-column input
- array semantics beyond what HannsDB already supports honestly
- `alter_column(..., field_schema=...)` migration requests
- any schema-mutation semantics that require full data rewrite/migration logic beyond the current scalar rename path

## Error Strategy
- **Shape is valid but semantics are not yet supported** → `NotImplementedError`
- **Value/type/nullable mismatch** → `ValueError`
- **Core execution/path failures** → preserve current native/core error mapping

This keeps the user-facing boundary clear and consistent with the project’s honesty rule.

## Architecture

### Preferred implementation strategy
**Python facade + PyO3 bridge parse and normalize the constant-expression subset, while core receives a small, already-classified backfill instruction/value rather than a general expression language.**

Why this is preferred:
- it keeps scope bounded
- it avoids turning core into a general expression engine
- it matches the user’s “small honest subset” requirement
- it is easier to test and to keep explicit at the public boundary

### Responsibilities by layer

#### Python/public surface
Files likely involved:
- `crates/hannsdb-py/python/hannsdb/model/collection.py`
- `crates/hannsdb-py/python/hannsdb/model/param/`
- top-level/model exports

Responsibilities:
- expose richer schema-mutation contract shape
- normalize inputs
- reject clearly unsupported public shapes before they pretend to work

#### PyO3/native bridge
Files likely involved:
- `crates/hannsdb-py/src/lib.rs`

Responsibilities:
- bridge `field_schema + expression + option`
- recognize/parse the constant-expression subset
- reject unsupported non-constant expression shapes explicitly

#### Core
Only change core if needed for the minimal constant-backfill execution path.

Core should **not** become a general expression interpreter in this slice.
If core needs a new interface, it should be a small “backfill with constant value” path, not “evaluate arbitrary expression string.”

## Minimal Execution Slices

### Slice A — contract shape
- richer public signature for `add_column(...)`
- richer public signature for `alter_column(...)`
- honest explicit errors still allowed

### Slice B — constant-expression execution
- real support for constant backfill in `add_column(..., expression=...)`
- strict type/nullability validation

### Slice C — tests + smoke
- positive tests for supported constant expressions
- negative tests for unsupported expression shapes
- smoke flow that proves end-to-end behavior

## Candidate Approaches Considered

### Option A — facade/bridge parse constants, core receives normalized constant value (**recommended**)
**Pros**
- safest scope
- strongest honesty boundary
- easiest to test

**Cons**
- future generalized expression support would likely require a later contract extension

### Option B — core parses constant-expression strings directly
**Pros**
- expression semantics live closer to execution
- may be a more natural foundation for future expansion

**Cons**
- more likely to sprawl into a partial expression engine
- more risk than needed for this slice

### Option C — contract shape only, no real expression execution
**Pros**
- smallest change

**Cons**
- does not satisfy the chosen requirement of “shape + one real execution subset”

## Recommendation
Choose **Option A**.

It is the only option that simultaneously:
- keeps the lane bounded
- provides real forward motion
- preserves honest parity
- avoids overcommitting the core

## Testing Strategy

### Public-surface tests
- import/export of the richer mutation contract
- supported constant-expression shapes accepted
- unsupported shapes explicitly rejected

### Facade/native tests
- `add_column(..., expression=<constant>)` succeeds for supported scalar types
- nullability/type mismatches fail correctly
- `alter_column(..., field_schema=...)` still fails explicitly

### End-to-end smoke
One lightweight flow should prove:
1. create collection
2. insert documents
3. `add_column(..., expression=<constant>)`
4. fetch/query documents
5. verify backfilled values are present and correct

## Risks

### Risk 1: Scope slips into a partial expression engine
**Mitigation:** keep execution subset strictly to single constants only.

### Risk 2: Public shape suggests more support than exists
**Mitigation:** explicit `NotImplementedError` for unsupported shapes.

### Risk 3: Type/nullability validation becomes inconsistent across layers
**Mitigation:** centralize normalization/validation logic and lock it with tests.

## Bottom Line
The right next schema-mutation lane is:

> **richer zvec-like mutation contract shape + real constant-expression backfill for `add_column(...)`, with strict explicit failure for everything beyond that subset**

This is the smallest honest step that both improves parity and preserves HannsDB’s current engineering discipline.
