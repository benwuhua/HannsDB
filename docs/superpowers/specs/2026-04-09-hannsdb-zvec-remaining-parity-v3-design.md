# HannsDB zvec Remaining Parity V3 Design

**Date:** 2026-04-09

**Goal:** Define the next implementation wave after the V2 parity work so HannsDB closes the remaining engine-native gaps against zvec without re-opening already-shipped API surface work.

## Context

Recent V2 work on `main` already closed a large amount of surface parity:

- field-oriented schema metadata
- typed query transport and Python collection facade
- `include_vector`, `group_by`, field-aware `query_by_id`
- secondary-vector typed fast paths
- segment-aware delete and `delete_by_filter`
- daemon transport for the capabilities already landed

The remaining gaps are now concentrated in deeper engine behavior rather than missing wrappers.

## Remaining Gaps To Plan Around

The plan should focus on these unresolved areas:

1. The core document/storage model is still not fully field-uniform because `Document` retains a `vector + vectors` split.
2. `update` and schema mutation (`add_column`, `drop_column`, `alter_column`) are not native core capabilities.
3. Typed query planning is still intentionally narrow compared with zvec:
   - richer filter grammar is missing
   - ordering/projection support is limited
   - reranking remains largely Python-owned
   - mixed multi-vector query shapes are only partially native
4. Per-field indexed runtime is still asymmetric:
   - primary vector paths are persisted and privileged
   - secondary vector ANN paths still rely on lazy/runtime assembly in some cases
5. The type system is still much narrower than zvec, especially for post-V2 missing scalar and vector metadata types.
6. Sparse vector support does not exist yet.
7. Daemon/runtime depth should only be revisited after the core gaps above are closed.

## Scope

The V3 plan should cover only the remaining substantive parity work above.

The plan should **not**:

- re-plan already completed V2 API surface work
- broaden into unrelated benchmarking or performance projects
- attempt a full zvec SQL-engine clone
- mix daemon/runtime refactors into early tasks unless the earlier core work makes them necessary

## Decomposition Rules

- Prioritize core-native correctness before additional wrappers.
- Prefer field-uniform runtime behavior over more primary-vector special casing.
- Keep tasks small, TDD-first, and individually committable.
- Delay daemon/runtime locking refactors until the core feature gaps are closed.
- Treat sparse vectors as a dedicated workstream, not an incidental extension of dense-vector tasks.

## Validation Expectations

The resulting plan should explicitly include:

- failing tests before each substantive implementation task
- exact verification commands and expected pass/fail outcomes
- file-level ownership for each task
- commit checkpoints after each task
- chunk boundaries suitable for plan-document review
