# Segment-Aware Delete Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite core `delete(ids)` so it mutates the segment-aware runtime layout instead of only legacy flat-layout files.

**Architecture:** Keep the public API and WAL format unchanged. Rework `delete_internal(...)` to resolve deletes through `SegmentManager` + per-segment tombstones, matching the same newest-row shadowing semantics already used by read/query paths.

**Tech Stack:** Rust, `hannsdb-core`, segment metadata/tombstone files, targeted cargo tests.

---

## Chunk 1: Segment-Aware Delete Core

### Task 1: Lock behavior with failing tests

**Files:**
- Modify: `crates/hannsdb-core/tests/collection_api.rs`
- Modify: `crates/hannsdb-core/tests/wal_recovery.rs`

- [ ] **Step 1: Write the failing collection-api test**

Add a test that:
- creates a collection
- rewrites it into a two-segment layout
- deletes ids that are visible in the active segment, shadowed in an immutable segment, and already tombstoned in the active segment
- asserts only the newest visible rows are affected

- [ ] **Step 2: Run the targeted collection-api test to verify it fails**

Run: `cargo test -p hannsdb-core --test collection_api collection_api_delete_uses_segment_aware_runtime_layout -- --nocapture`
Expected: FAIL because `delete_internal(...)` still mutates only flat-layout tombstones.

- [ ] **Step 3: Write the failing WAL-recovery test**

Add a test that:
- creates a two-segment layout
- performs `delete(ids)`
- simulates a crash by removing mutated tombstone metadata
- reopens the database
- asserts replay restores the correct delete outcome by routing through the segment-aware delete path

- [ ] **Step 4: Run the targeted wal-recovery test to verify it fails**

Run: `cargo test -p hannsdb-core --test wal_recovery wal_recovery_replays_segment_aware_delete_outcome -- --nocapture`
Expected: FAIL because replay still routes through flat-layout delete behavior.

### Task 2: Implement minimal segment-aware delete

**Files:**
- Modify: `crates/hannsdb-core/src/db.rs`
- Test: `crates/hannsdb-core/tests/collection_api.rs`
- Test: `crates/hannsdb-core/tests/wal_recovery.rs`

- [ ] **Step 1: Add the smallest helper needed to resolve visible rows per segment**

Use the existing segment ordering from `SegmentManager::segment_paths()` and the same newest-row shadowing semantics already used by read helpers.

- [ ] **Step 2: Rewrite `delete_internal(...)`**

Requirements:
- iterate segment paths instead of top-level flat files
- tombstone only the newest visible row for each requested id
- skip ids already shadowed by a newer tombstone
- update changed segment metadata only
- keep WAL behavior and cache invalidation intact

- [ ] **Step 3: Run the targeted tests to verify they pass**

Run:
- `cargo test -p hannsdb-core --test collection_api collection_api_delete_uses_segment_aware_runtime_layout -- --nocapture`
- `cargo test -p hannsdb-core --test wal_recovery wal_recovery_replays_segment_aware_delete_outcome -- --nocapture`

Expected: PASS

- [ ] **Step 4: Run broader regression checks**

Run:
- `cargo test -p hannsdb-core --test collection_api -- --nocapture`
- `cargo test -p hannsdb-core --test wal_recovery -- --nocapture`

Expected: PASS

- [ ] **Step 5: Sanity-check the diff**

Run:
- `git diff --check`
- `git status --short`

Expected:
- no whitespace/errors from `git diff --check`
- only intended tracked changes before commit, then clean after commit

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-04-08-segment-aware-delete-design.md \
        docs/superpowers/plans/2026-04-08-segment-aware-delete.md \
        crates/hannsdb-core/src/db.rs \
        crates/hannsdb-core/tests/collection_api.rs \
        crates/hannsdb-core/tests/wal_recovery.rs
git commit -m "feat(core): make delete segment-aware"
```
