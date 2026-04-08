# Segment-Aware Delete Design

**Date:** 2026-04-08

**Goal:** Make `HannsDb::delete(ids)` operate on the collection's segment-aware runtime view instead of the legacy flat-layout files, without introducing `delete_by_filter` or schema-mutation work.

## Context

`fetch_documents`, typed query execution, and the search-state loaders already read through `SegmentManager` + `VersionSet`. `delete_internal(...)` still mutates the top-level `segment.json` / `ids.bin` / `tombstones.json` paths only. That is correct for legacy single-segment layouts but wrong once a collection has been rewritten into a multi-segment runtime layout.

The immediate risk is inconsistent behavior:

- reads/searches observe the segment-aware view
- `delete(ids)` mutates only the flat legacy files
- WAL replay for deletes reuses the same flat delete path

That makes `delete_by_filter` unsafe to implement before `delete(ids)` is fixed.

## Chosen Approach

Rewrite `delete_internal(...)` so it walks segment paths in the same shadowing order used by read paths:

1. Iterate `segment_manager.segment_paths()` in runtime order.
2. For each requested external id, only the newest visible row may be deleted.
3. If the newest visible row is already tombstoned, the id counts as absent and older rows remain untouched.
4. Persist `tombstones.json` and `segment.json.deleted_count` for each changed segment only.
5. Keep WAL shape unchanged: `WalRecord::Delete { ids }` still replays through `delete_internal(...)`.
6. Invalidate the collection search cache after the mutation.

## Non-Goals

- No `delete_by_filter`
- No new WAL record types
- No auto-rollover / mutable multi-segment writer work
- No schema/catalog changes

## Required Correctness Properties

- Multi-segment delete removes the newest visible row for a target id, not an older shadowed row.
- If a newer segment already tombstones an id, deleting that id again must not resurrect or alter older rows.
- Legacy single-segment collections must keep working.
- WAL replay after a crash must rebuild the same segment-aware delete state.

## Validation

- Add a failing `collection_api` test proving `delete(ids)` works against a two-segment layout and respects shadowing/tombstones.
- Add a failing `wal_recovery` test proving a crash after multi-segment delete replays into the correct tombstone state.
- After implementation, run targeted `collection_api` and `wal_recovery` commands plus `git diff --check` and `git status --short`.
