# HannsDB Phase 6 Design Draft: Multi-Segment Management

Date: 2026-03-21

## Current single-segment limitations

HannsDB currently stores each collection in one append-only segment, so `records.bin` and companion files grow without physical cleanup. Deletions are soft tombstones, which preserves correctness but accumulates dead rows and extra scan cost. ANN state is rebuilt from the whole live dataset during optimize, so maintenance cost rises with collection size.

## 1. Rollover trigger rules

Phase 6 introduces explicit active/immutable segment states. Writes go only to one active segment per collection; once rollover condition is met, the active segment is sealed and a new active segment is created.

V1 rule uses simple OR-thresholds:
- row count >= 200_000
- vector file size (`records.bin`) >= 1 GiB
- tombstone ratio in active segment >= 0.20

This keeps decision logic deterministic, local-only, and easy to reason about. Threshold values are config constants in core (not user-tunable in v1), and can be revisited after benchmark evidence.

## 2. Multi-segment query strategy

Brute-force path: iterate segments sequentially, skip tombstoned rows, compute distances, and maintain one global bounded top-k heap. This preserves current scoring semantics and avoids loading all segment hits into memory before merge.

HNSW path: query each segment that has optimized ANN state, fetch per-segment top-k, then perform k-way merge into global top-k. Segments without ANN state use brute-force fallback in the same request and are merged into the same global heap.

For v1, execution remains single-process local-first, with optional bounded parallelism deferred to v2 unless profiling shows clear single-core regression.

## 3. Compaction semantics

Compaction rewrites selected immutable segments into one new immutable segment containing only live rows, then atomically swaps metadata and retires old segment directories.

When to run:
- manual trigger via admin API/CLI (`optimize_collection` extended behavior)
- optional daemon maintenance trigger when collection-level dead-row ratio exceeds threshold (default 0.30) and dead rows >= 50_000

Naming/management:
- segment IDs are monotonic (`seg-000001`, `seg-000002`, ...)
- compacted output gets a fresh new ID
- manifest/collection metadata tracks segment list and active segment ID
- old segments are moved to a temporary retire list and removed after successful metadata commit

## 4. ANN state maintenance rules

V1 does not implement incremental in-place HNSW mutation across all historical data. Instead:
- active segment accepts writes and serves brute-force
- sealed segments may have per-segment ANN state built/rebuilt
- global query merges ANN + brute-force partial results

Rebuild policy:
- sealing a segment allows background/offline ANN build for that segment
- compaction output requires ANN rebuild for the new compacted segment
- if tombstone ratio of an ANN-enabled immutable segment exceeds 0.30, that segment is a compaction candidate instead of incremental graph repair

This keeps implementation simple and aligned with current “ANN as derived state” principle.

## 5. External API/daemon changes

`flush_collection` under multi-segment means:
- all collection metadata and segment metadata are durably written and readable
- active segment append files and tombstone states are visible and consistent
- no promise of ANN build completion

`optimize_collection` under multi-segment means:
- seal current active segment if non-empty
- optionally run compaction plan selection and execution (v1: basic heuristic)
- build ANN state for eligible immutable segments (or compacted result segment)

Daemon minimal admin controls in v1:
- `POST /collections/:name/admin/flush` (existing semantics expanded for segment set)
- `POST /collections/:name/admin/optimize` (seal + compact + ANN rebuild workflow)
- `GET /collections/:name/admin/segments` (list segment stats, live/dead counts, ANN readiness)

## 6. Filesystem layout changes

Collection layout changes from single flat files to segment directories:

```text
collections/<name>/
  collection.json
  segment_set.json
  segments/
    seg-000001/
      segment.json
      records.bin
      ids.bin
      payloads.jsonl
      tombstones.json
      ann/                 # optional derived state files/markers
    seg-000002/
      ...
  wal.jsonl
```

`segment_set.json` tracks:
- active segment ID
- ordered immutable segment IDs
- per-segment summary counters
- compaction generation/version

## V1 scope

Must do:
- explicit segment rollover and active/immutable segment state
- multi-segment search merge for brute-force and ANN mix
- basic compaction pipeline (manual + simple threshold trigger)
- extended flush/optimize semantics and minimal segment admin inspection API
- filesystem layout migration for new collections (no complex auto-migrate of legacy data in v1)

Deferred to v2:
- adaptive per-collection tuning knobs for thresholds
- parallel multi-segment query scheduler
- incremental HNSW graph maintenance for mutable segments
- advanced compaction planner/cost model
- rich daemon scheduling and policy framework

## Key files expected to change

- `crates/hannsdb-core/src/db.rs`
- `crates/hannsdb-core/src/segment/*` (segment metadata/load/store paths)
- `crates/hannsdb-core/src/catalog/*` (collection + segment set metadata)
- `crates/hannsdb-core/src/query/*` (multi-segment merge path)
- `crates/hannsdb-index/src/*` (per-segment ANN lifecycle hooks)
- `crates/hannsdb-daemon/src/routes.rs` and request/response schema files
- `crates/hannsdb-py/src/lib.rs` (if admin/inspect surfaces are mirrored to Python)
