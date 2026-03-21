# HannsDB Project Design

**Date:** 2026-03-20

## 1. Project intent

HannsDB is a local lightweight agent database built in Rust.

It has three simultaneous goals:

1. Build a usable local database core for agent-oriented data management.
2. Reuse `knowhere-rs` as the ANN engine instead of reimplementing vector indexing from scratch.
3. Reach a real, benchmarkable integration path through `VectorDBBench`.

This means HannsDB is not just “a wrapper around ANN”, and it is also not intended to become a heavy remote-first vector service in v1.

The current direction is:

- local-first
- embedded-core first
- daemon as an optional control plane
- Python binding first for benchmark integration
- continuous verification of `knowhere-rs` as a first-class project goal

## 2. Product shape

HannsDB is designed as a `dual-surface` system built on one shared core.

### Embedded surface

The embedded surface is the primary data plane.

- Rust code can open the database directly from a local path.
- Python uses PyO3 bindings over the same core.
- This is the fastest path for local agents and for `VectorDBBench`.

### Daemon surface

The daemon is a thin local control plane.

- It wraps the same `hannsdb-core`.
- It is useful for multi-process access, local service style workflows, and admin operations.
- It is not the source of truth for data.

## 3. Repository layout

Current workspace structure:

- `crates/hannsdb-core`
  - catalog, collection metadata, segment files, tombstones, query path, main DB API
- `crates/hannsdb-index`
  - narrow ANN adapter layer
  - in-memory fallback backend
  - `knowhere-rs`-backed HNSW backend
- `crates/hannsdb-py`
  - embedded Python surface for benchmark and local agent-data access
- `crates/hannsdb-daemon`
  - local HTTP API over the shared core
- `docs/`
  - benchmark notes, smoke docs, dated design/plan docs, and now top-level project docs

External but coupled repos:

- `/Users/ryan/Code/knowhere-rs`
  - ANN dependency under active verification
- `/Users/ryan/Code/VectorDBBench`
  - benchmark harness HannsDB must run
- `/Users/ryan/Code/zvec`
  - reference for embedded Python benchmark integration shape

## 4. Core architecture

### 4.1 `hannsdb-core`

`hannsdb-core` owns the database semantics.

Current responsibilities:

- open/create/drop/list collections
- persist manifest and collection metadata
- persist segment metadata
- append vectors, ids, and scalar payload rows to local files
- soft delete through tombstones
- in-process search cache
- reconstruct canonical documents from local files
- optional optimize step that builds an ANN-backed search state

Current DB API in [`db.rs`](/Users/ryan/Code/HannsDB/crates/hannsdb-core/src/db.rs):

- `open`
- `create_collection`
- `create_collection_with_schema`
- `drop_collection`
- `list_collections`
- `get_collection_info`
- `insert`
- `insert_documents`
- `upsert_documents`
- `fetch_documents`
- `delete`
- `search`
- `query_documents`
- `flush_collection`
- `optimize_collection`

Current v1 `flush_collection()` guarantee is intentionally narrow:

- collection metadata is readable
- `segment.json` is readable
- `tombstones.json` is readable
- `wal.jsonl` is present and readable

It is a visibility/consistency boundary for local files, not a claim of full crash-safe durability.

### 4.1.1 Canonical v1 record model

HannsDB now has an explicit document model for the agent-data path.

- one primary vector field per collection in v1
- typed scalar fields with a narrow value set: `string`, `int64`, `float64`, `bool`
- append-only row model: `upsert` tombstones old rows and appends a new row
- ANN state stays derived from the primary vector field

This is intentionally narrower than `zvec` in full generality. The current v1 rule is:

- multi-vector ANN is not supported yet
- scalar secondary indexes are not supported yet
- correctness of filtered query is preferred over ANN acceleration for filtered query

### 4.2 `hannsdb-index`

`hannsdb-index` is intentionally small.

Its job is to keep `knowhere-rs` behind a HannsDB-owned boundary so that:

- score semantics stay under HannsDB control
- backend substitutions remain possible
- `knowhere-rs` quirks are isolated

Current backends:

- `InMemoryHnswIndex`
- `KnowhereHnswIndex` behind `knowhere-backend`

Current important integration facts:

- `L2` from `knowhere-rs` already finalizes to Euclidean distance
- `IP` semantics must be mapped back to HannsDB’s public distance semantics
- small-fixture stability required pinning `random_seed = 42`

### 4.3 `hannsdb-py`

The Python layer now serves two roles:

1. preserve the current `VectorDBBench` embedded path
2. expose a small but usable local agent-data surface

Current exposed lifecycle:

- `init`
- `create_and_open`
- `open`
- `Collection.insert`
- `Collection.upsert`
- `Collection.fetch`
- `Collection.delete`
- `Collection.query`
- `Collection.flush`
- `Collection.stats`
- `Collection.optimize`
- `Collection.destroy`

`Collection.flush` inherits the same minimal v1 semantics from the Rust core; it does not currently imply stronger fsync-style durability.

Current `Doc` shape includes:

- `id`
- optional `score`
- typed `fields`
- named `vectors`

This is still intentionally narrower than a full `zvec` parity story, but it is no longer only a benchmark shell.

### 4.4 `hannsdb-daemon`

The daemon exposes a thin HTTP surface today.

Current routes in [`routes.rs`](/Users/ryan/Code/HannsDB/crates/hannsdb-daemon/src/routes.rs):

- `GET /health`
- `GET /collections`
- `POST /collections`
- `GET /collections/:collection`
- `GET /collections/:collection/stats`
- `DELETE /collections/:collection`
- `POST /collections/:collection/admin/flush`
- `POST /collections/:collection/records`
- `POST /collections/:collection/records/upsert`
- `POST /collections/:collection/records/fetch`
- `DELETE /collections/:collection/records`
- `POST /collections/:collection/search`

The daemon is still intentionally thin. It now mirrors the current core capabilities for:

- document insert/upsert
- fetch by id
- filtered search with optional `output_fields`
- collection stats and flush

Background scheduling, compaction, and richer admin flows are still not the v1 center of gravity.

## 5. Storage model

The current on-disk model is simple and explicit.

Per database root:

- `manifest.json`
- `collections/<name>/collection.json`
- `collections/<name>/segment.json`
- `collections/<name>/records.bin`
- `collections/<name>/ids.bin`
- `collections/<name>/payloads.jsonl`
- `collections/<name>/tombstones.json`

Important design rule:

- structured metadata and record files are the durable source of truth
- ANN state is derived and rebuildable

Current record layout is row-aligned across:

- `ids.bin`
- `records.bin`
- `payloads.jsonl`
- `tombstones.json`

Compatibility rule:

- legacy benchmark-style raw vector inserts still work
- when legacy raw inserts are used, HannsDB appends empty payload rows to keep later document fetch and upsert aligned

At the moment, the implementation is still effectively single-segment per collection. Minimal WAL mutation logging and `open()` replay for WAL-owned collection lifecycles now exist, but full crash-style recovery guarantees, compaction, and multi-segment orchestration are still future work.

## 6. Query and optimize model

There are currently two search modes:

1. brute-force search over local records
2. optimized ANN-backed search after `optimize_collection()`

The optimize path builds a cached ANN search state and then future queries can use that in-memory optimized path.

There are also now two document query modes:

1. no-filter query can keep using the existing ANN/no-filter search path
2. filtered query uses a correct brute-force path over live rows and typed scalar payloads

This split is deliberate. In v1, filtered query correctness is preferred over trying to fake ANN + post-filter semantics.

This matters because benchmark behavior changed substantially over time:

- early bottleneck: repeated record loading and brute-force cosine search
- later bottleneck: `knowhere-rs` HNSW build cost during `optimize()`

That shift is real progress: it means the system is now spending time in the intended ANN path rather than in obvious accidental overhead.

## 7. Benchmark strategy

The project uses a staged benchmark strategy.

### Stage 1: internal optimize benchmark

Use `scripts/run_hannsdb_optimize_bench.sh` to isolate:

- create
- insert
- optimize
- search
- total

This is the fastest way to answer “did the local optimize path get better or worse?” without the noise of full `VectorDBBench`.

### Stage 2: tiny `VectorDBBench` smoke

Use `scripts/run_vdbb_hannsdb_smoke.sh` to verify:

- Python surface shape
- client wiring
- insert/query/optimize lifecycle
- recall/latency correctness on a tiny local dataset

This stage is already passing.

### Stage 3: standard benchmark case

Use `scripts/run_vdbb_hannsdb_perf1536d50k.sh` and the direct `VectorDBBench` command path for the standard `Performance1536D50K` case.

This stage is not yet closed end-to-end at target performance because `optimize()` on `50K / 1536 / cosine` is still expensive.

## 7.1 Performance ownership decision (2026-03-19)

The current bottleneck is in `knowhere-rs` HNSW build cost during `optimize()`, not in HannsDB glue. The remaining HannsDB-side work is bounded to adapter-boundary trims: avoid unnecessary copies/flattening, keep score semantics explicit at the boundary, and preserve deterministic integration behavior.

Next verification should prove two things:

1. `50K / 1536 / cosine` optimize time moves down in the `knowhere-rs` hotpath without changing HannsDB score semantics.
2. HannsDB-side adapter changes stay minor and do not reintroduce copy/load overhead into the benchmark path.

## 8. knowhere-rs verification as a project goal

This is not a side note. It is part of the project definition.

Every HannsDB ANN milestone should answer both questions:

1. Did HannsDB get closer to a usable benchmarkable local DB?
2. Did we verify that `knowhere-rs` is still stable, performant, and semantically compatible for this path?

Current verified `knowhere-rs` findings already influenced HannsDB design:

- score mapping cannot be assumed blindly
- HNSW behavior under tiny fixtures needed deterministic seeding
- cosine build hot path can be improved through targeted norm reuse

## 9. Current implementation status

### Implemented

- Rust workspace with four crates
- manifest and collection metadata persistence
- segment metadata, record append/load, tombstones
- core collection lifecycle and search path
- ANN adapter with `knowhere-rs` feature-gated backend
- minimal Python benchmark surface
- minimal daemon API
- HannsDB client integration in `VectorDBBench`
- tiny smoke benchmark path
- canonical document model with typed scalar payloads
- embedded Python document APIs: `upsert`, `fetch`, `delete`, `query(filter)`, `flush`, `stats`
- thin daemon document APIs: insert/upsert/fetch/filter-search
- standalone optimize benchmark path
- minimal append-only WAL at `<root>/wal.jsonl`
- WAL mutation logging for collection/data writes
- minimal `open()` replay for WAL-owned collection lifecycles without duplicate materialization on normal reopen

### Partially implemented

- ANN-backed optimize/search path
- daemon admin surface
- benchmark tooling and notes
- `knowhere-rs` performance verification
- durability foundation (`flush_collection()` semantics and stronger recovery guarantees still pending)

### Not implemented yet

- multi-segment lifecycle
- compaction
- richer schema/payload indexing
- filter execution beyond current minimal shape
- full production-grade daemon responsibilities
- full crash-style recovery semantics and durable flush guarantees

## 10. Main risks

### Risk 1: optimize cost dominates at target scale

The current biggest technical risk is not correctness. It is the cost of building the ANN state for `50K / 1536 / cosine`.

### Risk 2: design docs and implementation drift

This was already happening. The repo had detailed dated docs and detailed benchmark notes, but no single current source-of-truth overview. This new document is intended to close that gap.

### Risk 3: core durability is still incomplete

The project now has minimal WAL logging and replay, but durability is still not finished until `flush_collection()` has explicit guarantees and crash-style recovery cases are proven.

### Risk 4: benchmark success can mask product gaps

Running `VectorDBBench` is necessary, but it does not automatically mean HannsDB is already a good local agent database. Product work still needs record/payload/session-oriented capabilities beyond benchmark minimums.

## 11. Current design decision summary

The current project direction is:

- keep `hannsdb-core` as the source of truth
- treat ANN as derived acceleration state
- benchmark through the Python embedded path first
- keep daemon thin until core durability and benchmark path are stable
- treat `knowhere-rs` verification as part of the product roadmap, not external maintenance

## 12. Companion docs

Detailed working docs remain useful:

- [`2026-03-19-hannsdb-design.md`](/Users/ryan/Code/HannsDB/docs/superpowers/specs/2026-03-19-hannsdb-design.md)
- [`2026-03-19-hannsdb-v1.md`](/Users/ryan/Code/HannsDB/docs/superpowers/plans/2026-03-19-hannsdb-v1.md)
- [`vector-db-bench-notes.md`](/Users/ryan/Code/HannsDB/docs/vector-db-bench-notes.md)

This file should be treated as the top-level architectural overview for the current state of the project.
