# HannsDB Design

**Date:** 2026-03-19

**Goal:** Build HannsDB as a local lightweight agent database in Rust, reusing `knowhere-rs` for ANN search, supporting both embedded and daemon surfaces, and reaching a benchmarkable `VectorDBBench` integration through a Python binding first. HannsDB development should also continuously validate that `knowhere-rs` remains a viable and semantically compatible ANN dependency for this stack.

## Scope

### In Scope for v1

- Single-node local database
- Dense vector search with HNSW only
- Persistent collections on local disk
- Record upsert, delete, get, and top-k search
- Soft delete through tombstones or bitset filtering
- Rust embedded API
- Python binding with a `zvec`-compatible minimal subset for benchmark integration
- Local daemon API for process separation and background work
- `VectorDBBench` CLI/backend integration for one initial performance case

### Out of Scope for v1

- Distributed deployment
- Multi-tenant auth model
- Full-text search as a complete standalone subsystem
- IVF-PQ or DiskANN parity
- Rich SQL layer
- Cloud-native service operation

## Recommended Architecture

HannsDB should use an `embedded-core first, daemon wraps core` architecture.

The core idea is:

- The source of truth is HannsDB local storage, not the daemon.
- `knowhere-rs` is the ANN engine, not the database.
- The daemon is a control plane and transport layer over the same core used by embedded callers.
- `VectorDBBench` should hit the Python binding first because that is the fastest path to a real benchmarkable integration.

## System Layout

### `hannsdb-core`

Responsibilities:

- collection catalog and schema metadata
- record model
- segment metadata and lifecycle
- WAL and manifest replay
- query orchestration across segments
- delete/update semantics

Suggested modules:

- `src/catalog/`
- `src/storage/`
- `src/segment/`
- `src/record/`
- `src/query/`
- `src/error.rs`
- `src/lib.rs`

### `hannsdb-index`

Responsibilities:

- narrow adapter over `knowhere-rs`
- expose only the index capabilities HannsDB needs
- map HannsDB requests into `knowhere-rs` HNSW operations

Suggested first boundary:

- `IndexAdapter::insert`
- `IndexAdapter::search`
- `IndexAdapter::delete_mask`
- `IndexAdapter::save`
- `IndexAdapter::load`

Only HNSW should be implemented in v1.

### `hannsdb-py`

Responsibilities:

- PyO3-based Python binding
- expose a minimal `zvec`-like benchmark surface
- provide local path open/create/query/optimize lifecycle

This surface is intentionally smaller than the full Rust API.

### `hannsdb-daemon`

Responsibilities:

- local process boundary
- background flush, compact, rebuild
- HTTP or Unix socket control API
- WebSocket event stream only for notifications

The daemon should reuse `hannsdb-core` directly and must not own a separate data model.

## Reuse from `knowhere-rs`

Directly reusable in v1:

- `IndexConfig`, `IndexType`, `MetricType`, `IndexParams`
- legal config validation
- `HnswIndex`
- `Dataset`
- `BitsetView`
- `Interrupt`

Not sufficient on their own:

- current admin/storage helpers
- FFI surface as a primary integration boundary
- broad multi-index parity assumptions

HannsDB should integrate through a narrow internal adapter instead of exposing `knowhere-rs` types everywhere.

Validation rule:

- every new HannsDB ANN integration milestone should include a targeted `knowhere-rs`-backed verification pass
- if HannsDB and `knowhere-rs` differ on score or id semantics, the mismatch must be recorded explicitly and resolved at the HannsDB adapter/core boundary

## Data Model

### Collection

- `collection_id`
- `name`
- `dimension`
- `metric`
- `index_kind`
- `index_params`
- `created_at`
- `format_version`

### Record

- `id`
- `vector`
- `payload`
- `text`
- `created_at`
- `updated_at`
- `deleted_at`

### Segment

- `segment_id`
- `state`
- `record_count`
- `deleted_count`
- `index_files`
- `manifest_version`

## Storage Model

The local disk layout should be explicit and recoverable:

```text
<db-root>/
  manifest.json
  wal/
  collections/
    <collection-name>/
      collection.json
      segments/
        <segment-id>/
          segment.json
          records.bin
          index.hnsw
          tombstone.bin
```

Design rules:

- manifest and collection metadata are authoritative
- ANN index files are rebuildable derived state
- WAL exists to recover the latest acknowledged writes
- delete is soft delete first

## Read/Write Flow

### Insert

1. validate collection and dimension
2. append write intent to WAL
3. write record payload into active segment
4. add vectors to HNSW adapter
5. mark WAL entry committed
6. update manifest/segment metadata

### Search

1. load collection metadata
2. build search request for active and sealed segments
3. apply tombstone bitset per segment
4. search each segment through index adapter
5. merge top-k results across segments
6. return records and scores

### Delete

1. append delete intent to WAL
2. mark tombstone in segment metadata
3. expose delete mask to search path
4. compact later

## Surface Design

### Embedded API

Rust API should be the full-fidelity local surface.

Python API should provide the minimal benchmark-compatible subset first:

- `init(...)`
- `open(path, option)`
- `create_and_open(path, schema, option)`
- `collection.insert(...)`
- `collection.query(...)`
- `collection.optimize(...)`
- `collection.destroy()`

### Daemon API

v1 daemon routes:

- `POST /collections`
- `GET /collections`
- `POST /collections/{name}/records:upsert`
- `POST /collections/{name}/records:delete`
- `POST /collections/{name}/search`
- `POST /collections/{name}/admin/flush`
- `GET /health`

WebSocket is optional in the first implementation slice and should only emit events like flush progress or compaction completion.

## `VectorDBBench` Integration Strategy

The first benchmark path should mirror the existing `zvec` integration:

- local path config
- HNSW parameters
- Python import into benchmark process
- backend and CLI wiring first
- Streamlit/frontend mapping later

This keeps the first benchmark result focused on HannsDB core behavior instead of protocol overhead.

## Main Risks

- `knowhere-rs` maturity is asymmetric across index families; v1 must avoid IVF-PQ and DiskANN expansion.
- Persistence semantics differ by index family; HannsDB must own recovery logic.
- Python benchmark path requires safe repeated open/init behavior under multi-process runners.
- Daemon work can sprawl if started before core storage and binding are stable.

## Verification Strategy

Verification should be layered:

- unit tests for metadata, manifest, tombstone, query merge
- integration tests for create/open/recover/search/delete
- Python smoke tests for binding lifecycle
- daemon API smoke tests
- one `VectorDBBench` CLI smoke run for a small performance case

## Delivery Order

1. workspace bootstrap and crate boundaries
2. collection metadata and local storage layout
3. HNSW adapter over `knowhere-rs`
4. minimal embedded CRUD + search
5. Python benchmark surface
6. `VectorDBBench` adapter
7. daemon control API
8. filter support and broader benchmarks
