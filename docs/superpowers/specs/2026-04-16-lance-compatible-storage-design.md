# Lance-Compatible Storage Design

## Goal

Make HannsDB a Lance-compatible local vector database facade: HannsDB owns the DB/API/product surface and Hanns owns ANN acceleration, while upstream Lance owns the canonical persisted dataset format.

The first implementation slice is deliberately narrow: prove HannsDB can create and append rows to a real Lance dataset using upstream Lance crates, and prove that Lance tooling can open and scan the result without HannsDB-specific readers.

## Non-Goals for P0

- Do not replace the existing HannsDB segment runtime yet.
- Do not implement delete, upsert, update, compaction, or schema mutation on Lance storage yet.
- Do not integrate Hanns indices with Lance index metadata yet.
- Do not write Lance manifest, fragment, or data-file bytes by hand.
- Do not require external Lance applications to understand Hanns-specific sidecars.

## Current State

HannsDB currently persists collections with a HannsDB-owned layout:

- `collection.json`
- `segment_set.json`
- per-segment `segment.json`
- `records.bin`
- `ids.bin`
- `payloads.jsonl` / `payloads.arrow`
- `vectors.jsonl` / `vectors.arrow`
- `sparse_vectors.jsonl`
- `tombstones.json`
- `forward_store.json`
- `forward_store.arrow`
- `forward_store.parquet`

This is useful as an append-friendly embedded store, and the forward store is already Lance-like because it materializes unified rows to Arrow IPC and Parquet. However, it is not a Lance dataset. It lacks Lance manifest versions, fragments, data files, deletion files, row-id metadata, index metadata, and transaction semantics.

## Target Architecture

```text
HannsDB API / Python / daemon
        |
        v
hannsdb-core collection runtime
        |
        +-- LanceDatasetStore        # canonical persisted storage for new Lance-compatible collections
        |     - creates / opens / appends via upstream Lance Dataset APIs
        |     - never hand-writes Lance manifest or data-file bytes
        |
        +-- HannsIndexProvider       # future ANN acceleration layer
        |     - builds Hanns indexes from Lance row ids + vector columns
        |     - stores Hanns artifacts in a way Lance readers can ignore
        |
        +-- LegacyMigrationAdapter   # future one-way migration from current HannsDB layout
```

The storage contract is:

1. Lance owns the durable data format.
2. HannsDB owns user-facing semantics and API compatibility.
3. Hanns owns ANN execution.
4. External Lance applications can always read HannsDB-committed data without loading HannsDB.

## P0 Design

Add an optional `lance-storage` feature to `hannsdb-core`.

Under that feature, add a small storage boundary:

- `storage/lance_schema.rs`
  - Maps HannsDB `CollectionSchema` into Arrow/Lance schema.
  - P0 supports numeric document ids, declared scalar fields, and dense f32 vector fields.
  - P0 rejects unsupported fields explicitly instead of silently degrading.

- `storage/lance_store.rs`
  - Converts HannsDB `Document` rows into an Arrow `RecordBatch`.
  - Writes datasets using upstream `lance::Dataset::write`.
  - Appends using Lance `WriteMode::Append`.
  - Opens datasets using upstream `lance::Dataset::open`.

P0 writes these columns:

- `id`: `Int64`, HannsDB document id.
- one column per scalar field:
  - `String` -> UTF-8
  - `Bool` -> Boolean
  - `Int32` -> Int32
  - `Int64` -> Int64
  - `UInt32` -> UInt32
  - `UInt64` -> UInt64
  - `Float` -> Float32
  - `Float64` -> Float64
- one column per dense vector field:
  - `VectorFp32` -> fixed-size list of `Float32`

P0 should reject:

- sparse vectors
- fp16 vector storage
- array scalar fields
- nullable scalar fields, unless the value is present for every inserted row
- string primary key mode
- schema mutation

These rejections are intentional. They keep the first slice focused on Lance dataset compatibility, not full HannsDB semantics.

## Data Flow

Create:

1. HannsDB caller supplies `CollectionSchema`.
2. `lance_schema` maps it to Arrow schema.
3. P0 writes the first non-empty batch with `WriteMode::Create`.
4. Lance creates the manifest, fragments, data files, and version metadata.

Append:

1. HannsDB caller supplies documents matching the existing schema.
2. `lance_store` converts documents to a `RecordBatch`.
3. Lance appends a new fragment with `WriteMode::Append`.
4. External Lance readers see the appended committed version after reopen.

Read validation:

1. Test opens the dataset with `lance::Dataset::open`.
2. Test scans or counts rows through Lance APIs.
3. Test verifies schema and row content without using HannsDB readers.

## Compatibility Contract

P0 compatibility means:

- A dataset written by HannsDB can be opened by upstream Lance.
- Lance can scan rows and project fields.
- Dense vector columns have Lance-readable Arrow fixed-size-list representation.
- HannsDB does not create custom required files for Lance readers.

P0 does not mean:

- LanceDB can use Hanns indexes.
- Lance can observe uncommitted HannsDB active buffers.
- Existing HannsDB collections are automatically Lance datasets.
- Delete/upsert semantics are complete.

## Future Phases

### P1: HannsDB API Backed by Lance Reads

Wire selected HannsDB create/insert/fetch/search APIs to `LanceDatasetStore` behind a feature flag. Brute-force search reads the Lance vector column and keeps HannsDB's current score semantics.

### P2: Lance Delete / Upsert / Versioning

Map HannsDB deletes and upserts to Lance deletion and append/update semantics. External Lance readers must see the same live row set as HannsDB after commit.

### P3: Hanns Index Provider

Build Hanns ANN indexes over Lance row ids and vector columns. First store them as HannsDB-private, optional sidecars that Lance readers can ignore. Later evaluate Lance `IndexMetadata` integration if upstream Lance supports custom or ignored index metadata safely.

### P4: Legacy Migration

Add one-way migration from HannsDB's current segment layout into Lance dataset format. Existing collections remain readable until migrated.

## Risks

- Upstream Lance crates are large and may significantly increase build time.
- Path dependency on a sibling Lance checkout is acceptable for early development but should become a git dependency or vendored workspace decision later.
- Lance schema support for nested/list/vector columns must be verified with real scans, not assumed from Arrow type mapping alone.
- If Lance index metadata cannot tolerate custom Hanns index types, Hanns index artifacts must remain private sidecars.

## Success Criteria for P0

- `cargo test -p hannsdb-core --features lance-storage lance_compat` passes.
- Test creates a HannsDB-style schema, writes documents through `LanceDatasetStore`, opens the directory with upstream `lance::Dataset::open`, scans it, and verifies row content.
- Test appends a second batch and verifies external Lance row count increases.
- No production HannsDB API is switched to Lance storage in P0.
