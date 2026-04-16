# Lance-Compatible Storage P0 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a feature-gated Lance storage prototype that writes real Lance datasets from HannsDB schemas/documents and proves upstream Lance can open and scan them.

**Architecture:** Add an optional `lance-storage` feature to `hannsdb-core`. P0 creates a narrow `LanceDatasetStore` boundary and schema/batch conversion helpers, without rewiring existing HannsDB collection APIs or Hanns index execution.

**Tech Stack:** Rust, HannsDB core, upstream Lance Rust crate, Arrow arrays, Tokio tests, tempfile.

---

## File Structure

- Modify `Cargo.toml`
  - Add workspace dependencies needed by the P0 Lance feature, if direct imports require them.
- Modify `crates/hannsdb-core/Cargo.toml`
  - Add optional `lance-storage` feature.
  - Add optional path dependency on upstream Lance.
  - Add `tokio` dev dependency if async tests require it.
- Modify `crates/hannsdb-core/src/storage/mod.rs`
  - Export Lance storage modules behind `#[cfg(feature = "lance-storage")]`.
- Create `crates/hannsdb-core/src/storage/lance_schema.rs`
  - Convert HannsDB collection schema to Arrow schema.
  - Validate P0-supported field set.
- Create `crates/hannsdb-core/src/storage/lance_store.rs`
  - Convert documents to Arrow `RecordBatch`.
  - Create and append Lance datasets using upstream Lance APIs.
- Create `crates/hannsdb-core/tests/lance_compat.rs`
  - Black-box compatibility tests that open HannsDB-written data with upstream Lance.

## Chunk 1: Dependency and Feature Boundary

### Task 1: Add optional Lance storage feature

**Files:**
- Modify: `Cargo.toml`
- Modify: `crates/hannsdb-core/Cargo.toml`

- [ ] **Step 1: Write the expected feature check**

Run:

```bash
cargo check -p hannsdb-core --features lance-storage
```

Expected before implementation: FAIL because `lance-storage` does not exist.

- [ ] **Step 2: Add optional dependency and feature**

In `crates/hannsdb-core/Cargo.toml`, add:

```toml
[features]
default = []
hanns-backend = ["hannsdb-index/hanns-backend"]
lance-storage = ["dep:lance"]

[dependencies]
lance = { path = "../../../lance/rust/lance", optional = true }
```

If direct Arrow subcrates are needed for `RecordBatchIterator` or concrete array imports, prefer existing `arrow` first. Add `arrow-array` / `arrow-schema` workspace dependencies only if the compiler requires them.

- [ ] **Step 3: Verify feature resolves**

Run:

```bash
cargo check -p hannsdb-core --features lance-storage
```

Expected: May fail on unused/missing code until modules are added, but should no longer fail with "feature does not exist".

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml crates/hannsdb-core/Cargo.toml
git commit -m "Add Lance storage feature boundary"
```

Use the repo's Lore commit protocol.

## Chunk 2: Schema Mapping

### Task 2: Add schema conversion tests

**Files:**
- Create: `crates/hannsdb-core/src/storage/lance_schema.rs`
- Modify: `crates/hannsdb-core/src/storage/mod.rs`

- [ ] **Step 1: Write unit tests in `lance_schema.rs`**

Add tests behind `#[cfg(test)]` and module behind `#[cfg(feature = "lance-storage")]`.

Test cases:

```rust
#[test]
fn lance_schema_maps_supported_scalar_and_vector_fields() {
    // Build CollectionSchema with:
    // - scalar: title String, year Int64, score Float64, active Bool
    // - vector: dense VectorFp32 dimension 3
    // Assert output Arrow schema has:
    // - id Int64
    // - title Utf8
    // - year Int64
    // - score Float64
    // - active Boolean
    // - dense FixedSizeList(Float32, 3)
}

#[test]
fn lance_schema_rejects_sparse_vectors_in_p0() {
    // Build schema with VectorSparse.
    // Assert conversion returns InvalidInput with "sparse vectors are not supported".
}
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cargo test -p hannsdb-core --features lance-storage storage::lance_schema -- --nocapture
```

Expected: FAIL because conversion functions do not exist.

- [ ] **Step 3: Implement schema conversion**

Create public-in-crate API:

```rust
#[cfg(feature = "lance-storage")]
pub(crate) fn arrow_schema_for_lance(
    schema: &CollectionSchema,
) -> io::Result<Arc<arrow::datatypes::Schema>>;
```

Rules:

- Always include `id: Int64`.
- Map supported scalar fields.
- Map `VectorFp32` to fixed-size list of `Float32`.
- Reject `VectorFp16`, `VectorSparse`, arrays, and unsupported nullable behavior with `io::ErrorKind::InvalidInput`.

- [ ] **Step 4: Export module**

In `crates/hannsdb-core/src/storage/mod.rs`:

```rust
#[cfg(feature = "lance-storage")]
pub(crate) mod lance_schema;
```

- [ ] **Step 5: Run schema tests**

Run:

```bash
cargo test -p hannsdb-core --features lance-storage storage::lance_schema -- --nocapture
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/hannsdb-core/src/storage/mod.rs crates/hannsdb-core/src/storage/lance_schema.rs
git commit -m "Map HannsDB schemas to Lance-compatible Arrow"
```

Use Lore commit protocol.

## Chunk 3: Document Batch Conversion

### Task 3: Convert documents to Arrow record batches

**Files:**
- Modify: `crates/hannsdb-core/src/storage/lance_store.rs`
- Test: `crates/hannsdb-core/tests/lance_compat.rs`

- [ ] **Step 1: Write failing batch conversion test**

Create `crates/hannsdb-core/tests/lance_compat.rs` with:

```rust
#![cfg(feature = "lance-storage")]

use hannsdb_core::document::{CollectionSchema, Document, FieldValue, ScalarFieldSchema, VectorFieldSchema};
use hannsdb_core::storage::lance_store::documents_to_lance_batch;

#[test]
fn lance_batch_conversion_preserves_ids_scalars_and_vectors() {
    // Build schema: title String, year Int64, dense dim=3.
    // Build two Document values.
    // Convert to RecordBatch.
    // Assert num_rows == 2 and schema fields exist.
}
```

If `storage::lance_store` cannot be public, expose only `#[cfg(test)]` helpers or place integration tests through `LanceDatasetStore` in Chunk 4.

- [ ] **Step 2: Verify RED**

Run:

```bash
cargo test -p hannsdb-core --features lance-storage --test lance_compat -- --nocapture
```

Expected: FAIL because `lance_store` does not exist.

- [ ] **Step 3: Implement `lance_store.rs` batch conversion**

Create:

```rust
#[cfg(feature = "lance-storage")]
pub(crate) fn documents_to_lance_batch(
    schema: &CollectionSchema,
    documents: &[Document],
) -> io::Result<arrow::record_batch::RecordBatch>;
```

Rules:

- Validate all documents against the schema before conversion.
- `id` column comes from `Document.id`.
- Missing non-null P0 scalar fields return `InvalidInput`.
- Dense vector dimensions must match the schema.
- Preserve row order.

- [ ] **Step 4: Export module**

In `storage/mod.rs`:

```rust
#[cfg(feature = "lance-storage")]
pub(crate) mod lance_store;
```

- [ ] **Step 5: Run conversion test**

Run:

```bash
cargo test -p hannsdb-core --features lance-storage --test lance_compat -- --nocapture
```

Expected: PASS for conversion test.

- [ ] **Step 6: Commit**

```bash
git add crates/hannsdb-core/src/storage/lance_store.rs crates/hannsdb-core/src/storage/mod.rs crates/hannsdb-core/tests/lance_compat.rs
git commit -m "Convert HannsDB documents to Lance record batches"
```

Use Lore commit protocol.

## Chunk 4: Write and Open Real Lance Dataset

### Task 4: Add LanceDatasetStore create/open/append

**Files:**
- Modify: `crates/hannsdb-core/src/storage/lance_store.rs`
- Test: `crates/hannsdb-core/tests/lance_compat.rs`

- [ ] **Step 1: Write failing Lance create/open test**

Add async test:

```rust
#[tokio::test]
async fn lance_dataset_store_writes_dataset_openable_by_lance() {
    let temp = tempfile::tempdir().unwrap();
    let uri = temp.path().join("docs.lance");

    // Build schema and documents.
    // LanceDatasetStore::create(&uri, &schema, &docs).await.unwrap();
    // let dataset = lance::Dataset::open(uri.to_str().unwrap()).await.unwrap();
    // Assert dataset.count_rows(None).await.unwrap() == docs.len().
}
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cargo test -p hannsdb-core --features lance-storage --test lance_compat lance_dataset_store_writes_dataset_openable_by_lance -- --nocapture
```

Expected: FAIL because `LanceDatasetStore` does not exist.

- [ ] **Step 3: Implement `LanceDatasetStore`**

Add:

```rust
#[cfg(feature = "lance-storage")]
pub(crate) struct LanceDatasetStore {
    uri: String,
    schema: CollectionSchema,
}

impl LanceDatasetStore {
    pub(crate) fn new(uri: impl Into<String>, schema: CollectionSchema) -> Self;
    pub(crate) async fn create(&self, documents: &[Document]) -> io::Result<()>;
    pub(crate) async fn append(&self, documents: &[Document]) -> io::Result<()>;
    pub(crate) async fn open_lance(&self) -> io::Result<lance::Dataset>;
}
```

Implementation sketch:

- Convert documents to `RecordBatch`.
- Use `arrow::record_batch::RecordBatchIterator::new(vec![Ok(batch)], schema)`.
- Use `lance::Dataset::write(reader, uri, Some(write_params)).await`.
- Use `WriteMode::Create` for `create`.
- Use `WriteMode::Append` for `append`.
- Map Lance errors into `io::Error`.

- [ ] **Step 4: Run create/open test**

Run:

```bash
cargo test -p hannsdb-core --features lance-storage --test lance_compat lance_dataset_store_writes_dataset_openable_by_lance -- --nocapture
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/hannsdb-core/src/storage/lance_store.rs crates/hannsdb-core/tests/lance_compat.rs
git commit -m "Write HannsDB rows as Lance datasets"
```

Use Lore commit protocol.

## Chunk 5: Append and Scan Compatibility

### Task 5: Prove external Lance scan sees appended rows

**Files:**
- Modify: `crates/hannsdb-core/tests/lance_compat.rs`

- [ ] **Step 1: Add failing append/scan test**

Add:

```rust
#[tokio::test]
async fn lance_dataset_store_append_is_visible_to_lance_scan() {
    // Create dataset with first batch.
    // Append second batch.
    // Open with lance::Dataset::open.
    // Assert count_rows == total rows.
    // Build scanner and collect batches.
    // Assert projected id/title/vector content is present.
}
```

- [ ] **Step 2: Run and verify RED or existing pass**

Run:

```bash
cargo test -p hannsdb-core --features lance-storage --test lance_compat lance_dataset_store_append_is_visible_to_lance_scan -- --nocapture
```

Expected: FAIL until append and scan assertions are implemented correctly.

- [ ] **Step 3: Implement missing append/scan support**

If `append` already works from Chunk 4, this step may only require test fixes around Lance scanner APIs.

- [ ] **Step 4: Run full Lance compatibility test file**

Run:

```bash
cargo test -p hannsdb-core --features lance-storage --test lance_compat -- --nocapture
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/hannsdb-core/tests/lance_compat.rs crates/hannsdb-core/src/storage/lance_store.rs
git commit -m "Verify Lance can scan HannsDB-written datasets"
```

Use Lore commit protocol.

## Chunk 6: Documentation and Final Gates

### Task 6: Document the P0 compatibility contract

**Files:**
- Modify: `README.md`
- Modify: `docs/hannsdb-project-design.md`
- Optional: `docs/superpowers/specs/2026-04-16-lance-compatible-storage-design.md`

- [ ] **Step 1: Add README note**

Document:

```markdown
Experimental Lance-compatible storage is available behind `--features lance-storage`.
P0 writes committed rows as real Lance datasets and keeps Hanns index integration out of scope.
```

- [ ] **Step 2: Add design doc note**

In `docs/hannsdb-project-design.md`, add a short architecture note that HannsDB's long-term storage target is upstream Lance dataset compatibility, with Hanns as ANN provider.

- [ ] **Step 3: Run final gates**

Run:

```bash
cargo fmt --check
cargo check -p hannsdb-core --features lance-storage
cargo test -p hannsdb-core --features lance-storage --test lance_compat -- --nocapture
cargo test --workspace
git diff --check
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/hannsdb-project-design.md docs/superpowers/specs/2026-04-16-lance-compatible-storage-design.md docs/superpowers/plans/2026-04-16-lance-compatible-storage-p0.md
git commit -m "Document Lance-compatible storage P0"
```

Use Lore commit protocol.

## Open Questions for Implementation

- If `lance::Dataset::write` pulls in too many default features or creates build friction, should `lance-storage` be dev-only for P0 or a normal optional feature?
- Does Lance scanner preserve `FixedSizeList<Float32>` exactly for vector columns in the expected shape, or should HannsDB use Lance's preferred vector column representation helper if one exists?
- Should `id` be treated as a normal column in P0, or should stable row id metadata be enabled immediately? Recommendation: normal column in P0; stable row id metadata later.

## Completion Criteria

- A real Lance dataset is written by HannsDB code under `lance-storage`.
- Upstream Lance opens and scans the dataset in tests.
- Append creates a committed Lance-visible version.
- Existing HannsDB storage behavior is unchanged when `lance-storage` is disabled.
