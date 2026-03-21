# HannsDB Architecture Analysis

_Date: 2026-03-20_

## 1. Executive Summary

HannsDB is a local-first, lightweight vector database built in Rust. It is
designed for agent-oriented data management and uses **knowhere-rs** as its
pluggable ANN engine. The project ships four coordinated crates—core engine,
ANN adapter, HTTP daemon, and Python bindings—sharing a single Rust workspace.

**Key architectural bets:**

| Decision | Rationale |
|----------|-----------|
| ANN index as derived state | Vectors on disk are the source of truth; indices are rebuilt on demand |
| Single-segment per collection (v1) | Simplifies implementation; WAL + multi-segment planned for v2 |
| Soft deletes via tombstones | Avoids costly data compaction; tombstones are cleaned on rebuild |
| knowhere-rs feature-gated | Allows fallback to brute-force when ANN backend unavailable |
| Dual surface (Python + HTTP) | One Rust core exposed through PyO3 bindings and Axum REST API |

**Execution note:** This document is the target architecture baseline for the
next implementation tranche. Where current code still differs, follow the
design intent here only after the corresponding change is explicitly planned and
verified.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Clients                          │
│     Python (PyO3)          HTTP (REST)              │
│     hannsdb-py             hannsdb-daemon            │
└──────────┬──────────────────────┬───────────────────┘
           │                      │
           ▼                      ▼
┌─────────────────────────────────────────────────────┐
│                  hannsdb-core                        │
│  ┌──────────────────────────────────────────────┐   │
│  │  HannsDb                                     │   │
│  │  ├── Collection lifecycle (create/drop/list)  │   │
│  │  ├── Document ops (insert/upsert/delete)      │   │
│  │  ├── Query engine (search + filter)           │   │
│  │  └── Segment I/O (persist + reload)           │   │
│  └──────────────────────────────────────────────┘   │
│          │                                          │
│          ▼                                          │
│  ┌──────────────┐  ┌──────────────────────────┐     │
│  │ catalog/     │  │ segment/                 │     │
│  │  manifest    │  │  records.bin  (vectors)  │     │
│  │  collection  │  │  ids.bin      (ext IDs)  │     │
│  │  version     │  │  payloads.jsonl (fields) │     │
│  └──────────────┘  │  tombstones.json         │     │
│                    │  segment.json            │     │
│                    └──────────────────────────┘     │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│                  hannsdb-index                       │
│  ┌──────────────────────────────────────────────┐   │
│  │  HnswBackend trait                            │   │
│  │  ├── InMemoryHnswIndex  (brute-force fallback)│   │
│  │  └── KnowhereHnswIndex  (knowhere-rs HNSW)    │   │
│  └──────────────────────────────────────────────┘   │
│                           │                         │
│                           ▼                         │
│  ┌──────────────────────────────────────────────┐   │
│  │  knowhere-rs (optional, feature-gated)        │   │
│  │  ├── IndexConfig / SearchRequest / SearchResult│  │
│  │  ├── HnswIndex (1.789× faster than C++)       │   │
│  │  ├── SIMD distance kernels                    │   │
│  │  └── Thread-pool parallel build               │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 3. Crate Breakdown

### 3.1 hannsdb-core

**Purpose:** Core database engine — collection management, storage, query.

**Key modules:**

| Module | Files | Responsibility |
|--------|-------|----------------|
| `db.rs` (825 lines) | Main entry | `HannsDb` struct with all public methods |
| `catalog/` | manifest, collection, version | Metadata persistence (JSON) |
| `segment/` | records, metadata, tombstone, payloads | Binary/JSONL storage per collection |
| `query/` | search, filter | Distance metrics, brute-force scan, filter parser |
| `document.rs` | | Document model, `FieldValue`, `CollectionSchema` |

**Storage layout per collection:**

```
<db_root>/collections/<collection_name>/
├── collection.json     # name, dimension, metric, schema
├── segment.json        # segment ID, record count, deleted count
├── records.bin         # f32 vectors, row-major, append-only
├── ids.bin             # i64 external IDs, append-only
├── payloads.jsonl      # one JSON object per row
└── tombstones.json     # deletion bitset
```

**Data operations:**

- **Insert:** O(1) append to records.bin + ids.bin + payloads.jsonl
- **Delete:** O(1) tombstone flip (soft delete)
- **Upsert:** tombstone old + append new
- **Search (brute):** O(n) linear scan with distance calculation
- **Search (ANN):** Approximate nearest-neighbor search via knowhere-rs HNSW
  after `optimize_collection()`

**Supported metrics:** L2, Cosine (1 − similarity), IP (negated)

**Filter expressions:** `field op value [and field op value]`
- Operators: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Field types: String, Int64, Float64, Bool

### 3.2 hannsdb-index

**Purpose:** Adapter layer between core and ANN backends.

**Key abstraction:**

```rust
trait HnswBackend {
    fn insert(&mut self, vectors: &[(u64, Vec<f32>)]) -> Result<()>;
    fn insert_flat(&mut self, ids: &[u64], vectors: &[f32], dim: usize) -> Result<()>;
    fn insert_flat_identity(&mut self, vectors: &[f32], dim: usize) -> Result<()>;
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<HnswSearchHit>>;
}
```

**Implementations:**

| Backend | Feature gate | Strategy |
|---------|-------------|----------|
| `InMemoryHnswIndex` | always available | Brute-force scan (correctness baseline) |
| `KnowhereHnswIndex` | `knowhere-backend` | knowhere-rs HNSW with `ef_search=64, ef_construction=128, M=16` |

**knowhere-rs integration details:**
- Creates `IndexConfig(Hnsw, metric, dim)` → `train()` → `add()` → `search()`
- `random_seed=42` for deterministic test behavior
- Distance mapping: L2 passed through, IP negated, Cosine computed as `1 − dot/(‖a‖·‖b‖)`

### 3.3 hannsdb-daemon

**Purpose:** HTTP REST API wrapping the core engine.

**Framework:** Axum 0.7 + Tokio

**Routes:**

| Method | Path | Operation |
|--------|------|-----------|
| GET | `/health` | Health check |
| GET/POST | `/collections` | List / Create collection |
| GET/DELETE | `/collections/:name` | Info / Drop collection |
| GET | `/collections/:name/stats` | Collection statistics |
| POST | `/collections/:name/admin/flush` | Validate minimal flush boundary (`collection.json`, `segment.json`, `tombstones.json`, `wal.jsonl` are readable) |
| POST/DELETE | `/collections/:name/records` | Insert / Delete records |
| POST | `/collections/:name/records/upsert` | Upsert records |
| POST | `/collections/:name/records/fetch` | Fetch by ID |
| POST | `/collections/:name/search` | Search (with optional filter and optional `output_fields`) |

### 3.4 hannsdb-py

**Purpose:** Python bindings via PyO3 (abi3-py39+).

Exposes the same collection lifecycle, CRUD, and query operations as the core,
including `insert`, `upsert`, `fetch`, `delete`, `query`, `flush`, `stats`,
and `optimize`, wrapped in a Python-friendly `Collection` class with `init()`,
`open()`, and `create_and_open()` module-level functions.

---

## 4. knowhere-rs Architecture

knowhere-rs is a production-grade Rust replacement for Milvus KnowHere (C++).

### 4.1 Public API

```rust
// Key types re-exported from lib.rs
IndexConfig, IndexType, MetricType   // Configuration
SearchRequest, SearchResult          // Query interface
Index trait                          // Polymorphic index abstraction
Dataset, DataType                    // Vector data (f32, bf16, f16, binary)
```

### 4.2 Index Trait

The `Index` trait in `src/index.rs` defines the unified interface:

```rust
trait Index: Send + Sync {
    fn train(&mut self, dataset: &Dataset) -> Result<()>;
    fn add(&mut self, dataset: &Dataset) -> Result<usize>;
    fn search(&self, query: &Dataset, top_k: usize) -> Result<SearchResult>;
    fn search_with_bitset(...) -> Result<SearchResult>;  // filtered search
    fn save(&self, path: &str) -> Result<()>;
    fn load(&mut self, path: &str) -> Result<()>;
    // ...
}
```

### 4.3 Supported Index Types

| Index | File | Status | Notes |
|-------|------|--------|-------|
| **HNSW** | `faiss/hnsw.rs` (330KB) | Leading | 1.789× faster than C++ KnowHere |
| **AISAQ/DiskANN** | `faiss/diskann_aisaq.rs` (147KB) | Functional | Beam search + PQ; exact rerank TBD |
| HNSW-PQ | `faiss/hnsw_pq.rs` | Active | Product quantization variant |
| HNSW-SQ | `faiss/hnsw_quantized.rs` | Active | Scalar quantization variant |
| IVF-Flat | `faiss/ivf_flat.rs` | Active | Clustering + exhaustive |
| IVF-PQ | `faiss/ivfpq.rs` | Blocked | Recall < 0.8 |
| ScaNN | `faiss/scann.rs` | Active | Google ScaNN parity |
| Sparse | `faiss/sparse*.rs` | Active | Inverted, WAND |
| Flat | `faiss/mem_index.rs` | Baseline | Brute-force reference |

### 4.4 Performance Infrastructure

- **SIMD kernels** (`simd.rs`, 86KB): L2, IP, Cosine, Hamming, PQ distance
- **BFloat16 storage**: Optional compressed vector storage in HNSW
- **Thread-local scratch buffers**: Avoid mutex contention in parallel search
- **Layer-0 flat graph**: Cache-friendly memory layout for hot search path
- **FFI C bindings** (`ffi.rs`, 205KB): Full C API for Milvus integration

---

## 5. Data Flow

### 5.1 Insert Path

```
Client → insert(collection, ids, vectors, payloads)
  → Validate dimension match
  → Append vectors to records.bin (binary f32, row-major)
  → Append IDs to ids.bin (binary i64)
  → Append payloads to payloads.jsonl (one JSON per line)
  → Update segment metadata (record count)
  → Persist metadata/tombstone side files as needed
  (Full durability semantics remain limited until WAL exists)
```

### 5.2 Search Path (Brute-Force)

```
Client → search(collection, query_vector, top_k, filter?)
  → Load all vectors from records.bin
  → Load tombstone mask
  → For each non-deleted vector:
      → If filter: evaluate filter expression against payload
      → Compute distance (L2 / Cosine / IP)
      → Collect candidate hits
  → Sort by distance and truncate to top-k
```

### 5.3 Search Path (ANN / Optimized)

```
Client → optimize_collection(collection)
  → Load all non-deleted vectors
  → Create knowhere-rs HnswIndex(config)
  → Train + Add vectors to HNSW
  → Cache index in memory

Client → search(collection, query_vector, top_k)
  → If ANN cache exists: use KnowhereHnswIndex.search()
  → Else: fall back to brute-force
```

### 5.4 Delete Path

```
Client → delete(collection, ids)
  → Linear-scan stored IDs to find matching live rows
  → Set tombstone bits
  → Persist tombstones.json
  → Increment deleted count in segment metadata
  (No physical data removal — cleaned on next optimize/rebuild)
```

---

## 6. Design Trade-offs and Rationale

### 6.1 ANN as Derived State

The ANN index is **not** the source of truth. Raw vectors persisted in
`records.bin` are authoritative. The HNSW index is built on-demand via
`optimize_collection()` and cached in memory.

**Pros:** Simple crash recovery (just reload raw data); index format changes
don't require migration; can swap ANN backends without data loss.

**Cons:** Cold start requires full index rebuild; no incremental index updates
(must rebuild after inserts).

### 6.2 Single-Segment Model

Each collection has exactly one segment containing all records.

**Pros:** Simple implementation; no merge/compaction logic; predictable I/O.

**Cons:** Append-only growth with tombstones wastes space; no parallel segment
processing; large collections pay O(n) reload cost.

### 6.3 Soft Deletes Only

Deletions mark a tombstone bit; physical data remains until rebuild.

**Pros:** O(1) delete; no file rewriting; safe concurrent reads.

**Cons:** Storage bloat over time; brute-force search scans deleted rows
(mitigated by tombstone check).

### 6.4 Feature-Gated knowhere-rs

The `knowhere-backend` Cargo feature controls whether KnowhereHnswIndex is
compiled in. Without it, InMemoryHnswIndex (brute-force) is used.

**Pros:** Core can be built/tested without C++ toolchain; clear dependency
boundary; CI can test with and without ANN.

**Cons:** Two code paths to maintain; feature-gated code may have different
bugs than the default path.

---

## 7. Integration Points

### 7.1 HannsDB → knowhere-rs

```
hannsdb-core
  └── calls HnswAdapter (hannsdb-index)
        └── calls KnowhereHnswIndex
              └── calls knowhere_rs::HnswIndex
                    └── uses IndexConfig, Dataset, SearchResult
```

**Parameter mapping:**

| HannsDB | knowhere-rs |
|---------|-------------|
| `dim: usize` | `IndexConfig::dim` |
| `metric: "l2"` | `MetricType::L2` |
| `metric: "cosine"` | `MetricType::Cosine` |
| `metric: "ip"` | `MetricType::Ip` |
| `top_k: usize` | `SearchRequest::top_k` |

**Distance semantics reconciliation:**
- L2: pass-through (both use Euclidean distance)
- IP: HannsDB negates knowhere-rs result (distance = −dot_product)
- Cosine: HannsDB computes `1 − (dot / (‖a‖·‖b‖))`

### 7.2 VectorDBBench Integration

HannsDB targets the VectorDBBench standard benchmarks for validation:
- Performance1536D50K (1536-dim, 50K vectors, cosine metric)
- Custom dataset smoke tests
- Python bindings enable direct VectorDBBench integration

---

## 8. Testing Strategy

| Layer | Test File | Coverage |
|-------|-----------|----------|
| Core API | `hannsdb-core/tests/collection_api.rs` | Create, insert, search, delete, reopen, recovery |
| Document model | `hannsdb-core/tests/document_api.rs` | Typed fields, schema validation |
| Storage | `hannsdb-core/tests/segment_storage.rs` | Segment persistence and reload |
| Catalog | `hannsdb-core/tests/catalog_manifest.rs` | Manifest read/write |
| ANN adapter | `hannsdb-index/tests/hnsw_adapter.rs` | Backend trait compliance |
| HTTP API | `hannsdb-daemon/tests/http_smoke.rs` | Route smoke tests |

**Test patterns:** Temp directory per test for isolation; error case coverage
(zero dimension, mismatched counts); reopen-after-crash simulation.

---

## 9. Current Status and Gaps

### Completed

- Full four-crate workspace with consistent interfaces
- Collection lifecycle and vector persistence
- Tombstone-based soft deletes
- Brute-force and ANN search (knowhere-rs HNSW)
- Typed scalar payloads with filter expressions
- Python PyO3 bindings
- HTTP daemon with full REST API
- VectorDBBench integration path

### In Progress

- Stabilizing knowhere-rs HNSW build performance at 50K/1536/cosine scale
- Closing Performance1536D50K benchmark gate

### Not Yet Implemented

- Write-ahead log (WAL) and crash recovery
- Multi-segment management and compaction
- Secondary scalar indexes
- Multi-vector per collection
- Background compaction / index rebuild tasks
- Incremental index updates (currently full rebuild required)
- Production hardening of daemon (auth, rate limiting, TLS)

---

## 10. Dependency Graph

```
hannsdb-py ──────────┐
                     ▼
hannsdb-daemon ──► hannsdb-core ──► hannsdb-index ──► knowhere-rs (optional)
     │                  │
     │                  ├── serde, serde_json
     │                  ├── anyhow, thiserror
     │                  └── tempfile (tests)
     │
     ├── axum 0.7
     └── tokio 1.38
```

**Rust edition:** 2021 | **MSRV:** 1.75+ | **License:** Apache-2.0
