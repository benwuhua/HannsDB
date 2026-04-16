<div align="center">

# HannsDB

**Zero-dependency embedded vector database for local AI agents.**

Pure Rust engine · PyO3 bindings · Axum HTTP · Hanns ANN

[![Rust tests](https://img.shields.io/badge/rust%20tests-459-passing)](crates/hannsdb-core/tests) [![Python tests](https://img.shields.io/badge/python%20tests-325-passing)](crates/hannsdb-py/tests) [![Lines of Rust](https://img.shields.io/badge/rust-44%2C469%20loc-blue)](crates)

</div>

---

## Why HannsDB?

Most vector databases are built for distributed clusters with gRPC, Kubernetes, and 47 microservices.
**Local AI agents don't need any of that.**

HannsDB is a single-process, zero-config, file-based vector database designed for the machine your agent is already running on.
No Docker. No server to manage. No network hop. Just `pip install` and go.

```
# Three lines. That's the entire setup.
import hannsdb
db = hannsdb.create_and_open("./my_data", schema)
db.insert(docs)
```

---

## Performance

Benchmarks on 50K vectors · 1536 dimensions · Cosine metric (VectorDBBench):

| Metric | Value |
|--------|-------|
| **Search serial p99** | **0.7 ms** |
| **Recall@10** | **94.65%** |
| **Concurrent QPS** | **1,537** |

Single-vector search in **128 μs** at the Rust layer (x86, ef_search=64).
Zero-copy `Arc<Vec>` architecture — no per-query data cloning.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Your Agent                      │
│          (Python / Rust / HTTP)                  │
├────────────┬────────────────┬───────────────────┤
│  PyO3 API  │   Rust FFI     │   HTTP (Axum)     │
├────────────┴────────────────┴───────────────────┤
│                  hannsdb-core                     │
│  ┌──────────┬──────────┬───────────┬──────────┐ │
│  │ Catalog  │ Segment  │  Query    │   WAL    │ │
│  │ Metadata │ Storage  │ Executor  │ Recovery │ │
│  └──────────┴──────────┴───────────┴──────────┘ │
│  ┌─────────────────────────────────────────────┐ │
│  │          hannsdb-index (pluggable)           │ │
│  │  HNSW-SQ │ HNSW-HVQ │ IVF-USQ │ Brute-force │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

**Key design decisions:**

- **Soft deletes** — Tombstones mask deleted rows; compaction is deferred
- **ANN as cache** — Brute-force always works; `optimize()` builds HNSW on demand
- **WAL with fsync** — Every write is `sync_all()`'d to disk; survives power loss
- **Multi-segment** — Automatic rollover at 200K rows; background compaction
- **Forward store** — Arrow IPC + Parquet snapshots for fast columnar reads

---

## Features

### Vector Search

| Feature | Support |
|---------|---------|
| Dense vectors (f32 / f16) | ✅ |
| Sparse vectors (BM25) | ✅ |
| Distance metrics (L2 / Cosine / IP) | ✅ |
| ANN backends (HNSW-SQ / HNSW-HVQ / IVF-USQ) | ✅ |
| Filtered search | ✅ |
| Multi-vector fields per collection | ✅ |

### Data Model

| Feature | Support |
|---------|---------|
| 8 scalar types (String, Bool, Int32/64, Float/64, UInt32/64) | ✅ |
| Array fields (list of scalars) | ✅ |
| Nullable fields | ✅ |
| Schema mutation (add / drop / rename / widen columns) | ✅ |
| String primary keys | ✅ |
| Per-field vector schemas | ✅ |

### Reliability

| Feature | Support |
|---------|---------|
| Write-ahead log with fsync | ✅ |
| Crash recovery (40+ test scenarios) | ✅ |
| Tombstone-based soft deletes | ✅ |
| Multi-segment compaction | ✅ |
| WAL mid-line corruption tolerance | ✅ |
| Forward store authority on reopen | ✅ |

### Interfaces

| Interface | Support |
|-----------|---------|
| Python (PyO3 / maturin) | ✅ |
| Rust (core crate) | ✅ |
| HTTP REST API (28 endpoints) | ✅ |
| VectorDBBench integration | ✅ |

---

## Quick Start

### Python

```bash
pip install hannsdb
```

```python
import hannsdb
from hannsdb import CollectionSchema, FieldSchema, VectorSchema, DataType

schema = CollectionSchema(
    fields=[
        FieldSchema("title", DataType.STRING),
        FieldSchema("year", DataType.INT64),
    ],
    vectors=[
        VectorSchema("dense", dimension=768),
    ],
)

db = hannsdb.create_and_open("./agent_data", schema)
col = db.collection()

# Insert
col.insert([
    {"title": "Attention Is All You Need", "year": 2017, "dense": [0.1] * 768},
    {"title": "BERT", "year": 2018, "dense": [0.2] * 768},
])

# Search
from hannsdb import VectorQuery
results = col.query(vectors=[VectorQuery(vector=[0.15] * 768, field_name="dense")], topk=10)
for hit in results:
    print(hit.id, hit.score, hit.fields)

# Filtered search
results = col.query(
    vectors=[VectorQuery(vector=[0.15] * 768, field_name="dense")],
    topk=10,
    filter="year >= 2018",
)

# Build ANN index for fast search
col.optimize()

# Schema evolution — add column with backfill
col.add_column(FieldSchema("category", DataType.STRING), fill="uncategorized")

# Close
db.close()
```

### Rust

```rust
use hannsdb_core::db::HannsDb;
use hannsdb_core::document::{CollectionSchema, Document, FieldValue};

let db = HannsDb::open("./agent_data")?;
db.create_collection("docs", 768, "cosine")?;

let ids = vec![1, 2, 3];
let vectors = vec![0.1f32; 768 * 3];
db.insert("docs", &ids, &vectors)?;

let hits = db.search("docs", &[0.1; 768], 10)?;
for hit in hits {
    println!("id={}, distance={}", hit.id, hit.distance);
}
```

### HTTP API

```bash
# Start daemon
cargo run -p hannsdb-daemon -- --port 19530 --data-dir ./agent_data

# Create collection
curl -X POST http://localhost:19530/collections \
  -H 'Content-Type: application/json' \
  -d '{"name":"docs","dimension":768,"metric":"cosine"}'

# Insert
curl -X POST http://localhost:19530/collections/docs/records \
  -H 'Content-Type: application/json' \
  -d '{"ids":[1,2],"vectors":[[0.1;768],[0.2;768]]}'

# Search
curl -X POST http://localhost:19530/collections/docs/search \
  -H 'Content-Type: application/json' \
  -d '{"vector":[0.15;768],"top_k":10}'
```

---

## Project Structure

```
crates/
├── hannsdb-core/     # Database engine (3,500 LOC)
│   ├── catalog/      #   JSON metadata management
│   ├── segment/      #   Binary segment I/O
│   ├── storage/      #   Compaction, tombstone, persist, WAL, recovery
│   ├── query/        #   Distance metrics, filter parser, executor
│   └── forward_store/ #  Arrow IPC / Parquet columnar snapshots
├── hannsdb-index/    # ANN adapter layer (4,900 LOC)
│   ├── hnsw.rs       #   HNSW (brute-force / Hanns backend)
│   ├── hnsw_sq.rs    #   HNSW with scalar quantization
│   ├── hnsw_hvq.rs   #   HNSW with hierarchical vector quantization
│   ├── ivf_usq.rs    #   IVF with ultra-scalar quantization
│   ├── scalar.rs     #   Inverted scalar index (10 variants)
│   └── sparse.rs     #   Sparse index (BM25, WAND)
├── hannsdb-py/       # Python bindings (4,200 LOC)
└── hannsdb-daemon/   # HTTP API (2,800 LOC)
```

**44,469 lines of Rust · 784 tests · 0 TODO markers**

---

## Testing

| Layer | Tests | Coverage |
|-------|-------|----------|
| Core engine | 381 | WAL recovery, compaction, schema mutation, filter, multi-segment |
| Index | 36 | HNSW-SQ ef_search override, serialization round-trip |
| Daemon | 42 | HTTP CRUD lifecycle, error handling |
| Python | 325 | dtype round-trip, exception handling, concurrency, schema mutation |
| **Total** | **784** | |

---

## Build

```bash
# Default build (brute-force ANN)
cargo build --release

# With Hanns ANN backend
cargo build --release --features hanns-backend

# Experimental Lance-compatible storage prototype
cargo test -p hannsdb-core --features lance-storage --test lance_compat -- --nocapture

# Run tests
cargo test --workspace

# Python bindings
cd crates/hannsdb-py && maturin develop --release
```

---

## License

Private project. All rights reserved.
