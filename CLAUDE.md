# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build all crates (default backend: in-memory brute-force)
cargo build

# Build with knowhere ANN backend
cargo build --features knowhere-backend

# Release build
cargo build --release --features knowhere-backend

# Run all tests
cargo test -p hannsdb-core
cargo test -p hannsdb-core --features knowhere-backend

# Run a single test by name
cargo test -p hannsdb-core <test_name>

# Run optimize benchmark entry (prints timing to stdout)
HANNSSDB_OPT_BENCH_N=2000 HANNSSDB_OPT_BENCH_DIM=256 HANNSSDB_OPT_BENCH_METRIC=cosine \
  cargo test -p hannsdb-core collection_api_optimize_benchmark_entry -- --nocapture

# Benchmark helper scripts
./scripts/run_hannsdb_optimize_bench.sh N=2000 DIM=256 METRIC=cosine REPEATS=3 PROFILE=release FEATURES=knowhere-backend
./scripts/run_vdbb_hannsdb_smoke.sh        # VectorDBBench smoke test
./scripts/run_vdbb_hannsdb_perf1536d50k.sh # Full 1536-dim/50K benchmark
```

## Project Structure

Cargo workspace with four crates under `crates/`:

| Crate | Role |
|-------|------|
| `hannsdb-core` | Database engine — storage, query execution, collection management |
| `hannsdb-index` | ANN adapter layer — pluggable HNSW backends behind a trait |
| `hannsdb-daemon` | Optional HTTP API (Axum) — thin wrapper over core |
| `hannsdb-py` | Python bindings (PyO3/maturin) for embedded use and benchmarks |

All surfaces share the same `hannsdb-core` engine with identical semantics.

## Architecture

### Core Engine (`hannsdb-core`)

`HannsDb` (`db.rs`) is the single public entry point. Internal modules:

- **`catalog/`** — JSON metadata: manifest (database-level), collection schema, segment version
- **`segment/`** — Binary/JSONL files: `records.bin` (f32 vectors), `ids.bin` (i64 IDs), `payloads.jsonl` (scalar fields), `tombstones.json` (deletion bitset)
- **`query/`** — Distance metrics (L2, Cosine, IP), brute-force scan, filter expression parser/evaluator
- **`wal.rs`** — Append-only write-ahead log for crash recovery
- **`document.rs`** — `FieldValue`, `CollectionSchema`, `Document` model

**Storage layout per collection:**
```
<db_root>/collections/<name>/
├── collection.json    # schema, metric, dimension
├── segment.json       # record count, deleted count
├── records.bin        # f32 vectors, row-major, append-only
├── ids.bin            # i64 external IDs, append-only
├── payloads.jsonl     # one JSON object per row
├── tombstones.json    # deletion bitset
└── wal.jsonl          # mutation log
```

### ANN Index (`hannsdb-index`)

`HnswBackend` trait (`adapter.rs`) gates two implementations:

- **`InMemoryHnswIndex`** — always available; brute-force baseline
- **`KnowhereHnswIndex`** — behind `knowhere-backend` feature; wraps knowhere-rs with `ef_search=64`, `ef_construction=128`, `m=16`, deterministic `random_seed=42`

The HNSW index is **derived state**, not source of truth. Raw vectors in `records.bin` are authoritative. `optimize_collection()` builds the in-memory index on demand; subsequent searches use it until the collection mutates.

### Key Design Decisions

- **Soft deletes:** Tombstones mask deleted rows; physical compaction deferred to `optimize_collection()`.
- **Single segment (v1):** Multi-segment management is deferred to v2.
- **ANN as cache:** Brute-force search always works; `optimize_collection()` populates an in-memory HNSW cache.
- **Feature-gated backend:** `--features knowhere-backend` enables knowhere-rs; without it, fallback is in-memory linear scan.
- **One primary vector per document (v1):** Multi-vector ANN not yet supported.

### Filter Syntax

Filter expressions passed to `search` / `query_documents`:

```
field op value [and field op value ...]
```

Operators: `==`, `!=`, `<`, `<=`, `>`, `>=`. Field types: String, Int64, Float64, Bool.

## External Dependencies

- **knowhere-rs** — expected at `/Users/ryan/Code/knowhere-rs`; required for `--features knowhere-backend`
- **VectorDBBench** — at `/Users/ryan/Code/VectorDBBench`; used by benchmark scripts

## Benchmark Environment Variables

| Variable | Purpose |
|----------|---------|
| `HANNSSDB_OPT_BENCH_N` | Number of vectors |
| `HANNSSDB_OPT_BENCH_DIM` | Dimensionality |
| `HANNSSDB_OPT_BENCH_METRIC` | Distance metric (`l2`, `cosine`, `ip`) |
| `HANNSSDB_OPT_BENCH_TOPK` | Top-k results |

## Documentation

Design docs live in `docs/`:
- `hannsdb-project-design.md` — architectural rationale and current state
- `hannsdb-project-plan.md` — implementation roadmap (phases 1–8) and progress tracking
- `architecture-analysis.md` — detailed data flow and trade-off analysis
- `vector-db-bench-notes.md` — benchmarking history and bottleneck analysis
