# HannsDB + Hanns vs zvec Capability Report

**Date:** 2026-04-12  
**Scope:** Combined `Hanns engine + HannsDB product surface` vs `zvec`  
**Method:** Evidence-only comparison from source code, tests, docs, and existing benchmark notes. Unsupported claims are marked `unknown`.

---

## 1. Executive Summary

### Proven strengths of `Hanns + HannsDB` over zvec

1. **Hanns has broader quantized ANN internals than zvec’s currently exposed public RabitQ family.**  
   Hanns already contains a reusable `UsqQuantizer`, a full `IvfUsq` path, and an HNSW+USQ/HVQ path, while zvec’s public quantized surface is centered on `HnswRabitq*`.  
   Evidence:
   - `Hanns/src/quantization/usq/quantizer.rs`
   - `Hanns/src/faiss/ivf_usq.rs`
   - `Hanns/src/faiss/hnsw_hvq.rs`
   - `zvec/python/zvec/model/param/__init__.pyi`
   - `zvec/python/tests/test_collection_hnsw_rabitq.py`

2. **At the same tested HNSW benchmark shape, HannsDB currently shows better recall than zvec, even though zvec is still faster.**  
   In the documented `Performance1536D50K / cosine / M=16 / ef_construction=64 / ef_search=32` comparison, HannsDB recall is higher on both Mac and x86, while zvec still wins on build/load/p99.  
   Evidence:
   - `docs/vector-db-bench-notes.md` sections 38 and “VectorDBBench Integration Fixes and First Full Run (2026-03-23)”

### Near-parity areas where zvec no longer has a basic-functionality lead

3. **HannsDB has moved past the “missing basics” phase.**  
   Current evidence shows real support for:
   - `update`
   - `add_column` / `drop_column` / `alter_column`
   - `query_by_id`
   - `group_by`
   - sparse query/runtime
   - built-in reranker combinations
   Evidence:
   - `docs/hannsdb-vs-zvec-gap-analysis.md`
   - `crates/hannsdb-core/tests/zvec_parity_sparse.rs`
   - `crates/hannsdb-core/tests/zvec_parity_string_pk.rs`
   - `crates/hannsdb-py/tests/test_query_executor.py`
   - `crates/hannsdb-py/tests/test_schema_mutation_surface.py`
   - `crates/hannsdb-py/tests/test_collection_facade.py`

### Where zvec still clearly leads

1. **Public API productization breadth**
2. **Storage/runtime maturity**
3. **Standard-case build/load/p99 latency**

Those are the real gaps now—not whether HannsDB has basic query or mutation support at all.

---

## 2. Capability Matrix

Legend:

- **Hanns** = engine capability
- **HannsDB** = exposed product/API capability
- Matrix model is explicitly layered around:
  - `Hanns 已有`
  - `HannsDB 已暴露`
  - `两边都没有`
- **Judgment** uses:
  - `Hanns stronger`
  - `HannsDB stronger`
  - `zvec stronger`
  - `mixed`
  - `unknown`

| Dimension | Capability | Hanns | HannsDB | zvec | Judgment | Evidence | Notes |
|---|---|---|---|---|---|---|---|
| 接口 | Python/public param family breadth | N/A | Exposes `CollectionOption`, `OptimizeOption`, `Flat/Hnsw/IVF/IVFQuery/InvertIndex/IvfUsq` plus schema/query/reranker surfaces | Exposes richer public family including `AddColumnOption`, `AlterColumnOption`, `Flat`, `Hnsw`, `HnswRabitq`, `IVF`, `IVFQuery`, `InvertIndexParam`, `SegmentOption` | zvec stronger | `crates/hannsdb-py/python/hannsdb/__init__.py`, `crates/hannsdb-py/python/hannsdb/model/param/index_params.py`, `zvec/python/zvec/model/param/__init__.pyi` | HannsDB has improved materially again, but zvec still exposes a broader stable public contract |
| 接口 | Quantized ANN public surface honesty | `IvfUsq`, `UsqQuantizer`, `HnswHvq` exist internally | Public `IvfUsq*` and `HnswHvq*` now exist with real runtime wiring; `HnswHvq` is intentionally narrow (`ip`-only) | `HnswRabitqIndexParam` / `HnswRabitqQueryParam` are executable public surfaces with collection tests | mixed | `Hanns/src/quantization/usq/quantizer.rs`, `Hanns/src/faiss/ivf_usq.rs`, `Hanns/src/faiss/hnsw_hvq.rs`, `crates/hannsdb-py/tests/test_ivf_usq_surface.py`, `crates/hannsdb-py/tests/test_hnsw_hvq_surface.py`, `zvec/python/tests/test_collection_hnsw_rabitq.py` | Hanns is stronger internally; zvec still leads in breadth and maturity of delivered quantized productization |
| 接口 | Query surface combinations | N/A | `query_by_id`, `group_by`, `order_by`, built-in reranker combinations, and custom reranker + `query_by_id` / `group_by` are publicly tested | zvec query surface is mature, but this session did not verify all equivalent combination tests directly | mixed | `crates/hannsdb-py/tests/test_query_executor.py`, `crates/hannsdb-py/tests/test_collection_facade.py` | zvec side is not `unknown` in general, but direct apples-to-apples combination evidence is still thinner in this pass |
| 功能 | Basic mutation support | Engine supports required field/vector mutation primitives | `update`, `add_column`, `drop_column`, `alter_column` are publicly present and tested, and `add_column(..., expression=...)` now supports a constant-expression subset | Present | mixed | `docs/hannsdb-vs-zvec-gap-analysis.md`, `crates/hannsdb-py/tests/test_schema_mutation_surface.py`, `zvec/python/zvec/model/param/__init__.pyi` | HannsDB no longer trails on basic presence; remaining gap is depth, especially migration semantics |
| 功能 | `query_by_id` + string PK | Engine supports string-key resolution | Public alphanumeric `query_by_id` is tested | zvec string PK model is mature | mixed | `crates/hannsdb-core/tests/zvec_parity_string_pk.rs`, `crates/hannsdb-py/tests/test_collection_facade.py`, `docs/hannsdb-vs-zvec-gap-analysis.md` | HannsDB has closed much of the earlier gap here |
| 功能 | Sparse query/runtime | Engine sparse runtime and tests exist | Public sparse usage is exercised in collection facade tests | Present | mixed | `crates/hannsdb-core/tests/zvec_parity_sparse.rs`, `crates/hannsdb-py/tests/test_collection_facade.py`, `docs/hannsdb-vs-zvec-gap-analysis.md` | No longer a “missing feature” gap |
| 功能 | Quantized ANN family breadth | `UsqQuantizer`, `IvfUsq`, `HnswHvq`, PCA/USQ variants exist | `IvfUsq` and `HnswHvq` are now exposed and runtime-backed in HannsDB, though `HnswHvq` remains intentionally narrow (`ip`-only) | Public quantized family is narrower internally but more complete at product level (`HnswRabitq*`) | Hanns stronger | `Hanns/src/quantization/usq/quantizer.rs`, `Hanns/src/faiss/ivf_usq.rs`, `Hanns/src/faiss/hnsw_hvq.rs`, `Hanns/src/faiss/hnsw_pca_usq.rs`, `crates/hannsdb-py/tests/test_ivf_usq_surface.py`, `crates/hannsdb-py/tests/test_hnsw_hvq_surface.py`, `zvec/python/zvec/model/param/__init__.pyi` | Hanns/HannsDB narrowed the product gap, but zvec still has the broader mature quantized family |
| 性能 | Standard-case HNSW build/load/p99 (`1536D50K`, cosine, same params) | Unknown as engine-only row in this report | Slower build/load and p99 than zvec in documented apples-to-apples runs | Faster build/load and p99 | zvec stronger | `docs/vector-db-bench-notes.md` sections 38 and 2026-03-23 benchmark summary | Documented comparison is evidence-backed |
| 性能 | Standard-case recall (`1536D50K`, cosine, same params) | Unknown as engine-only row in this report | Higher recall than zvec in documented same-param runs | Lower recall than HannsDB in those runs | HannsDB stronger | `docs/vector-db-bench-notes.md` 2026-03-23 benchmark summary | Stronger on recall, weaker on latency/build |
| 性能 | Latest optimize/load improvement trend | Hanns internals likely contributed, but exact attribution is partial | HannsDB has documented optimize/load/p99 improvements over earlier baseline | No same-date zvec re-baseline in current evidence set | unknown | mempal search results (`drawer_hannsdb_default_*`), `docs/vector-db-bench-notes.md` | Trend is real, but latest direct zvec comparison is missing |
| 工程 | Storage/runtime maturity | Rich ANN internals, but not the whole DB runtime | `SegmentManager` / `VersionSet` exist, and active segments now materialize Arrow snapshots on flush with invalidation on later writes; primary persistence is still rooted in `records.bin`, `payloads.jsonl`, `vectors.jsonl`, `ids.bin` | `VersionManager` + Arrow/Parquet forward store + mature segment/runtime stack | zvec stronger | `crates/hannsdb-core/src/segment/manager.rs`, `crates/hannsdb-core/src/segment/version_set.rs`, `crates/hannsdb-core/tests/collection_api.rs`, `zvec/src/db/index/common/version_manager.h`, `zvec/src/db/index/storage/forward_writer.h`, `docs/hannsdb-vs-zvec-gap-analysis.md` | HannsDB improved active-segment Arrow materialization, but zvec still leads on overall storage/runtime maturity |
| 工程 | Service/daemon surface | N/A | HannsDB has an `axum` daemon with route and HTTP smoke coverage | Service layer not established from inspected evidence | unknown | `crates/hannsdb-daemon/src/routes.rs`, `crates/hannsdb-daemon/tests/http_smoke.rs`, search across `zvec/` | Verified HannsDB capability; zvec side remains unknown in this pass |

---

## 3. Current Gap Summary

### A. What is already stronger than zvec

1. **Quantized ANN engine inventory is broader on the Hanns side.**  
   Hanns has a more varied USQ-based internal family than zvec’s currently inspected public quantized family.

2. **Recall is stronger in the documented same-param HNSW benchmark runs.**  
   HannsDB beats zvec on recall in the `1536D50K / cosine / M=16 / ef_construction=64 / ef_search=32` comparisons currently documented.

### B. What is no longer a meaningful “missing feature” gap

1. Basic mutation support
2. `query_by_id`
3. `group_by`
4. sparse query/runtime
5. built-in reranker combinations
6. custom reranker + `query_by_id` / `group_by`
7. constant-expression `add_column(...)` backfill subset

The remaining differences are mostly **maturity/productization**, not 0→1 feature absence.

### C. Where zvec is still ahead

1. **Public API breadth and completeness**
   - broader param families
   - more mature quantized surface
   - richer schema option surface

2. **Storage/runtime engineering**
   - Arrow/Parquet forward store
   - stronger version/segment lifecycle
   - more mature persistence model

3. **Standard-case performance**
   - shorter build/load path
   - lower p99 latency in current apples-to-apples documented runs

### D. Where the answer is still `unknown`

1. Direct service/daemon comparison vs zvec
2. Latest same-date HannsDB vs zvec benchmark comparison after the newest HannsDB improvements
3. Some fine-grained query-combination parity on the zvec side in this pass

---

## 4. P0 / P1 / P2 Roadmap

This roadmap is dependency-ordered and driven by the matrix above.

### P0 — Turn existing engine strengths into product-level advantages

1. **Productize Hanns-native quantized ANN strength inside HannsDB**
   - Why: Hanns already has stronger USQ/IVF-USQ/HNSW-HVQ internals, but HannsDB does not yet deliver that as a real public/runtime advantage
   - Depends on: stable exposure contract + truthful runtime wiring
   - Dimensions: 接口 / 功能 / 性能

2. **Close the storage/runtime maturity gap**
   - Why: zvec’s strongest enduring lead is engineering depth around versioning + forward store
   - Depends on: clear target runtime model for segment lifecycle and persistence
   - Dimensions: 工程 / 性能

3. **Re-run direct apples-to-apples benchmarks after P0-1/P0-2**
   - Why: current report proves recall strength but still shows zvec lead on build/load/p99
   - Depends on: P0-1 and P0-2, otherwise the benchmark gap remains partly structural
   - Dimensions: 性能

### P1 — Expand public/product surface to match delivered capability

1. **Continue broadening Python/public API to match real engine capability**
   - Why: HannsDB core and Hanns engine still outrun what the product surface can honestly expose, even after the latest `InvertIndexParam` / custom-reranker / constant-expression slices
   - Depends on: stable public API direction from previous parity slices
   - Dimensions: 接口 / 功能

2. **Deepen schema/query productization beyond the newly landed honest subsets**
   - Why: basic support exists, but zvec still leads on migration depth and broader query/schema maturity
   - Depends on: stable public API direction from previous parity slices
   - Dimensions: 接口 / 功能

### P2 — Finish maturity and completeness layers

1. **Complete advanced index-family and quantization productization**
   - Why: this is where the engine/product gap remains most visible
   - Depends on: P0 runtime + P1 public API alignment
   - Dimensions: 接口 / 功能 / 性能

2. **Resolve remaining engineering unknowns**
   - Why: service/operability comparison is still not fully evidence-backed
   - Depends on: explicit evidence collection for deployment/recovery/ops surfaces
   - Dimensions: 工程

---

## 5. Bottom Line

The combined `Hanns + HannsDB` system is **not** simply “behind zvec everywhere.”  
The evidence in this pass supports a more specific conclusion:

- **Hanns is already stronger in quantized ANN engine breadth**
- **HannsDB is already stronger on recall in the documented same-param HNSW benchmark runs**
- **HannsDB is no longer missing many of the basic features older gap analyses used to emphasize**

But zvec still holds the clearest overall lead in:

- **public API completeness**
- **storage/runtime maturity**
- **build/load/p99 latency on the current standard benchmark path**

So the correct next strategy is **not** “keep chasing generic parity.”  
It is:

1. convert verified Hanns engine strengths into real HannsDB product/runtime strengths
2. close the storage/runtime engineering gap
3. then measure again against zvec on the same path

Fresh local checkpoint after the latest runtime/productization work:
- repo-local release optimize proxy (`N=2000`, `DIM=256`, `cosine`, `REPEATS=3`) now records:
  - median `insert=14ms`
  - median `optimize=107ms`
  - median `total=122ms`
- this is only a proxy, not a replacement for the full `1536D50K` apples-to-apples recheck
- but it confirms the recent `IvfUsq/HnswHvq/Arrow-snapshot` slices did not introduce an obvious small-case optimize regression

Fresh standard-path checkpoint after the latest rerun:
- result file:
  - `result_20260413_hannsdb-p0-rerun-20260413_hannsdb.json`
- metrics:
  - `insert_duration=14.6432`
  - `optimize_duration=114.001`
  - `load_duration=128.6442`
  - `serial_latency_p99=0.0003`
  - `recall=0.9441`
- this is now a current end-to-end benchmark artifact, not just a local proxy note

Fresh remote full x86 checkpoint:
- result file:
  - `result_20260413_hannsdb-hk-x86-20260413_hannsdb.json`
- metrics:
  - `insert_duration=24.1242`
  - `optimize_duration=78.5678`
  - `load_duration=102.692`
  - `serial_latency_p99=0.0005`
  - `recall=0.9442`
- this confirms the full standard benchmark lane is now reproducible on the hk-x86 host as well

Fresh remote x86 proxy checkpoint:
- remote x86 host completed a larger release optimize proxy at `50K / 1536 / cosine`
- output:
  - `OPT_BENCH_TIMING_MS create=0 insert=3327 optimize=18606 search=0 total=21934`
- this is still a proxy rather than a full VectorDBBench result, but it confirms the current larger-shape optimize path is executable on x86 with the latest synced code

---

## 6. Evidence Index

- HannsDB public exports and param surfaces:
  - `crates/hannsdb-py/python/hannsdb/__init__.py`
  - `crates/hannsdb-py/python/hannsdb/model/param/index_params.py`
- HannsDB quantized public-surface tests:
  - `crates/hannsdb-py/tests/test_ivf_usq_surface.py`
- HannsDB query/schema/functionality tests:
  - `crates/hannsdb-py/tests/test_query_executor.py`
  - `crates/hannsdb-py/tests/test_collection_facade.py`
  - `crates/hannsdb-py/tests/test_schema_mutation_surface.py`
  - `crates/hannsdb-core/tests/zvec_parity_sparse.rs`
  - `crates/hannsdb-core/tests/zvec_parity_string_pk.rs`
- Hanns engine internals:
  - `/Users/ryan/Code/vectorDB/Hanns/src/quantization/usq/quantizer.rs`
  - `/Users/ryan/Code/vectorDB/Hanns/src/faiss/ivf_usq.rs`
  - `/Users/ryan/Code/vectorDB/Hanns/src/faiss/hnsw_hvq.rs`
  - `/Users/ryan/Code/vectorDB/Hanns/src/faiss/hnsw_pca_usq.rs`
- zvec public params and quantized collection tests:
  - `/Users/ryan/Code/vectorDB/zvec/python/zvec/model/param/__init__.pyi`
  - `/Users/ryan/Code/vectorDB/zvec/python/tests/test_collection_hnsw_rabitq.py`
- Storage/runtime comparison:
  - `crates/hannsdb-core/src/segment/manager.rs`
  - `crates/hannsdb-core/src/segment/version_set.rs`
  - `/Users/ryan/Code/vectorDB/zvec/src/db/index/common/version_manager.h`
  - `/Users/ryan/Code/vectorDB/zvec/src/db/index/storage/forward_writer.h`
- Existing synthesized docs and benchmark notes:
  - `docs/hannsdb-vs-zvec-gap-analysis.md`
  - `docs/vector-db-bench-notes.md`
- Historical status context:
  - mempal results for benchmark/gap history (`drawer_hannsdb_default_*`)
