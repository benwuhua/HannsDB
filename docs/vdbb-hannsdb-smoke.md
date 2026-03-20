# HannsDB VectorDBBench Tiny Smoke

Run from HannsDB repo root:

```bash
./scripts/run_vdbb_hannsdb_smoke.sh
```

Force dataset regeneration:

```bash
./scripts/run_vdbb_hannsdb_smoke.sh --regenerate
```

Defaults:
- Dataset dir: `/tmp/hannsdb-custom-ds-review`
- DB path: `/tmp/hannsdb-vdbb-smoke-db`
- Case: `PerformanceCustomDataset`
- HNSW: `m=16`, `ef_construction=32`, `ef_search=16`
- Search: `k=3`

Expected output:
- Script prints `result file: ...`
- Result file path pattern:
  `/Users/ryan/Code/VectorDBBench/vectordb_bench/results/HannsDB/result_YYYYMMDD_hannsdb-smoke_hannsdb.json`
