# Plan: Storage Active Segment Insert Hot-Path Fix

**Date:** 2026-04-15
**Spec:** `docs/superpowers/specs/2026-04-15-storage-active-segment-hotpath-design.md`
**Expected gain:** Insert throughput +50-70%（消除每次 insert 的 O(n) Arrow 重写）

---

## 根本原因

`segment/writer.rs` `append_batch()` 在每次 insert 热路径中调用 `persist_forward_store_artifacts()`，该函数对整个 segment 做 4 次全量 Arrow 写（forward_store.arrow + .parquet + payloads.arrow + vectors.arrow）。随 segment 增大线性变慢。

---

## Step 1: 读懂当前代码

读以下文件，理解当前完整 append_batch 流程：
- `crates/hannsdb-core/src/segment/writer.rs` 全文
- `crates/hannsdb-core/src/storage/segment_io.rs` — `persist_forward_store_artifacts()` 和 `materialize_forward_store_snapshot()`
- `crates/hannsdb-core/src/db.rs` — 确认 rollover、compaction、flush 的调用路径

---

## Step 2: 从 append_batch() 移除热路径 Arrow 写

**文件：** `crates/hannsdb-core/src/segment/writer.rs`

在 `append_batch()` 中：
1. 移除对 `persist_forward_store_artifacts()` 的调用（这是核心改动）
2. 移除对 `load_payloads()` 的全量加载调用（不再需要构建 forward_store，所以不需要全量 load）
3. 确认保留：
   - `append_payloads()` → JSONL append（O(m)）
   - `ids.bin` / `records.bin` append（O(m)）

**验证：** rollover path（segment rollover 触发时）仍然调用 `persist_forward_store_artifacts()`，这是正确的。

---

## Step 3: 确认 rollover/compaction/flush 路径完整

检查并确认以下场景都有调用 `persist_forward_store_artifacts()` 或 `materialize_forward_store_snapshot()`：
- Segment rollover（行数超过 threshold 时）
- Compaction finalization（`compact_collection_internal()` 末尾）
- Optimize（`optimize_collection()` 路径）

如果任何一个路径缺失调用，补上。

---

## Step 4: 测试

运行基准测试验证 insert 提速：
```bash
HANNSSDB_OPT_BENCH_N=2000 HANNSSDB_OPT_BENCH_DIM=256 HANNSSDB_OPT_BENCH_METRIC=cosine \
  cargo test -p hannsdb-core collection_api_optimize_benchmark_entry -- --nocapture
```

记录修改前后 insert 时间对比。

运行全量测试：
```bash
cargo test -p hannsdb-core 2>&1 | tail -30
```

所有测试通过（pre-existing failure 除外）。

新增测试（`crates/hannsdb-core/tests/collection_api.rs`）：
- `test_insert_large_batch_arrow_not_rewritten_per_batch` — 验证 insert 100 批次后 Arrow 文件时间戳不变（只在 rollover 后变）
- `test_rollover_produces_arrow_snapshot` — rollover 后 forward_store.arrow 存在且可查询

---

## 成功标准

- [ ] `append_batch()` 不再调用 `persist_forward_store_artifacts()`
- [ ] Rollover / compaction / optimize 后 Arrow 快照仍然正确生成
- [ ] Insert + query 场景正确性不退步
- [ ] 性能基准：`N=2000 DIM=256` insert 时间明显下降（预期 >30%）
- [ ] `cargo test -p hannsdb-core` 全部通过
