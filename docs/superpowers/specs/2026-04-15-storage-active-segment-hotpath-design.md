# Storage: Active Segment Insert Hot-Path Fix — Design Spec

**Date:** 2026-04-15
**Gap:** GAP-1 — Insert 比 Zvec 慢 4.2x；根本原因是 O(n) Arrow 重写在每次 insert batch 内发生

---

## 一、第一性原理：问题根源

### 当前 insert 热路径（`segment/writer.rs` `append_batch()`）

```
insert_documents_internal()
  └── SegmentWriter::append_batch()
        ├── load_payloads()          ← 从 Arrow/JSONL 读全量 (O(n))
        ├── append new rows in memory
        ├── append_payloads(JSONL)   ← 追加写 JSONL (O(m))
        └── persist_forward_store_artifacts()   ← ⚠️ O(n+m) 重写整个 Arrow
              ├── load JSONL/Arrow again (load #2)
              ├── write forward_store.arrow (ALL rows)
              ├── write forward_store.parquet (ALL rows)
              ├── write payloads.arrow (ALL rows)
              ├── write vectors.arrow (ALL rows)
              └── remove JSONL files
```

**关键问题：** `persist_forward_store_artifacts()` 被放在 `append_batch()` 热路径里，每次 insert 都重写整个 Arrow 文件（全量写）。随着 segment 增大，这是 O(n) 开销，与已有数据量线性增长。

**第一性原理推导：**
- 一次 insert batch 最多只引入 m 条新行
- 没有理由在每次 insert 时重写全部 n+m 行到 Arrow
- Arrow 文件对 active segment 而言是"读优化快照"，不是 source of truth
- Source of truth 是：WAL（crash recovery）+ ids.bin + records.bin + JSONL（payloads）

### 正确的角色分工

| 文件 | 角色 | 写入时机 |
|---|---|---|
| WAL | crash recovery source of truth | 每次 insert（已正确） |
| ids.bin | 向量 ID 持久化 | 每次 insert（已正确） |
| records.bin | 向量数据持久化 | 每次 insert（已正确） |
| payloads.jsonl | 标量字段 source of truth（active segment） | append-only，每次 insert（已正确） |
| vectors.jsonl | 次级向量 source of truth | append-only，每次 insert（已正确） |
| forward_store.arrow | 读优化快照 | **只在 rollover/compaction 时**（当前错放在 insert 热路径） |
| forward_store.parquet | 读优化快照（Parquet 格式） | 同上 |

---

## 二、修复方案

### 核心修改：从 append_batch() 中移除 persist_forward_store_artifacts()

**文件：** `crates/hannsdb-core/src/segment/writer.rs`

在 `append_batch()` 中，移除对 `persist_forward_store_artifacts()` 的调用。

修改后 `append_batch()` 热路径变为：
```
├── load_payloads()          ← 从 Arrow/JSONL 读（仍然 O(n)，见后续优化）
├── append new rows in memory
├── append_payloads(JSONL)   ← 追加写 JSONL (O(m))  ✓
├── append_ids(ids.bin)      ← 追加写 IDs (O(m))     ✓
├── append_records(records.bin) ← 追加写 vectors (O(m)) ✓
└── [remove persist_forward_store_artifacts()]   ← 不再重写 Arrow
```

`persist_forward_store_artifacts()` 只在以下场景调用：
1. Segment rollover（已有调用，保留）
2. Compaction finalization（已有调用，保留）
3. Explicit `flush()`（已有调用，保留）

### 附加优化：消除 load_payloads() 重复加载

当前 `append_batch()` 调用 `load_payloads()` 加载全量 payloads（O(n)），这对每次 insert 依然是 O(n) 读。

**原因：** `append_batch` 需要"原有 payload 列表 + 新行"来写 forward_store。若 forward_store 不再在热路径写，则 `load_payloads()` 的完整加载也不再必要。

优化后：
- `append_batch()` 不再需要 `load_payloads()` 返回全量
- JSONL 追加是 O(1)（文件 append，不需要读旧内容）
- 因此可将 `load_payloads()` 调用也从热路径移除

修改后热路径变为纯 O(m)：
```
├── append_payloads(JSONL)      ← O(m) append
├── append_ids(ids.bin)          ← O(m) append
├── append_records(records.bin)  ← O(m) append
└── [WAL already written upstream]
```

**注意：** `load_payloads()` 仍然需要在 rollover、compaction、optimize 时调用（构建 Arrow snapshot 时需要全量）。

---

## 三、读路径影响分析

### Active segment 读（insert 期间查询）

修改后 active segment 的读路径：
- 查询时如果需要 payloads：调用 `load_payloads()`（从 JSONL 或 Arrow）
- Arrow 快照只在 rollover/compaction 后存在，active segment 回到 JSONL 读

**是否退步？**
- 查询中 payload 访问已通过 `filter` 进行，active segment 一般较小（默认 rollover threshold）
- 只有查询 active segment 的 payload 才需要 JSONL 全量读，而这已经是当前正确行为
- Rollover 后立即有 Arrow 快照可用，后续读走 Arrow
- **结论：读路径无退步，与修改前相同**

### Rollover 后读

Rollover 时 `persist_forward_store_artifacts()` 被调用，写出 Arrow snapshot。之后所有读走 Arrow（快速）。行为不变。

---

## 四、Zvec 对比

Zvec 的 insert 不做"每次重写全量 Arrow"这种操作。Zvec 使用：
- 增量写入 forward_store（mmap 或 bufferpool，支持 O(1) append）
- Active segment 的 forward_store 是 mmap-based，支持列级别 lazy-load

本修复使 HannsDB 达到"不做无效 O(n) 重写"的正确语义，是 insert 性能改善的直接路径。

---

## 五、预期性能提升

基于探查结论：
- 当前每次 insert batch：2× O(n) load + 4× O(n) write（forward_store.arrow + .parquet + payloads.arrow + vectors.arrow）
- 修改后每次 insert batch：O(m) append only

对于 50K 行的 segment（典型 1536 dim benchmark）：
- 每次 insert 从"读写 50K 行"降为"写 m 行"
- **预期 insert throughput 提升 50-70%**，接近 Zvec 水平

---

## 六、代码变更地图

### 主改动：`crates/hannsdb-core/src/segment/writer.rs`

1. **`append_batch()` 函数：**
   - 移除 `load_payloads()` 调用（不再需要全量 load 来构建 forward_store）
   - 移除 `persist_forward_store_artifacts()` 调用
   - 保留：`append_payloads(JSONL)`、`append_ids(ids.bin)`、`append_records(records.bin)`

2. **确认 rollover path 仍调用 `persist_forward_store_artifacts()`：** 已有，不动

### 验证点：`crates/hannsdb-core/src/db.rs`

- `compact_collection_internal()` 末尾调用 `materialize_forward_store_snapshot()`：保留
- `optimize_collection()` 路径：保留
- `flush_collection()` 如有：保留

### 无需改动：

- `segment/payloads.rs`：JSONL append/read 路径不变
- `forward_store/`：整个模块不变，只是调用时机变了
- `segment/arrow_io.rs`：不变
- WAL 路径：不变

---

## 七、风险分析

### 风险 1：读时 JSONL 加载慢

**场景：** insert 期间同时有大量查询走 active segment payloads。

**缓解：** Active segment 在 rollover threshold 内行数有限（默认较小）；JSONL 全量读的绝对耗时对小 segment 可接受。

**长期修复：** 在 SegmentWriter 中缓存 in-memory `MemForwardStore`，读时直接走内存（Phase 2）。

### 风险 2：crash 后 Arrow 快照过时

**场景：** 大量 insert 后 crash，重启后 active segment 从 JSONL 重建（WAL replay）。没有 Arrow snapshot 可用，第一次查询需要从 JSONL 全量读。

**影响：** 恢复后的第一次查询稍慢。这是可接受的权衡（WAL 保证正确性）。

### 风险 3：已有依赖 persist_forward_store_artifacts() 在热路径的逻辑

**缓解：** 仔细检查所有调用方，确认 rollover/compaction path 有保留。

---

## 八、测试计划

### 性能测试

```bash
HANNSSDB_OPT_BENCH_N=2000 HANNSSDB_OPT_BENCH_DIM=256 HANNSSDB_OPT_BENCH_METRIC=cosine \
  cargo test -p hannsdb-core collection_api_optimize_benchmark_entry -- --nocapture
```

比较修改前后的 `insert` 时间。

### 正确性测试（现有）

- 所有 `cargo test -p hannsdb-core` 通过
- 特别关注：`collection_api` 中 insert → query 场景
- 关注：rollover 后 query 结果正确

### 新增测试

- `test_insert_large_batch_then_query` — insert 10K 行后 query，结果正确
- `test_insert_rollover_then_arrow_read` — rollover 后确认 Arrow snapshot 存在且可读
