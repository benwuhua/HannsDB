# HannsDB Performance Test Plan

## 目的

量化 HannsDB 各子系统的吞吐量和延迟基线，提供回归检测依据，并指导后续优化方向。

---

## 测试范围

| 编号 | 测试点                          | 对应代码入口                                             |
|------|-------------------------------|--------------------------------------------------------|
| P1   | 单批次 Insert 吞吐量            | `bench_suite::bench_insert_throughput`                 |
| P2   | 暴力搜索延迟（规模缩放）         | `bench_suite::bench_brute_force_search_scaling`        |
| P3   | Compaction 吞吐量               | `bench_suite::bench_compaction_timing`                 |
| P4   | Optimize + HNSW 搜索（集成）    | `collection_api::bench_optimize_benchmark_entry`       |
| P5   | VectorDBBench 标准场景端到端    | `scripts/run_vdbb_hannsdb_perf1536d50k.sh`            |

---

## P1 — Insert 吞吐量

### 目标
测量将 N 条向量写入单一平文件 segment（不触发 ANN 构建）的原始速度。

### 运行命令
```bash
HANNSDB_BENCH_N=5000  HANNSDB_BENCH_DIM=64   cargo test -p hannsdb-core bench_insert -- --nocapture
HANNSDB_BENCH_N=50000 HANNSDB_BENCH_DIM=128  cargo test -p hannsdb-core bench_insert -- --nocapture
HANNSDB_BENCH_N=50000 HANNSDB_BENCH_DIM=1536 cargo test -p hannsdb-core bench_insert -- --nocapture
```

### 输出格式
```
BENCH_INSERT_THROUGHPUT phase=insert n=<N> dim=<DIM> ms=<ms> rows_per_sec=<rows/s>
```

### 当前基线（debug 构建，Apple M 系列）
| N     | DIM  | rows/sec |
|-------|------|----------|
| 5,000 | 64   | ~7,000   |

### 验收阈值
- Release 构建下 50K / 128-dim 的 `rows_per_sec` 不低于 20,000。
- 连续两次测量的 `rows_per_sec` 差异 < 20 %（稳定性）。

---

## P2 — 暴力搜索延迟（规模缩放）

### 目标
量化无 ANN 索引时，单次 top-10 查询的延迟随集合规模的变化趋势，用于：
1. 确认线性扩展行为（O(N·DIM)）。
2. 设定"何时值得调用 optimize_collection"的决策阈值。

### 运行命令
```bash
HANNSDB_BENCH_N=50000 HANNSDB_BENCH_DIM=64  cargo test -p hannsdb-core bench_brute_force -- --nocapture
HANNSDB_BENCH_N=50000 HANNSDB_BENCH_DIM=128 cargo test -p hannsdb-core bench_brute_force -- --nocapture
```

测试内部在 1K / 10K / 50K 三个节点各记录一次延迟。

### 输出格式
```
BENCH_BRUTE_FORCE_SEARCH phase=search n=<N> dim=<DIM> ms=<ms> us=<us>
```

### 当前基线（debug 构建，DIM=64）
| N      | us    |
|--------|-------|
| 1,000  | ~3,000 |
| 10,000 | ~30,000 |
| 50,000 | ~154,000 |

### 验收阈值
- Release 构建，DIM=128，N=50K：单次搜索延迟 < 100 ms。
- 相对于 N=1K 的延迟，N=50K 延迟不超过 70× （允许内存局部性带来一定的超线性）。

---

## P3 — Compaction 吞吐量

### 目标
量化将 K 个 immutable segment（共 N 行）合并为 1 个 segment 的耗时。
Compaction 是读密集+顺序写操作，其性能影响碎片化写入后的恢复时间。

### 运行命令
```bash
# 5 段，每段 1000 行，DIM=128
HANNSDB_BENCH_N=5000 HANNSDB_BENCH_DIM=128 HANNSDB_BENCH_SEGS=5 \
  cargo test -p hannsdb-core bench_compaction -- --nocapture

# 10 段，每段 5000 行，DIM=128
HANNSDB_BENCH_N=50000 HANNSDB_BENCH_DIM=128 HANNSDB_BENCH_SEGS=10 \
  cargo test -p hannsdb-core bench_compaction -- --nocapture
```

### 输出格式
```
BENCH_COMPACTION phase=compact k_segs=<K> immutable_rows=<M> total_rows=<N> dim=<DIM> ms=<ms>
```

### 当前基线（debug 构建，DIM=64，K=4，N=5K）
| K_segs | N      | DIM | ms    |
|--------|--------|-----|-------|
| 4      | 5,000  | 64  | ~444  |

### 验收阈值
- Release 构建，K=5，N=50K，DIM=128：compaction 时间 < 2,000 ms。
- Tombstone 过滤正确性：compaction 后 `deleted_count == 0`（由 lifecycle 集成测试保证）。

---

## P4 — Optimize + ANN 搜索（集成路径）

### 目标
覆盖完整的"insert → optimize（build HNSW）→ search"路径，包含 knowhere-rs 后端。
这是 VectorDBBench 标准场景的本地代理。

### 运行命令
```bash
# 小规模快速验证（默认，无需 --features）
HANNSSDB_OPT_BENCH_N=2000 HANNSSDB_OPT_BENCH_DIM=256 HANNSSDB_OPT_BENCH_METRIC=cosine \
  cargo test -p hannsdb-core collection_api_optimize_benchmark_entry -- --nocapture

# 标准目标规模（需 knowhere-backend）
HANNSSDB_OPT_BENCH_N=50000 HANNSSDB_OPT_BENCH_DIM=1536 HANNSSDB_OPT_BENCH_METRIC=cosine \
  cargo test -p hannsdb-core --release --features knowhere-backend \
  collection_api_optimize_benchmark_entry -- --nocapture
```

### 输出格式
```
OPT_BENCH_CONFIG n=<N> dim=<DIM> metric=<M> top_k=<K>
OPT_BENCH_TIMING_MS create=<ms> insert=<ms> optimize=<ms> search=<ms> total=<ms>
```

### 当前基线（Release，knowhere-backend，N=50K，DIM=1536，cosine）
```
insert=121s  optimize=81s  serial_latency_p99=111ms  recall=1.0
```
（来源：`docs/vector-db-bench-notes.md`，2026-03-21）

### 验收阈值
- `optimize` 阶段 < 120 s（目标：< 60 s 后续优化）。
- `recall@10` = 1.0（精确模式验证）。
- P4 回归检查命令：`./scripts/run_hannsdb_optimize_bench.sh N=2000 DIM=256 METRIC=cosine REPEATS=3`

---

## P5 — VectorDBBench 端到端

### 目标
通过 VectorDBBench `Performance1536D50K` 场景确认 HannsDB 在 Python 嵌入路径下的完整性能。

### 运行命令
```bash
DB_LABEL=hannsdb-1536d50k-knowhere \
TASK_LABEL=hannsdb-1536d50k-knowhere \
DB_PATH=/tmp/hannsdb-vdbb-1536d50k-knowhere-db \
  bash scripts/run_vdbb_hannsdb_perf1536d50k.sh
```

### 输出位置
`vectordb_bench/results/HannsDB/result_<timestamp>_<label>_hannsdb.json`

### 当前基线（2026-03-21）
```
insert=121s  optimize=81s  serial_latency_p99=111ms  recall=1.0
```

### 验收阈值
- 完整 benchmark 运行结束（无异常退出）。
- `recall` ≥ 0.95（允许 ANN 近似误差）。
- `p99_latency` < 200 ms（目标：< 50 ms 后续优化）。

---

## 执行顺序建议

```
P1（Insert）→ P2（暴力搜索）→ P3（Compaction）
                                   ↓
              P4（Optimize bench，小规模验证）
                                   ↓
              P4（Optimize bench，目标规模）
                                   ↓
                            P5（VectorDBBench）
```

P1–P3 在 debug 构建下即可快速运行，适合每次 PR 前的冒烟检查。
P4 目标规模和 P5 在 release + knowhere-backend 下运行，作为里程碑验证。

---

## 回归检测工作流

1. **每次提交前**：运行 `cargo test -p hannsdb-core bench -- --nocapture`，对比当前 `rows_per_sec` 与上次记录值，差异 > 30 % 时视为回归候选。
2. **Phase/功能合并后**：运行 P4 小规模（N=2000/DIM=256），确认 optimize 时间无明显增长。
3. **里程碑前**：运行 P5，更新 `docs/vector-db-bench-notes.md` 中的基线记录。

---

## 未覆盖（后续扩展）

| 场景                          | 说明                                          |
|------------------------------|----------------------------------------------|
| 多段并发搜索                  | 需要多段写路径完成后才有意义                    |
| Upsert 吞吐量                 | 当前 upsert = delete + insert，延迟待量化       |
| 内存占用（peak RSS）          | 需要 OS-level 采样，未接入 bench 框架           |
| WAL 回放时间                  | 大 WAL 文件的 open() 延迟未量化                 |
| Knowhere 不同参数对比          | ef_search / m / ef_construction 敏感性分析     |
