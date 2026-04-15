# HannsDB vs Zvec Current Gap Analysis (2026-04-15)

## 一、结论先行

2026-04-15 更新：本版本基于对 Zvec 代码库和 HannsDB 最新未提交改动的双重深度分析，替代 04-12 快照。

**04-15 新落地的能力（未提交）：**
- `add_column(..., expression=...)` 完整支持常量字面量 backfill（string/int/float/bool/null），含 WAL 恢复
- `alter_column(...)` 支持 int32→int64 / uint32→uint64 / float→float64 三种 widening migration，含 rename+widening 组合
- `InvertIndexParam` 正式进入公共 API（`enable_range_optimization=False`, `enable_extended_wildcard=False`）
- custom reranker + `query_by_id` / `group_by` 组合解除封锁

**当前 HannsDB 相比 Zvec 的主要 gap，不再是”有没有某个功能”，而是以下几类成熟度差距：**

1. **InvertIndexParam 功能化** — 已暴露但两个 flag 均不可用（强制 False）
2. **底层 storage/runtime 深度** — Arrow forward store / 多段合并 / compaction 成熟度仍落后
3. **Schema mutation 表达式深度** — 只有常量，Zvec 支持 SQL 表达式（字段引用、计算表达式）
4. **主键模型** — Python 层仍偏 i64，Zvec 以 string ID 为一等公民
5. **RaBitQ / 量化索引产品化** — Zvec 已有 HNSW_RABITQ 完整实现
6. **Embedding 生态** — Zvec 有 OpenAI/Qwen/Jina/BM25/SentenceTransformer，HannsDB 无

---

## 二、当前对比概览

| 维度 | HannsDB 当前状态 | Zvec 当前状态 | 当前 gap |
|------|------------------|---------------|----------|
| 核心语言 | Rust | C++17 | — |
| 存储路径 | `records.bin` + `payloads.jsonl` + `vectors.jsonl` + `ids.bin`，辅以 segment/version 元数据 | Arrow IPC + Parquet forward store + version/segment runtime | **大** |
| 段管理 | 已有 `SegmentManager` / `VersionSet`，不再是纯单段 | 完整多段、版本管理、成熟 segment lifecycle | **大** |
| 向量索引族 | Flat / HNSW / HNSW-HVQ / IVF / IVF-USQ | Flat / HNSW / HNSW+RaBitQ / IVF 等更多产品化组合 | **中** |
| 稀疏索引 | core 已有 sparse runtime 和 descriptor | 产品化更完整 | **中** |
| 标量索引 | 内建倒排索引 | 更成熟的倒排与存储体系 | **中** |
| 数据类型 / 参数面 | core 与 Python 暴露面仍不完全一致，但已公开 `CollectionOption` / `OptimizeOption` / `FlatIndexParam` / `IVFQueryParam` / `InvertIndexParam` 等更诚实的常用 surface | Python/public surface 更完整 | **中** |
| Query surface | core 与 facade 的差距继续缩小，custom reranker + `query_by_id` / `group_by` 也已打通 | 统一 query executor 更完整 | **中** |
| Schema mutation | 已有 richer contract shape，`add_column(..., expression=...)` 也已有 constant-expression 最小执行子集 | option/expression / migration 更丰富 | **中** |
| 主键模型 | Python 层仍偏 `i64` doc id | 字符串 PK 模型成熟 | **中** |

---

## 三、已经不应再算 gap 的部分

下面这些结论如果还出现在对外分析里，会误导优先级判断。

### 1. 基础 mutation 已经落地

HannsDB 当前已经有可工作的：

- `update`
- `add_column`
- `drop_column`
- `alter_column`

这些不是计划项，也不是 stub。

### 2. `query_by_id` 和 `group_by` 已经具备

当前 core 里已经有对应 query AST 与执行路径，相关 parity tests 可通过。  
因此不能再把它们归类成“功能缺失”。

### 3. sparse 不是空白

HannsDB core 已经具备 sparse vector 数据结构、descriptor 与 runtime，且有 parity tests。  
当前 gap 不在“有没有 sparse”，而在“对外产品面和参数面是否达到 Zvec 水平”。

### 4. filter grammar 已经明显增强

当前 HannsDB 已经支持的范围，至少包括：

- `IN / NOT IN`
- `LIKE / NOT LIKE`
- `IS NULL / IS NOT NULL`
- `has_prefix / has_suffix`
- 数组 `contains` / `contains_any` / `contains_all`

因此“只支持简单比较表达式”的说法已经不成立。

### 5. “只有单段”这个判断已经过时

HannsDB 现在已经存在 segment manager / version set 路径，不应继续按“纯单段 v1 原型”描述。  
真实问题是：**虽然已经进入多段方向，但 runtime / persistence / lifecycle 的成熟度仍明显落后于 Zvec。**

---

## 四、当前真实 gap

### GAP-1: Storage / Runtime 架构深度 (P0)

这是当前最实质的差距。

**Zvec**

- 以 `SegmentManager`、`VersionManager`、forward store 为核心
- forward store 建立在 Arrow IPC / Parquet 之上
- 运行时、段版本、列存访问路径更加成熟

**HannsDB**

- 已有 segment manager / version set，不再是最早期单段模型
- 但活跃持久化路径仍明显依赖：
  - `records.bin`
  - `payloads.jsonl`
  - `vectors.jsonl`
  - `ids.bin`
- 版本与集合元数据仍偏轻量

**本质差距**

- 列存深度
- 版本管理成熟度
- segment lifecycle 完整度
- 长期可扩展性与性能上限

这也是当前 ANN optimize/build、load、QPS 收敛仍然困难的重要背景。

### GAP-2: Python / Public API 产品面 (P0)

HannsDB core 的能力，已经明显强于当前 Python/public surface。

**现状**

- Python `DataType` / option / param 暴露仍不完整，但已继续缩小，当前已新增或公开：
  - `CollectionOption`
  - `OptimizeOption`
  - `FlatIndexParam`
  - `IVFQueryParam`
  - `InvertIndexParam`
  - `IvfUsq` / `HnswHvq` 这类 Hanns-native quantized surface
- 没有对齐 Zvec 的更完整参数模型

**和 Zvec 的差距主要体现在**

- 缺少更完整的 index param 家族
- 缺少更完整的数据类型暴露
- 缺少更完整的 schema option / query option 产品面

因此当前真正落后的不是“core 没能力”，而是“对外能稳定交付的产品面不够宽”。

### GAP-3: Schema Mutation 深度 (P1)

HannsDB 已经能做基础 schema mutation，而且 public contract 也继续往前推进了一步，但深度还没达到 Zvec。

**HannsDB 当前更像**

- 基础列增删改
- richer public contract shape：
  - `add_column(field_schema, expression=..., option=...)`
  - `alter_column(old_name, new_name=None, field_schema=None, option=...)`
- `add_column(..., expression=...)` 已支持最小 constant-expression 子集
- 但 `field_schema` migration 仍未真实落地

**Zvec 更像**

- 带 `FieldSchema`
- 支持表达式/选项
- schema 变更与类型系统结合得更完整

所以这里不该再写成“没有 schema mutation”，而应写成“**只有基础版，没有 Zvec 等级的 schema mutation 深度**”。

### GAP-4: Query Surface 与组合能力 (P1)

当前 HannsDB 的一个典型问题仍然是：**core 能做的事情，仍然略多于 Python facade 已经诚实公开的事情。**

例如：

- core 已有 `query_by_id`
- core 已有 `group_by`
- core 已有 reranker 结构
- core 已有 `order_by` 相关结构

但对外 surface 仍存在剩余限制：

- 更深的 query/schema/PK 组合仍未完全产品化
- `field_schema` migration 还只是 contract shape，不是真执行能力
- facade 与 core 的能力边界虽然继续收敛，但还没完全整齐

所以这里的 gap 是“**组合与产品化不足**”，不是“底层完全没有这些功能”。

### GAP-5: 主键模型不对等 (P1)

HannsDB Python 层当前仍偏向把文档 ID 当作 `i64` 处理。  
这和 Zvec 的字符串主键模型存在真实差距。

这会直接影响：

- 外部业务主键映射
- 易用性
- 和上层应用数据模型的对接方式

### GAP-6: 索引族与量化产品化 (P2)

HannsDB 在 core descriptor/runtime 层面已经不止 `Flat/HNSW/IVF`，还包括 sparse 与 scalar inverted 的方向。  
但从“可交付产品面”看，仍落后于 Zvec。

**差距主要在**

- 更完整的 quantized index family
- 类似 `HnswRabitq*` 这类成熟产品化形态
- 更丰富稳定的公共参数模型

所以当前问题不是“完全没有索引扩展”，而是“**索引能力还没有整理成和 Zvec 对等的公开产品面**”。

---

## 五、建议优先级

### P0

1. **继续补强 storage/runtime 深度**
   - 现状已从“只有 JSONL sidecar”推进到“active segment flush 后物化 Arrow snapshot，并在后续写入时失效旧 snapshot”，但仍未达到 zvec 的 forward-store 深度
2. **继续让 Python/public API 追上 core 已有能力**

这是当前最影响外部感知和长期演进的两件事。

### P1

3. **补齐 schema mutation 的 migration / field_schema 深度**
4. **继续统一 query surface，减少剩余 facade 对组合能力的额外限制**
5. **重新设计主键模型，至少明确字符串 PK 的演进路径**

### P2

6. **整理并公开更完整的 index family / quantization 参数面**
7. **在产品层明确 sparse / scalar / quantized 的支持矩阵**

---

## 六、和性能结论的关系

从近期 benchmark 与 mempal 记录看，HannsDB 已经显著压低了部分 `optimize / load / p99` 指标。  
这说明项目当前已经不是“基本功能没通”，而是进入了“底层路径继续收敛、真实性能继续验证”的阶段。

当前最新的真实标准路径结果也已经更新：

- `Performance1536D50K` rerun (`hannsdb-p0-rerun-20260413`)
- result artifact: `result_20260413_hannsdb-p0-rerun-20260413_hannsdb.json`
- `insert_duration=14.6432`
- `optimize_duration=114.001`
- `load_duration=128.6442`
- `serial_latency_p99=0.0003`
- `recall=0.9441`

这说明当前代码路径不仅在 repo-local proxy 上没有回退，而且在真实 benchmark harness 下也已经重新跑通并落盘。

但要注意两点：

1. 某些异常漂亮的数据点曾被确认是 brute-force 路径，不应误当成真实 HNSW 结论。
2. 即便近期 `optimize / load / p99` 有明显改善，和 Zvec 相比，QPS、build 成本、runtime 成熟度仍然仍是主战场。

---

## 七、一句话判断

**当前 HannsDB 相比 Zvec 的主要 gap，不再是基础功能缺失，而是 storage/runtime、public API 产品面、query/schema/PK 深度这三类“成熟度差距”。**
