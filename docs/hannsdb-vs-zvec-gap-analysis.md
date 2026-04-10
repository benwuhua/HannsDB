# HannsDB vs Zvec Feature Gap Analysis (2026-04-10)

## 一、总体对比

| 维度 | HannsDB | Zvec | 差距 |
|------|---------|------|------|
| 核心语言 | Rust | C++17 | — |
| 存储引擎 | JSONL + 二进制 flat 文件 | Arrow IPC + Parquet 列存 | **大** |
| 架构 | 嵌入式 + HTTP daemon | 纯嵌入式（Python/Node.js） | — |
| 索引类型 | 3 (HNSW, IVF, Flat) | 8 (HNSW, HNSW+RaBitQ, IVF, Flat, Sparse HNSW, Sparse Flat, Inverted) | **大** |
| 数据类型 | 8 | 22 | **大** |
| 多向量 | 支持 | 支持 | 平 |
| 多段管理 | 单段 (v1) | 完整多段 + 自动分裂 (10M docs) | **大** |
| SQL 解析 | 简单自定义 parser | ANTLR4 完整 SQL parser | **中** |
| 量化 | 无 | FP16, INT8, INT4, RaBitQ | **大** |
| 稀疏向量 | 不支持 | 支持 (FP32, FP16) | **大** |

---

## 二、逐项 GAP 详细分析

### GAP-1: 存储层 (Critical)

**Zvec**: Arrow IPC + Parquet 列存，三种前向存储实现
- `MemForwardStore` — 内存 Arrow RecordBatch，写段缓存
- `MmapForwardStore` — mmap 读取 Parquet/IPC 文件
- `BufferPoolForwardStore` — 缓冲池管理读取

**HannsDB**:
- `records.bin` — f32 平铺二进制（追加写入）
- `payloads.jsonl` — JSON 行格式标量字段
- `vectors.jsonl` — JSON 行格式次向量
- `ids.bin` — i64 平铺二进制

**影响**:
- Insert 性能: Zvec 4.2x 快（列存追加 vs JSONL 序列化）
- 读取效率: Parquet 列存按需读列，JSONL 全行解析
- 磁盘空间: 列存压缩率远高于 JSONL

### GAP-2: 多段管理 (Critical)

**Zvec**: 完整多段架构
- 写段达到 `max_doc_count_per_segment`（默认 10M）自动 dump
- `SegmentManager` 管理不可变段
- `Optimize()` 自动合并删除率 > 30% 的段
- MVCC 式版本管理（protobuf Manifest）

**HannsDB**:
- 单段 v1，多段代码存在但 auto-rollover 被禁用
- `compact_collection()` 需手动调用
- 版本管理用 JSON `segment_set.json`

### GAP-3: 索引类型 (Major)

**Zvec** (8 种):
| 索引 | 说明 |
|------|------|
| FLAT | 暴力扫描 |
| HNSW | 默认 m=50, ef_construction=500 |
| HNSW_RABITQ | HNSW + RaBitQ 量化（Linux x86_64） |
| IVF | n_list=1024 |
| HNSW_SPARSE | 稀疏向量专用 HNSW |
| FLAT_SPARSE | 稀疏向量专用 Flat |
| INVERTED | 标量倒排（RocksDB） |

**HannsDB** (3 种):
| 索引 | 说明 |
|------|------|
| Flat | 暴力扫描 |
| HNSW | 默认 m=16, ef_construction=64 |
| IVF | — |

**缺失**:
- 稀疏向量索引
- RaBitQ 量化（阿里自研，需要 AVX2/AVX-512）
- 量化支持 (FP16, INT8, INT4)

### GAP-4: 数据类型 (Major)

**Zvec 支持 22 种数据类型**:

标量 (8): BOOL, INT32, INT64, UINT32, UINT64, FLOAT, DOUBLE, STRING

密集向量 (8): VECTOR_BINARY32/64, VECTOR_FP16/FP32/FP64, VECTOR_INT4/INT8/INT16

稀疏向量 (2): SPARSE_VECTOR_FP32, SPARSE_VECTOR_FP16

数组 (7): ARRAY_STRING, ARRAY_BOOL, ARRAY_INT32/INT64, ARRAY_UINT32/UINT64, ARRAY_FLOAT/DOUBLE

**HannsDB 支持 8 种**: String, Int64, Int32, UInt32, UInt64, Float, Float64, Bool

**缺失**:
- 所有数组类型 (ARRAY_*)
- 稀疏向量类型 (SPARSE_VECTOR_*)
- 二进制类型 (BINARY)
- 量化向量类型 (FP16, INT4, INT8, INT16)

### GAP-5: SQL 查询引擎 (Medium)

**Zvec**: ANTLR4 完整 SQL 解析器
- 支持: `=, !=, <, >, <=, >=, LIKE, NOT LIKE, IN, NOT IN`
- 支持: `CONTAIN_ALL, CONTAIN_ANY, NOT_CONTAIN_ALL, NOT_CONTAIN_ANY`
- 支持: `IS NULL, IS NOT NULL`
- 支持: 逻辑组合 `AND, OR, ()`
- 查询优化器自动选择 pre-filter vs post-filter

**HannsDB**: 自定义简单 parser
- 支持: `==, !=, <, >, <=, >=`
- 支持: `and, or, ()`
- 支持: `in, not in`（代码中实现但 parser 支持有限）
- 基本标量索引候选集查询

**缺失**: LIKE 模糊匹配、NULL 检查、数组包含操作、查询优化器

### GAP-6: 标量索引能力 (Medium)

**Zvec**: RocksDB 倒排索引
- 支持 EQ, NE, LT, LE, GT, GE, LIKE, IN, CONTAIN_ALL, CONTAIN_ANY
- IS_NULL, IS_NOT_NULL, HAS_PREFIX, HAS_SUFFIX
- 可选范围优化和扩展通配符

**HannsDB**: 内存倒排索引 `InvertedScalarIndex`
- 支持 EQ, NE, LT, LE, GT, GE, IN, NOT IN
- 无 LIKE、NULL 检查、数组操作

### GAP-7: ID 映射 (Medium)

**Zvec**: RocksDB
- 字符串 PK → uint64 doc_id
- 支持快照、列族、压缩

**HannsDB**: `ids.bin` 二进制文件
- i64 外部 ID，追加写入
- 无字符串 PK 支持

### GAP-8: 量化支持 (Major)

**Zvec**: 每个索引可选量化
- FP16: 半精度浮点
- INT8: 8-bit 整数量化
- INT4: 4-bit 整数量化
- RaBitQ: 阿里自研量化（64-4095 维）

**HannsDB**: 无量化支持

### GAP-9: 稀疏向量 (Major)

**Zvec**: 完整稀疏向量支持
- 存储格式: `{indices: Vec<u32>, values: Vec<f32>}`
- 专用索引: HNSW_SPARSE, FLAT_SPARSE
- Python API: `SPARSE_VECTOR_FP32`, `SPARSE_VECTOR_FP16`

**HannsDB**: 不支持

### GAP-10: 嵌入 & 重排 (Extension)

**Zvec Python 扩展**:

嵌入函数 (6 种):
- DefaultLocalDenseEmbedding (SentenceTransformers)
- OpenAIDenseEmbedding
- QwenDenseEmbedding (DashScope)
- JinaDenseEmbedding
- HTTPDenseEmbedding (通用 /v1/embeddings)
- BM25EmbeddingFunction (DashText)
- DefaultLocalSparseEmbedding (SPLADE)
- QwenSparseEmbedding

重排器 (4 种):
- RrfReRanker (Reciprocal Rank Fusion)
- WeightedReRanker (加权融合)
- DefaultLocalReRanker (Cross-Encoder)
- QwenReRanker (DashScope API)

**HannsDB**:
- 核心: RRF + Weighted 融合（QueryContext 中实现）
- 无嵌入函数
- 无 Cross-Encoder 重排

---

## 三、HannsDB 优势

| 维度 | HannsDB | Zvec |
|------|---------|------|
| HTTP API | 有 (Axum daemon, 20 routes) | 无 |
| Rust 生态 | 纯 Rust，可嵌入 Rust 应用 | C++，绑定有限 |
| Search QPS | 744.6 (3.8x 快) | 197.8 |
| P99 延迟 | 1.50ms (4.3x 快) | 6.52ms |
| ANN 持久化 | 二进制 blob + IDs 独立持久化 | 随段存储 |
| WAL | JSONL 格式，可读性好 | 二进制 CRC 格式 |

---

## 四、优先级排序

### P0 — 核心功能缺失（影响基本能力）
1. **多段管理** — 当前单段限制可扩展性
2. **Arrow 列存** — JSONL 是 insert 瓶颈的根因

### P1 — 重要功能差距（影响竞争力）
3. **稀疏向量支持** — RAG/搜索场景刚需
4. **量化支持** (FP16/INT8) — 降低内存占用，提升吞吐
5. **数组类型** — 多值字段常见需求

### P2 — 增强功能（锦上添花）
6. **SQL 增强查询** — LIKE, NULL 检查, 数组操作
7. **RaBitQ 量化** — 高性能量化（需要 AVX2）
8. **RocksDB ID 映射** — 字符串 PK 支持
9. **嵌入函数扩展** — Python 层 embedding function

### P3 — 不需要跟进
- HTTP daemon — Zvec 没有但 HannsDB 有，是差异化优势
- Node.js 绑定 — 暂不需要

---

## 五、性能差距回顾 (2026-04-10 Benchmark)

**机器**: knowhere-x86-hk-proxy (x86, 4 cores, 40G disk)
**数据集**: OpenAI Small 50K, 1536 dim, Cosine

| 指标 | HannsDB | Zvec | 分析 |
|------|---------|------|------|
| QPS | 744.6 | 197.8 | HannsDB 3.8x 快（m=16 vs m=50，低 recall 换速度）|
| P99 延迟 | 1.50ms | 6.52ms | 同上 |
| Recall@100 | 0.9756 | 0.9990 | Zvec 高 2.4%（m=50 vs m=16）|
| Insert | 19.52s | 4.64s | Zvec 4.2x 快（Arrow 列存 vs JSONL）|

**根因分析**:
- Search 快: HannsDB 用更激进的 HNSW 参数 (m=16)，牺牲 recall
- Insert 慢: JSONL 序列化每行 JSON + 文本写入，Zvec 用 Arrow 列式追加
