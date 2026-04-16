# Plan: Expose HnswSq (HNSW + SQ8) Index in HannsDB

**Date:** 2026-04-15
**Gap closed:** Zvec 有 `HnswRabitqIndexParam`（HannsDB 无法直接对应），但 Hanns 引擎有 `HnswSqIndex`（HNSW + 8-bit 标量量化）尚未暴露，可作为量化 HNSW 系列的重要补充
**Spec reference:** Full plan details in this file (plan agent output)

---

## 参数设计

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `metric_type` | `str\|None` | `None` | `"l2"`, `"ip"`, `"cosine"` 均支持（区别于 HnswHvq 的 ip-only） |
| `m` | `int` | `16` | HNSW 每节点最大连接数 |
| `ef_construction` | `int` | `200` | 构建时候选列表大小 |
| `ef_search` | `int` | `50` | 查询时候选列表大小 |

`sq_bits` 不暴露（Hanns 内部硬编码 8-bit）。

---

## 实施顺序（依赖序）

### Step 1: 新建 `crates/hannsdb-index/src/hnsw_sq.rs`

参照 `hnsw_hvq.rs` 结构，区别：
- 使用 `hanns::faiss::HnswSqIndex` + `hanns::faiss::HnswQuantizeConfig`
- Magic bytes: `b"HDBHSQ00"`
- 支持 L2 / IP / Cosine（无 metric 限制）
- `HnswSqIndex::new(dim)` 后设置 config（`max_neighbors`, `ef_construction`, `ef_search`, `sq_bit=8`）
- `from_bytes`: `HnswSqIndex::load(path, dim)` （需要 dim 参数）
- `external_ids: Vec<i64>`（`HnswSqIndex` 使用 i64 IDs）

```rust
pub struct HnswSqIndex {
    inner: hanns::faiss::HnswSqIndex,
    dim: usize,
    external_ids: Vec<i64>,
}
```

Stub（无 hanns-backend）：返回 `Err(AdapterError::Backend("hnsw_sq requires hanns-backend"))`

### Step 2: `crates/hannsdb-index/src/lib.rs`

添加 `pub mod hnsw_sq;`

### Step 3: `crates/hannsdb-index/src/descriptor.rs`

添加 `VectorIndexKind::HnswSq` 枚举变体（serde name: `"hnsw_sq"`）

### Step 4: `crates/hannsdb-index/src/factory.rs`

添加 `VectorIndexKind::HnswSq` 分支：读取 `m`/`ef_construction`/`ef_search` params，创建 `HnswSqIndex`

### Step 5: `crates/hannsdb-core/src/document.rs`

- 添加 `VectorIndexSchema::HnswSq { metric, m, ef_construction, ef_search }` 变体
- 添加构造函数 `hnsw_sq(metric, m, ef_construction, ef_search)`
- 更新所有 match 分支：`metric()`、`quantize_type()`、`hnsw_settings()`
- 添加 `validate_vector_index_descriptor` 中 HnswSq 分支（无 metric 限制，验证参数类型）
- 添加 `validate_schema_primary/secondary_vector_descriptor` 中 HnswSq → descriptor 转换

### Step 6: `crates/hannsdb-py/src/lib.rs`

- 添加 `HnswSqIndexParam` Rust struct + `IndexParam::HnswSq` 枚举变体
- 添加 `PyHnswSqIndexParam` PyO3 class（`metric_type=None, m=16, ef_construction=200, ef_search=50`）
- 更新三处 `index_param` 提取 site（`create_collection` + `add_vector_field` + 第三处）
- 注册到 `#[pymodule]`

### Step 7: `crates/hannsdb-py/python/hannsdb/model/param/index_params.py`

添加 `HnswSqIndexParam` dataclass（frozen, `__post_init__` 验证）+ 更新 `__all__`

### Step 8: `crates/hannsdb-py/python/hannsdb/__init__.py` 等

将 `HnswSqIndexParam` 加入公共导出

---

## 测试计划

**`crates/hannsdb-index/src/hnsw_sq.rs`（单元测试）：**
- `test_hnsw_sq_basic` — 500 向量 insert + search top-10
- `test_hnsw_sq_recall` — 对比 brute-force，recall >= 0.5
- `test_hnsw_sq_save_load_roundtrip` — 序列化/反序列化后查询一致
- `test_hnsw_sq_metric_variants` — L2 / IP / Cosine 均可构建

**`crates/hannsdb-core/src/document.rs` 测试：**
- 合法 params 验证通过
- 未知 key 报错
- zero ef 报错
- serde roundtrip

**Python 测试（`crates/hannsdb-py/tests/`）：**
- `HnswSqIndexParam()` 默认值正确
- 无效 metric_type 报 ValueError
- end-to-end: create → insert → search 返回结果

---

## 成功标准

- [ ] `HnswSqIndexParam` 进入公共 API（`hannsdb.HnswSqIndexParam`）
- [ ] 支持 L2 / IP / Cosine（与 HnswHvq 的 ip-only 区分）
- [ ] `cargo test -p hannsdb-index --features hanns-backend` 通过
- [ ] `cargo test -p hannsdb-core --features hanns-backend` 通过
- [ ] Python end-to-end 测试通过
