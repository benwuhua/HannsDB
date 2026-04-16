# InvertIndexParam Functional Implementation — Design Spec

**Date:** 2026-04-15
**Gap:** GAP-InvertIndex — `enable_range_optimization` / `enable_extended_wildcard` 当前被强制拒绝，Zvec 均可用

---

## 一、本质问题

HannsDB 已公开 `InvertIndexParam(enable_range_optimization, enable_extended_wildcard)`，但两个 flag 均被 PyO3 层显式拒绝（ValueError）。Zvec 的对等 flag 均已实现：

- `enable_range_optimization`：数值字段范围查询加速（RocksDB bitmap + BTree）
- `enable_extended_wildcard`：字符串字段 suffix/infix 通配符加速（非仅 prefix）

HannsDB 现状：
- **数值 range 查询**：inverted index 内部已用 BTreeMap，范围查询 `lookup_range()` 完整实现，filter evaluator 已接入。**事实上 range optimization 对数值字段已经生效，只是 flag 被拒绝进不来。**
- **字符串通配符加速**：HasPrefix / HasSuffix / Like 过滤器走暴力扫描，inverted index 对字符串只有 HashMap（等值查询），无 trie / 前缀树 / 后缀树结构。

---

## 二、Zvec 实现参考

Zvec `InvertIndexParam`（`/python/zvec/model/param/__init__.pyi`）：
```python
class InvertIndexParam:
    enable_range_optimization: bool   # 数值范围查询加速
    enable_extended_wildcard: bool    # suffix/infix 通配符加速
```

Zvec 内部实现（`src/db/index/column/inverted_column/`）：
- 数值字段用 B-Tree + CRoaring bitmap
- 字符串字段：prefix 总是开启；`enable_extended_wildcard=True` 额外开启 suffix/infix

---

## 三、两个 flag 的精确语义

### 3.1 enable_range_optimization

**语义：** 对该字段的倒排索引启用范围查询优化。

**现状分析：**
- 数值字段（Int32/Int64/UInt32/UInt64/Float/Float64）：BTreeMap 已在内部使用，`lookup_range()` 完整实现，filter evaluator 已调用。**功能已就绪，只需解除 PyO3 拒绝。**
- 字符串字段：HashMap，无法支持 range（lexicographic order），返回空集。启用后应将 HashMap 改为 BTreeMap，支持词典序范围查询。

**实现边界：**
- 允许 `True` 的条件：字段类型为数值或字符串（Bool 无意义，Array 不适用）
- 存储到 descriptor params：`{"enable_range_optimization": true}`
- 数值字段：no-op（已优化），只需描述符正确持久化
- 字符串字段：build 时选择 BTreeMap 而非 HashMap；filter evaluator 中为 String 字段的 Gt/Gte/Lt/Lte 调用 `lookup_range()`

### 3.2 enable_extended_wildcard

**语义：** 对字符串字段的倒排索引启用扩展通配符（suffix + infix + LIKE）查询加速。仅对字符串字段有意义。

**实现策略（最小可行方案）：**
- 选择数据结构：对字符串值构建 **后缀数组（suffix array）** 或 **BTreeMap of reversed strings**（用于 suffix 匹配）
- 最简方案：维护两个 map：
  - `forward: BTreeMap<String, BTreeSet<i64>>`（前缀 + 等值）
  - `reversed: BTreeMap<String, BTreeSet<i64>>`（后缀：key = value.chars().rev().collect()）
- LIKE 通配符：`%abc%` 这类 infix 仍走暴力扫描（复杂度不变），但 `abc%`（前缀）和 `%abc`（后缀）可加速

**实现边界：**
- 仅对 String 字段有意义；非 String 字段接受 True 但忽略（不报错）
- `HasPrefix(field, pattern)` → `forward.range(pattern..=prefix_upper_bound(pattern))` 取并
- `HasSuffix(field, pattern)` → `reversed.range(rev_pattern..=prefix_upper_bound(rev_pattern))` 取并
- LIKE `%abc%`（infix）：从 index 中取所有 candidate，再在内存中做 pattern match（仍比全表扫描快）

---

## 四、代码变更地图

### Layer 1: PyO3 Bridge (`crates/hannsdb-py/src/lib.rs`)

**变更 1.1：解除 flag 拒绝**
```rust
// REMOVE these error blocks (lines 2064-2073):
if enable_range_optimization { return Err(...) }
if enable_extended_wildcard { return Err(...) }
```

**变更 1.2：params JSON 生成（line 935）**
```rust
// 当前：忽略 index_param，输出 {}
// 改为：将 flag 写入 params JSON
fn scalar_index_catalog_json(field_name: &str, index_param: Option<&InvertIndexParam>) -> String {
    let params = match index_param {
        Some(p) => serde_json::json!({
            "enable_range_optimization": p.enable_range_optimization,
            "enable_extended_wildcard": p.enable_extended_wildcard,
        }),
        None => serde_json::json!({}),
    };
    format!(
        r#"{{"vector_indexes":[],"scalar_indexes":[{{"field_name":{},"kind":"inverted","params":{}}}]}}"#,
        json_string(field_name),
        params,
    )
}
```

### Layer 2: Scalar Index Build (`crates/hannsdb-index/src/scalar.rs`)

**变更 2.1：扩展 String variant，支持 BTreeMap（range optimization）**
```rust
pub enum InvertedScalarIndex {
    String {
        descriptor: ScalarIndexDescriptor,
        map: HashMap<String, BTreeSet<i64>>,      // enable_range_optimization=false
    },
    StringOrdered {
        descriptor: ScalarIndexDescriptor,
        map: BTreeMap<String, BTreeSet<i64>>,     // enable_range_optimization=true
    },
    StringWildcard {
        descriptor: ScalarIndexDescriptor,
        forward: BTreeMap<String, BTreeSet<i64>>, // 前缀加速
        reversed: BTreeMap<String, BTreeSet<i64>>,// 后缀加速
    },
    // ... existing numeric variants
}
```

**变更 2.2：build_from_payloads 读取 params**
```rust
fn params_range_opt(descriptor: &ScalarIndexDescriptor) -> bool {
    descriptor.params.get("enable_range_optimization")
        .and_then(|v| v.as_bool()).unwrap_or(false)
}
fn params_wildcard(descriptor: &ScalarIndexDescriptor) -> bool {
    descriptor.params.get("enable_extended_wildcard")
        .and_then(|v| v.as_bool()).unwrap_or(false)
}
```

**变更 2.3：新增查询方法**
- `lookup_prefix(&self, prefix: &str) -> Option<BTreeSet<i64>>`
- `lookup_suffix(&self, suffix: &str) -> Option<BTreeSet<i64>>`
- `lookup_range` 扩展：String/StringOrdered 字段支持 Gt/Gte/Lt/Lte（词典序）

### Layer 3: Filter Acceleration (`crates/hannsdb-core/src/db.rs`)

**变更 3.1：`try_scalar_index_candidates` 扩展（lines 2816-2880）**
```rust
FilterExpr::HasPrefix { field, pattern, negated } => {
    let index = scalar_cache.get(field)?;
    let ids = index.lookup_prefix(pattern)?;
    Some(if *negated { all_ids - ids } else { ids })
}
FilterExpr::HasSuffix { field, pattern, negated } => {
    let index = scalar_cache.get(field)?;
    let ids = index.lookup_suffix(pattern)?;
    Some(if *negated { all_ids - ids } else { ids })
}
```

---

## 五、WAL 和持久化

- `ScalarIndexDescriptor.params` 字段已存在（`serde_json::Value`），WAL replay 路径不需要改
- `create_scalar_index` 已将完整 descriptor 存入 `indexes.json`
- 重建索引时从 descriptor params 读取 flag → 无额外 WAL 改动

---

## 六、测试计划

### 单元测试（`crates/hannsdb-index/src/scalar.rs`）
- `test_range_opt_numeric_flag_accepted` — 数值字段 True 不报错，range 查询结果正确
- `test_range_opt_string_btree` — 字符串字段启用后 Gt/Lt 按词典序生效
- `test_wildcard_prefix_accelerated` — HasPrefix 利用 forward BTreeMap
- `test_wildcard_suffix_accelerated` — HasSuffix 利用 reversed BTreeMap

### 集成测试（`crates/hannsdb-core/tests/collection_api.rs`）
- `test_invert_index_range_opt_flag_roundtrip` — create_scalar_index(InvertIndexParam(range_opt=True)) → close → reopen → flag 持久化
- `test_invert_index_range_query_with_opt` — range filter on indexed numeric field with flag=True
- `test_invert_index_has_prefix_with_wildcard` — HasPrefix with enable_extended_wildcard=True

### Python 测试（`crates/hannsdb-py/tests/test_common_param_surface.py`）
- `InvertIndexParam(enable_range_optimization=True)` 不报错
- `InvertIndexParam(enable_extended_wildcard=True)` 不报错
- end-to-end: create collection → insert → create_scalar_index(InvertIndexParam(True, True)) → query with HasPrefix

---

## 七、实现优先级

1. **Phase 1（最小可行）：** 解除 flag 拒绝 + params 写入 descriptor + 数值字段 range_opt no-op。代码改动 < 30 行，立即让 `InvertIndexParam(enable_range_optimization=True)` 可被接受。
2. **Phase 2：** 字符串字段 BTreeMap（range_opt）+ `lookup_range` 扩展支持 String。
3. **Phase 3：** StringWildcard variant + lookup_prefix/suffix + filter evaluator 接入。

---

## 八、不做什么

- LIKE `%abc%` infix 加速：暂不实现完整 infix 加速（复杂度 O(n) 不变，只是索引加速 prefix/suffix）
- Numeric wildcard：无意义，不支持
- 正则表达式加速：超出本 slice 范围
