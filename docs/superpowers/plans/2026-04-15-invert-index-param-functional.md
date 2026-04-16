# Plan: InvertIndexParam Functional Implementation

**Date:** 2026-04-15
**Spec:** `docs/superpowers/specs/2026-04-15-invert-index-param-functional-design.md`
**Gap closed:** Zvec `InvertIndexParam(enable_range_optimization, enable_extended_wildcard)` 均可用；HannsDB 当前强制拒绝

---

## 实施顺序

### Step 1: 解除 PyO3 拒绝 + 修复 params JSON 写入
**文件：** `crates/hannsdb-py/src/lib.rs`

1.1 删除 `enable_range_optimization=True` 的 `PyValueError` block（~line 2064-2068）
1.2 删除 `enable_extended_wildcard=True` 的 `PyValueError` block（~line 2069-2073）
1.3 修复 `scalar_index_catalog_json()` 函数（~line 935）— 当前忽略 index_param，改为将 flag 序列化进 params JSON

验证：`InvertIndexParam(enable_range_optimization=True)` 和 `InvertIndexParam(enable_extended_wildcard=True)` 不再报错，descriptor params 字段中含正确 JSON。

---

### Step 2: 字符串字段 enable_range_optimization 支持
**文件：** `crates/hannsdb-index/src/scalar.rs`

2.1 添加 `params_range_opt(descriptor) -> bool` helper
2.2 添加 `params_wildcard(descriptor) -> bool` helper
2.3 为 String 类型添加 `StringOrdered` variant（`BTreeMap<String, BTreeSet<i64>>`）
2.4 `build_from_payloads` 中：String 字段 + range_opt=true → 构建 StringOrdered
2.5 `lookup_range` 对 StringOrdered 支持 Gt/Gte/Lt/Lte（词典序）

**文件：** `crates/hannsdb-core/src/db.rs`

2.6 `try_scalar_index_candidates` 中 String 字段的 Gt/Gte/Lt/Lte 调用 `lookup_range`（之前 String 不支持，返回 None）

---

### Step 3: enable_extended_wildcard 支持
**文件：** `crates/hannsdb-index/src/scalar.rs`

3.1 添加 `StringWildcard` variant：
   - `forward: BTreeMap<String, BTreeSet<i64>>`（用于 prefix 加速）
   - `reversed: BTreeMap<String, BTreeSet<i64>>`（key = value reversed，用于 suffix 加速）
3.2 `build_from_payloads` 中：String 字段 + wildcard=true → 构建 StringWildcard
3.3 实现 `lookup_prefix(&self, prefix: &str) -> Option<BTreeSet<i64>>`
3.4 实现 `lookup_suffix(&self, suffix: &str) -> Option<BTreeSet<i64>>`

**文件：** `crates/hannsdb-core/src/db.rs`

3.5 `try_scalar_index_candidates` 扩展：
   - `FilterExpr::HasPrefix` → 调用 `lookup_prefix()`
   - `FilterExpr::HasSuffix` → 调用 `lookup_suffix()`

---

### Step 4: 测试
**文件：** `crates/hannsdb-index/src/scalar.rs`（单元测试）
- `test_invert_index_string_range_with_opt`
- `test_invert_index_prefix_wildcard_acceleration`
- `test_invert_index_suffix_wildcard_acceleration`

**文件：** `crates/hannsdb-core/tests/collection_api.rs`（集成测试）
- `test_invert_index_range_opt_flag_accepted`
- `test_invert_index_range_opt_flag_persists_reopen`
- `test_invert_index_has_prefix_accelerated`
- `test_invert_index_has_suffix_accelerated`

**文件：** `crates/hannsdb-py/tests/test_common_param_surface.py`
- `InvertIndexParam(True, False)` / `InvertIndexParam(False, True)` / `InvertIndexParam(True, True)` 不报错
- end-to-end create_scalar_index with both flags

---

## 成功标准

- [ ] `InvertIndexParam(enable_range_optimization=True)` 被接受（不再 ValueError）
- [ ] `InvertIndexParam(enable_extended_wildcard=True)` 被接受（不再 ValueError）
- [ ] descriptor params 中包含正确的 flag JSON，且 reopen 后持久化
- [ ] 字符串字段 Gt/Gte/Lt/Lte filter + `create_scalar_index(InvertIndexParam(range_opt=True))` 使用索引加速
- [ ] `HasPrefix` / `HasSuffix` filter + `create_scalar_index(InvertIndexParam(wildcard=True))` 使用索引加速
- [ ] `cargo test -p hannsdb-core` 全部通过
- [ ] `cargo test -p hannsdb-py` 全部通过
