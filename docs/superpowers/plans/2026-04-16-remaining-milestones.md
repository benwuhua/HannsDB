# Plan: 剩余 5 项里程碑冲刺

**Date:** 2026-04-16
**目标:** 关闭项目计划中剩余 5 个 `[ ]` 项

---

## 优先级排序（第一性原理：哪项对"可用的本地 agent 数据库"最关键）

| # | 里程碑 | 可行性 | 影响 | 决定 |
|---|--------|--------|------|------|
| 1 | Array dtype 支持 | 高（2 文件 ~50 行改动） | 高（zvec 对齐） | **立即做** |
| 2 | 多 segment 运营加固 | 高（补测试为主） | 高（生产正确性） | **立即做** |
| 3 | 存储故事精化 | 中（~895 行提取） | 中（代码健康度） | 做 |
| 4 | Compaction/rebuild 工作流 | 中 | 中 | 部分（sparse compaction + 测试） |
| 5 | Full crash recovery | 低（需要 fuzz 测试） | 高 | 记录当前状态 |

---

## Step 1: Array dtype 支持

**改动量：** 2 文件，~50 行

### 1.1 `crates/hannsdb-py/src/lib.rs` — `py_dict_to_fields` 添加 PyList 分支

在 `py_dict_to_fields` 函数中，`else` 分支前添加 PyList 检测：
```rust
} else if let Ok(list) = value.downcast::<PyList>() {
    let items: Vec<FieldValue> = list.iter()
        .map(|item| /* 递归转换 */)
        .collect::<Result<_, _>>()?;
    FieldValue::Array(items)
}
```

### 1.2 `crates/hannsdb-core/src/forward_store/schema.rs` — 移除 array 拒绝守卫

`build_scalar_array` 中的 `return Err(Unsupported)` 替换为实际 Arrow `ListBuilder` 构建逻辑（参考 `arrow_io.rs` 已有实现）。

### 测试

- `test_array_string_field_roundtrip` — insert ["a","b","c"], fetch 回来一致
- `test_array_int64_field_roundtrip`
- `test_array_float_field_roundtrip`
- `test_array_field_in_filter` — ArrayContains 过滤

---

## Step 2: 多 segment 运营加固（补测试）

### 2.1 稀疏向量 compaction 数据丢失修复

`compact_collection_internal` 不搬运 sparse vectors → 添加 compacted_sparse 收集和写入。

### 2.2 补齐关键边界测试

在 `tests/` 下新增或扩展：

1. `compaction_preserves_sparse_vectors` — 验证 sparse vector 在 compaction 后存活
2. `rollover_during_upsert` — upsert 触发 rollover 后数据一致
3. `compaction_single_immutable_segment` — 只有一个 immutable segment 时 compaction
4. `multiple_compaction_rounds` — 两次 compaction 后数据正确
5. `rollover_at_exact_threshold` — 正好 200K 行触发 rollover

---

## Step 3: 存储故事精化（从 db.rs 提取）

按影响力分 3 批：

### 3.1 `storage/compaction.rs` — 提取 `compact_collection_internal`（~150 行）

### 3.2 `storage/tombstone.rs` — 提取 `delete_internal` + `mark_live_ids_deleted`（~180 行）

### 3.3 `storage/optimize.rs` — 提取 ANN 持久化逻辑（~235 行）

每步后跑 `cargo test -p hannsdb-core` 确认无回归。

---

## Step 4: Crash recovery 记录

不实施新功能，但记录当前 WAL recovery 覆盖范围：
- 已有 6+ WAL recovery 测试覆盖 post-rollover、stale files、missing segment meta
- 缺少：断电模拟（kill -9 + reopen）、并发写入时崩溃

---

## 成功标准

- [ ] Array dtype insert→fetch round-trip 通过（string/int64/float）
- [ ] Sparse vector 在 compaction 后存活
- [ ] 5 个新 segment 边界测试通过
- [ ] `compact_collection_internal` 从 db.rs 提取到 `storage/compaction.rs`
- [ ] `cargo test -p hannsdb-core` 全部通过
- [ ] `.venv-hannsdb/bin/python -m pytest` 全部通过（预期 320+）
