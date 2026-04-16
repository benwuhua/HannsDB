# Plan: Zvec Integration Test Parity

**Date:** 2026-04-16
**Goal:** 补齐 HannsDB 缺失的 zvec 集成测试用例，覆盖异常处理、Doc 字段级访问、并发、输入校验四大缺口

---

## 缺口总览

| 域 | zvec 测试数 | HannsDB 已有 | 缺失 |
|---|---|---|---|
| 异常处理（缺失参数） | 11 有效 | 0 | 11 |
| 异常处理（正路径生命周期） | 2 | 部分（facade 有） | 1 |
| Doc 各 dtype get/set/has | 24 | 0 | 24 |
| Doc nullable=True/False | 2 | 0 | 2 |
| 并发操作 | 4 | 0 | 4 |
| Create/Open 输入校验 | ~10 | 部分 | 6 |
| **合计** | **~53** | — | **~48** |

---

## 实施顺序

### Step 1: 新建 `test_doc_field_access.py`

**新增测试文件**，对齐 zvec `test_doc.py` 的 TestCppDoc。

HannsDB 无 `_Doc` native 类，测试对象改为 `Doc` 类的 `fields` / `vectors` 构造和访问：

1. **各标量 dtype round-trip：** 构造 Doc 时传入各 dtype 的 field 值，通过 `doc.field(name)` 读回并验证
   - STRING → `"Tom"`
   - BOOL → `True`
   - INT32 → `19`
   - INT64 → `1111111111111111111`
   - FLOAT → `60.5`（`math.isclose`）
   - FLOAT64/DOUBLE → `1.77777777777`（`math.isclose, rel_tol=1e-9`）
   - UINT32 → `4294967295`
   - UINT64 → `18446744073709551615`

2. **各 array dtype round-trip：**
   - ARRAY_STRING → `["tag1", "tag2", "tag3"]`
   - ARRAY_INT32 → `[1, 2, 3]`
   - ARRAY_INT64 → `[1, 2, 3]`
   - ARRAY_FLOAT → `[1.0, 2.0, 3.0]`
   - ARRAY_DOUBLE → `[1.0, 2.0, 3.0]`
   - ARRAY_BOOL → `[True, False, True]`

3. **各 vector dtype round-trip：**
   - VECTOR_FP32 → `[1.111111, 2.222222, 3.333333]`
   - VECTOR_FP16 → `[1.0, 2.0, 3.0]`
   - VECTOR_INT8 → `[1, 2, 3]`
   - SPARSE_VECTOR_FP32 → `{1: 1.111111, 2: 2.222222, 3: 3.333333}`
   - SPARSE_VECTOR_FP16 → `{1: 1.1, 2: 2.2, 3: 3.3}`

4. **has_field / has_vector：** 对已设和未设的字段/向量做断言

5. **nullable field 行为：**
   - nullable=True + 值为 None → `doc.field("x")` 返回 None（或字段不在 dict 中）
   - nullable=False + 值为 None → 构造时被拒绝（取决于 HannsDB 行为，测试实际行为）

**估计：~30 个测试方法**

---

### Step 2: 新建 `test_exception_handling.py`

**新增测试文件**，对齐 zvec `test_collection_exception.py`。

每个测试用 `pytest.raises` 断言缺失必选参数时抛出异常：

1. `test_create_and_open_missing_path` — `hannsdb.create_and_open()` 无 path → TypeError
2. `test_create_and_open_missing_schema` — `hannsdb.create_and_open(path=...)` 无 schema → TypeError
3. `test_open_missing_path` — `hannsdb.open()` 无 path → TypeError
4. `test_insert_missing_docs` — `collection.insert()` → TypeError
5. `test_update_missing_docs` — `collection.update()` → TypeError
6. `test_upsert_missing_docs` — `collection.upsert()` → TypeError
7. `test_delete_missing_ids` — `collection.delete()` → TypeError
8. `test_fetch_missing_ids` — `collection.fetch()` → TypeError
9. `test_query_missing_vectorquery_field_name` — `VectorQuery()` 无 field_name → 验证报错
10. `test_add_column_missing_field_schema` — `collection.add_column()` → TypeError
11. `test_alter_column_missing_old_name` — `collection.alter_column(new_name=...)` → TypeError
12. `test_alter_column_missing_new_name` — `collection.alter_column(old_name=...)` → TypeError
13. `test_drop_column_missing_field_name` — `collection.drop_column()` → TypeError
14. `test_empty_collection_operations` — 空集合上 fetch/query/update 不崩溃
15. `test_resource_management` — 完整 CRUD 生命周期 + `result.ok()` 验证

**估计：~15 个测试方法**

---

### Step 3: 新增并发测试到 `test_collection_concurrency.py`

在已有文件中追加 4 个测试：

1. `test_concurrent_insert_and_query` — 多线程同时 insert + query，数据一致
2. `test_concurrent_insert_and_delete` — 多线程交替 insert/delete，最终计数正确
3. `test_read_write_locking` — 写操作持锁期间读操作阻塞但不死锁
4. `test_race_condition_detection` — 快速交替 insert/fetch/query，无崩溃

**估计：+4 个测试方法**

---

### Step 4: 新增 Create/Open 输入校验测试

在 `test_schema_surface.py` 或新建 `test_create_validation.py` 中追加：

1. `test_create_rejects_empty_collection_name` — `name=""` → ValueError
2. `test_create_rejects_invalid_collection_name` — `name="a/b"` → ValueError
3. `test_create_rejects_duplicate_field_names` — 两个 field 同名 → ValueError
4. `test_create_rejects_empty_fields_and_vectors` — fields=[] + vectors=[] → ValueError 或允许
5. `test_create_rejects_zero_dimension_vector` — dimension=0 (非 sparse) → ValueError
6. `test_create_rejects_invalid_field_name` — field name 含特殊字符 → ValueError

**估计：+6 个测试方法**

---

## 成功标准

- [ ] `test_doc_field_access.py` 覆盖所有 8 标量 + 6 数组 + 5 向量 dtype round-trip
- [ ] `test_exception_handling.py` 覆盖 13 个缺失参数异常 + 2 个正路径
- [ ] `test_collection_concurrency.py` 新增 4 个并发测试
- [ ] 输入校验测试新增 6 个
- [ ] 全部测试通过：`.venv-hannsdb/bin/python -m pytest crates/hannsdb-py/tests/ -q`
- [ ] 总测试数从 263 增长到 ~320+
