# Spec: Zvec Integration Test Parity — Design

**Date:** 2026-04-16
**Plan:** `docs/superpowers/plans/2026-04-16-zvec-test-parity.md`

---

## 设计原则

1. **测试 HannsDB 实际行为，不 mock** — 对齐 zvec 测试意图，但验证 HannsDB 自己的 API 语义
2. **TypeError vs ValueError** — zvec 统一用 `pytest.raises(Exception)` 捕获；HannsDB Python 层缺参数时 Python 自身抛 `TypeError`，我们用 `TypeError` 精确匹配
3. **Doc 字段通过 Collection round-trip** — 先 insert 再 fetch 读回，验证端到端 dtype 保持
4. **并发用 threading** — 与现有 `test_collection_concurrency.py` 一致

---

## 文件 1: `test_doc_field_access.py`

### Fixture

```python
@pytest.fixture
def col(tmp_path):
    schema = CollectionSchema(
        name="dtype_test",
        fields=[
            FieldSchema("s", DataType.STRING),
            FieldSchema("b", DataType.BOOL),
            FieldSchema("i32", DataType.INT32),
            FieldSchema("i64", DataType.INT64),
            FieldSchema("f", DataType.FLOAT),
            FieldSchema("f64", DataType.FLOAT64),
            FieldSchema("u32", DataType.UINT32),
            FieldSchema("u64", DataType.UINT64),
            FieldSchema("nullable_s", DataType.STRING, nullable=True),
            FieldSchema("arr_s", DataType.ARRAY_STRING),
            FieldSchema("arr_i32", DataType.ARRAY_INT32),
            FieldSchema("arr_i64", DataType.ARRAY_INT64),
            FieldSchema("arr_f", DataType.ARRAY_FLOAT),
            FieldSchema("arr_f64", DataType.ARRAY_DOUBLE),
            FieldSchema("arr_b", DataType.ARRAY_BOOL),
        ],
        vectors=[
            VectorSchema("vec", DataType.VECTOR_FP32, dimension=3),
        ],
    )
    return hannsdb.create_and_open(str(tmp_path / "db"), schema)
```

### 测试结构

每个 dtype 一个测试方法，模式一致：
```python
def test_{}_field_roundtrip(self, col):
    doc = Doc(id="1", fields={"field_name": VALUE}, vectors={"vec": [0.1, 0.2, 0.3]})
    col.insert(doc)
    fetched = col.fetch("1")
    assert fetched.field("field_name") == VALUE  # 或 math.isclose
```

### Nullable 测试

```python
def test_nullable_field_accepts_none(self, col):
    doc = Doc(id="1", fields={"nullable_s": None}, vectors={"vec": [0.1, 0.2, 0.3]})
    col.insert(doc)
    fetched = col.fetch("1")
    # nullable field omitted or None

def test_non_nullable_field_rejects_none_via_validation(self, col):
    # 构造含 None 的 required field → _validate_doc_nullable_fields raises
```

---

## 文件 2: `test_exception_handling.py`

### Fixture

```python
@pytest.fixture
def col(tmp_path):
    schema = CollectionSchema(
        name="exc_test",
        fields=[FieldSchema("name", DataType.STRING)],
        vectors=[VectorSchema("dense", DataType.VECTOR_FP32, dimension=4)],
    )
    return hannsdb.create_and_open(str(tmp_path / "db"), schema)
```

### 异常模式

所有缺失参数测试统一用：
```python
def test_insert_missing_docs(self, col):
    with pytest.raises(TypeError):
        col.insert()
```

正路径生命周期：
```python
def test_resource_management(self, col):
    doc = Doc(id="0", fields={"name": "test"}, vectors={"dense": [1.0, 2.0, 3.0, 4.0]})
    result = col.insert(doc)
    assert result.ok()
    fetched = col.fetch("0")
    assert fetched.field("name") == "test"
    # update, delete, etc.
```

---

## 文件 3: `test_collection_concurrency.py`（追加）

### 新增测试

```python
def test_concurrent_insert_and_query(self, tmp_path):
    """Multiple threads insert while others query — no crashes, no data loss."""

def test_concurrent_insert_and_delete(self, tmp_path):
    """Interleaved insert/delete — final doc_count matches expected."""

def test_read_write_locking(self, tmp_path):
    """Write holds lock; reads queue but don't deadlock."""

def test_race_condition_detection(self, tmp_path):
    """Rapid fire insert/fetch/query cycle — no crashes."""
```

---

## 文件 4: 输入校验测试（追加到 `test_schema_surface.py`）

```python
def test_create_rejects_empty_collection_name(self, tmp_path):
    schema = CollectionSchema(name="", fields=[], vectors=[...])
    with pytest.raises((ValueError, RuntimeError)):
        hannsdb.create_and_open(str(tmp_path / "db"), schema)

# ... 其余 5 个类似
```

---

## 注意事项

- `ARRAY_*` dtype 需要 HannsDB 核心支持 array 字段的 insert/fetch round-trip；如果核心不支持，对应测试用 `pytest.mark.skip` 标注
- `VECTOR_FP16` / `SPARSE_VECTOR_FP16` 同理，取决于 HannsDB 核心对这些 dtype 的支持程度
- 并发测试不验证性能，只验证正确性和无崩溃
- 输入校验测试可能发现 HannsDB 当前不校验某些输入的情况——记录实际行为即可
