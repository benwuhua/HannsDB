# String Primary Key Fix — Design Spec

**Date:** 2026-04-15
**Gap:** GAP-5 — Python 层 insert/upsert 丢弃 public key，string PK registry 写入路径不完整

---

## 一、本质问题

HannsDB 的 `PrimaryKeyRegistry` 基础设施已经完整：
- `crates/hannsdb-core/src/pk.rs`：双向 BTreeMap（string↔i64），auto-increment internal ID
- `assign_internal_ids_for_public_keys`：已实现 numeric fast-path + string mode 升级
- `insert_documents_with_primary_keys`：已存在，正确写入 registry
- `fetch` / `delete` / `query_by_id`：均走 string-keyed core 方法，已正确

**Bug 所在：** `Collection::insert` 和 `Collection::upsert`（`crates/hannsdb-py/src/lib.rs`）调用 `core_documents_from_docs` 分配了 internal ID，但随即**丢弃** public key，改用 `db.insert_documents` / `db.upsert_documents`（绕过 registry）。

另外，`upsert_documents_with_primary_keys` 方法不存在，所以即使修 insert，upsert 也缺对等方法。

---

## 二、修复架构

保持 `Document.id: i64` 内部不变，string→i64 映射由 `PrimaryKeyRegistry` 负责。无 hash 映射（有碰撞风险），使用权威 BTreeMap。

**向后兼容：** 纯数字 string ID（`"1"`, `"42"`）走 numeric fast-path，registry 不更新，现有数据零迁移。

---

## 三、代码变更

### 3.1 新增 `upsert_public_keys_with_internal_ids`
**文件：** `crates/hannsdb-core/src/storage/primary_keys.rs`

与 `register_public_keys_with_internal_ids` 逻辑相同，但允许 key 已存在且 ID 相同（idempotent overwrite）。若 key 存在但 ID 不同则报错（数据一致性违规）。

```rust
pub(crate) fn upsert_public_keys_with_internal_ids(
    paths: &CollectionPaths,
    collection_meta: &mut CollectionMetadata,
    public_keys: &[String],
    internal_ids: &[i64],
) -> io::Result<()>
```

### 3.2 新增 `upsert_documents_with_primary_keys`
**文件：** `crates/hannsdb-core/src/db.rs`

镜像 `insert_documents_with_primary_keys`，但调用 `upsert_documents_internal` + `upsert_public_keys_with_internal_ids`。

```rust
pub fn upsert_documents_with_primary_keys(
    &mut self,
    collection: &str,
    keyed_documents: &[(String, Document)],
) -> io::Result<usize>
```

### 3.3 修复 insert_documents_with_primary_keys 的幂等性
**文件：** `crates/hannsdb-core/src/db.rs`

`core_documents_from_docs` 内部已经调用 `assign_internal_ids_for_primary_keys` 写 registry，然后 `insert_documents_with_primary_keys` 再次写 registry 会报 "key already exists" 错误。

修复：`insert_documents_with_primary_keys` 改用 `upsert_public_keys_with_internal_ids`（幂等），或从 `core_documents_from_docs` 中移除 registry 写入。

**最简修法：** `insert_documents_with_primary_keys` 内部调用 `upsert_public_keys_with_internal_ids` 代替 `register_public_keys_with_internal_ids`。

### 3.4 修复 `Collection::insert`
**文件：** `crates/hannsdb-py/src/lib.rs`

```rust
// Before:
let documents: Vec<_> = core_documents_from_docs(self, docs)?
    .into_iter().map(|(_, doc)| doc).collect();
self.db.insert_documents(&self.collection_name, &documents)

// After:
let keyed_documents = core_documents_from_docs(self, docs)?;
self.db.insert_documents_with_primary_keys(&self.collection_name, &keyed_documents)
```

### 3.5 修复 `Collection::upsert`
**文件：** `crates/hannsdb-py/src/lib.rs`

```rust
// Before:
let documents: Vec<_> = core_documents_from_docs(self, docs)?
    .into_iter().map(|(_, doc)| doc).collect();
self.db.upsert_documents(&self.collection_name, &documents)

// After:
let keyed_documents = core_documents_from_docs(self, docs)?;
self.db.upsert_documents_with_primary_keys(&self.collection_name, &keyed_documents)
```

---

## 四、WAL 正确性

WAL record 存储 `Vec<Document>` with `Document.id: i64`。Registry 在 WAL write 之前写盘。crash after registry write / before WAL write → registry 有孤立分配（harmless，不影响查询）。crash after WAL write → replay 用已分配 ID，registry 已有映射。安全。

---

## 五、测试计划

### `crates/hannsdb-core/tests/collection_api.rs`
- `test_string_pk_insert_and_fetch` — insert with UUID string IDs, fetch back by same strings
- `test_string_pk_upsert_overwrite` — upsert same string key, verify latest value
- `test_string_pk_delete_by_string` — delete by string key
- `test_string_pk_query_by_id` — query_by_id with string PK
- `test_numeric_string_pk_backward_compat` — `id="42"` 走 fast-path，registry 不变

### `crates/hannsdb-py/tests/test_collection_facade.py`
- `test_string_pk_insert_fetch_roundtrip`
- `test_string_pk_upsert_overwrite`
- `test_numeric_string_id_backward_compat`

### `crates/hannsdb-core/tests/wal_recovery.rs`
- `test_string_pk_survives_recovery` — insert string-keyed docs, close without flush, reopen, fetch by string keys

---

## 六、不做的事

- 不改 `CoreDocument.id: i64` 内部类型
- 不加 Python 层 hash 映射
- 不做 schema migration（纯数字集合无需迁移）
- 不改 Doc.id 的 Python 类型（已经是 str）
