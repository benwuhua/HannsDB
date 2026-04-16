# Plan: String Primary Key Fix

**Date:** 2026-04-15
**Spec:** `docs/superpowers/specs/2026-04-15-string-pk-fix-design.md`
**Bug:** insert/upsert 丢弃 public key → string PK registry 不完整 → fetch/delete by string key 在 insert 之后无法找到文档

---

## Step 1 — `upsert_public_keys_with_internal_ids` (`storage/primary_keys.rs`)

在 `register_public_keys_with_internal_ids` 之后新增幂等版本：
- key 已存在且 ID 相同：no-op
- key 已存在但 ID 不同：io::Error（数据一致性违规）
- key 不存在：正常注册

---

## Step 2 — `upsert_documents_with_primary_keys` (`db.rs`)

镜像 `insert_documents_with_primary_keys` 新增 upsert 版本：
1. `require_write()`
2. 提取 public_keys + internal_ids
3. 调用 `upsert_public_keys_with_internal_ids`
4. 调用 `upsert_documents_internal`

---

## Step 3 — 修复 `insert_documents_with_primary_keys` 幂等性 (`db.rs`)

将内部对 `register_public_keys_with_internal_ids` 的调用改为 `upsert_public_keys_with_internal_ids`，避免 `core_documents_from_docs` 已写 registry 后再次写入报错。

---

## Step 4 — 修复 `Collection::insert` (`lib.rs`)

单行修复：用 `insert_documents_with_primary_keys` 替换 `insert_documents`，传 `keyed_documents` 而非裸 `documents`。

---

## Step 5 — 修复 `Collection::upsert` (`lib.rs`)

单行修复：用 `upsert_documents_with_primary_keys` 替换 `upsert_documents`。

---

## Step 6 — 测试

**`crates/hannsdb-core/tests/collection_api.rs`：**
- `test_string_pk_insert_and_fetch`
- `test_string_pk_upsert_overwrite`
- `test_numeric_string_pk_backward_compat`

**`crates/hannsdb-core/tests/wal_recovery.rs`：**
- `test_string_pk_survives_recovery`

**`crates/hannsdb-py/tests/test_collection_facade.py`：**
- `test_string_pk_insert_fetch_roundtrip`
- `test_string_pk_upsert_overwrite`

---

## 成功标准

- [ ] insert with UUID string IDs + fetch by same string → 返回正确文档
- [ ] upsert same string key → 返回最新版本，fetch OK
- [ ] `id="42"` 等纯数字 string → 走 fast-path，现有测试无回退
- [ ] WAL recovery 后 string PK 可被 fetch
- [ ] `cargo test -p hannsdb-core` 全部通过
- [ ] `cargo test -p hannsdb-py` 全部通过
