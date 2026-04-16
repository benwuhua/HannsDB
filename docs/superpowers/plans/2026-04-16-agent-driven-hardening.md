# Plan: Agent-Driven Hardening Sprint

**Date:** 2026-04-16
**Goal:** 从第一性原理出发，优先做对"本地 agent 数据库"最有价值的事

---

## 优先级排序

| # | 项 | 影响 | 难度 | 决定 |
|---|---|------|------|------|
| 1 | WAL fsync 持久化保证 | **Critical** — 无 fsync 则 WAL 对断电零保护 | 低（~20 行） | **P0** |
| 2 | Daemon HTTP 服务启动 | **Critical** — main.rs 无 axum::serve，daemon 不可用 | 低（~30 行） | **P0** |
| 3 | 存储 tombstone 提取 | 中 — db.rs 减 115 行，代码健康度 | 中 | **P1** |
| 4 | 存储 persist 提取 | 中 — db.rs 减 348 行，ANN 序列化独立 | 中 | **P1** |
| 5 | Crash-during-compaction 测试 | 高 — 验证最危险操作的中断恢复 | 中 | **P1** |
| 6 | WAL 中间行容错 | 高 — 单行损坏当前导致不可恢复 | 中 | **P2** |

---

## Step 1: WAL fsync（P0）

**文件：** `crates/hannsdb-core/src/wal.rs`

WAL `append` 当前只做 `BufWriter::flush()`，不保证数据落盘。
添加 `file.sync_all()` 在每次 WAL 写入后（或可选：每 N 次 fsync 一次）。

```rust
// append_wal_record 末尾
writer.flush()?;
writer.get_ref().sync_all()?;  // 确保落盘
```

**测试：** 新增 `wal_fsync_persists_records_to_disk` — 验证 sync_all 被调用。

---

## Step 2: Daemon HTTP 服务启动（P0）

**文件：** `crates/hannsdb-daemon/src/main.rs`

当前 main.rs 只打印 banner。需要：
1. 解析端口参数（默认 19530）
2. 创建 `HannsDb::open` 实例
3. 调用 `build_router()` 并启动 `axum::serve`

```rust
#[tokio::main]
async fn main() {
    let port = std::env::var("HANNSDB_PORT")
        .unwrap_or_else(|_| "19530".to_string())
        .parse::<u16>()
        .expect("invalid port");
    let db = HannsDb::open(...).expect("open db");
    let app = build_router(Arc::new(RwLock::new(db)));
    let listener = TcpListener::bind(("0.0.0.0", port)).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

---

## Step 3: 存储层 tombstone 提取（P1）

**新文件：** `crates/hannsdb-core/src/storage/tombstone.rs`

从 `db.rs` 提取：
- `delete_internal`（67 行, L1442-1508）
- `mark_live_ids_deleted_across_segments`（48 行, L1623-1670）

函数签名改为自由函数，接收 paths + collection_meta 等参数，返回 `TombstoneResult`。

---

## Step 4: 存储层 persist 提取（P1）

**新文件：** `crates/hannsdb-core/src/storage/persist.rs`

从 `db.rs` 提取：
- `optimize` 的核心存储逻辑（ANN/scalar/sparse 序列化，235 行）
- `try_load_persisted_ann_state`（113 行）

---

## Step 5: Crash-during-compaction 测试（P1）

**文件：** `crates/hannsdb-core/tests/compaction.rs`

模拟 compaction 中断：
1. 创建 2 个 immutable segments
2. 手动执行部分 compaction（写入新 segment 但不更新 segment_set）
3. Reopen 后验证 WAL replay 恢复正确状态

---

## Step 6: WAL 中间行容错（P2）

**文件：** `crates/hannsdb-core/src/wal.rs`

`load_wal_records` 当前在中间行 parse 失败时返回 Err。
改为：中间行失败时 log warning 并停止读取（与 tail truncation 相同策略）。

---

## 验证

```bash
cargo test -p hannsdb-core           # 380+ tests pass
cargo test -p hannsdb-index           # 36 tests pass
cargo test -p hannsdb-daemon          # daemon compiles
cargo check -p hannsdb-daemon         # no compile errors
```

---

## 成功标准

- [ ] WAL fsync 保证数据落盘
- [ ] `cargo run -p hannsdb-daemon` 启动 HTTP 服务并响应 `/health`
- [ ] tombstone 逻辑从 db.rs 提取到 storage/tombstone.rs
- [ ] ANN 持久化逻辑从 db.rs 提取到 storage/persist.rs
- [ ] crash-during-compaction 测试通过
- [ ] `cargo test -p hannsdb-core` 全部通过
