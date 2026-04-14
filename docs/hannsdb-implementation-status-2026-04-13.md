# HannsDB 实现状态更新（2026-04-13）

## 本轮已完成

### 1. IvfUsq 真实 runtime/productization
- 新增独立 `IvfUsq` kind
- HannsDB 已可真实执行 `create/open/optimize/query/reopen`
- 不再是仅 public surface 占位
- `bitset` 路径已补 correctness-first 处理

### 2. HnswHvq honest public/runtime slice
- Hanns 上游已补 `save/load` 基础能力
- HannsDB 已接入独立 `HnswHvq` family
- 当前对外 contract 为 `ip`-only，保持 honest
- `create/open/optimize/query/reopen` 已打通

### 3. active segment Arrow snapshot
- `flush_collection()` 后会为 active segment 物化：
  - `payloads.arrow`
  - `vectors.arrow`
- 后续 active 写入会失效旧 Arrow snapshot
- 下次 flush 会重新生成 snapshot

### 4. 主 smoke / 文档同步
- `scripts/run_zvec_parity_smoke.sh` 已纳入：
  - `test_ivf_usq_surface.py`
  - `test_hnsw_hvq_surface.py`
  - `lifecycle` core suite
  - 完整 `compaction` suite
  - 完整 `collection_api` suite
  - 完整 `wal_recovery` suite
  - 完整 `segment_storage` suite
  - feature-on ANN 状态合同
- 正式文档已同步：
  - `docs/hannsdb-vs-zvec-gap-analysis.md`
  - `docs/hannsdb-hanns-vs-zvec-capability-report.md`
  - `docs/vector-db-bench-notes.md`

### 9. repo-local optimize proxy 增强
- `scripts/run_hannsdb_optimize_bench.sh` 现在支持 `INDEX_KIND`
- `scripts/run_hannsdb_optimize_bench.sh` 现在也支持 `QUERY_EF_SEARCH / QUERY_NPROBE`
- 当前已验证的 honest benchmark 代理路径：
  - `hnsw`
  - `ivf_usq` (`METRIC=l2`)
  - `hnsw_hvq` (`METRIC=ip`)
- 非法组合会在脚本层直接拒绝：
  - 未知 `INDEX_KIND`
  - `hnsw_hvq + 非 ip`
  - `QUERY_NPROBE + 非 IVF 家族`

### 5. 本轮 deslop / 硬化
- 删除 `hannsdb-core` 本轮改动范围内的未使用缓存字段与死代码
- 清理未使用 helper / import / variable，减少本仓 warning 噪音
- 范围保持在：
  - `crates/hannsdb-core/src/db.rs`
  - `crates/hannsdb-core/src/query/planner.rs`
  - `crates/hannsdb-core/src/segment/index_runtime.rs`

### 6. active Arrow snapshot reopen 修复
### 7. compacted segment reopen 回归合同
### 8. auto-rollover 生命周期回归解封
- 旧的 ignored 生命周期测试已验证为真实通过，现已解封
- 新增 reopen 后 `search` 回归，证明 tombstone-ratio 触发 rollover 后 collection 仍可稳定读取

- 为 compacted immutable segment 新增 reopen + search/fetch 回归合同
- 由于该用例使用手工 segment fixture（绕过 WAL），在 reopen 前显式截断 WAL，模拟 clean shutdown，避免 recovery 抹掉非 WAL fixture 数据
- 该调整是测试夹具修正，不是产品路径降级

- 新增回归测试：`flush_collection()` 后即使移除 active segment 的
  `payloads.jsonl` / `vectors.jsonl`，仍可通过 Arrow snapshot reopen + fetch
- 修复 empty-schema Arrow sentinel 的读路径：
  - 兼容新的 0 字节 sentinel
  - 兼容历史 `_dummy` schema sentinel
  - 通过同目录 `ids.bin` 推导真实行数，避免把非空 segment 误读成 0 行

### 10. ANN 状态合同修复
- `index_completeness` 不再只依赖内存 `search_cache`
- 在 `hanns-backend` 下，若已存在持久化 ANN blob，重启后仍会继续报告 `1.0`
- 数据写入（insert / insert_documents / upsert / delete）现在会失效持久化 ANN blob，避免 `ann_ready` 假阳性
- daemon 包已增加 `hanns-backend` feature forwarding，`http_smoke` 可直接验证：
  - optimize 后 `ann_ready=true`
  - 后续写入后 `ann_ready=false`
- `collection_info` 也已补齐对应状态合同：
  - optimize 后 `index_completeness=1.0`
  - 重启后若 persisted ANN 仍存在，则保持 `1.0`
  - 后续写入后重新回到 `0.0`

### 11. storage 模块切片起步
- 新增显式 `crates/hannsdb-core/src/storage/`
- 当前先把现有 WAL / recovery helper 收编到：
  - `storage::wal`
  - `storage::recovery`
- 这一步暂时不改行为，目的是给后续 durability/storage story 整理提供稳定落脚点
- 后续又把 `WalReplayPlan` 与 `collection_name_for_wal_record` 也移入 `storage::recovery`
- active / immutable segment 的一批读盘 helper 也开始从 `db.rs` 收拢到：
  - `storage::segment_io`
  - 当前已包含 `load_records_or_empty / load_record_ids_or_empty / append_documents`
  - 以及 `load_shadowed_live_records / load_shadowed_live_vector_records / load_all_collection_ids / persisted_ann_exists`
  - 以及 `has_live_id / latest_live_row_index / latest_row_index_for_id / mark_live_id_deleted`
- primary-key registry 的文件读写与 numeric-key 扫描也开始收拢到：
  - `storage::primary_keys`
  - 当前已包含 `load_primary_key_registry / save_primary_key_registry / ensure_string_primary_key_mode`
  - 以及 `parse_numeric_public_key / display_key_for_internal_id`
  - 以及 `assign_internal_ids_for_public_keys / resolve_public_keys_to_internal_ids / register_public_keys_with_internal_ids`
- 到这一轮为止，`db.rs` 里原先那批 storage / primary-key 辅助函数已经基本抽空，剩余更多是 orchestration 与业务合同，而不是文件布局细节
- 文档层也同步确认：`validate_documents` 已移到 `document.rs`，进一步减少 `db.rs` 的输入校验细节
- schema/index descriptor 的输入校验也已移到 `document.rs`：
  - `validate_vector_index_descriptor`
  - `validate_schema_primary_vector_descriptor`
  - `validate_schema_secondary_vector_descriptors`
- `field_value_to_scalar` 也已移到 `document.rs`，进一步减少 `db.rs` 对 payload/value 细节转换的直接承担
- `next_compacted_segment_id` 也已移到 `storage::segment_io`，继续把 segment layout/命名细节从 `db.rs` 剥离
- `sort_document_hits_by_field` / `compare_field_values_for_sort` 已删除，`db.rs` 现在复用 `query::executor` 的排序 helper，减少重复排序语义
- `project_document_hits` 也已删掉，`db.rs` 现在复用 `query::executor::project_hits_output_fields`
- `RankedDocumentHit` 现在也复用 `query::executor::compare_hits`，继续消除命中排序语义的重复定义
- `RankedDocumentHit` 本身也已收回到局部作用域，不再作为 `db.rs` 的顶层辅助类型长期悬挂
- 到当前这一步，`db.rs` 文件级顶层私有辅助定义已经明显缩到只剩少数状态字段与 handle/container 定义，不再堆着大量 storage/query helper
- filtered segment-aware query 路径里最后一处手写 `distance/id` 排序也已改为复用 `compare_hits`
- 占位性质的 `IndexRegistry` 也已删除，`CollectionHandle` 不再额外挂一个未实际承载状态的 registry 壳
- `CachedDocumentState` 这个单字段 wrapper 也已删除，document cache 现在直接缓存 `Arc<HashMap<i64, Document>>`

### 12. rollover 之后的 active write 路径修复
- 这一步继续补 storage/runtime maturity，而不是扩新的 public surface
- 之前 multi-segment 读路径已经是 segment-aware，但部分写路径仍然会在 `segment_set.json` 存在后继续盯着 collection 根目录文件
- 现在：
  - `insert`
  - `insert_documents`
  - `upsert_documents`
  都会在进入 multi-segment 模式后，改为通过 `SegmentManager::active_segment_path()` 追加到真正的 active segment
- live-id / duplicate-id 检查也不再只看 root-level `ids.bin`，而是按 collection 的 live view 做跨 segment 判断
- 新增回归测试覆盖：
  - rollover 之后的新写入确实落在 active segment
  - reopen 之后仍可读到 post-rollover 新写入
  - rollover 之后对旧 live id 的重复插入会被拒绝

### 13. active-segment mutation authority 收拢
- 这一轮继续往 storage/runtime orchestration 靠拢，不再让 `db.rs` 自己重复实现 append/rollover/sealing mechanics
- 当前 authority split 明确为：
  - `SegmentWriter`：active segment append、首轮 rollover 迁移、旧 active sealing（`JSONL -> Arrow`）
  - `VersionSet` / `SegmentManager`：segment topology 与 active/immutable path authority
  - `db.rs`：WAL、duplicate/live-view policy、ANN invalidation、compaction policy
- 这意味着：
  - `insert`
  - `insert_documents`
  - `upsert_documents`
  已改为通过统一的 writer-owned mutation path 落盘，而不是继续各自在 `db.rs` 里维护一份 append/rollover 细节
- 同时补了两类额外回归：
  - post-rollover `update_documents` 仍会把最新 live row 追加到 active segment
  - post-rollover `delete` 仍保持 latest-live 语义，但没有把 delete 逻辑错误地收进 writer authority
- 为避免 ad-hoc payload 在 sealed Arrow segment 中退化成 JSON 字符串，本轮也补了 Arrow payload ad-hoc field 的类型推断；常见 scalar/bool/array ad-hoc 字段在 rollover/reopen 后保持原始语义
- 未被使用的 `CollectionHandle::name()` accessor 也已删掉，避免再保留无调用的薄包装 API
- `FieldValue` 排序比较逻辑也已收拢到 `document.rs`，`query::executor` 不再内联维护那段 match 细节
- `CollectionHandle` 不再单独保存 `root`; collection path / wal path 现在直接从 `segment_manager.collection_dir()` 推导，减少一份重复状态
- `CollectionHandle` 也不再保存单独的 `name`; 日志里的 collection 名称现在直接从 `segment_manager.collection_dir()` 推导
- `CollectionHandle::collection_name()` 这个局部 helper 也已删掉，直接复用 `segment_manager.collection_name()`
- optimize / persisted-ANN load 路径里的 collection-name 日志也已做局部缓存，避免同一函数里反复重复取值
- `storage::paths::root_for_collection_dir()` 这个单点 helper 也已内联回 `wal_path_for_collection_dir()`，继续压缩路径工具表面
- `wal_path_for_collection_dir()` 也已继续收口；当前 flush 路径直接在使用点推导 collection-root 的 WAL 路径，不再额外挂一个单点 wrapper
- `load_wal_records_or_empty()` 也已从 `storage::recovery` 收回到 `storage::wal`，让 WAL 相关 helper 更集中在同一个落脚点
- `SearchHit` 的默认排序也开始统一复用，`query/search.rs` 不再在 dense/sparse 两条 brute-force 路径各自重复写一份相同排序闭包
- `hannsdb-core` 里的 `core_bootstrap_marker()` 占位导出与其自测也已删除，避免继续保留无外部引用的 bootstrap 噪音
- `DocumentHit` 也已从 `db.rs` 顶层移到 `query` 模块承接，`db.rs` 仅保留兼容 re-export，进一步消除 query 对 db 的反向耦合
- `SearchHit` 也已与 `DocumentHit` 一起集中到 `query::hits`，`query::search` 只保留搜索实现而不再自带结果类型定义
- `compare_search_hits` 也已跟着 `SearchHit` 一起落到 `query::hits`，让结果类型与其默认比较规则放在同一落脚点
- `CollectionInfo / CollectionSegmentInfo` 也已移到独立 `db_types` 模块承接，`db.rs` 只保留兼容 re-export，继续缩小 `db.rs` 顶层类型面
- `hannsdb-core` crate 顶层也开始直接 re-export `CollectionInfo / CollectionSegmentInfo / DocumentHit / SearchHit`，daemon 已改用更短的顶层类型路径
- `HannsDb` 本体也已开始从 crate 顶层 re-export，daemon 侧已改用 `hannsdb_core::HannsDb`，继续减弱对 `db` 子模块路径的耦合
- reranker 里重复的 fused-score 排序也已统一成单一 helper，`rrf` / `weighted` 不再各自维护一份相同 `sort_by`
- `cargo check -p hannsdb-core --features hanns-backend` 与 `wal_recovery` 全套仍保持绿色

## 最新验证证据

### Python focused surfaces
```bash
cd crates/hannsdb-py && source .venv/bin/activate && \
python -m pytest tests/test_ivf_usq_surface.py tests/test_hnsw_hvq_surface.py -q
```
结果：`9 passed`

### 主 parity smoke
```bash
bash scripts/run_zvec_parity_smoke.sh
```
结果：
- Rust parity ✅
- `lifecycle` ✅
- `compaction` ✅
- `collection_api` ✅
- `wal_recovery` ✅
- `segment_storage` ✅
- feature-on ANN 状态合同 ✅
- daemon `http_smoke` ✅
- Python `168 passed, 4 skipped` ✅

### core check（deslop 后）
```bash
cargo check -p hannsdb-core --features hanns-backend
```
结果：通过 ✅

### active Arrow reopen 回归
```bash
cargo test -p hannsdb-core \
  collection_api_reopen_fetch_documents_uses_active_arrow_snapshots_when_jsonl_sidecars_are_missing \
  -- --nocapture
```
结果：通过 ✅

### optimize proxy（最新一轮）
```bash
N=2000 DIM=256 METRIC=cosine TOPK=10 REPEATS=3 FEATURES=hanns-backend PROFILE=release \
  bash scripts/run_hannsdb_optimize_bench.sh
```
结果：
- `create=0`
- `insert=14`
- `optimize=107`
- `search=0`
- `total=122`

### quantized optimize proxy（最新一轮）
```bash
N=200 DIM=64 METRIC=l2 TOPK=10 INDEX_KIND=ivf_usq REPEATS=1 FEATURES=hanns-backend PROFILE=release \
  bash scripts/run_hannsdb_optimize_bench.sh
```
结果：
- `create=0`
- `insert=0`
- `optimize=3`
- `search=0`
- `total=5`

```bash
N=200 DIM=64 METRIC=ip TOPK=10 INDEX_KIND=hnsw_hvq REPEATS=1 FEATURES=hanns-backend PROFILE=release \
  bash scripts/run_hannsdb_optimize_bench.sh
```
结果：
- `create=0`
- `insert=1`
- `optimize=16`
- `search=0`
- `total=18`

### query-param optimize proxy（最新一轮）
```bash
N=200 DIM=64 METRIC=l2 TOPK=10 INDEX_KIND=ivf_usq QUERY_NPROBE=5 REPEATS=1 FEATURES=hanns-backend PROFILE=release \
  bash scripts/run_hannsdb_optimize_bench.sh
```
结果：
- `OPT_BENCH_QUERY_PARAMS ef_search=none nprobe=5`
- `create=2`
- `insert=2`
- `optimize=3`
- `search=0`
- `total=11`

```bash
N=200 DIM=64 METRIC=cosine TOPK=10 INDEX_KIND=hnsw QUERY_EF_SEARCH=48 REPEATS=1 FEATURES=hanns-backend PROFILE=release \
  bash scripts/run_hannsdb_optimize_bench.sh
```
结果：
- `OPT_BENCH_QUERY_PARAMS ef_search=48 nprobe=none`
- `create=2`
- `insert=3`
- `optimize=7`
- `search=0`
- `total=15`

## 当前状态判断

当前这条 Ralph 主线的核心目标已经达到：
- 量化 ANN 产品化从 `IvfUsq` 扩展到 `HnswHvq`
- storage/runtime 在 active segment 层面向 Arrow snapshot 更进一步
- 主 smoke 全绿
- 已拿到新的真实 `Performance1536D50K` 成功 rerun 结果

## 仍需注意
- 当前工作区仍有历史未提交改动，不适合直接做“只包含本轮改动”的干净提交
- 部分 warning 仍存在，但目前 fresh build/test/smoke 均为绿色
- 当前剩余 warning 主要在上游 `Hanns` 仓库，不是这轮新收敛出的 HannsDB 局部 warning
- 真实 `Performance1536D50K` 已在当前环境重新跑通；上一轮失败是 `/tmp` 空间耗尽导致的运行时环境问题，不是当前代码路径的固定 blocker

### 真实 `Performance1536D50K` rerun（2026-04-13）
结果文件：
- `/Users/ryan/Code/vectorDB/VectorDBBench/vectordb_bench/results/HannsDB/result_20260413_hannsdb-p0-rerun-20260413_hannsdb.json`

关键指标：
- `insert_duration=14.6432`
- `optimize_duration=114.001`
- `load_duration=128.6442`
- `serial_latency_p99=0.0003`
- `serial_latency_p95=0.0003`
- `recall=0.9441`
- `ndcg=0.9506`

备注：
- 首次同日尝试曾因 `/tmp` 空间耗尽失败并写出 0 字节结果占位文件
- 清理陈旧 `/tmp` benchmark DB 后，`hannsdb-p0-rerun-20260413` 成功完成并落盘

### 远端 x86 release optimize proxy（2026-04-13）
命令：
- `N=50000 DIM=1536 METRIC=cosine bash scripts/sync-remote.sh knowhere-bench`

结果：
- `OPT_BENCH_CONFIG n=50000 dim=1536 metric=cosine top_k=10 index_kind=hnsw`
- `OPT_BENCH_TIMING_MS create=0 insert=3327 optimize=18606 search=0 total=21934`

说明：
- 这是远端 x86 的 release optimize proxy，不是完整 VectorDBBench run
- 但它确认了最新代码在远端 x86 上的较大规模路径可执行

### 远端 hk-x86 完整 `Performance1536D50K`（2026-04-13）
结果文件：
- `/data/work/VectorDBBench/vectordb_bench/results/HannsDB/result_20260413_hannsdb-hk-x86-20260413_hannsdb.json`

关键指标：
- `insert_duration=24.1242`
- `optimize_duration=78.5678`
- `load_duration=102.692`
- `serial_latency_p99=0.0005`
- `serial_latency_p95=0.0004`
- `recall=0.9442`
- `ndcg=0.9507`

说明：
- 这是 hk-x86 远端完整标准 benchmark，不是 proxy
- 该结果是在修复远端 repo 路径、补齐 sibling `Hanns`、重建 Linux venv 并安装 `hannsdb` 扩展后得到的

### 远端统一 watchdog 入口验证（2026-04-13）
结果文件：
- `/data/work/VectorDBBench/vectordb_bench/results/HannsDB/result_20260413_hannsdb-remote-watchdog-check_hannsdb.json`

关键指标：
- `insert_duration=22.057`
- `optimize_duration=79.6991`
- `load_duration=101.7561`
- `serial_latency_p99=0.0005`
- `serial_latency_p95=0.0004`
- `recall=0.9442`
- `ndcg=0.9507`

说明：
- 这是 `scripts/sync-remote.sh vdbb-watchdog` 新统一入口的成功验证结果
- 说明远端 bootstrap + 统一 watchdog 路线已经可以完整复现标准 benchmark

### 远端统一 watchdog fast-path 复跑（2026-04-13）
结果文件：
- `/data/work/VectorDBBench/vectordb_bench/results/HannsDB/result_20260413_hannsdb-remote-watchdog-fastpath_hannsdb.json`

关键指标：
- `insert_duration=22.1823`
- `optimize_duration=79.0168`
- `load_duration=101.1991`
- `serial_latency_p99=0.0005`
- `serial_latency_p95=0.0004`
- `recall=0.9442`
- `ndcg=0.9507`

说明：
- 这是同一统一入口在远端环境已准备好的前提下的 fast-path 复跑
- 证明 `sync-remote.sh vdbb-watchdog` 现在既能负责首次 bootstrap，也能复用现成环境稳定重复执行
