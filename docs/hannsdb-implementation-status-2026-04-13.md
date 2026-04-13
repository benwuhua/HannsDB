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
