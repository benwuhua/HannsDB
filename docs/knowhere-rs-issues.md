# knowhere-rs Issues (2026-03-21)

## P0: Cosine Path Missing TLS Scratch Reuse

**问题描述**  
在 `src/faiss/hnsw.rs` 的 `search_single`（约 `5317+`）中，cosine 无 filter 常见路径仍走 generic 分支，并在路径中创建新的 `SearchScratch`（历史位置约 `5349`）。相比之下，L2 已有 `search_single_l2_unfiltered_with_scratch` + TLS 包装（`search_single_l2_unfiltered`，约 `5486+ / 5530+`）来复用 scratch。

**预期行为**  
cosine 无 filter 路径应与 L2 一样复用 thread-local scratch，避免每次 query 的 scratch/heap 初始化与分配成本。

**影响（1536-dim cosine 估计）**  
属于查询热路径固定开销，预计对 p99 有中等偏高影响（量级约毫秒到十毫秒级，取决于 ef 与候选扩展规模）。

**建议修法**  
新增 `search_single_cosine_unfiltered_with_scratch`（或等价函数）+ TLS 包装函数，并在 `search_single` 的 cosine + no-filter 分支直接走该路径。

---

## P1: Per-Query Result Buffer Allocation in `HnswIndex::search`

**问题描述**  
`HnswIndex::search`（`src/faiss/hnsw.rs`，约 `3710+`）每次都分配 `all_ids` / `all_dists`（约 `3731-3732`），随后再做一轮距离后处理循环（约 `3748+`）。在 `n_queries=1` 的主流场景，这条路径存在可避免的额外分配与拷贝。

**预期行为**  
单 query 应有 fast path：直接写 top-k 结果，尽量避免统一批量容器与二次循环。

**影响（1536-dim cosine 估计）**  
固定成本累积明显，尤其在高 QPS/短查询批次下影响 p95/p99 尾延迟。

**建议修法**  
为 `n_queries == 1` 增加专用输出路径；统一路径保留批量场景，单 query 跳过不必要的 `Vec` 分配与额外 copy/normalization。

---

## P2: Unfiltered Path Carries Filter Clone/Closure Overhead

**问题描述**  
`HnswIndex::search` 先执行 `let filter = req.filter.clone();`（约 `3729`），再把它传入 `search_single`（约 `3738`）。在上层没有 filter 的场景，仍沿用 filter-capable generic 控制流，带来额外分支/闭包判断开销。

**预期行为**  
无 filter 时应走显式的 `None` 快速路径，避免无意义的 filter 复制和判定逻辑。

**影响（1536-dim cosine 估计）**  
单次影响较小，但在尾延迟与高频调用下会累积；与 P0/P1 叠加后可见收益。

**建议修法**  
在 `search` 和 `search_single` 增加 no-filter 专用分支，避免 `filter.clone()` 与通用 filter 闭包路径进入热循环。
