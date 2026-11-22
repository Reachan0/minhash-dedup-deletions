# 运行汇总（analyze_selection.py 自动生成）
生成时间（UTC）：2026-11-20T05:40:46Z
全部数值由 `newsnap/*.jsonl` 重新计算，本文件不代表任何手工测量。

## 输入行数
- 质量行（runs.jsonl）：32
- 计时行（timing.jsonl，仅成功进程）：16
- 进程树 RSS 采样行（timing_sidecar.jsonl）：29
- 失败记录（failed.jsonl）：7

## 已完成的运行
- A τ=0.8：pathcold×3（中位 73.84s，峰值 RSS 4198 MB）；incremental×5（中位 4.34s，峰值 RSS 4197 MB）
- A τ=0.75：pathcold×3（中位 73.97s，峰值 RSS 4198 MB）；incremental×3（中位 4.31s，峰值 RSS 4197 MB）

## 缺失/失败的运行
- literal_B_pathcold：退出码 1，墙钟 75.763s，进程树峰值 10957.7 MB；末行日志：numpy._core._exceptions._ArrayMemoryError: Unable to allocate 28.0 GiB for an array with shape (3764559835,) and data type int64
- literal_W_pathcold：退出码 1，墙钟 73.748s，进程树峰值 10958.3 MB；末行日志：numpy._core._exceptions._ArrayMemoryError: Unable to allocate 28.0 GiB for an array with shape (3764559835,) and data type int64
- literal_B_incremental：退出码 1，墙钟 8.022s，进程树峰值 10957.1 MB；末行日志：numpy._core._exceptions._ArrayMemoryError: Unable to allocate 28.0 GiB for an array with shape (3764559835,) and data type int64
- literal_W_incremental：退出码 1，墙钟 8.024s，进程树峰值 10956.9 MB；末行日志：numpy._core._exceptions._ArrayMemoryError: Unable to allocate 28.0 GiB for an array with shape (3764559835,) and data type int64
- literal_C_incremental：退出码 1，墙钟 38.128s，进程树峰值 22186.7 MB；末行日志：MemoryError: Allocation failed (probably too large).
- literal_C_pathcold：退出码 1，墙钟 415.128s，进程树峰值 21447.5 MB；末行日志：MemoryError: Allocation failed (probably too large).
- snapA_textcold_t0.8_r1：退出码 1，墙钟 952.711s，进程树峰值 16965.3 MB；末行日志：ValueError: too many values to unpack (expected 4)

## 场景判定（阈值取自 SELECTION_PREREG.md §3）
### S1 — 每次删除都要可审计（直达见证是硬约束）
- 推荐路径：W；是否满足全部硬约束与质量容差：否
- 可执行路径：A
- 满足全部约束的路径：无
- 实测最便宜合格路径：无
### S2 — 内存受限服务（峰值 RSS ≤ 6 GiB，增量摄入）
- 推荐路径：B；是否满足全部硬约束与质量容差：否
- 可执行路径：A
- 满足全部约束的路径：无
- 实测最便宜合格路径：无
### S3 — 签名缓存已存在（预处理已付费）
- 推荐路径：B；是否满足全部硬约束与质量容差：否
- 可执行路径：A
- 满足全部约束的路径：无
- 实测最便宜合格路径：无

## 采样诊断（§4）
- 样本 2000 文档，候选对 58683，每文档 29.34 对；告警阈值 1.90（10×C4 池率 0.19）；是否触发：是
- 采样成本：4.908s，峰值 RSS 3743.5 MB
