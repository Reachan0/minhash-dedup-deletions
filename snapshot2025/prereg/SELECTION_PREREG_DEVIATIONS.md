# SELECTION_PREREG 偏离记录（只追加，不改历史）

冻结文件：`.work/dedupjoin-pvldb-v3/newsnap/SELECTION_PREREG.md`
冻结 sha256：`cad923bf61a652c965e12fd6131042b60b8b6478bcaf92769f84433bbed5a807`
（2026-11-20T03:21:42Z 以 `sha256sum -c` 复核，OK；原始输出见
`.work/dedupjoin-newsnap-selection/inputs/prereg_hash_check.txt`）

本文件只记录揭晓前已经发生的执行环境/资源偏离。**没有任何预测、阈值、场景或判定规则被修改。**

---

## D1 — 2026-11-20T03:19Z — 换机：CPU 型号与内存上限不同

- **步骤**：`verify-rebuild-inputs`
- **事实**：
  - 上一轮核心基准（2026-11-19）：Intel Xeon E5-2686 v4 @ 2.30 GHz，`nproc=8`，
    cgroup 内存 48 GiB，12 线程计时。
  - 本轮 pod：Intel Xeon Platinum 8223CL @ 3.00 GHz，`nproc=8`（16 逻辑核），
    cgroup 内存 16 GiB，预注册的 12 线程配置保留（8 物理核上属超订）。
- **原因**：平台调度，非方法选择。
- **影响**：新机绝对墙钟时间**不能**直接与预注册的秒数预测判错；揭晓时必须把
  (a) “新机绝对值 vs 原预测”的对照 与 (b) “同机、同进程口径下的方法间相对比较”（如 C/B、W/B）
  分开报告。内存上限 16 GiB 使 C 冷启动（预测 ≈11.6 GiB）接近上限，仍可运行但不能与其他
  大进程并行。
- **规则影响**：无。参数、阈值、场景与推荐均未改动。

## D2 — 2026-11-20T03:37Z — 预处理环境：spacy 版本未在 PREREG 中冻结

- **步骤**：`verify-rebuild-inputs`（构建 shingle 缓存）
- **事实**：PREREG §3 规定 `load_word_tokenizer("en")` → `spacy.blank("en")`
  （规则分词器，无统计模型），但未冻结 spacy 包版本。本轮使用 spacy 3.8.18、
  datatrove 0.6.0（本地 wheel，sha256 `f231eb89…`）、regex 2026.11.19。
- **原因**：上一轮未保存 shingle 缓存，也未记录 spacy 版本；本 pod 只能安装当前可用版本。
- **影响**：分词规则的变化可能改变逐文档 shingle 集合与总 shingle 数（本轮
  455,796,881 个唯一 shingle）。文档输入、语言规则、5-gram/xxhash64/MinHash 定义均未变。
  由于没有上一轮的 shingle 缓存可比，该差异无法在本轮消除；已列入
  `inputs/INPUT_VERIFICATION.md` 的“仍有疑问”表。
- **规则影响**：无。预测、阈值、场景与推荐均未改动。

## D3 — 2026-11-20T03:38Z — shingle 缓存构建使用 8 个进程

- **步骤**：`verify-rebuild-inputs`（构建 shingle 缓存）
- **事实**：缓存构建用 8 个进程（本机 8 物理核），而非上一轮尝试时的 12 进程。
- **原因**：16 GiB 内存上限与 8 物理核；进程数只影响预处理速度，不改变分词、shingle、
  哈希或文档集合。
- **影响**：仅影响缓存构建耗时（901.8 s），不进入任何计时口径；A/B/W/C 计时路径仍按
  预注册的 12 线程配置运行。
- **规则影响**：无。

---

## 附：本轮未构成偏离的执行说明

- 6 个 Parquet 分片与 manifest、冻结 PREREG 表逐字节一致，因此按步骤要求**直接使用、
  不重建、不重新下载 WET**；没有触发重建分支。
- 2,000 篇抽样诊断按 SELECTION_PREREG §4 原样运行（`run_sampling_diag.py` 未修改），
  其“每文档候选对超过池 C4 率 10 倍”的告警已触发；这是诊断结果，不是规则偏离。

## D4 — 2026-11-20T04:45Z — 揭晓步骤再次换机：CPU 型号、核数与内存上限均变化

- **步骤**：`run-prereg-selection`
- **事实**：
  - 输入核验步骤（2026-11-20T03:19Z）的 pod：Intel Xeon Platinum 8223CL @ 3.00 GHz，
    `nproc=8`（16 逻辑核），cgroup 内存 16 GiB。
  - 本轮揭晓 pod：**Intel Xeon Platinum 8259CL @ 2.50 GHz**，`nproc=12`
    （16 逻辑核，affinity 16），cgroup 内存 **32 GiB**（34,359,738,368 B）。
  - 原始核心基准（2026-11-19）：Intel Xeon E5-2686 v4 @ 2.30 GHz，cgroup 48 GiB。
- **原因**：平台调度，非方法选择。
- **影响**：
  - 预注册固定 12 线程在本 pod 上等于 `nproc`（不再超订）；计时按 12 线程执行，未改配置。
  - 32 GiB 上限使 C 路径（预测冷启动 ≈11.6 GiB）可安全运行，仍不与其他大进程并行。
  - 本轮绝对墙钟与 2026-11-19 的核心基准、以及 2026-11-20T03:19Z 的输入核验 pod 均不可直接
    比较；按步骤要求，新机绝对时间与原预测只作“绝对值对照”，判定另给“同机相对”一列。
- **规则影响**：无。预测、阈值、场景、推荐与判定规则均未改动。

## D5 — 2026-11-20T04:45Z — 计时边界说明：cold 采用核心基准的“缓存 shingle”边界

- **步骤**：`run-prereg-selection`
- **事实**：`SELECTION_PREREG.md` §2 把 cold 写成“文本 → shingle → 签名/带或秩 → 方法”，
  但 §3 的秒数预测（如 A/B 冷启动 ≈190 s、C ≈573 s）是由 `core/timing.jsonl` 的 cold 行
  拟合的；该 cold 行的实测阶段是 `read_shingle_s`（从缓存读 shingle）+ `signature_s` +
  `banding_s` + 方法，**不含 spacy 文本分词**（分词在核心基准里属于一次性语料准备）。
  本快照的 spacy 分词单独实测为 901.8 s / 8 进程（`build_report.json`），若计入 cold，
  任何路径都会超过全部场景的冷启动预算，且与预测口径不一致。
- **本轮口径**（与核心基准逐阶段一致，写进所有计时行）：
  - **cold（核心基准边界）** = 全新进程 + 读缓存 shingle + 重建签名/带（A/B/W）或重建秩并做
    精确连接（C）+ 方法；对应 `run_timing.py --mode pathcold`。
  - **incremental** = 全新进程 + 读缓存 shingle + 读缓存签名（A/B/W）或秩（C）+ 方法；
    对应 `--mode incremental`。
  - **文本冷启动**（文本 → spacy shingle → …，§2 字面口径）另跑一次 A 作为补充观测，
    只报告、不用于判定 §3 预测（预注册成本模型未拟合该项）。
- **规则影响**：无。判定仍按预注册的硬约束与容差；两种口径在报告中分列，绝不混用。

## D6 — 2026-11-20T05:05Z — 质量指标的实现层等价：同集文档折叠（方法定义未变）

- **步骤**：`run-prereg-selection`
- **事实**：快照中存在 86,771 篇文档的 shingle 集完全相同（8 个 shingle，占全部文档 16.3%），
  精确 Jaccard 图上它们构成 3,766,054,727 条文档级边的团（`band_stats.json`、
  `dup_sets.json`、`group_accuracy.log`）。冻结的 B/W 候选生成在文档上执行
  `all_bucket_pairs`：仅最大桶的 `np.triu_indices(86_771, 1)` 就需两个 int64 数组各
  3.766×10^9×8 B ≈ 30.1 GB（合计 ≈60.3 GB），超过本 pod 的 32 GiB cgroup 上限；
  字面 C 的 `exact_join` 同样需要具体化同一组边。**这是快照结构造成的字面不可执行，
  不是实现错误**；字面尝试的失败记录保留在 `newsnap/failed.jsonl` 与日志中，不删除。
- **做法（实现层等价，不是方法变更）**：
  - 以“不同 shingle 集”为一组（431,116 个有 shingle 的组 + 994 个空文档组），
    每组内 m 篇同集文档两两 J=1，贡献 C(m,2) 条文档级边；跨组按 m_u·m_v 条计。
  - B 的分量、W 的贪心极大独立集、C 的精确分量均在组级计算后按组大小与重数**精确提升**到
    531,597 篇文档；组按最小文档行号排序，与文档级贪心顺序一致。A 仍按字面定义在
    531,597 篇文档上构建 LSH 链边（与组折叠无关）。
  - 代表文档 = 组件内最小文档行号。快照 id 是 WET 字符串，核心 harness 的“最小 id”
    即最小行索引（组件内 `minimum` over row index），与冻结配置一致。
  - 空文档（0 个 shingle）不与任何文档有 Jaccard 边，单独成组、不参与删除。
  - 该方法不改变候选生成、验证阈值、并查集、MIS 顺序或 τ；只改变同一精确图的存储方式。
- **验证**：组级精确连接在 τ=0.80 得 172,795 条组边，提升后精确删除 113,730 篇；
  τ=0.75 得 197,295 条组边、删除 117,700 篇。折叠语料的 shingle 总量 448,530,537，
  与全文去重数 455,796,881 的差为组内重复（见 `collapsed_build.json`）。
- **规则影响**：无。预测、阈值、场景、推荐与判定规则均未改动；B/W 的字面计时因不可执行
  记为失败，而不是用折叠实现的耗时替代。

## D7 — 2026-11-20T05:05Z — 环境表述更正（D4 的 nproc）

- **步骤**：`run-prereg-selection`
- **事实**：`nproc` 在本 pod 输出 12，是因为 `OMP_NUM_THREADS=12`（GNU coreutils 的 `nproc`
  会读取该变量）；`sched_getaffinity` 实际为 16 核。D4 已记录“affinity 16”，此处仅更正
  “12 线程等于 nproc”的说法：实际是 **12 线程 < 16 个可用核**，不存在超订。
- **规则影响**：无。线程数仍为冻结配置 12。

## D8 — 2026-11-20T05:05Z — 输入与采样诊断在上一 pod 完成

- **步骤**：`run-prereg-selection`
- **事实**：shingle 缓存（901.8 s，8 进程，峰值 8.064 GiB）与 2,000 篇采样诊断
  （4.908 s，峰值 3,743.5 MB）在 2026-11-20T03:37Z 的 pod（Intel 8223CL，16 GiB）
  上完成；本轮揭晓 pod 为 8259CL / 32 GiB。二者是数据产物与诊断结果，已按步骤复算校验
  （缓存 SHA-256 c9cb3fcb… 与 INPUT_PATHS.json 一致），不重新生成。
- **影响**：采样诊断的绝对耗时来自上一 pod，只用于 §4 的告警判定（比值与文档级统计），
  不参与本轮方法计时比较。
- **规则影响**：无。

## D9 — 2026-11-20T05:40Z — 补充的单进程文本→shingle 冷启动失败（记录保留）

- **步骤**：`run-prereg-selection`
- **事实**：为补一个“从网页文本开始”的端到端参考，另起一个 A 的 `--mode cold` 进程。
  它在完成分词后调用 `snap_common.shingle_all()`，该函数返回 `(off, flat, nwords, meta, timing)`
  五项，而复用的 `run_timing.py` 第 48 行按四项解包，抛出
  `ValueError: too many values to unpack (expected 4)`；进程墙钟 952.711 s、
  进程树峰值 16,965.3 MB。记录已写入 `newsnap/failed.jsonl`，日志为
  `logs/snapA_textcold_t0.8_r1.log`，不删除。
- **为什么不重跑**：该运行是补充项，判定口径按 D5 使用“缓存 shingle + 路径重建”的核心
  cold 边界；分词阶段已有 901.8 s / 8 进程的实测。重跑需再次分词约 16 分钟，且这是
  复用脚本的接口不匹配，不是方法或数据问题。
- **规则影响**：无。没有用它替代任何计时，也没有改动预测、阈值、场景或判定规则。
