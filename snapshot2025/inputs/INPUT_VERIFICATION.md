# 输入核验记录：CC-MAIN-2025-30 独立快照

- **任务**：DedupJoin v3 预注册选型验证（步骤 `verify-rebuild-inputs`）
- **核验时间（UTC）**：2026-11-20T03:19:19Z – 2026-11-20T03:43Z
- **核验对象**：上一轮冻结的 6 个 Parquet 分片、`corpus/manifest.json`、`PREREG.md`、`SELECTION_PREREG.md`、104 段 WET 的下载与抽取记录
- **本步范围**：只核验输入、构建 shingle 缓存、运行 SELECTION_PREREG §4 的 2,000 篇抽样诊断；**未运行 A/B/W/C**，未改动任何预测、阈值、场景或判定规则。

## 1. 结论摘要

1. **输入可用，无需重建。** 6 个 Parquet 文件均存在，字节数、sha256、行数与 `manifest.json` 逐项一致；行数合计 531,597（100,000×5 + 31,597）。六个文件的 sha256 同时与冻结于 2026-11-19 的 `PREREG.md` 分片表一致（该文件自身 sha256 已由 `PREREG.sha256` 校验通过），因此这是冻结锚点而非事后自证。
2. **内容级核验全部通过。** 531,597 个文档 ID 无重复；文本无空值/空串；`nchar` 与文本长度逐篇一致；全部文档满足 `lang='en'` 且 `lang_score ≥ 0.8`；总词数 522,207,916、均值 982.3379665、中位数/分位数/极值均与 manifest 完全一致；104 个 segment 的逐段英文文档数与 manifest、`segment_stats.jsonl` 全部一致，且 segment 首次出现顺序与 rank 0–103 顺序一致。
3. **段抽样可复现。** `wet.paths.gz` 的 sha256 与 `PREREG.md` 记录一致；用 `random.Random(20261119).shuffle` 重算 100,000 条路径的顺序与 `sampling_plan.json` 完全一致；manifest 的 104 段正是该顺序的前 104 位。
4. **shingle 缓存已按冻结定义建成。** datatrove 0.6.0 `simplify_text` + `spacy.blank("en")` 分词、5-gram、xxhash64、逐文档去重集合；531,597 篇共 455,796,881 个唯一 shingle（去重前 532,196,386 个 5-gram 位置，篇内重复占 14.36%），994 篇不足 5 个 token（空 shingle 集合）。构建耗时 901.8 s（8 进程），父进程峰值 RSS 8.064 GiB。
5. **抽样诊断触发预注册的告警。** 2,000 篇等距抽样产生 58,683 个候选对，**29.34 对/文档**，远高于池 C4 率 0.19 的 10 倍阈值（1.9），`flag_bucket_expansion_exceeds_10x = true`；最大桶 343 篇；候选对中 58,666 对在 τ=0.80 下精确验证成立。该结果只用于提示 B 的桶扩展可能超出预测，**不改变 §3 的冻结规则**。
6. **机器已更换。** 上一轮核心基准在 Intel Xeon E5-2686 v4 @ 2.30 GHz、cgroup 48 GiB 上运行 12 线程；本机为 Intel Xeon Platinum 8223CL @ 3.00 GHz、nproc 8（16 逻辑核）、cgroup 16 GiB。后续揭晓必须把“新机绝对计时与旧预测的对照”和“同机方法间相对比较”分开报告。

## 2. 机器与环境（与上一轮对比）

| 项目 | 上一轮（核心基准，2026-11-19T05:45Z 记录） | 本轮（2026-11-20T03:19Z） | 是否变化 |
|---|---|---|---|
| CPU 型号 | Intel Xeon E5-2686 v4 @ 2.30 GHz | Intel Xeon Platinum 8223CL @ 3.00 GHz | **已换机** |
| `nproc` / 逻辑核 | 8 / 16 | 8 / 16 | 否 |
| cgroup 内存上限 | 48 GiB（51,539,607,552 B） | 16 GiB（17,179,869,184 B） | **是** |
| OS | 未记录 | Debian GNU/Linux 12 (bookworm)，kernel 6.18.51 | 未知 |
| Python | 3.11.2 | 3.11.2 | 否 |
| numpy / pyarrow / pandas | 2.4.6 / 26.0.1 / 3.1.1 | 2.4.6 / 26.0.1 / 3.1.1 | 否 |
| xxhash / scipy / scikit-learn | 4.0.2 / 1.17.1 / 1.9.1 | 4.0.2 / 1.17.1 / 1.9.1 | 否 |
| numba / datatrove | 0.68.1 / 0.6.0 | 0.68.1 / 0.6.0 | 否 |
| spacy | 未记录（上一轮未冻结） | 3.8.18（本轮安装） | **未冻结项** |
| 计时线程配置 | 12 | 12（预注册固定值；本机 8 物理核，属超订） | 否（配置保留） |

- 环境记录：`env/machine_and_env.txt`（机器 + 安装前 `pip freeze`）、`env/pip_freeze_post_install.txt`（安装后完整 169 个包）、`env/machine_current.json`、`env/machine_previous.json`、`env/pip_install.log`。
- datatrove 0.6.0 从上一轮保留的本地 wheel 安装（`datatrove-0.6.0-py3-none-any.whl`，sha256 `f231eb894fd79189a4c96b4d2188ece1a3804a922dc49bdf3ddcecd274f68ab9`）；spacy 3.8.18、regex 2026.11.19、numba 0.68.1 从 PyPI 安装（日志见 `env/pip_install.log`）。
- **记录不一致（旧记录）**：`output/dedupjoin-pvldb-v3/experiment/analysis/reproduction/ACCEPTANCE.md` 写“4 vCPU”，而同目录 `environment.txt` 追加的核心基准记录写 `nproc=8`、16 逻辑核。本记录采用 `environment.txt` 的基准条目，并把该不一致列为“仍有疑问”项（不影响输入身份）。

## 3. 逐分片文件级核验

机器可读表：`shard_hashes.csv`；原始测量：`shard_sha256_measured.txt`。

| 文件 | 行数（实测 = manifest） | 字节（实测 = manifest） | sha256（实测 = manifest = PREREG 表） |
|---|---:|---:|---|
| shard-00000.parquet | 100,000 | 238,383,754 | `615d46e156e749e89fbcfcc833d243eeb682e729d5e8cb40145549731d0a470c` |
| shard-00001.parquet | 100,000 | 234,577,118 | `25690a668b5898f7d67ec4457d81d5514ce504adb62b4b2ee6f4eb4906f6e3d6` |
| shard-00002.parquet | 100,000 | 230,052,485 | `005703987d1d2080b530d70d8eaf1cae3fff8333c9b5bc831af9edd63759d511` |
| shard-00003.parquet | 100,000 | 236,561,775 | `0d84f7cda02a69c33247b2e439e15eeee214487c4a6304ddf80c5df251bc8b66` |
| shard-00004.parquet | 100,000 | 239,822,036 | `ae015a2aa9894c21ac874900f5436230afa57ecb6d921c3b7ad067677f1e23be` |
| shard-00005.parquet | 31,597 | 72,886,812 | `cd86ac62afa2bb0618703c6768804440c08c78d636a8783d5c87e1d01c35f140` |
| **合计** | **531,597** | **1,252,283,980** | 6/6 一致 |

- Parquet 元数据：每个分片 1 个 row group，`created_by = parquet-cpp-arrow version 26.0.1`；schema 为 `id, segment, record_index, url, fetched_at, text, nchar, nwords, lang, lang_score`。
- 总字节数与 `PREREG.md` 的 “Corpus size … 1,252,283,980 B” 一致；总行数与 “531,597 English-detected documents” 一致。

## 4. 逐文档内容级核验

| 指标 | 结果 | 判定 |
|---|---|---|
| 文档数 | 531,597 | 等于 manifest |
| 唯一 ID / 重复 ID | 531,597 / 0 | 通过 |
| 空文本 / 空文本串 | 0 / 0 | 通过 |
| `nchar` 与 `len(text)` 不一致 | 0 | 通过 |
| `lang='en'` 且 `lang_score ≥ 0.8` | 531,597 / 531,597；最小分数 0.80000019 | 通过 |
| 总词数 | 522,207,916 | 等于 manifest |
| 词数均值 | 982.3379665423244 | 等于 manifest |
| 词数中位数 / p10 / p90 / min / max | 586 / 12 / 2035.4 / 1 / 399,477 | 全部等于 manifest |
| 逐段英文文档数 | 104 段、0 处不符、无缺失/多余段 | 通过 |
| segment 首次出现顺序 | 等于 rank 0–103 | 通过 |

**文档级身份哈希**（供下一步与复现使用）：

- ID 序列 sha256：`5ee1b3cef926e31c125b22c17f367f888dad180d3bdbeb26b70c4530ddff52fd`
  （定义：按分片顺序逐行，`LE32(len(utf8(id))) || utf8(id)` 拼接后取 sha256）
- 内容 sha256：`ad4e1646fb09b5e4b9b1a625b84ccf6510d955a5e88a2e56c05873efc3860c1f`
  （定义：每个分片内按行拼接 `LE64(xxh64_intdigest(utf8(text)))` 取 sha256，再按分片顺序拼接取 sha256；逐文档值保存在 `doc_text_xxh64.npy`，共 531,597 个 uint64）
- 逐分片内容 sha256：

| 文件 | 内容 sha256 |
|---|---|
| shard-00000.parquet | `4c396b6cb1c6ddd33e661de6c117972c897564cda0f23298896a13b6bca92776` |
| shard-00001.parquet | `68a619c4a166a792cbc8e901ff17bf11788cdd90b5bfa5c4baa2d45f85bf8fea` |
| shard-00002.parquet | `6194de168096a0c5da59192c480ebd70b8e905964d668f49c04a7be28e0aa3cb` |
| shard-00003.parquet | `6365e768d047a5927af37988404761947286761c54e8df4332bfdd8ba7845e3c` |
| shard-00004.parquet | `656602191e9150395cb4078cdd3394435742b6c16ccd6aa90d93c98fb24badc0` |
| shard-00005.parquet | `8cde1abd532088d191a91b1334f6318a2540ae97770ca688cc08068d16e934e0` |

**输入身份声明**：本次 6 个文件与 manifest、PREREG 表逐字节一致，因此**文件 sha256 即输入身份**；上面的 ID/内容哈希是第二重、顺序敏感的身份，若将来 Parquet 被重写但文档内容不变，可据此证明内容等价。

## 5. WET 段下载与抽取核验

- `wet.paths.gz` sha256：`b00996a663900fbb38df54c3dd29be792bdc58cb488401ed00bc59056cf53338`，与 `PREREG.md` 一致；文件内 100,000 条路径。
- 重算 `random.Random(20261119).shuffle` 的 100,000 条顺序与 `sampling_plan.json` **完全一致**；manifest 的 104 段等于该顺序前 104 位，rank 连续 0–103（`sampling_plan_check.json`）。
- 104 段 WET 记录总数 2,456,751；检出英文文档 531,597（21.6%）；压缩 WET 总字节 7,418,462,506 B。
- `segment_stats.jsonl` 共 104 行，逐段 `english_docs` 求和 531,597，与 manifest 和分片实测一致；104 段均记录了 `source_sha256`，缺失 0。
- 原始下载/抽取日志 `wet_logs/fetch_extract.log`：13 个批次全部成功（每批 8 段），**无失败事件**；`wet_logs/build.log` 是上一轮 shingle 构建到 250,000 篇时的中断记录（与本轮无关）。
- 逐段清单：`segment_wet_checks.csv`（rank、segment、source_url、source_bytes、source_sha256、records、english_docs、状态）。

## 6. shingle 缓存（冻结定义）

| 项目 | 值 |
|---|---|
| 分词/规范化 | datatrove 0.6.0 `simplify_text`（默认 `TextNormConfig`）→ `load_word_tokenizer("en")`（`spacy.blank("en")` + sentencizer） |
| shingle | 连续 5-gram，xxhash64，逐文档去重为集合 |
| 文档数 | 531,597 |
| 5-gram 位置数（篇内去重前） | 532,196,386 |
| 唯一 shingle 数（篇内去重后） | 455,796,881 |
| 篇内重复占比 | 14.36% |
| 空 shingle 文档（<5 token） | 994 |
| 构建耗时 / 进程数 | 901.8 s / 8 进程 |
| 父进程峰值 RSS | 8.064 GiB |
| 缓存文件 | `snapshot_spacy.npz`，3,654,881,460 B，sha256 `c9cb3fcb64f0304db44bbb2243b91d49bdfe38d79a8564c610a015b94ee9fbdd` |
| 元数据 | `snapshot_meta.parquet`，sha256 `148b914fd6d7366312abd300206d8a5c6228d4366f4874495299970942b2b182` |

**与预注册 S ≈ 519.9 million 的关系**：519.9M 是 5-gram **位置**的估计（522.2M 词 − 4×531,597 ≈ 520.1M，接近该值）；缓存按 PREREG §3 存**逐文档集合**，篇内重复的 5-gram 被去掉，故集合大小为 455.8M。分词器把 “cannot” 等拆开也使 token 位置数（532.2M）略高于项目词数。**预测、阈值和规则不变**；揭晓时把该输入属性与预测偏差一起报告。

## 7. SELECTION_PREREG §4 抽样诊断

脚本：`code/newsnap/run_sampling_diag.py`（未修改）；结果：`sampling_diag.json` / `sampling_diag.jsonl`；日志：`logs/sampling_diag.log`。

| 指标 | 实测 |
|---|---|
| 抽样方式 | `floor(k·n/2000)`，k=0…1999，n=531,597，确定性、不涉及相似度 |
| 样本文档 / 有效 / 空 shingle | 2,000 / 1,992 / 8 |
| 样本 shingle 数 | 1,694,857 |
| 候选对 / 每文档 | 58,683 / **29.3415** |
| 有候选的文档 | 372 |
| 最大桶 | 343 |
| 桶内对总数 | 821,291 |
| τ=0.80 精确验证成立的候选对 | 58,666 |
| 池 C4 率 / 10 倍阈值 | 0.19 / 1.9 |
| **`flag_bucket_expansion_exceeds_10x`** | **true** |
| 墙钟 / 峰值 RSS | 4.908 s / 3,743.5 MB（= 3.66 GiB） |
| 线程 | 12（预注册固定配置） |

解释与限制：

- 29.34 对/文档 是池 C4 率 0.19 的约 154 倍；几乎全部候选对（58,666/58,683）在 τ=0.80 下通过精确验证。最大桶 343 篇单独贡献 C(343,2)=58,653 对，说明样本中存在一个 343 篇的模板/镜像簇。按预注册 §4，揭晓报告必须标注“B 的桶扩展可能超出预测，建议部署前做小规模精确审计”。**该诊断不改变 §3 的规则与推荐。**
- 预注册估计抽样“至多约 2 s、约 0.1 GiB”；实测 4.908 s、3.66 GiB。原因是脚本从完整 shingle 缓存切片：npz 解压加载与 3.65 GB 数组访问本身约 2.4 s，峰值 RSS 由缓存文件决定而非 2,000 篇样本。只有样本部分的计算时间约 2.5 s（签名 1.57 s、分带 0.25 s、候选 0.12 s、验证 0.57 s）。该差异记入“仍有疑问/偏差说明”，不改规则。

## 8. 已验证 / 仍有疑问

| # | 项目 | 状态 | 证据 / 说明 |
|---|---|---|---|
| 1 | 6 个 Parquet 存在且字节数一致 | **已验证** | `shard_hashes.csv` |
| 2 | 6 个 Parquet sha256 与 manifest 一致 | **已验证** | 同上 |
| 3 | 6 个 Parquet sha256 与冻结 PREREG 表一致 | **已验证** | `input_verification.json → prereg_cross_check`；PREREG.sha256 校验 OK |
| 4 | 总行数 531,597 | **已验证** | 100,000×5 + 31,597 |
| 5 | 逐段英文文档数 = manifest = segment_stats | **已验证** | 104 段，0 不符，无缺失/多余 |
| 6 | 文档 ID 唯一、顺序与 rank 顺序 | **已验证** | 重复 0；`id_sequence_sha256`；首次出现顺序=rank 0–103 |
| 7 | 文本逐篇内容哈希 | **已验证（本轮建立基线）** | `content_sha256`、`doc_text_xxh64.npy`；无旧文件可直接对比，但文件 sha256 已与冻结锚点一致 |
| 8 | 语言规则 en / ≥0.8 | **已验证** | 531,597/531,597；最小 0.80000019 |
| 9 | 总词数与长度统计 | **已验证** | 522,207,916；均值/中位数/分位数/极值全等 |
| 10 | 段抽样种子与顺序 | **已验证** | `sampling_plan_check.json`；100,000 条乱序可重算 |
| 11 | Parquet 元数据/压缩/行组差异 | **已验证（无差异）** | 每片 1 row group，`parquet-cpp-arrow 26.0.1`；文件哈希一致，无需解释字节差异 |
| 12 | WET 原始文件可重新校验 | **仍有疑问（不可重验）** | 抽取后原文件已删除；`source_sha256` 是下载时实测值，未与上游独立校验值比对。输入身份由 Parquet 冻结哈希保证，不受影响 |
| 13 | 分词器版本 | **仍有疑问（未冻结）** | PREREG 只规定 `spacy.blank("en")`，未冻结 spacy 版本；本轮 3.8.18。无上一轮 shingle 缓存可对比；影响的是 shingle 计数而非文档输入 |
| 14 | S ≈ 519.9M 与实测 455.8M | **已验证差异并解释** | 位置数 vs 逐文档集合数；预测不改，揭晓时报告 |
| 15 | 抽样耗时/内存超出预注册估计 | **已验证差异并解释** | 4.908 s / 3.66 GiB vs ~2 s / ~0.1 GiB；因加载完整缓存，诊断规则不变 |
| 16 | 旧机器记录 4 vCPU vs nproc=8 | **仍有疑问（记录不一致）** | 采用 `environment.txt` 基准条目；不影响本轮输入核验，但影响旧计时口径的解读 |

## 9. 缺项与未做事项

- 本轮**未重新下载 104 段 WET**：6 个分片已与冻结哈希一致，按步骤要求“一致则直接使用，不重建”。因此没有新的下载日志；保留上一轮的下载/抽取日志作为记录（无失败）。
- 本轮**未运行 A/B/W/C**：按步骤目标，四条路径与两个阈值的运行留给 `run-prereg-selection`。
- 原始 WET 文件与逐段 Parquet 中间件已删除，无法在本 pod 重算 WET/逐段 Parquet 的 sha256；已在上表第 12 项注明。

## 10. 复现命令

```bash
# 1) 前置：若环境缺 datatrove / spacy / numba
pip install .work/dedupjoin-pvldb-v3/code/audit/_wheel/datatrove-0.6.0-py3-none-any.whl spacy regex numba==0.68.1

# 2) 分片、逐文档内容、逐段计数核验（约 17 s）
python3 .work/dedupjoin-newsnap-selection/inputs/verify_inputs.py

# 3) 段抽样种子与顺序核验
python3 .work/dedupjoin-newsnap-selection/inputs/verify_sampling_plan.py

# 4) 构建 shingle 缓存（约 15 min，写 3.65 GB）
SNAP_WORKERS=8 python3 .work/dedupjoin-newsnap-selection/inputs/build_cache.py

# 5) 2,000 篇抽样诊断
python3 .work/dedupjoin-newsnap-selection/code/newsnap/run_sampling_diag.py

# 6) 汇总核验产物
python3 .work/dedupjoin-newsnap-selection/inputs/summarize_inputs.py
```

预期校验值：6 个分片 sha256 见第 3 节；ID 序列 `5ee1b3ce…`；内容 `ad4e1646…`；总行数 531,597；总词数 522,207,916；104 段全匹配；抽样 58,683 对 / 29.3415 对每文档 / 最大桶 343 / 告警 true。

## 11. 机器可读产物索引

| 文件 | 内容 |
|---|---|
| `input_verification.json` | 全部核验结果（分片、内容、逐段、PREREG 交叉核对、抽样、机器对比） |
| `shard_hashes.csv` | 逐分片行数/字节/sha256（实测与 manifest） |
| `segment_counts_measured.csv` | 逐段实测英文文档数 vs manifest vs segment_stats |
| `segment_wet_checks.csv` | 逐段 WET 来源、校验值、状态 |
| `sampling_plan_check.json` | 100,000 条路径的种子乱序复算结果 |
| `sampling_diag.json` / `.jsonl` | 2,000 篇抽样诊断原始结果 |
| `doc_text_xxh64.npy` | 531,597 个逐文档文本 xxh64（uint64，分片顺序） |
| `INPUT_PATHS.json` | 下一步使用的已验证输入路径与哈希 |
| `prereg_hash_check.txt` | `sha256sum -c` 原始输出 |
| `wet_logs/` | 上一轮 WET 下载/抽取日志、segment_stats、构建中断日志 |
| `verify_inputs.py` / `verify_sampling_plan.py` / `build_cache.py` / `summarize_inputs.py` | 本步全部核验脚本 |
