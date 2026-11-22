# 正确性账本脚本

本目录是 `correctness-ledger-bounded` 步骤的可重跑脚本。它们在任务工作区内运行，
读取 `.work/dedupjoin/` 下冻结的 shingle 缓存、签名缓存与逐种子删除掩码，
把结果写成 `data/correctness/` 下的逐运行记录，再由 `build_report.py` 生成
`analysis/correctness/` 的表格与中文说明。所有数字均来自这些记录文件。

## 运行顺序

```bash
C=.work/dedupjoin-pvldb-v3/core/correctness
python3 $C/run_ledger.py --corpus c4
python3 $C/run_ledger.py --corpus fw
python3 $C/run_ledger.py --corpus mix
python3 $C/run_ledger.py --corpus wet
python3 $C/run_flips.py
python3 $C/run_variance.py
python3 $C/run_components.py
python3 $C/make_component_figure.py
python3 $C/build_report.py
```

`run_ledger.py` 每算完一个 (语料, τ, 种子) 单元就追加并 flush 一行，
重跑时按行键跳过已完成的单元，因此中断后可直接续跑。
`--out` 可指定另一个输出文件；`--taus`、`--seeds` 可缩小范围。

## 内存约束

- 单个大 npz（2.0 GB 与 4.1 GB 的 shingle 包）从不整体 `np.load`：
  `correctness_lib.extract_flat` 顺序抽取 `flat.npy` 成员到本地缓存，
  再用 `np.lib.format.open_memmap` 只读映射，语料处理完立即删除。
- 签名用 `.npy` 的 memmap 打开；每个 (语料, 种子) 单元算完就释放边与分量，
  不跨种子累积。
- 逐单元峰值 RSS 写入每行的 `peak_rss_mb`；本次 32 次运行最高约 2.6 GB。

## 依赖

- `code/core/vendor/lsh.py`（datatrove 0.6.0 哈希族与 band 键）随 `code/core/` 一起发布；
  `correctness_lib._vendor_dir` 同时识别工作区与发布目录布局。
- 需要 `numpy`、`numba`、`pyarrow`、`matplotlib`；中文字体用镜像自带的 Noto Sans CJK SC。
- `correctness_lib._find_root` 向上查找含 `.work/dedupjoin` 的工作区根目录，
  因此脚本在发布目录中可直接运行的前提是任务工作区仍存在。

## 文件对应

| 脚本 | 记录文件 | 说明 |
|---|---|---|
| `run_ledger.py` | `data/correctness/ledger.jsonl` | A/B/W/C 三层核对、W 顺序实验、分量断言 |
| `run_flips.py` | `data/correctness/flips.jsonl` | 种子翻转与期望删除集对称差 |
| `run_variance.py` | `data/correctness/variance_decomposition*.json*` | 召回方差的分量分解 |
| `run_components.py` | `data/correctness/component_cases.json` | Draft 2 反例的两类真实分量 |
| `make_component_figure.py` | `analysis/correctness/figures/component_classes.*` | 两类分量示意图 |
| `build_report.py` | `analysis/correctness/*.csv`、`*.md` | 汇总表与中文说明 |
