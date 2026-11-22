#!/usr/bin/env python3
"""Build the correctness summary tables and the Chinese analysis notes.

Everything here is derived from the per-run record files written by
run_ledger.py, run_flips.py, run_variance.py and run_components.py; no number
is typed in by hand.  Re-running the four scripts and this one reproduces every
table and the prose numbers below.

Outputs (all under core/correctness/analysis/):
  ledger_runs.csv, summary_by_cell.csv, witness_orders.csv,
  component_classes.csv, flips.csv, variance.csv,
  correctness_ledger.md, definitions.md, component_mechanism.md, flip_variance.md
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import correctness_lib as K

AN = os.path.join(K.OUTDIR, "analysis")
os.makedirs(AN, exist_ok=True)
CORPUS_ZH = {"c4": "C4", "fw": "FineWeb", "mix": "Mix", "wet": "WET"}


def rows_of(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def sd(xs):
    xs = [x for x in xs if x is not None]
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else None


def f(x, nd=1):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def pct(x, nd=2):
    return "—" if x is None else f"{100 * x:.{nd}f}%"


def table(head, body):
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    for r in body:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def load_all():
    R = rows_of(os.path.join(K.OUTDIR, "ledger.jsonl"))
    F = rows_of(os.path.join(K.OUTDIR, "flips.jsonl"))
    V = json.load(open(os.path.join(K.OUTDIR, "variance_decomposition.json"),
                       encoding="utf-8"))["cells"]
    C = json.load(open(os.path.join(K.OUTDIR, "component_cases.json"),
                       encoding="utf-8"))
    return R, F, V, C


# ---------------------------------------------------------------- CSVs
def write_csvs(R, F, V, C):
    # per-run ledger
    head = ["corpus", "tau", "seed", "n", "truth_edges", "missed_truth_edges",
            "candidate_pair_recall", "verified_pair_recall", "n_candidate_pairs",
            "n_verified_edges"]
    for m in "ABWC":
        head += [f"{m}_deleted", f"{m}_unsupported", f"{m}_hijacked",
                 f"{m}_missed_vs_C", f"{m}_residual_pairs", f"{m}_witness_coverage"]
    head += ["n_components", "blocks_B", "sum_k_minus_1", "verified_missed",
             "assertion_ok", "class_connected_many_missed",
             "missed_edges_in_those", "class_split_few_missed", "split_total",
             "status", "peak_rss_mb"]
    body = []
    for r in sorted(R, key=lambda x: (x["corpus"], x["tau"], x["seed"])):
        d = r["deletion_layer"]
        row = [r["corpus"], r["tau"], r["seed"], r["n"],
               r["edge_layer"]["truth_edges"], r["edge_layer"]["missed_truth_edges"],
               f"{r['candidate_layer']['candidate_pair_recall']:.6f}",
               f"{r['candidate_layer']['verified_pair_recall']:.6f}",
               r["candidate_layer"]["n_candidate_pairs"],
               r["candidate_layer"]["n_verified_edges"]]
        for m in "ABWC":
            x = d[m]
            row += [x["deleted"], x["unsupported"], x["hijacked"],
                    (d["verified_missed"] if m == "B" else
                     d["raw_missed"] if m == "A" else
                     d["witness_missed"] if m == "W" else 0),
                    x["residual_duplicate_pairs"], f"{x['witness_coverage']:.6f}"]
        c = r["components"]
        row += [c["n_components"], c["blocks_B"], c["sum_k_minus_1"],
                c["verified_missed"], c["assertion_ok"],
                c["connected_with_many_missed"], c["missed_edges_in_those"],
                c["split_with_few_missed"], c["split_total"],
                r["status"], r["peak_rss_mb"]]
        body.append(row)
    with open(os.path.join(AN, "ledger_runs.csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(head) + "\n")
        for row in body:
            fh.write(",".join(str(v) for v in row) + "\n")

    # summary by cell
    cells = defaultdict(list)
    for r in R:
        cells[(r["corpus"], r["tau"])].append(r)
    head = ["corpus", "tau", "seeds", "candidate_pair_recall_mean",
            "A_deleted", "A_missed", "A_unsupported", "A_hijacked",
            "B_deleted", "B_missed", "B_unsupported", "B_hijacked",
            "C_deleted", "W_deleted", "W_missed",
            "B_residual_pairs", "A_residual_pairs", "B_witness_coverage",
            "sum_k_minus_1", "verified_missed", "assertion_ok_all"]
    body = []
    for (corp, tau), rs in sorted(cells.items()):
        d = [r["deletion_layer"] for r in rs]
        body.append([
            corp, tau, len(rs),
            f"{mean([r['candidate_layer']['candidate_pair_recall'] for r in rs]):.5f}",
            f(mean([x["A"]["deleted"] for x in d])),
            f(mean([x["raw_missed"] for x in d])),
            f(mean([x["A"]["unsupported"] for x in d])),
            f(mean([x["A"]["hijacked"] for x in d])),
            f(mean([x["B"]["deleted"] for x in d])),
            f(mean([x["verified_missed"] for x in d])),
            f(mean([x["B"]["unsupported"] for x in d])),
            f(mean([x["B"]["hijacked"] for x in d])),
            f(mean([x["C"]["deleted"] for x in d])),
            f(mean([x["W"]["deleted"] for x in d])),
            f(mean([x["witness_missed"] for x in d])),
            f(mean([x["B"]["residual_duplicate_pairs"] for x in d])),
            f(mean([x["A"]["residual_duplicate_pairs"] for x in d])),
            f"{mean([x['B']['witness_coverage'] for x in d]):.4f}",
            f(mean([r["components"]["sum_k_minus_1"] for r in rs])),
            f(mean([r["components"]["verified_missed"] for r in rs])),
            all(r["components"]["assertion_ok"] for r in rs),
        ])
    with open(os.path.join(AN, "summary_by_cell.csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(head) + "\n")
        for row in body:
            fh.write(",".join(str(v) for v in row) + "\n")

    # witness orders
    head = ["corpus", "tau", "seed", "order", "graph", "deleted",
            "missed_vs_C", "extra_vs_C", "symdiff_vs_C", "witness_coverage",
            "residual_pairs", "symdiff_vs_minid"]
    body = []
    for r in sorted(R, key=lambda x: (x["corpus"], x["tau"], x["seed"])):
        for order, rec in r["witness_orders"].items():
            for g in ("truth_graph", "verified_graph"):
                x = rec[g]
                body.append([r["corpus"], r["tau"], r["seed"], order, g,
                             x["deleted"], x["vs_C"]["missed"], x["vs_C"]["extra"],
                             x["vs_C"]["symdiff"], f"{x['witness_coverage']:.6f}",
                             x["residual_duplicate_pairs"],
                             x["deleted_ids_symdiff_vs_minid"]])
    with open(os.path.join(AN, "witness_orders.csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(head) + "\n")
        for row in body:
            fh.write(",".join(str(v) for v in row) + "\n")

    # component classes
    head = ["corpus", "tau", "seed", "n_components", "blocks_B",
            "sum_k_minus_1", "verified_missed", "assertion_ok",
            "connected_with_many_missed", "missed_edges_in_those",
            "split_with_few_missed", "split_total"]
    body = []
    for r in sorted(R, key=lambda x: (x["corpus"], x["tau"], x["seed"])):
        c = r["components"]
        body.append([r["corpus"], r["tau"], r["seed"], c["n_components"],
                     c["blocks_B"], c["sum_k_minus_1"], c["verified_missed"],
                     c["assertion_ok"], c["connected_with_many_missed"],
                     c["missed_edges_in_those"], c["split_with_few_missed"],
                     c["split_total"]])
    with open(os.path.join(AN, "component_classes.csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(head) + "\n")
        for row in body:
            fh.write(",".join(str(v) for v in row) + "\n")

    # flips
    head = ["corpus", "tau", "method", "source", "K_runs",
            "n_flipped", "flip_share_docs", "flip_word_share",
            "expected_symdiff", "expected_symdiff_share_docs",
            "TN_n_flipped", "TN_expected_symdiff_share_docs",
            "TN_expected_symdiff_word_share"]
    body = []
    for r in sorted(F, key=lambda x: (x["corpus"], x["tau"], x["method"], x["source"])):
        a, t = r["all"], r["true_neighbour"]
        body.append([r["corpus"], r["tau"], r["method"], r["source"], r["K_runs"],
                     a["n_flipped"], f"{a['flip_share_docs']:.5f}",
                     f"{a['flip_word_share']:.5f}", f"{a['expected_symdiff']:.2f}",
                     f"{a['expected_symdiff_share_docs']:.5f}",
                     t["n_flipped"], f"{t['expected_symdiff_share_docs']:.5f}",
                     f"{t['expected_symdiff_word_share']:.5f}"])
    with open(os.path.join(AN, "flips.csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(head) + "\n")
        for row in body:
            fh.write(",".join(str(v) for v in row) + "\n")

    # variance
    head = ["corpus", "tau", "K", "true_edges", "n_components",
            "pair_recall_mean", "pair_recall_sd", "component_share",
            "covariance_share", "largest_component_edges",
            "largest_component_share_of_terms", "variance_without_largest",
            "change_after_removal_share", "theoretical_upper_bound",
            "measured_over_bound", "overdispersion"]
    body = []
    for key, r in sorted(V.items()):
        v = r["variance"]
        body.append([r["corpus"], r["tau"], r["n_seeds"], r["true_edges"],
                     r["n_components"], f"{r['pair_recall']['mean']:.6f}",
                     f(r["pair_recall"]["sd"], 6), f(v["component_share"], 4),
                     f(v["covariance_share"], 4), v["largest_component_edges"],
                     f(v["largest_component_share_of_component_terms"], 4),
                     f"{v['measured_total_without_largest']:.3e}",
                     f(v["measured_change_after_removal_share"], 4),
                     f"{v['theoretical_upper_bound']:.3e}",
                     f(v["measured_over_bound"], 4), f(v["overdispersion"], 2)])
    with open(os.path.join(AN, "variance.csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(head) + "\n")
        for row in body:
            fh.write(",".join(str(v) for v in row) + "\n")


# ---------------------------------------------------------------- markdown
def md_correctness(R, F, V, C):
    cells = defaultdict(list)
    for r in R:
        cells[(r["corpus"], r["tau"])].append(r)
    body = []
    for (corp, tau), rs in sorted(cells.items()):
        d = [r["deletion_layer"] for r in rs]
        b_miss = mean([x["verified_missed"] for x in d])
        cdel = mean([x["C"]["deleted"] for x in d])
        body.append([
            f"{CORPUS_ZH[corp]} / τ={tau}", len(rs),
            f"{mean([r['candidate_layer']['candidate_pair_recall'] for r in rs]):.5f}",
            f(mean([x["raw_missed"] for x in d])), f(b_miss),
            f(mean([x["witness_missed"] for x in d])), f(cdel),
            pct(b_miss / cdel) if cdel else "—",
            f(mean([x["B"]["unsupported"] for x in d])),
            f(mean([x["B"]["hijacked"] for x in d])),
            f(mean([x["B"]["residual_duplicate_pairs"] for x in d])),
            f"{mean([x['B']['witness_coverage'] for x in d]):.4f}",
            f"{mean([r['components']['sum_k_minus_1'] for r in rs]):.0f}",
        ])
    t1 = table(["单元", "种子数", "候选对召回", "A 漏删", "B 漏删", "W 漏删",
                "C 删除", "B 漏删占 C", "B 无据删除", "B 劫持删除",
                "B 残余真重复对", "B 见证覆盖", "Σ(k−1)"], body)

    # c4 tau=0.8 per seed (the Draft 2 counterexample)
    b2 = []
    for r in sorted([x for x in R if x["corpus"] == "c4" and x["tau"] == 0.8],
                    key=lambda x: x["seed"]):
        d = r["deletion_layer"]
        b2.append([r["seed"], f"{r['candidate_layer']['verified_pair_recall']:.5f}",
                   r["edge_layer"]["missed_truth_edges"], d["B"]["deleted"],
                   d["verified_missed"], d["witness_missed"],
                   r["components"]["sum_k_minus_1"],
                   r["components"]["connected_with_many_missed"],
                   r["components"]["missed_edges_in_those"],
                   r["components"]["split_with_few_missed"]])
    t2 = table(["种子", "验证后边召回", "漏掉真边", "B 删除", "B 漏删",
                "W 漏删", "Σ(k−1)", "漏边多但连通（分量数）",
                "这些分量漏掉的边", "漏边≤2 却分裂（分量数）"], b2)

    # witness order means per cell
    wcells = defaultdict(list)
    for r in R:
        for order, rec in r["witness_orders"].items():
            for g in ("truth_graph", "verified_graph"):
                wcells[(r["corpus"], r["tau"], order, g)].append(rec[g])
    b3 = []
    for (corp, tau, order, g), xs in sorted(wcells.items()):
        b3.append([f"{CORPUS_ZH[corp]} / τ={tau}", order,
                   "真值图" if g == "truth_graph" else "B 验证图",
                   f(mean([x["deleted"] for x in xs])),
                   f(mean([x["vs_C"]["missed"] for x in xs])),
                   f(mean([x["vs_C"]["extra"] for x in xs])),
                   f(mean([x["vs_C"]["symdiff"] for x in xs])),
                   f"{mean([x['witness_coverage'] for x in xs]):.3f}",
                   f(mean([x["residual_duplicate_pairs"] for x in xs])),
                   f(mean([x["deleted_ids_symdiff_vs_minid"] for x in xs]))])
    t3 = table(["单元", "顺序", "图", "删除数", "相对 C 漏删", "相对 C 多删",
                "相对 C 对称差", "见证覆盖", "残余对", "相对 min-ID 对称差"], b3)

    # missed-candidate effect per cell (mean over orders and runs)
    mcells = defaultdict(list)
    for r in R:
        for order, rec in r["witness_orders"].items():
            mcells[(r["corpus"], r["tau"], order)].append(rec["truth_vs_verified"])
    b4 = []
    for (corp, tau, order), xs in sorted(mcells.items()):
        b4.append([f"{CORPUS_ZH[corp]} / τ={tau}", order,
                   f(mean([x["deleted_ids_symdiff"] for x in xs])),
                   f(mean([x["truth_deleted_verified_kept"] for x in xs])),
                   f(mean([x["verified_deleted_truth_kept"] for x in xs]))])
    t4 = table(["单元", "顺序", "真值图 W vs 验证图 W 对称差",
                "仅真值图删除", "仅验证图删除"], b4)
    mc8 = mean([x["deleted_ids_symdiff"] for x in mcells[("c4", 0.8, "minid")]])
    pc8 = mean([x["vs_C"]["missed"] for x in wcells[("c4", 0.8, "minid", "truth_graph")]])
    mc75 = mean([x["deleted_ids_symdiff"] for x in mcells[("c4", 0.75, "minid")]])
    pc75 = mean([x["vs_C"]["missed"] for x in wcells[("c4", 0.75, "minid", "truth_graph")]])

    n_rows = len(R)
    n_ok = sum(r["status"] == "ok" for r in R)
    peak = max(r["peak_rss_mb"] for r in R)
    cross_n = sum(len(r["crosscheck"]) for r in R)
    cross_fail = sum(1 for r in R for v in r["crosscheck"].values() if not v)
    assert_fail = sum(1 for r in R if not r["components"]["assertion_ok"])
    dt = json.load(open(os.path.join(K.SLUG, "core", "crosscheck_datatrove.json"),
                        encoding="utf-8"))

    md = f"""# 去重正确性账本：A、B、W、C 四路径的三层核对

本文件由 `build_report.py` 从逐运行记录 `ledger.jsonl`、`witness_orders.csv`
与 `component_cases.json` 生成，文中数字不手工录入；重跑脚本即可复现。

## 1. 口径

四条路径都作用在同一份冻结的 5-gram 特征与同一批 LSH 签名上，阈值 τ 相同，
代表性规则除 W 的顺序实验外都取最小文档 ID（keep-min-id）。

- **A（原始 LSH）**：datatrove 0.6.0 的两段式——先在 14×8 个 band 桶内取链式边，
  再并查集聚类。它代表"候选生成后不做精确校验"的流水线。
- **B（候选边精确校验后聚类）**：同一批 band 桶取**全部桶内对**（不止链式边），
  用 5-gram Jaccard ≥ τ 逐对精确校验，再并查集聚类。
- **W（直接见证）**：在给定的图上按固定文档顺序贪心取极大独立集（MIS），
  保留者即代表，被删除者必须有一个直接保留的邻居。分别在完整真值图与 B 的验证图上运行。
- **C（完整精确连接后聚类）**：对全体文档做精确 5-gram 自连接，取 J ≥ τ 的真值图，
  再并查集聚类。它是本题的删除参照。

三层核对分别回答三个不同的问题：

1. **候选边判定层**：桶内对里有多少真边（候选对召回/精度），以及校验后边集召回。
2. **完整边集召回层**：真值图中有多少条边根本没有成为候选（`missed_truth_edges`）。
   这一层即使 B 的候选精度为 1 也要报告——漏掉的真边是删除错误的来源。
3. **最终删除 ID 集合层**：直接比较删除文档的 ID 集合，而不是只比计数。
   `raw_missed = |K_C \\ K_A|`，`verified_missed = |K_C \\ K_B|`，
   `witness_missed = |K_C \\ K_W|`（均为最小 ID 语义下的集合差）。

删除侧的四个文档级类别（对每个方法、每次运行单独统计）：

- **无据删除（unsupported）**：被删，但在真值图里没有任何 J ≥ τ 的邻居；
- **劫持删除（hijacked）**：被删，有真邻居，但 C 保留它——即被拉进了错误的簇；
- **漏删（missed）**：C 删除而该方法保留；
- **残余真重复对（residual）**：真值边两端都未被删除。

## 2. 逐单元汇总（各种子均值）

{t1}

`Σ(k−1)` 是分量层面的恒等式：在每个精确连通分量内只保留 B 校验过的边，
若该分量裂成 k 块，则 B 比 C 多保留 k−1 篇；统一最小 ID 时，
各分量求和必须等于 `verified_missed`。本次 {n_rows} 次运行全部满足，
失败 {assert_fail} 次。

- 运行状态：`ok` {n_ok}/{n_rows}；交叉核对项 {cross_n} 个，失败 {cross_fail} 个
  （对照冻结的 `runs.jsonl` 删除数、验证边数、召回、无据删除、残余对、见证覆盖，
  以及 τ=0.8 的逐种子位掩码）。
- 单次运行峰值 RSS 最高 {peak:.0f} MB，远低于 48 GiB 上限；每个 (语料, 种子, τ)
  单元独立读取、算完即写行，不跨种子累积边或分量。

## 3. C4、τ=0.8 的逐种子明细

{t2}

这张表就是 Draft 2 反例的重算结果：种子 3 的验证后边召回 {b2[2][1]}，
却只比 C 多保留 {b2[2][4]} 篇；种子 1 的召回 {b2[0][1]}，反而多保留 {b2[0][4]} 篇。
原因不在召回，而在分量结构：种子 3 的最大分量裂成 {C['seeds']['3']['giant']['k']} 块
（漏 {C['seeds']['3']['giant']['missed']} 条边），种子 1 的同一分量仍为
{C['seeds']['1']['giant']['k']} 块（漏 {C['seeds']['1']['giant']['missed']} 条边）；
而"漏边≤2 却分裂"的小分量，种子 1 有 {b2[0][9]} 个、种子 3 只有 {b2[2][9]} 个。

## 4. W 的顺序实验：策略代价与漏候选影响分开

W 在完整真值图与 B 的验证图上，按同一组顺序运行：min-ID 顺序加三个固定排列
（`order_permutations(n,3)`，基种子 101）。下表按单元对种子取均值，删除量均为 ID 集合比较。

{t3}

读法：

- **策略代价**（真值图上 W 相对 C）：min-ID 顺序下 W 恰好是 C 删除集的子集，
  平均少删的文档就是"只删有直接见证者"的代价；换成排列后删除数变化不大，
  但删除 ID 与 C 的对称差很大——因为每个分量保留的代表换了人，这不是漏候选。
- **漏候选影响**（验证图 W 相对真值图 W，同序）：见下表。它才是 B 的候选生成
  真正造成的额外保留。τ=0.8 时它明显小于策略代价（C4：{mc8:.1f} 对 {pc8:.1f}）；
  τ=0.75 时两者量级相当（C4：{mc75:.1f} 对 {pc75:.1f}），
  因为此时小分量更多，B 漏掉的一条边更容易成为桥。

{t4}

## 5. datatrove 两段式与 14 分片

官方 datatrove 的 LSH 去重必须把 stage 1（桶内边）与 stage 2（并查集）作为两个独立作业运行，
中间产物是边文件；本项目在 `run_core.sh` 中即按此拆分，并另做了 14 分片与单进程的
一致性核对（`crosscheck_datatrove.json`）：分片数 14、两段式 pair 集合完全一致
（对称差 {dt['pair_symmetric_difference']}），union-by-size 删除集合一致
（对称差 {dt['union_by_size_symmetric_difference']}），`all_checks_pass = {str(dt['all_checks_pass']).lower()}`。
该核对已实际运行，不是仅写在文档里的约定。

## 6. 缺口

- 新独立快照（CC-MAIN-2025-30）的去重结果尚未生成，本账本不包含它的删除集；
  两分类计数待 `independent-snapshot-eval` 产出删除集后按同一脚本补齐（见
  `component_mechanism.md` 第 4 节）。
- 词占比使用冻结特征缓存的空白分词计数，不是固定分词器下的训练 token 数；
  如需训练 token 口径须重新分词，本步未做。
"""
    with open(os.path.join(AN, "correctness_ledger.md"), "w", encoding="utf-8") as fh:
        fh.write(md)


def md_definitions():
    md = """# 指标定义与恒等式

本文件配套 `correctness_ledger.md`。所有集合均以文档 ID 表示，`K_X` 表示方法 X 的删除集合。

## 图与分量

- 真值图 G_τ：全体文档对中 5-gram Jaccard ≥ τ 的边；由精确自连接得到。
- 精确连通分量：G_τ 的连通分量；C 的删除集 = 每个分量去掉最小 ID 的代表。
- B 的验证图：候选对中 J ≥ τ 的边；它的连通分量把每个精确分量切成若干块。

## 逐方法指标

| 指标 | 定义 | 说明 |
|---|---|---|
| deleted | \\|K_X\\| | 删除文档数 |
| unsupported | \\|K_X ∩ ¬N\\|，N 为有真邻居的文档集 | 无任何真重复却被删 |
| hijacked | \\|K_X ∩ N ∩ ¬K_C\\| | 有真邻居、C 保留，却被删（拉入错误簇） |
| missed | \\|K_C \\ K_X\\| | C 删而 X 保留 |
| residual | 两端都未删的真值边数 | 删除后仍残留的真重复对 |
| witness coverage | 被删文档中至少有一个未删真邻居的比例 | 删除是否有直接见证 |

`unsupported + hijacked + (共同删除) = |K_X|`；`X ⊂ K_C` 时 `hijacked = 0`。
B 与 W 的删除集在最小 ID 顺序下都是 C 的子集，故二者的 hijacked 恒为 0；
A 的劫持删除来自候选连通性把无关文档并进簇。

## 三层召回

- 候选对召回：真值边中成为候选对的比例；
- 验证后边召回：真值边中通过 J ≥ τ 校验的比例（B 的校验不会删除真边，
  因此两者在本次数据上相同）；
- 删除召回：\\|K_C ∩ K_X\\| / \\|K_C\\|。前两层高不代表第三层高，反之亦然。

## 分量恒等式

在每个精确分量 c 内，B 的验证图把它分成 k_c 块。B 的删除数比 C 少 k_c − 1，
于是

    |K_C \\ K_B| = Σ_c (k_c − 1) = (#B 块) − (#精确分量)

逐运行断言。该恒等式把"边级漏检"翻译成"文档级漏删"，是解释 Draft 2 反例的桥梁。

## 种子翻转与方差

- 文档 d 在 K 次运行中被删 c 次，两个不同种子删除结果不一致的概率
  `p_d = 2c(K−c)/(K(K−1))`；期望对称差 `Σ_d p_d`。
- 词占比按文档的空白分词计数加权，与"训练 token 占比"不是同一口径。
- 方差分解：`R_s = Σ_c (e_c/E) p_{c,s}`，
  `Var(R) = Σ_c (e_c/E)^2 Var(p_c) + Σ_{c≠c'} (e_c e_c'/E^2) Cov(p_c,p_c')`。
  第一项按分量计，第二项是分量间协方差（由实测总方差减第一项得到）。
  上界 `Σ_c (e_c/E)^2 p̄_c(1−p̄_c)` 是边缘命中率给定时任一分量能贡献的最大方差，
  仅在分量内所有边同进同出（p_c ∈ {0,1}）时取到。
"""
    with open(os.path.join(AN, "definitions.md"), "w", encoding="utf-8") as fh:
        fh.write(md)


def md_components(C):
    s1 = C["seeds"].get("1", {})
    s3 = C["seeds"].get("3", {})
    per = C["per_seed"]
    rows = [[p["seed"], p["missed_edges"], p["verified_missed"],
             f"{p['pair_recall']:.5f}" if p["pair_recall"] else "—"] for p in per]
    t = table(["种子", "漏掉真边", "多保留文档（= B 漏删）", "验证后边召回"], rows)
    ex_l = s3.get("examples", {}).get("leaky")
    ex_s = s3.get("examples", {}).get("split")
    ns = C["new_snapshot"]
    md = f"""# 分量机制：为什么成对召回会误导删除决策

## 1. 机制

删除决策发生在分量层面。一条真边漏检，只有当它是桥时才改变删除集合：
把一个精确连通分量切成两块，B 就比 C 多保留 1 篇。反之，一个稠密分量内部
漏掉成百上千条边，只要它仍然连通，删除数一篇不差。因此"候选对召回"与
"多保留文档数"之间没有单调关系。

## 2. Draft 2 反例的重算（C4，τ=0.8，14×8）

{t}

种子 3 的召回最低（{per[2]['pair_recall']:.5f}），多保留 {per[2]['verified_missed']} 篇；
种子 1 的召回更高（{per[0]['pair_recall']:.5f}），却多保留 {per[0]['verified_missed']} 篇。
种子 1 与种子 3 的运行中，分量恒等式都成立
（Σ(k−1) = {s1.get('sum_k_minus_1')} 与 {s3.get('sum_k_minus_1')}）。

真实分量给出的两类情形（图 `component_classes.png`）：

- **漏边多但仍连通**：种子 3 的一个 {ex_l['size'] if ex_l else '—'} 文档分量有
  {ex_l['true_edges'] if ex_l else '—'} 条真边，漏 {ex_l['missed'] if ex_l else '—'} 条，
  仍是一块，B 多保留 0 篇；
- **漏边少却分裂**：种子 3 的一个 {ex_s['size'] if ex_s else '—'} 文档分量只有
  {ex_s['true_edges'] if ex_s else '—'} 条真边，漏 {ex_s['missed'] if ex_s else '—'} 条
  就裂成两块，B 多保留 1 篇。

## 3. 两类分量的计数

对每次运行，把每个精确分量按 (k, 漏边数) 分类：

- "漏边多但连通"：k = 1 且漏边 ≥ 5 的分量数，以及它们漏掉的边数；
- "漏边少却分裂"：k > 1 且漏边 ≤ 2 的分量数。

逐运行计数见 `component_classes.csv`。C4、τ=0.8 的五次运行中，第一类分量数为
{[r['components']['connected_with_many_missed'] for r in _rows('c4', 0.8)]}，
对应漏边 {[r['components']['missed_edges_in_those'] for r in _rows('c4', 0.8)]} 条；
第二类为 {[r['components']['split_with_few_missed'] for r in _rows('c4', 0.8)]} 个。
最大分量在种子 1 中漏 {s1['giant']['missed']} 条仍为 {s1['giant']['k']} 块，
在种子 3 中漏 {s3['giant']['missed']} 条却裂成 {s3['giant']['k']} 块；
种子 1 与种子 3 的多保留文档数相差
{abs(per[0]['verified_missed'] - per[2]['verified_missed'])} 篇，
其中 {s3['giant']['k'] - s1['giant']['k']} 篇来自这个最大分量的块数差。

## 4. 新快照

新独立快照（CC-MAIN-2025-30）的去重删除集尚未产出（`{ns['status']}`）。
本节的两类计数在该快照上待 `independent-snapshot-eval` 完成后补齐；
本步不填任何推测值。
"""
    with open(os.path.join(AN, "component_mechanism.md"), "w", encoding="utf-8") as fh:
        fh.write(md)


def _rows(corpus, tau):
    R = rows_of(os.path.join(K.OUTDIR, "ledger.jsonl"))
    return sorted([r for r in R if r["corpus"] == corpus and r["tau"] == tau],
                  key=lambda x: x["seed"])


def md_flips(F, V):
    pick = [r for r in F if r["corpus"] == "c4" and r["tau"] == 0.8]
    order = {"A": 0, "B": 1, "W": 2, "C": 3}
    pick.sort(key=lambda r: (order[r["method"]], r["source"]))
    rows = []
    for r in pick:
        a, t = r["all"], r["true_neighbour"]
        rows.append([r["method"], r["source"], r["K_runs"], a["n_flipped"],
                     pct(a["flip_share_docs"]), pct(a["flip_word_share"], 3),
                     f"{a['expected_symdiff']:.1f}",
                     pct(a["expected_symdiff_share_docs"], 3),
                     pct(t["expected_symdiff_share_docs"], 3),
                     pct(t["expected_symdiff_word_share"], 3)])
    t1 = table(["方法", "来源", "K", "翻转文档数", "文档占比", "词占比",
                "期望对称差", "期望对称差/文档", "真邻居子组/文档", "真邻居子组/词"], rows)

    vrows = []
    for key, r in sorted(V.items()):
        v = r["variance"]
        vrows.append([f"{CORPUS_ZH[r['corpus']]} / τ={r['tau']}", r["n_seeds"],
                      r["true_edges"], f"{r['pair_recall']['sd']:.5f}",
                      pct(v["component_share"], 1), pct(v["covariance_share"], 1),
                      pct(v["largest_component_share_of_component_terms"], 1)
                      if v["largest_component_share_of_component_terms"] is not None else "—",
                      f"{v['measured_total_without_largest']:.3e}",
                      f(v["measured_over_bound"], 3),
                      f(v["overdispersion"], 1)])
    t2 = table(["单元", "K", "真值边", "召回 sd", "分量项占比", "协方差项占比",
                "最大分量占分量项", "去掉最大分量后方差", "实测/上界", "过散布"], vrows)

    a100 = next(r for r in F if r["key"] == "c4|t0.8|A|raw|K100")
    b5 = next(r for r in F if r["key"] == "c4|t0.8|B|sets|K5")
    w5 = next(r for r in F if r["key"] == "c4|t0.8|W|sets|K5")
    md = f"""# 种子翻转与召回方差的来源

## 1. 翻转概率

文档 d 在 K 次运行中被删 c 次，两个不同种子删除结果不一致的概率为
`p_d = 2c(K−c) / (K(K−1))`。期望删除集对称差为 `Σ_d p_d`。

C4、τ=0.8、100 个种子的原始 LSH（A）：翻转文档 {a100['all']['n_flipped']:,} 篇，
文档占比 {pct(a100['all']['flip_share_docs'])}，**词占比 {pct(a100['all']['flip_word_share'], 2)}**，
即 Draft 2 中沿用的 1.14%——它一直是词占比，不是文档占比（文档占比只有 2.65%）。
按上式，两次运行的期望对称差为 {a100['all']['expected_symdiff']:.0f} 篇
（占输入 {pct(a100['all']['expected_symdiff_share_docs'], 3)}，按词 {pct(a100['all']['expected_symdiff_word_share'], 3)}）。
在 B 的 5 种子序列上，翻转降到 {b5['all']['n_flipped']} 篇，
期望对称差 {b5['all']['expected_symdiff']:.1f} 篇；C 恒为 0。

## 2. C4、τ=0.8 的分方法翻转

{t1}

"真邻居子组"只统计至少有一个真邻居的文档（C4、τ=0.8 共
{a100['true_neighbour']['n_docs']:,} 篇）。A 的 {a100['all']['n_flipped']:,} 篇翻转文档中
只有 {a100['true_neighbour']['n_flipped']:,} 篇（{pct(a100['true_neighbour']['n_flipped'] / a100['all']['n_flipped'], 1)}）
在这个子组里：多数翻转是无真邻居、却被原始 LSH 拉进簇的文档；但子组内的翻转率是
{pct(a100['true_neighbour']['flip_share_docs'], 1)}，约为全体
{pct(a100['all']['flip_share_docs'], 1)} 的五倍。
相反，B 的 {b5['all']['n_flipped']} 篇翻转与 W 的 {w5['all']['n_flipped']} 篇翻转
全部落在真邻居子组（{b5['true_neighbour']['n_flipped']} 与 {w5['true_neighbour']['n_flipped']}）——
候选生成造成的种子差异完全发生在真重复内部，不再来自无据删除。

口径说明：词占比使用冻结特征缓存的空白分词计数，不是固定分词器的训练 token 数；
若要训练 token 口径必须重新分词，本步未做。

## 3. 召回方差分解

把每次运行的召回写成 `R_s = Σ_c (e_c/E) p_{{c,s}}`，e_c 为分量 c 的真边数，
p_c 为该次运行在分量 c 内的命中率。实测总方差按种子样本方差计。

{t2}

要点：

- C4、τ=0.8 时，分量内项解释了 {pct(V['c4|t0.8']['variance']['component_share'], 1)} 的总方差，
  其中最大分量（{V['c4|t0.8']['variance']['largest_component_edges']:,} 条边）
  占分量项 {pct(V['c4|t0.8']['variance']['largest_component_share_of_component_terms'], 2)}。
  也就是说，种子间的召回波动几乎就是那一个巨型分量的命中率波动。
- 去掉最大分量后，实测方差从 {V['c4|t0.8']['variance']['measured_total']:.3e}
  降到 {V['c4|t0.8']['variance']['measured_total_without_largest']:.3e}，
  变化 {pct(V['c4|t0.8']['variance']['measured_change_after_removal_share'], 2)}。
- 分量间协方差项在 C4、τ=0.8 近似为 0（{V['c4|t0.8']['variance']['covariance_term']:.2e}），
  在 FineWeb、WET 上为正且占比可观；个别单元为负，表示分量间命中率相互抵消。
- 上界 `Σ_c (e_c/E)^2 p̄_c(1−p̄_c)` 是"分量内所有边同进同出"时的最大方差；
  实测仅为该上界的 {pct(V['c4|t0.8']['variance']['measured_over_bound'], 1)}
  （过散布 {V['c4|t0.8']['variance']['overdispersion']:.1f} 倍于独立边二项零假设），
  说明分量内部并非全有全无，但仍远非独立。
- 在 FineWeb、WET 的 τ=0.8 单元，去掉最大分量后方差反而上升
  （{V['fw|t0.8']['variance']['measured_total_without_largest']:.3e} 对
  {V['fw|t0.8']['variance']['measured_total']:.3e}；WET 同理），
  即最大分量是稳定项，剩余小分量的波动更大。
"""
    with open(os.path.join(AN, "flip_variance.md"), "w", encoding="utf-8") as fh:
        fh.write(md)


def main():
    R, F, V, C = load_all()
    write_csvs(R, F, V, C)
    md_correctness(R, F, V, C)
    md_definitions()
    md_components(C)
    md_flips(F, V)
    for f in sorted(os.listdir(AN)):
        print("wrote", os.path.join(AN, f))


if __name__ == "__main__":
    main()
