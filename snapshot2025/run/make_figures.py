#!/usr/bin/env python3
"""Data figures for the reveal report (matplotlib only; numbers from JSONL-derived
newsnap/selection_results.json). Chinese labels, PNG for the DOCX/PDF pipeline."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
SLUG = os.path.dirname(HERE)
SNAP = os.path.join(SLUG, "newsnap")
OUT = os.path.join(SLUG, "report", "figures")
os.makedirs(OUT, exist_ok=True)

R = json.load(open(os.path.join(SNAP, "selection_results.json"), encoding="utf-8"))
Q = R["quality"]
TAU = 0.8
methods = ["A", "B", "W", "C"]
labels = {"A": "A（链边并查集）", "B": "B（桶对+精确验证）",
          "W": "W（验证图贪心独立集）", "C": "C（精确连接）"}
del_tot, uns, sup = [], [], []
for m in methods:
    a = Q[m][str(TAU)]["aggregate"]
    d = a["deleted"]
    u = a["unsupported"]
    del_tot.append(d)
    uns.append(u)
    sup.append(d - u)

# --- Figure 1: deletion composition
fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=200)
x = np.arange(len(methods))
b1 = ax.bar(x, sup, width=0.55, color="#5B7DB1", label="有支持删除（有精确近邻）")
b2 = ax.bar(x, uns, width=0.55, bottom=sup, color="#E4572E", label="无支持删除（无精确近邻）")
exact = Q["C"][str(TAU)]["aggregate"]["exact_deleted"]
ax.axhline(exact, color="#2A9D8F", lw=1.2, ls="--",
           label=f"精确删除集大小 {exact:,.0f}")
for i, m in enumerate(methods):
    a = Q[m][str(TAU)]["aggregate"]
    share = a["unsupported_share_del"]
    ax.text(i, del_tot[i] * 1.02, f"{del_tot[i]:,.0f}\n无支持 {share*100:.2f}%",
            ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels([labels[m] for m in methods], fontsize=8)
ax.set_ylabel("删除文档数（τ=0.80）", fontsize=9)
ax.set_ylim(0, max(del_tot) * 1.22)
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", lw=0.3, alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_deletions.png"))
plt.close(fig)

# --- Figure 2: candidate-pair density against the pre-registered alarm
b = R["duplicate_density"]
sd = R["sampling_diag"]
items = [
    ("告警阈值（10×C4 池率）", sd["threshold_pairs_per_doc"], "#9AA5B1"),
    ("2,000 篇等距抽样", sd["pairs_per_doc"], "#F2B134"),
    ("全量·跨带去重后", b["n_candidate_pairs_doc_unique"] / b["n_documents"], "#5B7DB1"),
    ("全量·按带求和", b["candidate_pairs_total"] / b["n_documents"], "#E4572E"),
]
fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=200)
y = np.arange(len(items))[::-1]
ax.barh(y, [v for _, v, _ in items], color=[c for _, _, c in items], height=0.55)
for yi, (name, v, _) in zip(y, items):
    ax.text(v * 1.08, yi, f"{v:,.2f}", va="center", fontsize=8)
ax.set_yticks(y)
ax.set_yticklabels([n for n, _, _ in items], fontsize=8.5)
ax.set_xscale("log")
ax.set_xlabel("每文档候选对数（对数坐标）", fontsize=9)
ax.set_xlim(1, sd["full_snapshot_pairs_per_doc"] * 3)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", lw=0.3, alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_candidates.png"))
plt.close(fig)
print("wrote", OUT)
