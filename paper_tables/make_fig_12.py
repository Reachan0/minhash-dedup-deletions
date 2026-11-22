#!/usr/bin/env python3
"""Figures 1 and 2 of v5 from the v3 result files.

fig_01_decision: unsupported share and cold time against incremental time, one point per corpus
  and path (v3 data/main_decision_table.json, tau 0.8, min-id, medians of independent processes on
  one seed). B nearly coincides with W, so B is drawn as a hollow marker on top. W has no cold run.
fig_03_counterexample: (a) pair recall against exact deletions missed per seed on C4 at tau 0.8;
  (b) missed edges inside the largest exact component for seeds 1 and 3, labelled with its number
  of blocks and the share of the run's missed exact deletions that comes from it
  (v3 data/correctness/component_cases.json).
Usage: python make_fig_12.py [paper dir]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
P = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "out")
for _d in ("tables", "figures"):
    os.makedirs(os.path.join(P, _d), exist_ok=True)
V3 = os.path.join(HERE, "..", "results", "original_corpora")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 400, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
COL = {"A": "#1f77b4", "B": "#ff7f0e", "W": "#2ca02c", "C": "#d62728"}
MK = {"c4": "o", "fw": "s", "mix": "^", "wet": "D"}
NAME = {"c4": "C4", "fw": "FineWeb", "mix": "Mix", "wet": "WET"}


def point(ax, x, y, path, corpus):
    if path == "B":
        ax.scatter([x], [y], s=70, facecolors="none", edgecolors=COL["B"], linewidths=1.6,
                   marker=MK[corpus], zorder=5)
    else:
        ax.scatter([x], [y], s=30, color=COL[path], marker=MK[corpus], zorder=3 if path != "W" else 4)


def fig1():
    rows = [r for r in json.load(open(os.path.join(V3, "main_decision_table.json")))
            if r["tau"] == 0.8 and r["representative"] == "minid"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for r in rows:
        p, c = r["method"], r["workload"]
        point(a, r["incremental_median_s"], 100 * r["unsupported_share_del_mean"], p, c)
        if r.get("cold_median_s"):
            point(b, r["incremental_median_s"], r["cold_median_s"], p, c)
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xlabel("Incremental time (s)")
        ax.set_xticks([2, 3, 4, 6, 10])
        ax.set_xticklabels(["2", "3", "4", "6", "10"])
        ax.minorticks_off()
    a.set_ylabel("Unsupported deletions (%)")
    a.set_ylim(-4, 100)
    a.set_title("(a) Quality", fontsize=10)
    b.set_yscale("log")
    b.set_yticks([100, 200, 300, 500])
    b.set_yticklabels(["100", "200", "300", "500"])
    b.yaxis.set_minor_formatter(NullFormatter())
    b.set_ylim(70, 650)
    b.set_ylabel("Cold time (s)")
    b.set_title("(b) Cost", fontsize=10)
    hp = [Line2D([], [], marker="o", ls="", color=COL[p], ms=5, label=f"Path {p}") for p in "ACW"]
    hp.insert(1, Line2D([], [], marker="o", ls="", mfc="none", mec=COL["B"], mew=1.4, ms=6, label="Path B"))
    hc = [Line2D([], [], marker=MK[c], ls="", color="#777777", ms=5, label=NAME[c]) for c in MK]
    fig.legend(handles=hp + hc, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.06),
               handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(P, "figures", f"fig_01_decision.{ext}"), bbox_inches="tight")


def fig2():
    d = json.load(open(os.path.join(V3, "correctness", "component_cases.json")))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for r in d["per_seed"]:
        a.scatter([r["pair_recall"]], [r["verified_missed"]], s=30, color="#1f77b4", zorder=3)
        a.annotate(f"seed {r['seed']}", (r["pair_recall"], r["verified_missed"]), textcoords="offset points",
                   xytext=(5, 4), fontsize=8)
    a.set_xlabel("Verified pair recall")
    a.set_ylabel("Exact deletions missed")
    a.set_ylim(14, 28)
    a.set_title("(a) Recall does not order the outcome", fontsize=10)
    seeds = ["1", "3"]
    ys = [d["seeds"][s]["giant"]["missed"] for s in seeds]
    b.bar([0, 1], ys, width=0.5, color="#9467bd")
    for i, s in enumerate(seeds):
        g, tot = d["seeds"][s]["giant"], d["seeds"][s]["total_verified_missed"]
        blocks = "1 block" if g["k"] == 1 else f"{g['k']} blocks"
        b.annotate(f"{blocks}\n{g['k'] - 1} of the run's {tot} missed", (i, ys[i]), textcoords="offset points",
                   xytext=(0, 4), ha="center", va="bottom", fontsize=8)
    b.set_xticks([0, 1])
    b.set_xticklabels([f"seed {s}" for s in seeds])
    b.set_ylim(0, 5000)
    b.set_ylabel("Missed edges in the largest component")
    b.set_title("(b) The same component, two seeds", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(P, "figures", f"fig_03_counterexample.{ext}"), bbox_inches="tight")


if __name__ == "__main__":
    fig1()
    fig2()
    print("figures written")
