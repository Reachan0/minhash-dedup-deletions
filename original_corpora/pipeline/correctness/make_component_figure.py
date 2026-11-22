#!/usr/bin/env python3
"""Draw the two component classes behind the Draft 2 counterexample (Chinese).

Input : core/correctness/component_cases.json  (real components, c4 tau=0.8)
Output: core/correctness/figures/component_classes.png/.pdf

The figure has three panels:
  (a) a real component where several true edges were missed yet the component
      stays connected, so B keeps no extra document;
  (b) a real component split in two by a single missed edge, so B keeps one
      extra document although recall barely moved;
  (c) per seed: missed true edges vs extra kept documents, the non-monotone
      relation that pair recall cannot express.

Usage: python3 make_component_figure.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import correctness_lib as K

plt.rcParams["font.family"] = ["Noto Sans CJK SC", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False

CASES = os.path.join(K.OUTDIR, "component_cases.json")
FIGDIR = os.path.join(K.OUTDIR, "figures")
BLOCK_COLORS = ["#3b6fb6", "#d1495b", "#5a9367", "#e08e3c", "#7b5ea7", "#4a9aa8"]

EDGE_MISS = dict(color="#d1495b", linestyle=(0, (3, 2)), linewidth=1.4)
EDGE_HIT = dict(color="#8a8f98", linewidth=1.0)


def circular_layout(ex):
    """Nodes ordered by block so each block occupies a contiguous arc."""
    order = []
    for b in ex["blocks"]:
        order.extend(b)
    edge_set = {(e[0], e[1]) for e in ex["edges"]}
    # keep graph vertices in a sensible order: greedy DFS over the block order
    pos = {}
    m = len(order)
    for idx, v in enumerate(order):
        ang = 2 * np.pi * idx / max(m, 1) - np.pi / 2
        pos[v] = (np.cos(ang), np.sin(ang))
    return pos


def draw_component(ax, ex, title, note):
    pos = circular_layout(ex)
    for a, b, det in ex["edges"]:
        x = [pos[a][0], pos[b][0]]
        y = [pos[a][1], pos[b][1]]
        ax.plot(x, y, **(EDGE_HIT if det else EDGE_MISS), zorder=1)
    block_of = {}
    for bi, block in enumerate(ex["blocks"]):
        for v in block:
            block_of[v] = bi
    for v, (x, y) in pos.items():
        ax.scatter([x], [y], s=210, color=BLOCK_COLORS[block_of[v] % len(BLOCK_COLORS)],
                   edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(x, y, str(v), ha="center", va="center", fontsize=6.5,
                color="white", zorder=4)
    # mark the kept (minimum-id) document of each block
    for v in ex["kept_local"]:
        x, y = pos[v]
        ax.scatter([x], [y], s=340, facecolors="none", edgecolors="#1f1f1f",
                   linewidth=1.3, zorder=5)
    for bi, block in enumerate(ex["blocks"]):
        cx = np.mean([pos[v][0] for v in block])
        cy = max(pos[v][1] for v in block)
        ax.text(cx, cy + 0.18, f"B块{bi + 1}" if ex["k"] > 1 else "同一 B 块",
                ha="center", va="bottom", fontsize=8,
                color=BLOCK_COLORS[bi % len(BLOCK_COLORS)],
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                zorder=6)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.text(0, -1.38, note, ha="center", va="top", fontsize=8.2, color="#333333")


def main():
    cases = json.load(open(CASES, encoding="utf-8"))
    seeds = cases["seeds"]
    # panel (a)/(b) use seed 3 so both panels come from the run whose overall
    # counterexample behaviour the note describes
    def pick(kind, pref="3"):
        if pref in seeds and kind in seeds[pref]["examples"]:
            return pref
        cand = [s for s in seeds if kind in seeds[s]["examples"]]
        return cand[0] if cand else None
    leak_seed = pick("leaky")
    split_seed = pick("split")

    fig = plt.figure(figsize=(11.2, 4.1), dpi=200)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.22,
                          left=0.03, right=0.975, top=0.84, bottom=0.13)

    if leak_seed is not None:
        ex = seeds[leak_seed]["examples"]["leaky"]
        draw_component(
            fig.add_subplot(gs[0, 0]), ex,
            "(a) 漏边多但仍连通",
            f"C4 τ=0.8 种子{leak_seed}：真实边 {ex['true_edges']} 条，漏 {ex['missed']} 条，"
            f"仍为 1 块\nB 比 C 多保留 0 篇")
    if split_seed is not None:
        ex = seeds[split_seed]["examples"]["split"]
        draw_component(
            fig.add_subplot(gs[0, 1]), ex,
            "(b) 漏边少却分裂",
            f"C4 τ=0.8 种子{split_seed}：真实边 {ex['true_edges']} 条，仅漏 {ex['missed']} 条，"
            f"裂成 {ex['k']} 块\nB 比 C 多保留 {ex['k'] - 1} 篇")

    ax = fig.add_subplot(gs[0, 2])
    per = cases["per_seed"]
    x = np.arange(len(per))
    missed_edges = np.array([p["missed_edges"] for p in per], float)
    verified = np.array([p.get("verified_missed") or 0 for p in per], float)
    bars = ax.bar(x, missed_edges / 1000.0, color="#9fb6d4", width=0.62,
                  label="漏掉的真实边（千条，左轴）")
    ax.set_xticks(x)
    ax.set_xticklabels([f"种子{p['seed']}" for p in per], fontsize=8)
    ax.set_ylabel("漏掉的真实边（千条）", fontsize=9)
    ax.set_ylim(0, max(missed_edges / 1000.0) * 1.35)
    for xi, v in zip(x, missed_edges / 1000.0):
        ax.text(xi, v + max(missed_edges / 1000.0) * 0.03, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5, color="#4a6f9c")
    ax2 = ax.twinx()
    ax2.plot(x, verified, marker="o", color="#d1495b", linewidth=1.6,
             label="多保留文档数（右轴）")
    for xi, v in zip(x, verified):
        ax2.annotate(f"{int(v)}", (xi, v), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=7.5, color="#a5313f")
    ax2.set_ylabel("B 相对 C 多保留的文档数", fontsize=9, color="#a5313f")
    ax2.set_ylim(0, max(verified) * 1.45)
    ax2.tick_params(axis="y", colors="#a5313f", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("(c) 漏边数与多保留文档数不成比例", fontsize=10.5, pad=8)

    handles = [
        Line2D([0], [0], color="#8a8f98", lw=1.2, label="候选边（已检出）"),
        Line2D([0], [0], color="#d1495b", lw=1.4, ls=(0, (3, 2)), label="漏掉的真实边"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3b6fb6",
               markersize=8, label="同色=同一 B 连通块"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="#1f1f1f", markersize=10, label="黑圈=该块保留的最小 ID"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.005))
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(FIGDIR, f"component_classes.{ext}")
        fig.savefig(p, bbox_inches="tight")
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
