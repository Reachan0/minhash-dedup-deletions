#!/usr/bin/env python3
"""figures/fig_skew.{pdf,png}: (a) within-bucket pairs per document on every corpus, log scale;
(b) share of A's unsupported deletions against corpus size on the nested corpora.
Inputs: <data>/rung_*.jsonl, <data>/b2_runs.jsonl, <v3 timing.jsonl>.
Palette: dataviz reference categorical slots 1-3 (validated; every mark is directly labelled).
Usage: python make_fig_skew.py <data dir> <v3 timing.jsonl> <out dir>
"""
import glob
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA, V3, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
N_FILTERED = 429221


def jl(p):
    return [json.loads(l) for l in open(p)]


def main():
    bars = []
    names = {"c4": "C4", "fw": "FineWeb", "mix": "Mix", "wet": "WET-2019"}
    seen = {}
    for r in jl(V3):
        w = r.get("workload")
        if r.get("tag") == "main" and r.get("method") == "A" and r.get("bucket_pairs_sum") and w in names and w not in seen:
            seen[w] = r["bucket_pairs_sum"] / r["n"]
    for w in ("c4", "fw", "mix", "wet"):
        bars.append((names[w], seen[w], BLUE))
    ladder = []
    for p in sorted(glob.glob(os.path.join(DATA, "rung_*.jsonl")), key=lambda x: int(x.split("_")[-1][:-6])):
        rows = jl(p)
        n = next(r for r in rows if r["kind"] == "corpus")["n"]
        A = [r for r in rows if r["kind"] == "run" and r["method"] == "A" and r["tau"] == 0.8]
        big = A[0]["largest_bucket"]
        allv = [100 * r["unsupported_share_del"] for r in A]
        outv = [100 * r["unsupported"] / (r["deleted"] - (big - 1)) for r in A]
        ladder.append((n, allv, outv))
        bars.append(("Snapshot-2025" if n < 600000 else f"2025, {n / 1e6:.1f}M", A[0]["bucket_pairs_sum"] / n, ORANGE))
    runs = jl(os.path.join(DATA, "b2_runs.jsonl"))
    lab = {"base": "2025, $\\geq$50 words", "exact_1000": "+ $10^3$ copies", "exact_10000": "+ $10^4$ copies",
           "exact_100000": "+ $10^5$ copies", "near_1000": "+ $10^3$ near-copies",
           "near_10000": "+ $10^4$ near-copies", "near_100000": "+ $10^5$ near-copies"}
    for v in lab:
        s = [r for r in runs if r["variant"] == v and r["kind"] == "stream"][-1]
        bars.append((lab[v], s["enumerated_pairs"] / N_FILTERED, AQUA))

    plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "serif"})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1.45, 1]))
    a = ax[0]
    ys = list(range(len(bars)))[::-1]
    a.barh(ys, [b[1] for b in bars], color=[b[2] for b in bars], height=0.7, edgecolor="white", linewidth=1)
    a.set_xscale("log")
    a.set_xlim(0.002, 5e6)
    a.set_yticks(ys)
    a.set_yticklabels([b[0] for b in bars], fontsize=6.5)
    for y, b in zip(ys, bars):
        v = b[1]
        a.text(v * 1.25, y, f"{v:,.0f}" if v >= 100 else (f"{v:.1f}" if v >= 1 else f"{v:.2g}"), va="center", fontsize=6, color=INK)
    a.set_xlabel("within-bucket pairs per document (log scale)")
    a.set_title("(a) verification load", fontsize=8, color=INK, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, AQUA)]
    a.legend(handles, ["original corpora", "2025 crawl, nested", "filtered, with injections"],
             fontsize=6, frameon=False, loc="upper right")

    b = ax[1]
    ns = [l[0] for l in ladder]
    for idx, col, txt in ((2, ORANGE, "outside the template group"), (1, BLUE, "all deletions")):
        mid = [st.mean(l[idx]) for l in ladder]
        err = [[m - min(l[idx]) for m, l in zip(mid, ladder)], [max(l[idx]) - m for m, l in zip(mid, ladder)]]
        b.errorbar(ns, mid, yerr=err, color=col, marker="o", ms=4, lw=2, capsize=2, label=txt)
        for x, m in zip(ns, mid):
            b.annotate(f"{m:.1f}", (x, m), textcoords="offset points", xytext=(0, -11 if idx == 2 else 6),
                       ha="center", fontsize=6, color=INK)
    b.axhline(35.9, color=MUTED, lw=1, ls="--")
    b.annotate("WET-2019 (175k documents)", (1.4e6, 35.9), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=6, color=MUTED)
    b.set_xscale("log")
    b.set_xticks(ns)
    b.set_xticklabels([f"{n / 1e6:.2g}M" for n in ns])
    b.minorticks_off()
    b.set_ylim(0, 42)
    b.set_xlabel("documents (2025 crawl, nested)")
    b.set_ylabel("A's deletions without support (%)")
    b.set_title("(b) size", fontsize=8, color=INK, loc="left")
    b.legend(fontsize=6, frameon=False, loc="center right")
    for x in ax:
        x.grid(axis="x" if x is a else "y", color=GRID, lw=0.6)
        x.set_axisbelow(True)
        for s in ("top", "right"):
            x.spines[s].set_visible(False)
    fig.tight_layout(w_pad=1.5)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "fig_skew.pdf"))
    fig.savefig(os.path.join(OUT, "fig_skew.png"), dpi=220)
    print("written", len(bars), "bars,", len(ladder), "sizes")


if __name__ == "__main__":
    main()
