#!/usr/bin/env python3
"""Figures and tables for v5 Sections 7-8 from the result files collected under data/.

Inputs (data/): rung_<N>.jsonl and b2_<variant>.jsonl (b1_rung.py output), b2_runs.jsonl
(b2_runs.py output), operator_E.jsonl, post_reveal.jsonl; v3 timing.jsonl for the older corpora.
Outputs: figures/fig_skew.pdf and .png, tables/tab_cost.tex, tables/tab_skew.tex.
Palette: dataviz reference categorical slots 1-3 (validated: all checks pass; aqua needs direct
labels, which every bar and line carries).
Usage: python make_v5_figs.py <paper dir>
"""
import glob
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
P = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "out")
for _d in ("tables", "figures"):
    os.makedirs(os.path.join(P, _d), exist_ok=True)
D = os.path.join(HERE, "..", "results", "paper")
V3 = os.path.join(HERE, "..", "results", "original_corpora", "timing.jsonl")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def corpus_rows(name):
    rows = jl(os.path.join(D, f"{name}.jsonl"))
    head = next((r for r in rows if r.get("kind") == "corpus"), None)
    runs = [r for r in rows if r.get("kind") == "run"]
    stab = [r for r in rows if r.get("kind") == "stability"]
    return head, runs, stab


def a_share(runs, tau=0.8, meth="A"):
    v = [r["unsupported_share_del"] for r in runs if r["method"] == meth and r["tau"] == tau]
    return (st.mean(v), min(v), max(v)) if v else None


def pairs_per_doc(runs):
    r = next((r for r in runs if r["tau"] == 0.8), None)
    return (r["bucket_pairs_sum"] / r["n"], r["largest_bucket"]) if r else None


def main():
    bars = []  # (label, pairs/doc, colour)
    seen = set()
    for r in jl(V3):
        w = r.get("workload")
        if r.get("method") == "A" and r.get("mode") == "incremental" and w not in seen and r.get("bucket_pairs_sum"):
            seen.add(w)
            bars.append(({"c4": "C4", "fw": "FineWeb", "mix": "Mix", "wet": "WET-2019"}[w],
                         r["bucket_pairs_sum"] / r["n"], BLUE))
    order = {"C4": 0, "FineWeb": 1, "Mix": 2, "WET-2019": 3}
    bars.sort(key=lambda b: order[b[0]])
    ladder = []
    for p in sorted(glob.glob(os.path.join(D, "rung_*.jsonl")), key=lambda x: int(x.split("_")[-1].split(".")[0])):
        name = os.path.basename(p)[:-6]
        head, runs, stab = corpus_rows(name)
        if not runs:
            continue
        ppd, big = pairs_per_doc(runs)
        lab = "Snapshot-2025" if head["n"] < 600000 else f"{head['n'] / 1e6:.0f}M"
        bars.append((lab, ppd, BLUE))
        ladder.append((head["n"], a_share(runs), stab, big, ppd))
    head, runs, _ = corpus_rows("b2_base")
    if runs:
        bars.append(("$\\geq$50 words", pairs_per_doc(runs)[0], BLUE))
    inj = {"exact": [], "near": []}
    for kind, col in (("exact", ORANGE), ("near", AQUA)):
        for m in (1000, 10000, 100000):
            head, runs, stab = corpus_rows(f"b2_{kind}_{m}")
            if runs:
                ppd, big = pairs_per_doc(runs)
                bars.append((f"{'copies' if kind == 'exact' else 'near-dup.'} $10^{{{len(str(m)) - 1}}}$", ppd, col))
                inj[kind].append((m, a_share(runs), stab))

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "serif"})
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw=dict(width_ratios=[1.7, 1, 1]))
    a = ax[0]
    xs = range(len(bars))
    a.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.72, edgecolor="white", linewidth=1)
    a.set_yscale("log")
    a.set_xticks(list(xs))
    a.set_xticklabels([b[0] for b in bars], rotation=60, ha="right", fontsize=6.5)
    a.set_ylabel("within-bucket pairs per document")
    for x, b in zip(xs, bars):
        a.text(x, b[1] * 1.25, f"{b[1]:.2g}", ha="center", va="bottom", fontsize=5.5, color=INK)
    a.set_title("(a) bucket-pair load", fontsize=8, color=INK, loc="left")

    b = ax[1]
    if ladder:
        ns = [l[0] for l in ladder]
        mid = [100 * l[1][0] for l in ladder]
        err = [[100 * (l[1][0] - l[1][1]) for l in ladder], [100 * (l[1][2] - l[1][0]) for l in ladder]]
        b.errorbar(ns, mid, yerr=err, color=BLUE, marker="o", ms=4, lw=2, capsize=2)
        for n_, m_ in zip(ns, mid):
            b.annotate(f"{m_:.1f}", (n_, m_), textcoords="offset points", xytext=(0, 6), ha="center",
                       fontsize=6, color=INK)
        b.set_xscale("log")
    b.set_xlabel("documents (one crawl, nested)")
    b.set_ylabel("A: unsupported deletions (%)")
    b.set_title("(b) size", fontsize=8, color=INK, loc="left")

    c = ax[2]
    for kind, col, mk, lab in (("exact", ORANGE, "s", "exact copies"), ("near", AQUA, "^", "near-duplicates")):
        pts = inj[kind]
        if pts:
            c.plot([p[0] for p in pts], [100 * p[1][0] for p in pts], color=col, marker=mk, ms=4, lw=2)
            c.annotate(lab, (pts[-1][0], 100 * pts[-1][1][0]), textcoords="offset points", xytext=(-4, 6),
                       ha="right", fontsize=6.5, color=INK)
    base_head, base_runs, _ = corpus_rows("b2_base")
    if base_runs:
        c.axhline(100 * a_share(base_runs)[0], color=MUTED, lw=1, ls="--")
        c.annotate("no injection", (1000, 100 * a_share(base_runs)[0]), textcoords="offset points",
                   xytext=(0, -9), fontsize=6, color=MUTED)
    c.set_xscale("log")
    c.set_xlabel("injected documents $m$ (fixed size)")
    c.set_ylabel("A: unsupported deletions (%)")
    c.set_title("(c) skew", fontsize=8, color=INK, loc="left")
    for x in ax:
        x.grid(axis="y", color=GRID, lw=0.6)
        x.set_axisbelow(True)
        for s in ("top", "right"):
            x.spines[s].set_visible(False)
    fig.tight_layout(w_pad=1.2)
    os.makedirs(os.path.join(P, "figures"), exist_ok=True)
    fig.savefig(os.path.join(P, "figures", "fig_skew.pdf"))
    fig.savefig(os.path.join(P, "figures", "fig_skew.png"), dpi=200)
    json.dump(dict(bars=[(b[0], b[1]) for b in bars], ladder=[(l[0], l[1], l[3], l[4]) for l in ladder],
                   injections={k: [(p[0], p[1]) for p in v] for k, v in inj.items()}),
              open(os.path.join(P, "figures", "fig_skew.json"), "w"), indent=1)
    print("figure written", len(bars), "bars,", len(ladder), "rungs")


if __name__ == "__main__":
    main()


def fmt_time(r):
    if not r:
        return "--"
    if r.get("status") == "timeout":
        return f"$>${r['limit_s'] // 60:.0f}\\,min"
    if r.get("status") != "ok":
        return "fails"
    return f"{r['t_method_s']:.1f}\\,s / {r['peak_rss_gib']:.1f}\\,GiB"


def tables():
    runs = jl(os.path.join(D, "b2_runs.jsonl"))
    get = lambda v, k, q=None: next((r for r in runs if r["variant"] == v and r["kind"] == k
                                     and (q is None or r.get("q") == q)), None)
    lines = []
    for v, lab in (("base", "Snapshot-2025, $\\geq$50 words"),
                   ("exact_1000", "+ $10^3$ exact copies"), ("exact_10000", "+ $10^4$ exact copies"),
                   ("exact_100000", "+ $10^5$ exact copies"), ("near_1000", "+ $10^3$ near-duplicates"),
                   ("near_10000", "+ $10^4$ near-duplicates"), ("near_100000", "+ $10^5$ near-duplicates")):
        head, rr, _ = corpus_rows(f"b2_{v}")
        ppd = pairs_per_doc(rr) if rr else None
        s, l, c = get(v, "stream"), get(v, "literal"), get(v, "cap", 1000)
        lines.append(" & ".join([
            lab, f"{ppd[1]:,}" if ppd else "--", f"{ppd[0]:.3g}" if ppd else "--", fmt_time(l), fmt_time(s),
            f"{s['verified_pairs']:,}" if s and s.get("verified_pairs") is not None else "--",
            f"{c['extra_retained']} / {c['unchecked_U']:,}" if c and c.get("extra_retained") is not None else "--"]) + " \\\\")
    body = "\n".join(lines)
    tex = r"""% generated by scripts/make_v5_figs.py from data/b2_runs.jsonl and data/b2_*.jsonl
\begin{table*}[t]
\centering
\caption{Cost of verification on the filtered Snapshot-2025 (429{,}221 documents) with injected documents, seed 1, $\tau=0.8$, cached shingles. \emph{Materialised} lists every within-bucket pair (24\,GiB limit); \emph{streaming} is the operator of Section~\ref{sec:streaming}, time after signatures. \emph{Cap} is capped B with $q=1000$: extra documents retained relative to B, and the bound $U$ (unchecked pairs).}
\label{tab:cost}
\small
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Corpus & Largest bucket & Pairs/doc & Materialised B & Streaming B & Verified pairs & Cap: extra / $U$ \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(os.path.join(P, "tables", "tab_cost.tex"), "w").write(tex)
    print("tab_cost written")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "tables":
    tables()
