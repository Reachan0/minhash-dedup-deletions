#!/usr/bin/env python3
"""Pair recall vs document recall across the pool corpora and the new snapshot.

Inputs (all produced by the same code):
  newsnap/analysis/cells.jsonl   c4, fw, mix, wet and the size-matched c4sub/fwsub
  newsnap/runs.jsonl             the 2025-30 snapshot cells (A/B min-id rows)

Outputs:
  newsnap/analysis/pair_vs_doc.csv / .md   per (workload, tau) medians over seeds
  newsnap/analysis/pair_vs_doc.png         three-panel figure (English text)
  newsnap/analysis/pair_vs_doc_stats.json  correlations and threshold statistics
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import snap_common as S  # noqa: E402

OUT = os.path.join(S.SNAP, "analysis")
CELLS = os.path.join(OUT, "cells.jsonl")
NAME_ORDER = ["c4", "fw", "mix", "wet", "c4sub", "fwsub", "snap"]


def load():
    rows = []
    if os.path.exists(CELLS):
        for line in open(CELLS):
            r = json.loads(line)
            rows.append(dict(source="cells", workload=r["name"], n=r["n"], tau=r["tau"],
                             seed=r["seed"],
                             pair_A=r.get("pair_recall_A_chain"),
                             pair_B=r.get("pair_recall_B_verified"),
                             doc_A=r.get("doc_recall_A"), doc_B=r.get("doc_recall_B"),
                             template_density=r.get("template_density"),
                             largest_share=r.get("largest_component_share_deleted"),
                             comp_p90=r.get("component_size_p90"),
                             comp_max=r.get("component_size_max")))
    if os.path.exists(S.RUNS):
        for line in open(S.RUNS):
            r = json.loads(line)
            m = r.get("method")
            if r.get("workload") != "snap" or r.get("representative") != "minid":
                continue
            if m not in ("A", "B") or r.get("key", "").count("|") != 4:
                continue
            rows.append(dict(source="runs", workload="snap", n=r["n"], tau=r["tau"],
                             seed=r.get("hash_seed"),
                             pair_A=r.get("pair_recall") if m == "A" else None,
                             pair_B=r.get("pair_recall") if m == "B" else None,
                             doc_A=r.get("doc_recall") if m == "A" else None,
                             doc_B=r.get("doc_recall") if m == "B" else None,
                             template_density=r.get("template_density"),
                             largest_share=r.get("largest_component_share_deleted"),
                             comp_p90=r.get("component_size_p90"),
                             comp_max=r.get("component_size_max")))
    return rows


def median(vals):
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def aggregate(rows):
    cells = {}
    for r in rows:
        cells.setdefault((r["workload"], r["tau"], r["n"]), []).append(r)
    out = []
    for (w, tau, n), rs in cells.items():
        # pair/doc for A come from A rows only, B from B rows only
        ra = [r for r in rs if r["pair_A"] is not None or r["doc_A"] is not None]
        rb = [r for r in rs if r["pair_B"] is not None or r["doc_B"] is not None]
        rec = dict(workload=w, n=n, tau=tau,
                   seeds=len({r["seed"] for r in rs if r["seed"] is not None}),
                   pair_A=median([r["pair_A"] for r in ra]),
                   doc_A=median([r["doc_A"] for r in ra]),
                   pair_B=median([r["pair_B"] for r in rb]),
                   doc_B=median([r["doc_B"] for r in rb]),
                   template_density=median([r["template_density"] for r in ra]),
                   largest_share=median([r["largest_share"] for r in ra]),
                   comp_p90=median([r["comp_p90"] for r in ra]),
                   comp_max=median([r["comp_max"] for r in ra]))
        if rec["pair_A"] is not None and rec["doc_A"] is not None:
            rec["gap_A"] = rec["doc_A"] - rec["pair_A"]
        else:
            rec["gap_A"] = None
        if rec["pair_B"] is not None and rec["doc_B"] is not None:
            rec["gap_B"] = rec["doc_B"] - rec["pair_B"]
        else:
            rec["gap_B"] = None
        out.append(rec)
    out.sort(key=lambda r: (NAME_ORDER.index(r["workload"]) if r["workload"] in NAME_ORDER else 9,
                            r["tau"]))
    return out


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    os.makedirs(OUT, exist_ok=True)
    agg = aggregate(load())
    with open(os.path.join(OUT, "pair_vs_doc.csv"), "w") as f:
        cols = ["workload", "n", "tau", "seeds", "pair_A", "doc_A", "gap_A", "pair_B",
                "doc_B", "gap_B", "template_density", "largest_share", "comp_p90", "comp_max"]
        f.write(",".join(cols) + "\n")
        for r in agg:
            f.write(",".join("" if r.get(c) is None else
                             (f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]))
                             for c in cols) + "\n")
    lines = ["| workload | n | tau | pair recall A | doc recall A | gap A | "
             "pair recall B | doc recall B | gap B | template density | largest component "
             "share of deletions | component p90 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in agg:
        def f(v, d=3):
            return "—" if v is None else (f"{v:.{d}f}" if v < 100 else str(int(v)))
        lines.append(f"| {r['workload']} | {r['n']:,} | {r['tau']:.2f} | {f(r['pair_A'])} | "
                     f"{f(r['doc_A'])} | {f(r['gap_A'])} | {f(r['pair_B'])} | "
                     f"{f(r['doc_B'])} | {f(r['gap_B'])} | {f(r['template_density'])} | "
                     f"{f(r['largest_share'])} | {f(r['comp_p90'], 1)} |")
    md = "\n".join(lines)
    with open(os.path.join(OUT, "pair_vs_doc.md"), "w") as f:
        f.write(md + "\n")

    # statistics: does the gap grow with template density / largest-cluster share?
    gA = [r for r in agg if r["gap_A"] is not None and r["template_density"] is not None]
    gB = [r for r in agg if r["gap_B"] is not None and r["template_density"] is not None]
    stats = dict(
        n_cells=len(agg),
        spearman_gapA_vs_template=spearman([r["gap_A"] for r in gA],
                                          [r["template_density"] for r in gA]),
        spearman_gapB_vs_template=spearman([r["gap_B"] for r in gB],
                                          [r["template_density"] for r in gB]),
        spearman_gapA_vs_largest=spearman([r["gap_A"] for r in gA],
                                          [r["largest_share"] for r in gA]),
        spearman_gapB_vs_largest=spearman([r["gap_B"] for r in gB],
                                          [r["largest_share"] for r in gB]),
        max_gap_A=max((r["gap_A"] for r in gA), default=None),
        max_gap_B=max((r["gap_B"] for r in gB), default=None),
        cells_gap_gt_010=[dict(workload=r["workload"], tau=r["tau"], gap=r["gap_A"],
                               template=r["template_density"])
                          for r in gA if abs(r["gap_A"]) > 0.10],
        cells_gap_gt_005=[dict(workload=r["workload"], tau=r["tau"], gap=r["gap_A"])
                          for r in gA if abs(r["gap_A"]) > 0.05],
    )
    json.dump(stats, open(os.path.join(OUT, "pair_vs_doc_stats.json"), "w"), indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print("matplotlib unavailable:", e, flush=True)
        print(md)
        print(json.dumps(stats, indent=1))
        return

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    styles = {"A": dict(marker="o", color="#2f6db3", label="path A (chain edges)"),
              "B": dict(marker="s", color="#c0504d", label="path B (verified edges)")}
    for key, meth in (("pair_A", "A"), ("pair_B", "B")):
        xs = [r[key] for r in agg if r[key] is not None and r["doc_" + meth] is not None]
        ys = [r["doc_" + meth] for r in agg if r[key] is not None and r["doc_" + meth] is not None]
        ax[0].scatter(xs, ys, s=46, alpha=0.85, **styles[meth])
        for r in agg:
            if r[key] is not None and r["doc_" + meth] is not None and r["workload"] == "snap":
                ax[0].annotate("snap", (r[key], r["doc_" + meth]), fontsize=7,
                               xytext=(4, -8), textcoords="offset points")
    lo = min(0.5, min([r["pair_A"] for r in agg if r["pair_A"] is not None] + [1.0]))
    ax[0].plot([lo, 1], [lo, 1], ls="--", lw=0.8, color="0.5")
    ax[0].set_xlabel("pair recall (share of exact duplicate pairs found)")
    ax[0].set_ylabel("document recall (share of exact deletions made)")
    ax[0].set_title("(a) pair vs document recall")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].grid(alpha=0.3)

    for gap, meth in (("gap_A", "A"), ("gap_B", "B")):
        pts = [r for r in agg if r[gap] is not None and r["template_density"] is not None]
        ax[1].scatter([r["template_density"] for r in pts], [r[gap] for r in pts],
                      s=46, alpha=0.85, **styles[meth])
    ax[1].axhline(0.10, ls="--", lw=0.8, color="0.4")
    ax[1].axhline(-0.10, ls="--", lw=0.8, color="0.4")
    ax[1].axhline(0.0, lw=0.8, color="0.3")
    ax[1].set_xlabel("template density (share of docs in LSH components of $\\geq$10 docs)")
    ax[1].set_ylabel("document recall $-$ pair recall")
    ax[1].set_title("(b) where the two metrics disagree")
    ax[1].grid(alpha=0.3)

    pts = [r for r in agg if r["gap_A"] is not None and r["largest_share"] is not None]
    ax[2].scatter([r["largest_share"] for r in pts], [r["gap_A"] for r in pts],
                  s=46, alpha=0.85, **styles["A"])
    ax[2].axhline(0.0, lw=0.8, color="0.3")
    ax[2].set_xlabel("largest raw-LSH component's share of deletions")
    ax[2].set_ylabel("document recall $-$ pair recall (path A)")
    ax[2].set_title("(c) single-cluster concentration")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pair_vs_doc.png"), dpi=160)
    fig.savefig(os.path.join(OUT, "pair_vs_doc.pdf"))
    print(md)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
