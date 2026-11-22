#!/usr/bin/env python3
"""Emit the method-comparison and variance-mechanism tables from the new run.

Consumes results/main_comparison.json (main_compare.py + aggregate_main.py) and
results/seed_variance.json (seed_variance.py analyze).  Writes
  tab1_main.tex      four corpora x four thresholds, five seeds, four methods
  tab6_methods.tex   method quality at tau=0.8: precision, residual pairs, witness,
                     content coverage, cost
  tab7_stages.tex    per-stage error attribution at tau=0.8
  tab8_variance.tex  100-seed dispersion, cluster/document weighting, design effect
and merges the entries into paper/tables/manifest.json (entries for other tables are
kept as they are).
"""
import hashlib
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RES = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/paper/tables")
LABEL = {"c4": "C4", "fw": "FineWeb", "mix": "Mix", "wet": "WET snap."}
TAUS = ["0.6", "0.7", "0.8", "0.9"]


def fmt(n):
    return f"{n:,.0f}"


def f2(x):
    return "--" if x is None else f"{x:.2f}"


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def cells(rec, key):
    v = rec.get(key)
    return "--" if not v else v["mean"]


def main():
    d = json.load(open(os.path.join(RES, "main_comparison.json")))
    v = json.load(open(os.path.join(RES, "seed_variance.json")))
    os.makedirs(OUT, exist_ok=True)
    corpora = [c for c in ("c4", "fw", "mix", "wet") if c in d["corpora"]]

    # ---------------------------------------------------------------- table 1
    rows = []
    for c in corpora:
        for tau in TAUS:
            m = d["corpora"][c]["methods"][tau]
            ex = m["exact_removed"]["mean"]
            rows.append(
                f"{LABEL[c]} & {float(tau):.1f} & {fmt(m['raw_removed']['mean'])} & "
                f"{fmt(m['ver_removed']['mean'])} & {fmt(m['wit_removed']['mean'])} & "
                f"{fmt(ex)} & {f3(m['pair_recall']['mean'])} & "
                f"{100 * cells(m, 'raw_wrong_isolated') / cells(m, 'raw_removed'):.1f} & "
                f"{100 * cells(m, 'raw_wrong_isolated') / d['corpora'][c]['n']:.3f} \\\\")
    tab1 = r"""\begin{table*}[t]
\centering
\caption{Deletion decisions of four methods on four corpora, datatrove's default
14$\times$8 banding, mean over five hash seeds. \emph{LSH}: raw MinHash-LSH deletions;
\emph{+ver.}: exact verification of candidate edges; \emph{Wit.}: witness rule (every
deleted document keeps a retained neighbour at $J\geq\tau$); \emph{Exact}: exact
self-join. \emph{recall}: pair recall of the LSH candidate edges; \emph{unsup.}:
deleted documents without a neighbour at exact $J\geq\tau$, as a share of LSH deletions
and of the input. Exact deletions are seed-independent.}
\label{tab:main}
\small
\setlength{\tabcolsep}{3.0pt}
\begin{tabular}{@{}llrrrrrrr@{}}
\toprule
Corpus & $\tau$ & LSH del. & +ver. & Wit. & Exact & recall & unsup.\,\% & unsup.\,\% in. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(os.path.join(OUT, "tab1_main.tex"), "w").write(tab1)

    # ---------------------------------------------------------------- table 6
    rows = []
    for c in corpora:
        m = d["corpora"][c]["methods"]["0.8"]
        for name, rk, mk, wkey in (
                ("LSH", "raw_removed", "raw", "raw"),
                ("+verify", "ver_removed", "ver", "ver"),
                ("Witness", "wit_removed", "wit", "wit"),
                ("Exact", "exact_removed", "exact", "exact")):
            rem = m[rk]["mean"]
            unsup = m["raw_wrong_isolated"]["mean"] if rk == "raw_removed" else 0.0
            miss = (m["exact_removed"]["mean"] * (1 - m[f"{mk}_deletion_recall"]["mean"])
                    if rk != "exact_removed" else 0.0)
            resid = m[f"{mk}_residual_pairs"]["mean"]
            wsh = m["witness_share"][wkey]["mean"]
            tok = m[f"{mk}_content_tokens_frac"]["mean"]
            uni = m[f"{mk}_content_uniq_frac"]["mean"]
            prec = 1 - unsup / rem if rem else None
            t = m["time"]["lsh_total_s"]["mean"] if rk != "exact_removed" else m["time"]["exact_total_s"]
            mem = m["time"]["peak_rss_mb"]["mean"]
            rows.append(
                f"{LABEL[c]} & {name} & {fmt(rem)} & {f3(prec)} & {fmt(max(miss, 0))} & "
                f"{fmt(resid)} & {f3(wsh)} & {100 * uni:.2f} & {t:.1f} \\\\")
        if c != corpora[-1]:
            rows.append(r"\addlinespace")
    tab6 = r"""\begin{table*}[t]
\centering
\caption{Method comparison at $\tau=0.8$, mean over five seeds. \emph{del.}:
deletions; \emph{prec.}: share of deletions with a neighbour at exact $J\geq\tau$;
\emph{missed}: exact deletions the method keeps; \emph{resid.}: true pairs left in the
retained set; \emph{witness}: share of deleted documents with a retained neighbour at
$J\geq\tau$; \emph{uniq.\,\%}: unique ($df=1$) shingles retained as a share of all
unique shingles; \emph{time}: seconds, signature/banding/verification for the LSH
variants (verification included in +verify and Witness) against global ranking plus
join for Exact; peak resident memory is shared across methods of one corpus.}
\label{tab:methods}
\small
\setlength{\tabcolsep}{3.0pt}
\begin{tabular}{@{}llrrrrrrr@{}}
\toprule
Corpus & Method & del. & prec. & missed & resid. & witness & uniq.\,\% & time (s) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(os.path.join(OUT, "tab6_methods.tex"), "w").write(tab6)

    # ---------------------------------------------------------------- table 7
    rows = []
    for c in corpora:
        m = d["corpora"][c]["methods"]["0.8"]
        rows.append(
            f"{LABEL[c]} & {fmt(cells(m, 'n_truth_pairs'))} & {fmt(m['exact_removed']['mean'])} & "
            f"{fmt(cells(m, 'n_cand_pairs'))} & {f3(m['pair_recall']['mean'])} & "
            f"{f3(m['cand_precision']['mean'])} & {fmt(cells(m, 'raw_wrong_isolated'))} & "
            f"{fmt(cells(m, 'raw_wrong_hijack'))} & {fmt(m['ver_missed']['mean'])} \\\\")
    tab7 = r"""\begin{table}[t]
\centering
\caption{Per-stage error attribution at $\tau=0.8$, mean over five seeds.
\emph{truth}: exact pairs; \emph{cand.}: candidate pairs emitted by the banding;
\emph{recall}/\emph{prec.}: candidate pair recall against the truth and share of
candidates with $J\geq\tau$; \emph{unsup.}/\emph{hij.}: raw deletions with no true
neighbour, and raw deletions that have one but that the exact join keeps as its
component representative; \emph{missed}: exact deletions kept by verification, the
residual error after verification.}
\label{tab:stages}
\small
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
Corpus & truth & exact del. & cand. & recall & prec. & unsup. & hij. & missed \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(OUT, "tab7_stages.tex"), "w").write(tab7)

    # ---------------------------------------------------------------- table 8
    dec = json.load(open(os.path.join(RES, "variance_decomposition.json")))["cells"]
    rows = []
    labels = {"c4": "C4", "fw": "FineWeb", "mix": "Mix", "wet": "WET"}
    want = [("c4", "14x8", "0.7"), ("c4", "14x8", "0.8"), ("c4", "14x8", "0.9"),
            ("c4", "20x5", "0.8"), ("c4", "10x10", "0.8"),
            ("fw", "14x8", "0.8"), ("mix", "14x8", "0.8"), ("wet", "14x8", "0.8")]
    for corpus, cfg, tau in want:
        key = f"{corpus}|{cfg}|{tau}"
        if key not in v["cells"]:
            continue
        x = v["cells"][key]
        d = dec.get(corpus) if (tau == "0.8" and cfg == "14x8") else None
        od = (f"{x['overdispersion']:.1f}$\\times$" if x["overdispersion"] else "n/a")
        bound = (f"{d['bound_over_observed']:.2f}" if d and d["tau"] == 0.8 else "n/a")
        cov = (f"{100 * d['covariance_var_share']:.0f}" if d and d["tau"] == 0.8 else "n/a")
        rows.append(
            f"{labels[corpus]} & {cfg} & {x['n_seeds']} & {float(tau):.1f} & "
            f"{x['pair_recall']['mean']:.4f} & {x['pair_recall']['std']:.4f} & "
            f"{x['pair_recall']['p05']:.4f} & {od} & {bound} & {cov} & "
            f"{x['cluster_weighted']['std']:.4f} \\\\")
    tab8 = r"""\begin{table*}[t]
\centering
\caption{Seed-to-seed dispersion of pair recall, independently rerun. \emph{mean},
\emph{sd}, \emph{p05}: over the seeds in the third column; \emph{overdisp.}: sd divided
by the binomial standard error at the number of true pairs; \emph{bound}: the
all-or-nothing standard deviation from exact-component edge counts (every edge of a
component sharing one hit), divided by the observed sd; \emph{cov.\,\%}: share of the
observed variance carried by covariance between components (the rest is
within-component); both are reported for $\tau=0.8$; \emph{cluster sd}: standard
deviation of the cluster-weighted recall.}
\label{tab:variance}
\small
\setlength{\tabcolsep}{3.0pt}
\begin{tabular}{@{}lllrrrrrrrr@{}}
\toprule
Corpus & Config & seeds & $\tau$ & mean & sd & p05 & overdisp. & bound & cov.\,\% & cluster sd \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(os.path.join(OUT, "tab8_variance.tex"), "w").write(tab8)

    # ------------------------------------------------------------- manifest
    entries = [
        dict(id="tab:main", path="tab1_main.tex", label="tab:main",
             generator="output/dedupjoin-pvldb/code/make_tables_step.py",
             data_sources=["output/dedupjoin-pvldb/results/main_comparison.json"],
             display="mean_only", n_seeds=5),
        dict(id="tab:methods", path="tab6_methods.tex", label="tab:methods",
             generator="output/dedupjoin-pvldb/code/make_tables_step.py",
             data_sources=["output/dedupjoin-pvldb/results/main_comparison.json"],
             display="mean_only", n_seeds=5),
        dict(id="tab:stages", path="tab7_stages.tex", label="tab:stages",
             generator="output/dedupjoin-pvldb/code/make_tables_step.py",
             data_sources=["output/dedupjoin-pvldb/results/main_comparison.json"],
             display="mean_only", n_seeds=5),
        dict(id="tab:variance", path="tab8_variance.tex", label="tab:variance",
             generator="output/dedupjoin-pvldb/code/make_tables_step.py",
             data_sources=["output/dedupjoin-pvldb/results/seed_variance.json"],
             display="mean_only", n_seeds=100),
    ]
    mpath = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(mpath))
    keep = [e for e in manifest["tables"] if e["id"] not in {x["id"] for x in entries}]
    for e in entries:
        with open(os.path.join(OUT, e["path"]), "rb") as fh:
            e["sha256"] = hashlib.sha256(fh.read()).hexdigest()
    manifest["tables"] = keep + entries
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
