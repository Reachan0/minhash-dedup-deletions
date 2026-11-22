"""Emit the LaTeX tables of Draft 0 from results/results.json.

Every number in the paper's tables is written by this script; no value is
typed into the manuscript by hand.  Run:
    python3 output/dedupjoin-pvldb/code/make_tables.py
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RES = os.path.join(ROOT, "output/dedupjoin-pvldb/results/results.json")
OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/paper/tables")


def fmt(n):
    return f"{n:,.0f}"


def main():
    d = json.load(open(RES))
    os.makedirs(OUT, exist_ok=True)

    # ---------------------------------------------------------------- table 1
    rows = []
    for c in d["main_cells"]:
        if not c:
            continue
        corpus = {"c4": "C4 700k", "fw": "FineWeb 1M", "mix": "Mix 1M"}[c["corpus"]]
        rows.append(
            f"{corpus} & {c['tau']:.1f} & {fmt(c['lsh_removed']['mean'])} & "
            f"{fmt(c['lshv_removed']['mean'])} & {fmt(c['exact_removed']['mean'])} & "
            f"{c['lshv_gap_to_exact']['mean']:.1f} & {c['recall']['mean']:.3f} & "
            f"{c['raw_pair_precision']['mean']:.3f} & "
            f"{100 * c['misdelete_frac']['mean']:.1f} & "
            f"{c['misdelete_per_input_pct']['mean']:.3f} \\\\")
    tab1 = r"""\begin{table*}[t]
\centering
\caption{Deletion decisions at the datatrove default 14$\times$8 banding, mean over three hash seeds. \emph{LSH del.}: raw MinHash-LSH deletions; \emph{+ver.}: deletions after exact verification of candidate edges; \emph{Exact}: deletions of the exact self-join; \emph{Gap}: exact minus verified (Eq.~\ref{eq:subset}); \emph{recall}/\emph{prec.}: pair-level recall and raw candidate precision; \emph{unsup.}: deleted documents with no neighbour at exact $J\geq\tau$, as a fraction of deletions and of the input.}
\label{tab:main}
\small
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{@{}llrrrrrrrr@{}}
\toprule
Corpus & $\tau$ & LSH del. & +ver. & Exact & Gap & recall & prec. & unsup.\,\% & unsup.\,\% in. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(os.path.join(OUT, "tab1_main.tex"), "w").write(tab1)

    # ---------------------------------------------------------------- table 2
    want = [("c4", "14x8"), ("c4", "20x5"), ("c4", "10x10"),
            ("fw", "14x8"), ("mix", "14x8")]
    rows = []
    labels = {"c4": "C4", "fw": "FineWeb", "mix": "Mix"}
    for corpus, cfg in want:
        for x in d["dispersion"]:
            if x["corpus"] == corpus and x["config"] == cfg and x["family"] == "dt":
                od = (f"{x['overdispersion']:.1f}$\\times$"
                      if x["overdispersion"] is not None else "n/a")
                rows.append(
                    f"{labels[corpus]} & {cfg} & {x['tau']:.1f} & "
                    f"{x['recall_mean']:.4f} & {x['recall_min']:.4f} & "
                    f"{x['recall_p05']:.4f} & {od} \\\\")
    tab2 = r"""\begin{table}[t]
\centering
\caption{Pair recall over 500 hash seeds against the exact edge set (dt hash family). \emph{Mean}: mean recall; \emph{worst}: minimum over seeds; \emph{p05}: 5th percentile; \emph{overdisp.}: observed standard deviation divided by the binomial standard error at the measured number of true pairs.}
\label{tab:seeds}
\small
\setlength{\tabcolsep}{3.4pt}
\begin{tabular}{@{}lllrrrr@{}}
\toprule
Corpus & Config & $\tau$ & Mean & Worst & p05 & overdisp. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(OUT, "tab2_seeds.tex"), "w").write(tab2)

    # ---------------------------------------------------------------- table 3
    lm = d["downstream_lm"]["arms"]
    evals = ["val_clean", "d1_lsh_only_removed", "d2_exact_only_removed"]
    rows = []
    for arm in ["exact", "lsh", "nodedup"]:
        if arm not in lm:
            continue
        e = lm[arm]["eval"]
        label = {"exact": "Exact join", "lsh": "Raw LSH", "nodedup": "No dedup"}[arm]
        cells = []
        for k in evals:
            cells.append(f"{e[k]['mean_loss']:.3f}\\,$\\pm$\\,{e[k]['std_loss']:.3f}")
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    tab3 = r"""% qa-allow-mixed-statistics: the no-dedup arm has a single seed, so its rows carry no +/- spread.
\begin{table}[t]
\centering
\caption{Downstream probe: validation loss (mean $\pm$ standard deviation over three seeds; the no-dedup arm has one seed) of a 51M-parameter model trained for 150M tokens on three variants of the same C4 sample. \emph{Clean}: held-out clean documents; \emph{$D_1$}: documents removed by raw LSH but not by exact verification; \emph{$D_2$}: documents removed only by the exact join. $D_2$ contains 1{,}024 evaluation tokens and is reported for completeness only.}
\label{tab:downstream}
\small
\setlength{\tabcolsep}{3.4pt}
\begin{tabular}{@{}lrrr@{}}
\toprule
Training corpus & Clean & $D_1$ & $D_2$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(OUT, "tab3_downstream.tex"), "w").write(tab3)

    # ---------------------------------------------------------------- table 4
    ss = json.load(open(os.path.join(ROOT, "output/dedupjoin-pvldb/results/seed_sensitivity.json")))
    labels = {"c4_700000": "C4 700k", "fw_1000000": "FineWeb 1M"}
    rows = []
    for cell in ("c4_700000", "fw_1000000"):
        for tau in ("0.8", "0.7"):
            d = ss["cells"][cell].get(tau)
            if not d:
                continue
            for key, name in (("shared_h117", "shared 117"), ("standalone_h112", "standalone 112")):
                x = d.get(key)
                if not x:
                    continue
                rows.append(
                    f"{labels[cell]} & {float(tau):.1f} & {name} & "
                    f"{fmt(x['mean_lsh_removed'])} & {fmt(x['mean_lshv'])} & "
                    f"{fmt(x['mean_exact'])} & {x['mean_gap']:.1f} "
                    f"({x['min_gap']}--{x['max_gap']}) & {x['mean_recall']:.4f} & "
                    f"{100 * x['share_of_deleted_mean_of_seeds']:.1f} \\\\")
    tab4 = r"""\begin{table*}[t]
\centering
\caption{Signature-draw sensitivity at the default 14$\times$8 banding, mean over three hash seeds. \emph{Shared 117}: the main grid's design, one 117-permutation draw per corpus whose first 112 columns are banded (all configurations share the permutations); \emph{standalone 112}: a fresh 112-permutation draw, as a standalone datatrove run performs it. \emph{Unsup.}: unsupported share of deletions (\%) with the same definition as Table~\ref{tab:main}.}
\label{tab:standalone}
\small
\setlength{\tabcolsep}{3.0pt}
\begin{tabular}{@{}llrrrrrrr@{}}
\toprule
Corpus & $\tau$ & draw & LSH del. & +ver. & Exact & Gap (range) & recall & unsup.\,\% \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(os.path.join(OUT, "tab4_signature_draw.tex"), "w").write(tab4)

    # ---------------------------------------------------------------- table 5
    q = json.load(open(os.path.join(
        ROOT, "output/dedupjoin-pvldb/results/quality_metrics_audited.json")))["cells"]
    rows = []
    for tag, label in [("c4_700000", "C4 700k"), ("fw_1000000", "FineWeb 1M")]:
        for tau in sorted(q[tag], reverse=True):
            r = q[tag][tau]
            rows.append(
                f"{label} & {float(tau):.1f} & {r['candidate_edge_recall']:.3f} & "
                f"{r['residual_duplicate_pairs_raw_lsh']} & "
                f"{r['residual_duplicate_pairs_lsh_verify']} & "
                f"{r['direct_witness_coverage_raw_lsh']:.3f} & "
                f"{r['direct_witness_coverage_lsh_verify']:.3f} \\\\")
    tab5 = r"""\begin{table}[t]
\centering
\caption{Three final-corpus metrics on the same exact pair list (audited cell, seed 1). (a)~candidate-edge recall; (b)~residual duplicate pairs ($J\geq\tau$) with both endpoints retained, for raw LSH and LSH with verification (the exact join has zero by construction); (c)~fraction of deleted documents with a direct $J\geq\tau$ edge to a retained document.}
\label{tab:quality}
\small
\setlength{\tabcolsep}{3.4pt}
\begin{tabular}{@{}llrrrrr@{}}
\toprule
Cell & $\tau$ & (a) recall & \multicolumn{2}{c}{(b) residual pairs} & \multicolumn{2}{c}{(c) witness coverage} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}
 & & & LSH & +ver. & LSH & +ver. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(OUT, "tab5_quality.tex"), "w").write(tab5)

    # ------------------------------------------------------------- manifest
    import hashlib

    entries = [
        {
            "id": "tab:main",
            "path": "tab1_main.tex",
            "label": "tab:main",
            "generator": "output/dedupjoin-pvldb/code/make_tables.py",
            "data_sources": ["output/dedupjoin-pvldb/results/results.json"],
            "display": "mean_only",
            "n_seeds": 3,
        },
        {
            "id": "tab:seeds",
            "path": "tab2_seeds.tex",
            "label": "tab:seeds",
            "generator": "output/dedupjoin-pvldb/code/make_tables.py",
            "data_sources": ["output/dedupjoin-pvldb/results/results.json"],
            "display": "mean_only",
            "n_seeds": 500,
        },
        {
            "id": "tab:standalone",
            "path": "tab4_signature_draw.tex",
            "label": "tab:standalone",
            "generator": "output/dedupjoin-pvldb/code/make_tables.py",
            "data_sources": ["output/dedupjoin-pvldb/results/seed_sensitivity.json"],
            "display": "mean_only",
            "n_seeds": 3,
        },
        {
            "id": "tab:quality",
            "path": "tab5_quality.tex",
            "label": "tab:quality",
            "generator": "output/dedupjoin-pvldb/code/make_tables.py",
            "data_sources": ["output/dedupjoin-pvldb/results/quality_metrics_audited.json"],
            "display": "raw",
            "n_seeds": 1,
        },
        {
            "id": "tab:downstream",
            "path": "tab3_downstream.tex",
            "label": "tab:downstream",
            "generator": "output/dedupjoin-pvldb/code/make_tables.py",
            "data_sources": ["output/dedupjoin-pvldb/results/results.json"],
            "display": "mean_std",
            "std_ddof": 1,
            "n_seeds": 3,
        },
    ]
    for e in entries:
        with open(os.path.join(OUT, e["path"]), "rb") as fh:
            e["sha256"] = hashlib.sha256(fh.read()).hexdigest()
    manifest = {"schema_version": 1, "tables": entries}
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
