#!/usr/bin/env python3
"""Emit the scale-ladder table and the selection guide from results/scale_ladder_summary.json.

  tab9_ladder.tex   six prefix rungs and five regime samples: cost by stage, peak
                    resident memory, and deletion quality at tau=0.8
  tab10_guide.tex   which method for which corpus size / length / duplication regime,
                    each recommendation tied to a measured number

The selection rules encoded here are also written to results/selection_guide.json so the
paper text can only copy them.
"""
import hashlib
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RES = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/paper/tables")
MANIFEST = os.path.join(OUT, "manifest.json")

LABEL = {
    "prefix:100000": "100k prefix",
    "prefix:300000": "300k prefix",
    "prefix:1000000": "1M prefix",
    "prefix:2000000": "2M prefix",
    "prefix:3000000": "3M prefix",
    "prefix:4000000": "4M prefix",
    "strata:uniform": "100k uniform",
    "strata:L1_lt50": "100k short ($<$50 w)",
    "strata:L5_ge1500": "100k long ($\\geq$1500 w)",
    "strata:D4_ge16": "high-dup (all 37.7k)",
    "strata:D1_isolated": "100k isolated",
}


def fnum(n):
    return f"{n:,}"


def main():
    L, R = chr(123), chr(125)  # braces for the LaTeX exponent, kept out of literals
    s = json.load(open(os.path.join(RES, "scale_ladder_summary.json")))
    recs = {r["workload_id"]: r for r in s["ladder"] + s["strata"]}
    rows = []
    ids = ["prefix:100000", "prefix:300000", "prefix:1000000", "prefix:2000000",
           "prefix:3000000", "prefix:4000000", "strata:uniform", "strata:L1_lt50",
           "strata:L5_ge1500", "strata:D4_ge16", "strata:D1_isolated"]
    for wid in ids:
        r = recs.get(wid)
        if not r:
            continue
        c = r["costs"]
        a = r.get("at_0.8", {})
        rows.append(
            f"{LABEL[wid]} & {fnum(r['n_docs'])} & {r['n_shingles']/1e6:.1f} & "
            f"{100*r['dup_density_14x8']:.2f} & {c['lsh_raw_s']:.1f} & "
            f"{c['lsh_verify_s']:.1f} & {c['exact_s']:.1f} & "
            f"{r['lsh_peak_mb']/1024:.2f} & {r['exact_peak_mb']/1024:.2f} & "
            f"{100*(a.get('raw_unsupported_share') or 0):.1f} & {a.get('ver_missed', '--')} \\\\")
    tab9 = r"""\begin{table*}[t]
\centering
\caption{Scale ladder at $\tau=0.8$, one protocol and 16 threads. Prefix rows are the
first $N$ documents of the pool in write order (FineWeb, C4, raw WET); regime rows are
100{,}000-document samples (all 37{,}713 members for the high-duplication stratum).
\emph{LSH}: signature, banding, candidate generation, bucket expansion and raw clustering.
\emph{+verify}: the same plus exact verification of every candidate edge.
\emph{exact}: global rank construction, prefix-filter join and clustering.
Peaks are resident-set high-water marks of the method's own stages.
\emph{unsup.}: share of raw LSH deletions with no exact neighbour at $\tau=0.8$;
\emph{resid.}: exact deletions that verification leaves undone (residual).}
\label{tab:ladder}
\small
\setlength{\tabcolsep}{2.0pt}
\begin{tabular}{@{}lrrrrrrrrrr@{}}
\toprule
Workload & docs & shingles (M) & dup (\%) & LSH (s) & +verify (s) & exact (s)
 & \multicolumn{2}{c}{peak GiB} & unsup.\ (\%) & resid. \\
\cmidrule(lr){8-9}
 & & & & & & & LSH & exact & & \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    with open(os.path.join(OUT, "tab9_ladder.tex"), "w") as f:
        f.write(tab9)

    # ---------------------------------------------------------------- guide
    def g(wid, key, default=None):
        r = recs.get(wid)
        if not r:
            return default
        if key in r:
            return r[key]
        return r.get("at_0.8", {}).get(key, default)

    big = max(s["ladder"], key=lambda r: r["n_docs"])
    ext = s["extrapolation"]
    pen = chr(123) + "31" + chr(125)  # LaTeX exponent braces, kept out of literals

    def num(wid, key, default=0):
        v = g(wid, key, default)
        return default if v is None else v

    def pct_of_exact(wid, key):
        ex = num(wid, "exact_removed", 1) or 1
        return 100 * num(wid, key, 0) / ex

    top = f"{big['n_docs']/1e6:.0f}M"
    guide = [
        dict(regime=f"Any corpus with an audit, $\\leq${top} documents",
             use="exact reference",
             why=f"end-to-end {big['costs']['exact_s']:.0f}\\,s at {top} documents with a "
                 f"{big['total_peak_rss_mb']/1024:.0f}\\,GiB peak; the only method whose deletions "
                 f"are exact by construction"),
        dict(regime="Corpus beyond five million documents",
             use="exact join with 64-bit or blocked ranks",
             why=f"the 31-bit rank index fails at about "
                 f"{ext['rank_index_limit_docs']/1e6:.1f}M documents (extrapolated: $V$ is "
                 f"{ext['rank_index_v_per_shingle']:.3f}$\\times$ the shingle count and rank construction "
                 f"asserts $V<2^{pen}-1$); memory alone would allow about "
                 f"{ext['projected_docs_at_64GiB']/1e6:.1f}M"),
        dict(regime="Production pipeline, $\\geq$1M docs",
             use="LSH + edge verification",
             why=f"cuts decisions that differ from the exact join from "
                 f"{num('prefix:1000000', 'symdiff_raw_exact'):,} documents "
                 f"({num('prefix:1000000', 'symdiff_raw_exact') / max(num('prefix:1000000', 'exact_removed'), 1):.1f}$\\times$ "
                 f"the {num('prefix:1000000', 'exact_removed')} exact deletions) to "
                 f"{num('prefix:1000000', 'symdiff_ver_exact')}, for one Jaccard check per candidate edge"),
        dict(regime="High duplication (D4)", needs=["strata:D4_ge16"],
             use="LSH + edge verification, not the witness rule",
             why=f"raw LSH leaves {100*num('strata:D4_ge16','raw_unsupported_share'):.1f}\\% of its "
                 f"deletions without a neighbour; verification leaves "
                 f"{num('strata:D4_ge16','ver_missed'):,} exact deletions undone "
                 f"({pct_of_exact('strata:D4_ge16','ver_missed'):.1f}\\%) but the witness rule gives up "
                 f"{num('strata:D4_ge16','wit_missed'):,} "
                 f"({pct_of_exact('strata:D4_ge16','wit_missed'):.1f}\\%): in components this large, "
                 f"keeping a neighbour behind every deletion retains many documents"),
        dict(regime="Short documents ($<$50 words)", needs=["strata:L1_lt50"],
             use="verification strongly recommended; witness costs coverage",
             why=f"unsupported share {100*num('strata:L1_lt50','raw_unsupported_share'):.1f}\\% and "
                 f"{num('strata:L1_lt50','n_truth_pairs'):,} true pairs from 100k documents: few shingles "
                 f"make the banding decision noisy, verification leaves "
                 f"{num('strata:L1_lt50','ver_missed'):,} exact deletion"
                 f"{'' if num('strata:L1_lt50','ver_missed') == 1 else 's'} undone and the witness rule "
                 f"{num('strata:L1_lt50','wit_missed'):,}"),
        dict(regime="Long documents ($\\geq$1500 words)", needs=["strata:L5_ge1500"],
             use="raw LSH closer to exact; verify if decisions are audited",
             why=f"unsupported share {100*num('strata:L5_ge1500','raw_unsupported_share'):.1f}\\%, the lowest "
                 f"regime measured; verification leaves {num('strata:L5_ge1500','ver_missed'):,} exact "
                 f"deletions undone in the 100k sample, at "
                 f"{num('strata:L5_ge1500','n_shingles')/1e6:.0f}M shingles"),
        dict(regime="No prior audit of deletions", needs=["strata:uniform", "strata:D4_ge16",
                                                           "strata:D1_isolated"],
             use="audit before deciding",
             why=f"there is no default to trust: the unsupported share is "
                 f"{100*num('strata:uniform','raw_unsupported_share'):.0f}\\% on a uniform sample, "
                 f"{100*num('strata:D4_ge16','raw_unsupported_share'):.0f}\\% on the high-duplication "
                 f"stratum and {100*num('prefix:1000000','raw_unsupported_share'):.0f}\\% on a FineWeb "
                 f"prefix, while on 100k isolated documents raw LSH deletes "
                 f"{num('strata:D1_isolated','raw_removed')} where exact truth deletes "
                 f"{num('strata:D1_isolated','exact_removed')}"),
    ]
    guide = [g for g in guide if all(w in recs for w in g.pop("needs", []))]
    with open(os.path.join(RES, "selection_guide.json"), "w") as f:
        json.dump(guide, f, indent=1)
    lines = [f"{x['regime']} & {x['use']} & {x['why']} \\\\" for x in guide]
    tab10 = r"""\begin{table*}[t]
\centering
\caption{Selection guide. Each recommendation is tied to a measured number in
Table~\ref{tab:ladder}; the rules are generated by the same script that builds both
tables.}
\label{tab:guide}
\footnotesize
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{@{}p{4.2cm}p{3.6cm}p{8.2cm}@{}}
\toprule
Regime & Use & Why (measured) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    with open(os.path.join(OUT, "tab10_guide.tex"), "w") as f:
        f.write(tab10)

    # ---------------------------------------------------------------- manifest
    man = json.load(open(MANIFEST))
    entries = {e["id"]: e for e in man["tables"]}
    for tid, path in (("tab:ladder", "tab9_ladder.tex"), ("tab:guide", "tab10_guide.tex")):
        p = os.path.join(OUT, path)
        entries[tid] = {
            "id": tid, "path": path, "label": tid,
            "generator": "output/dedupjoin-pvldb/code/make_tables_scale.py",
            "data_sources": ["output/dedupjoin-pvldb/results/scale_ladder_summary.json"],
            "display": "raw", "n_seeds": 1, "threads": 16,
            "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
        }
    man["tables"] = list(entries.values())
    with open(MANIFEST, "w") as f:
        json.dump(man, f, indent=1)
    print("wrote tab9_ladder.tex, tab10_guide.tex", "extrapolation:",
          {k: v for k, v in ext.items() if k != "points"})


if __name__ == "__main__":
    main()
