#!/usr/bin/env python3
"""Aggregate the per-seed method rows into results/main_comparison.json.

One place computes every number the paper's method-comparison section uses: per-corpus
dataset and truth statistics, per-method quality at each threshold (mean +- sd over the
five hash seeds), content coverage, per-stage error attribution, representative-choice
cost, witness coverage, timing and peak memory.
"""
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03"))
import main_compare as MC  # noqa: E402

OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/main")
CORPORA = ["c4", "fw", "mix", "wet"]
TAUS = ["0.6", "0.7", "0.8", "0.9"]


def run_nwords(corpus):
    """The token counts the pipeline itself uses (same array as the deletion masks):
    metadata nwords agree for C4/FineWeb/Mix but differ from the WET extraction column."""
    if corpus in ("c4", "fw"):
        z = np.load(os.path.join(MC.RERUN, f"{corpus}_{MC.CORPORA[corpus]}.npz"))
        return np.asarray(z["nwords"], np.int64)
    z = np.load(os.path.join(WRK, f"{corpus}_{MC.CORPORA[corpus]}.npz"))
    return np.asarray(z["nwords"], np.int64)


def ms(vals):
    v = np.array([x for x in vals if x is not None], np.float64)
    if len(v) == 0:
        return None
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                min=float(v.min()), max=float(v.max()), n=int(len(v)))


def rep_choice(corpus, tau, seed=None):
    """Representative choice: min-id (the pipeline rule) versus longest-document oracle."""
    n = MC.CORPORA[corpus]
    meta = pq.read_table(os.path.join(WRK, f"{corpus}_meta.parquet"),
                         columns=["nwords", "n_unique"]).to_pydict()
    nw = run_nwords(corpus)
    uniq = np.asarray(meta["n_unique"], np.int64)
    t = np.load(os.path.join(WRK, f"{corpus}_truth_t{tau}.npz"))
    root = np.asarray(t["root"])
    size = np.bincount(root, minlength=n)
    docs = np.flatnonzero(size[root] >= 2)
    # group docs by component
    order = np.argsort(root[docs], kind="stable")
    docs = docs[order]
    r = root[docs]
    brk = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    starts = np.r_[brk, len(docs)]
    n_diff = 0
    tok_min = tok_max = 0
    uniq_min = uniq_max = 0
    for i in range(len(brk)):
        mem = docs[starts[i]:starts[i + 1]]
        rep = int(root[mem[0]])
        longest = int(mem[np.argmax(nw[mem])])
        n_diff += rep != longest
        tok_min += int(nw[rep])
        tok_max += int(nw[longest])
        uniq_min += int(uniq[rep])
        uniq_max += int(uniq[longest])
    ncomp = len(brk)
    total_tok = int(nw.sum())
    total_uniq = int(uniq.sum())
    kept_tok = total_tok - int(nw[removed_mask(corpus, tau)].sum())
    return dict(n_components=int(ncomp),
                components_rep_not_longest=int(n_diff),
                share_rep_not_longest=float(n_diff / ncomp) if ncomp else None,
                component_rep_tokens=int(tok_min), component_longest_tokens=int(tok_max),
                token_gain_if_longest=int(tok_max - tok_min),
                token_gain_share_of_corpus=float((tok_max - tok_min) / total_tok) if total_tok else None,
                uniq_gain_if_longest=int(uniq_max - uniq_min),
                kept_tokens=int(kept_tok))


def removed_mask(corpus, tau):
    t = np.load(os.path.join(WRK, f"{corpus}_truth_t{tau}.npz"))
    return np.asarray(t["removed"])


def witness_coverage(corpus, tau, seed):
    """Share of deleted documents with at least one retained exact neighbour."""
    z = np.load(os.path.join(WRK, f"{corpus}_s{seed}_t{tau}_sets.npz"))
    n = int(z["n"])
    t = np.load(os.path.join(WRK, f"{corpus}_truth_t{tau}.npz"))
    ti, tj = np.asarray(t["i"]), np.asarray(t["j"])
    out = {}
    for name in ("raw", "ver", "wit", "exact"):
        rem = np.unpackbits(np.asarray(z[f"rem_{name}"]))[:n].astype(bool)
        deld = np.flatnonzero(rem)
        if len(deld) == 0:
            out[name] = None
            continue
        # a deleted doc is "witnessed" if some truth neighbour is retained
        m = rem[ti] ^ rem[tj]  # edges with exactly one endpoint deleted
        wit = np.zeros(n, bool)
        wit[ti[m & rem[ti]]] = True
        wit[tj[m & rem[tj]]] = True
        out[name] = float(wit[deld].mean())
    return out


def main():
    rows = [json.loads(l) for l in open(os.path.join(OUT, "main_compare_rows.jsonl"))]
    by = {}
    for r in rows:
        by.setdefault(r["corpus"], []).append(r)
    out = {"provenance": "independently rerun (this run)",
           "seeds": sorted({r["seed"] for r in rows}), "corpora": {}}
    for corpus in CORPORA:
        lst = sorted(by.get(corpus, []), key=lambda r: r["seed"])
        if not lst:
            continue
        prep = json.load(open(os.path.join(WRK, f"{corpus}_prepare.json")))
        meta = pq.read_table(os.path.join(WRK, f"{corpus}_meta.parquet"),
                             columns=["nwords", "n_unique", "nshingle"]).to_pydict()
        nw = run_nwords(corpus)
        uniq = np.asarray(meta["n_unique"], np.int64)
        total_tok, total_uniq = int(nw.sum()), int(uniq.sum())
        c = dict(n=prep["n"], n_shingles=prep["n_shingles"], t_rank_s=prep["t_rank_s"],
                 seeds=[r["seed"] for r in lst],
                 total_tokens=total_tok, total_unique_items=total_uniq,
                 truth=prep["truth"], methods={})
        for tau in TAUS:
            m = []
            for r in lst:
                x = dict(r["per_tau"][tau])
                x["pair_recall"] = x["edge_recall"]
                m.append(x)
            rec = {}
            for key in ("pair_recall", "cand_precision", "edge_recall", "raw_removed",
                        "raw_wrong", "raw_wrong_isolated", "raw_wrong_hijack",
                        "raw_wrong_total", "raw_missed", "raw_overlap",
                        "ver_removed", "ver_missed", "ver_wrong", "wit_removed",
                        "wit_missed", "wit_wrong", "exact_removed",
                        "raw_residual_pairs", "ver_residual_pairs", "wit_residual_pairs",
                        "exact_residual_pairs", "raw_deletion_recall", "ver_deletion_recall",
                        "wit_deletion_recall", "cluster_weighted_edge_recall",
                        "n_symdiff_raw_exact", "n_symdiff_ver_exact"):
                rec[key] = ms([x[key] for x in m])
            for key in ("raw_content", "ver_content", "wit_content", "exact_content"):
                rec[key + "_tokens_frac"] = ms([x[key]["tokens"] / total_tok for x in m])
                rec[key + "_uniq_frac"] = ms([x[key]["uniq"] / total_uniq for x in m])
            rec["witness_coverage"] = ms([witness_coverage(corpus, tau, s)[n]
                                          for s in [r["seed"] for r in lst]
                                          for n in ("raw", "ver", "wit", "exact")]) if False else None
            # witness coverage per method, averaged over seeds
            wc = {n: [] for n in ("raw", "ver", "wit", "exact")}
            for r in lst:
                w = witness_coverage(corpus, tau, r["seed"])
                for n in wc:
                    wc[n].append(w[n])
            rec["witness_share"] = {n: ms(v) for n, v in wc.items()}
            rec["n_truth_pairs"] = ms([prep["truth"][tau]["n_pairs"]] * len(m))
            rec["n_cand_pairs"] = ms([r["n_cand_pairs"] for r in lst])
            rec["time"] = dict(
                lsh_total_s=ms([r["t_sig_s"] + r["t_band_s"] + r["t_jaccard_s"] for r in lst]),
                exact_join_s=prep["truth"][tau]["t_join_s"],
                exact_total_s=prep["truth"][tau]["t_join_s"] + prep["t_rank_s"],
                peak_rss_mb=ms([r["peak_rss_mb"] for r in lst]),
            )
            c["methods"][tau] = rec
        c["representative_choice"] = {tau: rep_choice(corpus, tau) for tau in ("0.8",)}
        out["corpora"][corpus] = c
    # paired check if present
    p = os.path.join(OUT, "paired_before_after.json")
    if os.path.exists(p):
        out["paired_before_after"] = json.load(open(p))
    with open(os.path.join(OUT, "main_comparison.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results/main_comparison.json")
    for corpus, c in out["corpora"].items():
        m = c["methods"]["0.8"]
        print(corpus, {k: (round(m[k]["mean"], 3) if m[k] else None)
                       for k in ("pair_recall", "raw_wrong_total", "ver_missed", "exact_removed",
                                 "wit_removed", "raw_deletion_recall")})


if __name__ == "__main__":
    main()
