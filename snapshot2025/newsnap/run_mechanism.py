#!/usr/bin/env python3
"""Component mechanism and W order-sensitivity on the new snapshot.

Mechanism question: are deletions lost because a few missing LSH edges split a
true duplicate component, or are many missing edges harmless because redundant
paths keep the component connected?  For every exact-truth component (tau fixed)
we count (a) truth edges absent from the raw candidate graph, (b) how many raw
components it spans, and (c) how many of its exact deletions path A / path B
fail to delete.  Components are bucketed by the number of missing edges.

W order experiments: the witness rule is a greedy maximal independent set over
documents in increasing id order; the same graph processed in other fixed orders
gives different (still maximal) witness sets.  Run on the exact truth graph and
on the B verified graph, 5 fixed orders.

Rows append to newsnap/analysis/mechanism.jsonl and newsnap/analysis/order.jsonl.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import snap_common as S  # noqa: E402
K = S.K

OUT = os.path.join(S.SNAP, "analysis")
MECH = os.path.join(OUT, "mechanism.jsonl")
ORDER = os.path.join(OUT, "order.jsonl")
MASK_DIR = os.path.join(S.CACHE, "masks")
BUCKETS = [(1, 1), (2, 3), (4, 9), (10, 10 ** 9)]
ORDER_KINDS = ["id", "rev", "rnd7", "rnd11", "rnd13"]


def enc(i, j, n):
    lo = np.minimum(i, j).astype(np.int64)
    hi = np.maximum(i, j).astype(np.int64)
    return lo * n + hi


def order_perm(n, kind):
    if kind == "id":
        return np.arange(n)
    if kind == "rev":
        return np.arange(n)[::-1]
    return np.random.default_rng(int(kind[3:]) + 1000).permutation(n)


def mis_in_order(n, ei, ej, order):
    """Greedy MIS with documents visited in `order` (order[0] first)."""
    pos = np.empty(n, np.int64)
    pos[order] = np.arange(n)
    kept_pos = K.greedy_mis(n, pos[ei], pos[ej])
    kept = np.empty(n, bool)
    kept[order] = kept_pos
    return kept


def component_table(n, ti, tj, ex_root, graph_root, rem, cand_codes):
    """Per exact component: missing candidate edges, graph connectivity, losses.

    `graph_root` is the union-find root of the method graph under study (raw
    chain graph for A, verified graph for B); `rem` its deletion mask.
    """
    hnb = np.zeros(n, bool)
    hnb[ti] = True
    hnb[tj] = True
    docs = np.flatnonzero(hnb)
    uniq, inv = np.unique(ex_root[docs], return_inverse=True)
    size = np.bincount(inv)
    comp_of = np.full(n, -1, np.int64)
    comp_of[docs] = inv
    m = len(uniq)
    ci, cj = comp_of[ti], comp_of[tj]
    n_true = np.bincount(ci, minlength=m)
    codes = enc(ti, tj, n)
    pos = np.searchsorted(cand_codes, codes)
    pos[pos >= len(cand_codes)] = 0
    present = cand_codes[pos] == codes if len(cand_codes) else np.zeros(len(codes), bool)
    n_present = np.bincount(ci, weights=present.astype(np.float64), minlength=m)
    n_missing = n_true - n_present
    # method-graph components spanned by each exact component
    _, rinv = np.unique(graph_root[docs], return_inverse=True)
    nraw = int(rinv.max()) + 1 if len(rinv) else 0
    cpair = np.unique(inv.astype(np.int64) * max(nraw, 1) + rinv)
    n_raw_comp = np.bincount(cpair // max(nraw, 1), minlength=m).astype(np.int64)
    ex_rem = (ex_root != np.arange(n)) & hnb
    miss = np.zeros(m, np.int64)
    np.add.at(miss, comp_of[ex_rem & ~rem], 1)
    del_docs = np.bincount(comp_of[ex_rem], minlength=m)
    rows = []
    for lo, hi in BUCKETS:
        sel = (n_missing >= lo) & (n_missing <= hi)
        if not sel.any():
            continue
        rows.append(dict(
            missing_edges=f"{lo}+" if hi > 10 ** 8 else (f"{lo}" if lo == hi else f"{lo}-{hi}"),
            n_components=int(sel.sum()),
            connected_share=float((n_raw_comp[sel] == 1).mean()),
            docs=int(size[sel].sum()),
            doc_share=float(size[sel].sum() / size.sum()),
            truth_edges=int(n_true[sel].sum()),
            missing_truth_edges=int(n_missing[sel].sum()),
            missing_edge_share=float(n_missing[sel].sum() / max(n_missing.sum(), 1)),
            deleted_exact=int(del_docs[sel].sum()),
            missed_by_A=int(miss[sel].sum()),
            missed_share_of_all_A_misses=float(miss[sel].sum() / max(miss.sum(), 1)),
            doc_recall_A=float(1.0 - miss[sel].sum() / max(del_docs[sel].sum(), 1)),
            size_p50=float(np.median(size[sel])),
        ))
    # component-level detail for the largest few
    big = np.argsort(size)[::-1][:5]
    top = [dict(size=int(size[b]), truth_edges=int(n_true[b]),
                missing_edges=int(n_missing[b]), raw_components=int(n_raw_comp[b]),
                deleted_exact=int(del_docs[b]), missed_by_A=int(miss[b]))
           for b in big]
    # edge level: missing truth edges whose endpoints stay in one method component
    if present.any() or (~present).any():
        still = graph_root[ti[~present]] == graph_root[tj[~present]]
    else:
        still = np.zeros(0, bool)
    return dict(by_missing_bucket=rows, largest_components=top,
                missing_edges_total=int((~present).sum()),
                missing_edge_true_total=int(len(codes)),
                missing_but_connected_share=float(still.mean()) if len(still) else None,
                n_components=int(m), n_docs_in_components=int(size.sum()),
                n_candidate_edges_unique=int(len(cand_codes)))


def main():
    os.makedirs(OUT, exist_ok=True)
    K.set_threads(K.THREADS)
    off, flat, _ = S.load_snapshot()
    n = len(off) - 1
    valid = np.diff(off) > 0
    print(f"snapshot n={n}", flush=True)
    truth = {}
    for tau in S.TAUS:
        ti, tj, _ = S.compute_truth(tau, off=off, flat=flat)
        truth[tau] = (ti, tj, K.has_true_neighbour(n, ti, tj))
        print(f"truth t{tau}: {len(ti)} edges", flush=True)

    # ---- W on the exact truth graph, fixed orders
    for tau in S.TAUS:
        ti, tj, hnb = truth[tau]
        done = S.done_keys(ORDER)
        base = None
        for kind in ORDER_KINDS:
            key = f"worder|truth|t{tau}|s0|{kind}"
            if key in done:
                continue
            t0 = time.time()
            kept = mis_in_order(n, ti, tj, order_perm(n, kind))
            rem = (~kept) & valid
            m = K.method_metrics(n, rem, ti, tj, hnb)
            if base is None:
                base = rem
            row = dict(key=key, evidence_tier="rerun", graph="truth", tau=tau, seed=0,
                       order=kind, wall_s=round(time.time() - t0, 2),
                       symdiff_vs_id=int((rem ^ base).sum()), **m)
            S.append_row(ORDER, row)
            print(key, "deleted", m["deleted"], "unsup", m["unsupported"],
                  "residual", m["residual_duplicate_pairs"], "symdiff", row["symdiff_vs_id"],
                  flush=True)

    # ---- per seed: A/B graphs, mechanism + W order
    done_m = S.done_keys(MECH)
    done_o = S.done_keys(ORDER)
    for seed in S.SEEDS:
        sig, _ = K.signatures_cached(off, flat, "snap", seed)
        Kb = K.band_keys(sig)
        del sig
        ei, ej, bst = K.chain_edges(Kb, valid)
        raw_root = K.uf_minid(n, ei, ej)
        remA = (raw_root != np.arange(n)) & valid
        pi, pj, _ = K.all_bucket_pairs(Kb, valid, n)
        jac = K.L.jaccard_pairs(off, flat, pi, pj)
        print(f"s{seed}: chain {len(ei)} candpairs {len(pi)}", flush=True)
        for tau in S.TAUS:
            ti, tj, hnb = truth[tau]
            ver = jac >= tau - 1e-12
            vei, vej = pi[ver], pj[ver]
            ver_root = K.uf_minid(n, vei, vej)
            remB = (ver_root != np.arange(n)) & valid
            key = f"mech|t{tau}|s{seed}"
            if key not in done_m:
                ex_root = K.uf_minid(n, ti, tj)
                ctab = component_table(n, ti, tj, ex_root, raw_root, remA,
                                       np.unique(enc(ei, ej, n)))
                ctabB = component_table(n, ti, tj, ex_root, ver_root, remB,
                                        np.unique(enc(vei, vej, n)))
                row = dict(key=key, evidence_tier="rerun", tau=tau, seed=seed,
                           largest_bucket=int(bst["largest_bucket"]),
                           bucket_pairs_sum=int(bst["bucket_pairs_sum"]),
                           raw=ctab, verified=ctabB,
                           deleted_A=int(remA.sum()), deleted_B=int(remB.sum()),
                           truth_edges=int(len(ti)))
                S.append_row(MECH, row)
                print(key, "missing_but_connected", ctab["missing_but_connected_share"],
                      flush=True)
            # W orders on the B verified graph
            base = None
            for kind in ORDER_KINDS:
                okey = f"worder|B|t{tau}|s{seed}|{kind}"
                if okey in done_o:
                    continue
                t0 = time.time()
                kept = mis_in_order(n, vei, vej, order_perm(n, kind))
                rem = (~kept) & valid
                m = K.method_metrics(n, rem, ti, tj, hnb)
                if base is None:
                    base = rem
                row = dict(key=okey, evidence_tier="rerun", graph="B", tau=tau, seed=seed,
                           order=kind, wall_s=round(time.time() - t0, 2),
                           symdiff_vs_id=int((rem ^ base).sum()), **m)
                S.append_row(ORDER, row)
                print(okey, "deleted", m["deleted"], "unsup", m["unsupported"],
                      "wcov", round(m["witness_coverage"], 4), "symdiff", row["symdiff_vs_id"],
                      flush=True)
        del Kb, ei, ej, pi, pj, jac


if __name__ == "__main__":
    main()
