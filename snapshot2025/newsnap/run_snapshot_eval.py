#!/usr/bin/env python3
"""Pre-registered evaluation on the CC-MAIN-2025-30 snapshot.

PREREG.md: 5 MinHash seeds {1..5}, tau in {0.75, 0.80}, OFF/A/B/W/C on the same
documents and token stream, cluster-equal-weight and drop-largest variants,
bootstrap intervals.  Rows append to newsnap/runs.jsonl (one per unit), masks to
newsnap/cache/masks/ for the mechanism and order analyses.

Evidence tier of every row: rerun (computed from the snapshot inputs in this run).
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import snap_common as S  # noqa: E402
K = S.K

MASK_DIR = os.path.join(S.CACHE, "masks")


def cluster_variants(n, root, rem, has_nb, rng, n_boot=1000):
    """Cluster-equal-weight and drop-largest metrics over the raw LSH components.

    A component is a connected component of the raw (chain-edge) union-find graph.
    Deleted docs in the component = size - 1 (min-id representative survives).
    """
    size = np.bincount(root, minlength=n)
    comp = (size >= 2)
    # only components containing deleted docs matter; all size>=2 components do
    del_c = np.where(comp, size - 1, 0).astype(np.float64)
    uns = (rem & ~has_nb)
    uns_c = np.bincount(root[uns], minlength=n).astype(np.float64)
    shares = uns_c[comp] / del_c[comp]
    w_del = del_c[comp]
    largest = int(np.argmax(size))
    doc_share = float(uns.sum() / del_c.sum()) if del_c.sum() else None
    ew = float(shares.mean()) if len(shares) else None
    # cluster bootstrap (components weighted equally)
    bs = []
    if len(shares):
        idx = rng.integers(0, len(shares), size=(n_boot, len(shares)))
        bs = shares[idx].mean(axis=1)
    # drop-largest: remove every doc of the largest component
    keep = np.ones(n, bool)
    keep[root == largest] = False
    dl = float(uns[keep].sum() / np.maximum((rem & keep).sum(), 1))
    # weighted-by-deletion share is the document-level share of deletions
    return dict(
        n_components=int(comp.sum()),
        largest_component_docs=int(size[largest]),
        largest_component_deleted=int(del_c[largest]),
        largest_component_share_deleted=float(del_c[largest] / max(del_c.sum(), 1)),
        cluster_equal_weight_unsupported_share=ew,
        cluster_equal_weight_ci95=[float(np.quantile(bs, 0.025)),
                                   float(np.quantile(bs, 0.975))] if len(bs) else None,
        drop_largest_unsupported_share_del=dl,
        unsupported_share_del_docweighted=doc_share,
        cluster_share_p50=float(np.median(shares)) if len(shares) else None,
        template_density=float(np.isin(root, np.flatnonzero(size >= 10)).sum()
                               / max(int((size[root] > 1).sum()), 1)),
        component_size_p50=float(np.median(size[comp])) if comp.any() else None,
        component_size_p90=float(np.quantile(size[comp], 0.9)) if comp.any() else None,
        component_size_max=int(size[largest]),
    )


def doc_bootstrap_share_of_input(n, uns, n_boot=1000):
    """Exact document bootstrap for a binary indicator, as multinomial counts."""
    k = int(uns.sum())
    p = k / n
    draws = np.random.multinomial(n, [p, 1 - p], size=n_boot)[:, 0] / n
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def doc_bootstrap_share_of_del(rem, uns, n_boot=1000):
    idx = np.flatnonzero(rem)
    flags = uns[idx].astype(np.float64)
    if len(flags) == 0:
        return None, None
    rng = np.random.default_rng(20261119)
    bs = []
    for _ in range(n_boot):
        s = flags[rng.integers(0, len(flags), len(flags))].mean()
        bs.append(s)
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def main():
    os.makedirs(MASK_DIR, exist_ok=True)
    K.set_threads(K.THREADS)
    off, flat, nwords = S.load_snapshot()
    n = len(off) - 1
    valid = np.diff(off) > 0
    print(f"snapshot n={n} valid={valid.sum()} shingles={int(off[-1])}", flush=True)
    base = dict(evidence_tier="rerun", threads=K.THREADS, workload="snap", snapshot="CC-MAIN-2025-30",
                tokenizer="datatrove-spacy", hw=K.hw_facts())

    truth = {}
    exact_rem = {}
    for tau in S.TAUS:
        t0 = time.time()
        ti, tj, jac = S.compute_truth(tau, off=off, flat=flat)
        hnb = K.has_true_neighbour(n, ti, tj)
        truth[tau] = (ti, tj, hnb)
        ex_root = K.uf_minid(n, ti, tj)
        exact_rem[tau] = (ex_root != np.arange(n)) & valid
        print(f"truth tau={tau}: {len(ti)} edges, exact deleted {int(exact_rem[tau].sum())} "
              f"in {time.time()-t0:.0f}s", flush=True)

    done = S.done_keys(S.RUNS)
    rng = np.random.default_rng(20261119)

    for seed in S.SEEDS:
        key0 = f"snap|t0.8|A|minid|s{seed}"
        if key0 in done:
            print("seed", seed, "already done", flush=True)
            continue
        t0 = time.time()
        sig, _ = K.signatures_cached(off, flat, "snap", seed)
        t_sig = time.time() - t0
        t0 = time.time()
        Kb = K.band_keys(sig)
        del sig
        t_band = time.time() - t0
        print(f"s{seed}: sig {t_sig:.1f}s band {t_band:.1f}s", flush=True)

        # ---- A: chain edges
        t0 = time.time()
        ei, ej, bstats = K.chain_edges(Kb, valid)
        tA = time.time() - t0
        print(f"s{seed}: chain edges {len(ei)} largest bucket {bstats['largest_bucket']}", flush=True)
        for rep in ("minid", "dt"):
            root = K.uf_minid(n, ei, ej) if rep == "minid" else K.uf_datatrove(n, ei, ej)
            rem = (root != np.arange(n)) & valid
            t_uf = 0.0
            for tau in S.TAUS:
                key = f"snap|t{tau}|A|{rep}|s{seed}"
                if key in done:
                    continue
                ti, tj, hnb = truth[tau]
                m = K.method_metrics(n, rem, ti, tj, hnb)
                es = K.edge_stats(ti, tj, ei, ej, n)
                ex = exact_rem[tau]
                m.update(raw_missed=int((ex & ~rem).sum()),
                         hijacked_raw=int((rem & hnb & ~ex).sum()),
                         exact_deleted=int(ex.sum()),
                         doc_recall=float((ex & rem).sum() / max(ex.sum(), 1)))
                cv = cluster_variants(n, root, rem, hnb, rng) if rep == "minid" else {}
                ci_del = doc_bootstrap_share_of_del(rem, rem & ~hnb)
                ci_in = doc_bootstrap_share_of_input(n, rem & ~hnb)
                row = dict(base, key=key, tau=tau, method="A", representative=rep,
                           hash_seed=seed, n=n, valid_docs=int(valid.sum()),
                           n_candidate_edges=int(len(ei)),
                           largest_bucket=bstats["largest_bucket"],
                           bucket_pairs_sum=bstats["bucket_pairs_sum"],
                           truth_edges=int(len(ti)), t_signature_s=round(t_sig, 2),
                           t_band_s=round(t_band, 2), t_method_s=round(tA, 2),
                           unsupported_share_del_ci95=ci_del,
                           unsupported_share_input_ci95=ci_in,
                           **m, **es)
                if rep == "minid":
                    row.update(cv)
                S.append_row(S.RUNS, row)
            if rep == "minid":
                np.savez(os.path.join(MASK_DIR, f"raw_s{seed}.npz"), root=root, rem=rem)
        del ei, ej

        # ---- B / W: all bucket pairs + exact verification
        t0 = time.time()
        pi, pj, bstats = K.all_bucket_pairs(Kb, valid, n)
        t_cand = time.time() - t0
        t0 = time.time()
        jac = K.L.jaccard_pairs(off, flat, pi, pj)
        t_ver = time.time() - t0
        print(f"s{seed}: B pairs {len(pi)} cand {t_cand:.1f}s verify {t_ver:.1f}s "
              f"largest bucket {bstats['largest_bucket']}", flush=True)
        for tau in S.TAUS:
            ti, tj, hnb = truth[tau]
            ver = jac >= tau - 1e-12
            ei, ej = pi[ver], pj[ver]
            es = K.edge_stats(ti, tj, ei, ej, n)
            rem_b = (K.uf_minid(n, ei, ej) != np.arange(n)) & valid
            for rep in ("minid", "dt"):
                key = f"snap|t{tau}|B|{rep}|s{seed}"
                if key in done:
                    continue
                root = K.uf_minid(n, ei, ej) if rep == "minid" else K.uf_datatrove(n, ei, ej)
                rem = rem_b if rep == "minid" else (root != np.arange(n)) & valid
                m = K.method_metrics(n, rem, ti, tj, hnb)
                ex = exact_rem[tau]
                m.update(verified_missed=int((ex & ~rem).sum()),
                         hijacked_verified=int((rem & hnb & ~ex).sum()),
                         exact_deleted=int(ex.sum()),
                         doc_recall=float((ex & rem).sum() / max(ex.sum(), 1)))
                cv = cluster_variants(n, root, rem, hnb, rng) if rep == "minid" else {}
                row = dict(base, key=key, tau=tau, method="B", representative=rep,
                           hash_seed=seed, n=n, valid_docs=int(valid.sum()),
                           n_candidate_pairs=int(len(pi)), n_verified_edges=int(ver.sum()),
                           largest_bucket=bstats["largest_bucket"],
                           bucket_pairs_sum=bstats["bucket_pairs_sum"],
                           truth_edges=int(len(ti)), t_signature_s=round(t_sig, 2),
                           t_band_s=round(t_band, 2), t_candidates_s=round(t_cand, 2),
                           t_verify_s=round(t_ver, 2), **m, **es)
                if rep == "minid":
                    row.update(cv)
                    np.savez(os.path.join(MASK_DIR, f"ver_s{seed}_t{tau}.npz"),
                             rem=rem, verified_edges_i=ei, verified_edges_j=ej)
                S.append_row(S.RUNS, row)
            key = f"snap|t{tau}|W|minid|s{seed}"
            if key not in done:
                t0 = time.time()
                m, rem = K.witness_from_verified(n, ei, ej, valid, ti, tj, hnb)
                t_mis = time.time() - t0
                ex = exact_rem[tau]
                m.update(witness_missed=int((ex & ~rem).sum()),
                         verified_missed=int((ex & ~rem_b).sum()),
                         exact_deleted=int(ex.sum()),
                         doc_recall=float((ex & rem).sum() / max(ex.sum(), 1)))
                row = dict(base, key=key, tau=tau, method="W", representative="minid",
                           hash_seed=seed, n=n, valid_docs=int(valid.sum()),
                           n_verified_edges=int(ver.sum()), truth_edges=int(len(ti)),
                           t_signature_s=round(t_sig, 2), t_mis_s=round(t_mis, 2), **m, **es)
                np.savez(os.path.join(MASK_DIR, f"wit_s{seed}_t{tau}.npz"), rem=rem)
                S.append_row(S.RUNS, row)
        del pi, pj, jac, ei, ej
        del Kb

    # ---- C: exact self-join (seed-free).  C's edge set IS the truth edge set:
    # both are the exact Jaccard >= tau graph from the same frozen ranks, so the
    # truth join is reused instead of paying for a second identical join.
    for tau in S.TAUS:
        ti, tj, hnb = truth[tau]
        for rep in ("minid", "dt"):
            key = f"snap|t{tau}|C|{rep}|s0"
            if key in done:
                continue
            ei, ej = ti, tj
            tC = 0.0
            root = K.uf_minid(n, ei, ej) if rep == "minid" else K.uf_datatrove(n, ei, ej)
            rem = (root != np.arange(n)) & valid
            m = K.method_metrics(n, rem, ti, tj, hnb)
            m.update(exact_deleted=int(exact_rem[tau].sum()),
                     doc_recall=float((exact_rem[tau] & rem).sum()
                                      / max(exact_rem[tau].sum(), 1)))
            cv = cluster_variants(n, root, rem, hnb, rng) if rep == "minid" else {}
            row = dict(base, key=key, tau=tau, method="C", representative=rep, hash_seed=None,
                       n=n, valid_docs=int(valid.sum()), n_exact_edges=int(len(ei)),
                       truth_edges=int(len(ti)), t_method_s=round(tC, 2), **m)
            if rep == "minid":
                row.update(cv)
                np.savez(os.path.join(MASK_DIR, f"exact_t{tau}.npz"), rem=rem)
            S.append_row(S.RUNS, row)
        print(f"C tau={tau}: {len(ei)} edges", flush=True)


if __name__ == "__main__":
    main()
