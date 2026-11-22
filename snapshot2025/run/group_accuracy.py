#!/usr/bin/env python3
"""A/B/W/C quality on the snapshot, exactly, via the duplicate-collapsed graph.

Per seed s in {1..5} (C once, seed-free):
  * signatures/bands are computed once on the 431,116 distinct shingle sets and
    expanded to the 531,597 documents (identical sets have identical signatures);
  * A runs its literal chain-edge + union-find on the documents;
  * B/W verify the group-level candidate pairs and lift components / greedy MIS
    to documents with multiplicities;
  * C is the exact collapsed graph (see collapsed_truth.py).

Rows append to newsnap/runs.jsonl; masks to newsnap/cache/masks/.
Every metric is document-level (531,597 docs, 530,603 valid).
"""
import json
import os
import time

import numpy as np

import collapsed_lib as CL

S, K = CL.S, CL.K
TAUS = [0.8, 0.75]
SEEDS = [1, 2, 3, 4, 5]
MASK = os.path.join(S.CACHE, "masks")


def main():
    os.makedirs(MASK, exist_ok=True)
    K.set_threads(K.THREADS)
    z = CL.load_collapsed()
    off_u, flat_u = z["off_u"], z["flat_u"]
    group, m, rep, nsh = z["group_of_doc"], z["group_size"], z["group_rep"], z["group_nshingle"]
    n = len(group)
    G = len(m)
    valid = nsh[group] > 0
    valid_g = nsh > 0
    print(f"docs={n} valid={valid.sum()} groups={G} valid_groups={valid_g.sum()}", flush=True)
    base = dict(evidence_tier="rerun", threads=K.THREADS, workload="snap",
                snapshot="CC-MAIN-2025-30", tokenizer="datatrove-spacy",
                hw=K.hw_facts(), variant="dup_collapsed_exact")

    truth, state = {}, {}
    for tau in TAUS:
        ti, tj, jac = CL.truth_u(tau)
        truth[tau] = (ti, tj)
        st = CL.exact_state(z, ti, tj)
        state[tau] = st
        print(f"truth tau={tau}: {len(ti)} group edges, exact deleted {int(st['rem'].sum())}",
              flush=True)

    # ---- C (seed-free, once)
    for tau in TAUS:
        ti, tj = truth[tau]
        st = state[tau]
        met = CL.lift_metrics(st["rem"], z, ti, tj, st["has_nb_doc"], n, valid)
        met.update(exact_deleted=int(st["rem"].sum()),
                   doc_recall=float(st["rem"].sum() / max(st["rem"].sum(), 1)),
                   n_exact_group_edges=int(len(ti)))
        row = dict(base, key=f"snap|t{tau}|C|minid|s0", tau=tau, method="C",
                   representative="minid", hash_seed=None, n=n,
                   valid_docs=int(valid.sum()), **met)
        S.append_row(S.RUNS, row)
        np.savez(os.path.join(MASK, f"exact_t{tau}.npz"), rem=st["rem"])

    for seed in SEEDS:
        t0 = time.time()
        Kb_g = CL.group_band_keys(z, seed)
        t_sig = time.time() - t0
        Kb_full = CL.full_band_keys(Kb_g, z)
        print(f"s{seed}: group signatures+bands {t_sig:.1f}s", flush=True)

        # ---- A: literal chain edges on documents
        t0 = time.time()
        ei, ej, bstats = K.chain_edges(Kb_full, valid)
        rootA = K.uf_minid(n, ei, ej)
        remA = (rootA != np.arange(n)) & valid
        tA = time.time() - t0
        print(f"s{seed}: A chain edges {len(ei)} largest bucket {bstats['largest_bucket']} "
              f"{tA:.1f}s", flush=True)
        np.savez(os.path.join(MASK, f"raw_s{seed}.npz"), root=rootA.astype(np.int64), rem=remA)
        del ei, ej, rootA, Kb_full

        # ---- B/W: group-level candidate pairs + exact verification
        t0 = time.time()
        pi, pj, bstats_g = K.all_bucket_pairs(Kb_g, valid_g, G)
        t_cand = time.time() - t0
        t0 = time.time()
        jac = K.L.jaccard_pairs(off_u, flat_u, pi, pj)
        t_ver = time.time() - t0
        doc_cand = int((m[pi].astype(np.int64) * m[pj]).sum()
                       + (m.astype(np.int64) * (m - 1) // 2).sum())
        print(f"s{seed}: group pairs {len(pi)} ({t_cand:.1f}s) verify {t_ver:.1f}s; "
              f"doc-level candidate pairs {doc_cand}", flush=True)
        del Kb_g
        for tau in TAUS:
            ti, tj = truth[tau]
            st = state[tau]
            ver = jac >= tau - 1e-12
            ei, ej = pi[ver], pj[ver]
            # B: union-find over verified group edges; min doc of each component kept
            rootB = K.uf_minid(G, ei, ej)
            comp_min = np.full(G, np.iinfo(np.int64).max, np.int64)
            np.minimum.at(comp_min, rootB, rep)
            remB = valid & (np.arange(n) != comp_min[rootB[group]])
            met = CL.lift_metrics(remB, z, ti, tj, st["has_nb_doc"], n, valid)
            met.update(verified_missed=int((st["rem"] & ~remB).sum()),
                       hijacked_verified=int((remB & st["has_nb_doc"] & ~st["rem"]).sum()),
                       exact_deleted=int(st["rem"].sum()),
                       doc_recall=float((st["rem"] & remB).sum() / max(st["rem"].sum(), 1)),
                       n_candidate_pairs_group=int(len(pi)),
                       n_candidate_pairs_doc=int(doc_cand),
                       n_verified_group_edges=int(ver.sum()),
                       largest_bucket_group=int(bstats_g["largest_bucket"]))
            row = dict(base, key=f"snap|t{tau}|B|minid|s{seed}", tau=tau, method="B",
                       representative="minid", hash_seed=seed, n=n,
                       valid_docs=int(valid.sum()), t_signature_s=round(t_sig, 2),
                       t_candidates_s=round(t_cand, 2), t_verify_s=round(t_ver, 2),
                       **met)
            S.append_row(S.RUNS, row)
            np.savez(os.path.join(MASK, f"ver_s{seed}_t{tau}.npz"), rem=remB)

            # W: greedy MIS on group order (= min document index)
            kept = K.greedy_mis(G, ei, ej)
            remW = valid & ((~kept[group]) | (np.arange(n) != rep[group]))
            metw = CL.lift_metrics(remW, z, ti, tj, st["has_nb_doc"], n, valid)
            metw.update(witness_missed=int((st["rem"] & ~remW).sum()),
                        verified_missed=int((st["rem"] & ~remB).sum()),
                        exact_deleted=int(st["rem"].sum()),
                        doc_recall=float((st["rem"] & remW).sum() / max(st["rem"].sum(), 1)),
                        n_verified_group_edges=int(ver.sum()))
            roww = dict(base, key=f"snap|t{tau}|W|minid|s{seed}", tau=tau, method="W",
                        representative="minid", hash_seed=seed, n=n,
                        valid_docs=int(valid.sum()), t_signature_s=round(t_sig, 2),
                        t_mis_s=None, **metw)
            S.append_row(S.RUNS, roww)
            np.savez(os.path.join(MASK, f"wit_s{seed}_t{tau}.npz"), rem=remW)
        del pi, pj, jac, ei, ej

        # ---- A rows (tau-specific truth metrics; same deletion mask)
        for tau in TAUS:
            ti, tj = truth[tau]
            st = state[tau]
            met = CL.lift_metrics(remA, z, ti, tj, st["has_nb_doc"], n, valid)
            met.update(raw_missed=int((st["rem"] & ~remA).sum()),
                       hijacked_raw=int((remA & st["has_nb_doc"] & ~st["rem"]).sum()),
                       exact_deleted=int(st["rem"].sum()),
                       doc_recall=float((st["rem"] & remA).sum() / max(st["rem"].sum(), 1)),
                       largest_bucket=int(bstats["largest_bucket"]),
                       bucket_pairs_sum=int(bstats["bucket_pairs_sum"]),
                       n_candidate_edges=int(bstats["n_chain_pairs"]))
            row = dict(base, key=f"snap|t{tau}|A|minid|s{seed}", tau=tau, method="A",
                       representative="minid", hash_seed=seed, n=n,
                       valid_docs=int(valid.sum()), t_signature_s=round(t_sig, 2),
                       t_method_s=round(tA, 2), **met)
            S.append_row(S.RUNS, row)
        print(f"s{seed}: rows written", flush=True)


if __name__ == "__main__":
    main()
