#!/usr/bin/env python3
"""Accuracy campaign: deletion sets and metrics for A, B, W and C on identical inputs.

One JSONL row per (workload, tau, method, representative rule, seed).  Resumable: rows
already present in core/runs.jsonl are skipped.  Truth graphs are recomputed from the
frozen rank caches (tau=0.75 does not exist upstream) and cached in core/cache/.

Usage:
  python3 run_accuracy.py --workloads wet --taus 0.8
  python3 run_accuracy.py --workloads c4,fw --taus 0.8,0.75
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_lib as K

SEEDS_FULL = 5      # C4 and FineWeb at tau=0.8
SEEDS_MIN = 3       # everything else


def seeds_for(w, tau):
    return SEEDS_FULL if (w in ("c4", "fw") and abs(tau - 0.8) < 1e-9) else SEEDS_MIN


def rem_from_root(root, n, valid):
    return (root != np.arange(n)) & valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", default="c4,fw,mix,wet")
    ap.add_argument("--taus", default="0.8,0.75")
    ap.add_argument("--methods", default="A,B,W,C")
    args = ap.parse_args()
    workloads = [w for w in args.workloads.split(",") if w]
    taus = [float(t) for t in args.taus.split(",") if t]
    methods = set(args.methods.split(","))
    K.set_threads(K.THREADS)
    base = dict(evidence_tier="rerun", threads=K.THREADS,
                hash_config="xxhash64 5-gram, 112 perms, 14x8, seed in row",
                hw=K.hw_facts())

    for w in workloads:
        t_w0 = time.time()
        off, flat, nwords = K.load_shingles(w)
        n = len(off) - 1
        lens = np.diff(off)
        valid = lens > 0
        truth = {}
        for tau in taus:
            t0 = time.time()
            ti, tj, jac = K.compute_truth(w, tau, off)
            truth[tau] = (ti, tj, K.has_true_neighbour(n, ti, tj))
            print(f"[{w}] truth tau={tau}: {len(ti)} edges in {time.time()-t0:.1f}s", flush=True)
        done = K.done_keys(K.RUNS)

        for seed in range(1, max(seeds_for(w, t) for t in taus) + 1):
            need = any(seed <= seeds_for(w, t) for t in taus)
            if not need:
                continue
            expected = []
            for tau in taus:
                if seed > seeds_for(w, tau):
                    continue
                for m in methods:
                    reps = ("minid",) if m in ("W", "C") else ("minid", "dt")
                    expected += [f"{w}|t{tau}|{m}|{rep}|s{seed}" for rep in reps]
            if expected and all(k in done for k in expected):
                continue
            t0 = time.time()
            sig, _ = K.signatures_cached(off, flat, w, seed)
            t_sig = time.time() - t0
            t0 = time.time()
            Kb = K.band_keys(sig)
            del sig
            t_band = time.time() - t0
            print(f"[{w}] s{seed} sig {t_sig:.1f}s band {t_band:.1f}s", flush=True)

            # ---------- A: chain edges, both representative rules
            if "A" in methods:
                t0 = time.time()
                ei, ej, bstats = K.chain_edges(Kb, valid)
                tA = time.time() - t0
                for rep in ("minid", "dt"):
                    t1 = time.time()
                    root = (K.uf_minid(n, ei, ej) if rep == "minid"
                            else K.uf_datatrove(n, ei, ej))
                    rem = rem_from_root(root, n, valid)
                    tuf = time.time() - t1
                    for tau in taus:
                        if seed > seeds_for(w, tau):
                            continue
                        key = f"{w}|t{tau}|A|{rep}|s{seed}"
                        if key in done:
                            continue
                        ti, tj, hnb = truth[tau]
                        m = K.method_metrics(n, rem, ti, tj, hnb)
                        m.update(K.edge_stats(ti, tj, ei, ej, n))
                        row = dict(base, key=key, workload=w, tau=tau, method="A",
                                   representative=rep, hash_seed=seed, n=n,
                                   valid_docs=int(valid.sum()),
                                   n_candidate_edges=int(len(ei)),
                                   largest_bucket=bstats["largest_bucket"],
                                   bucket_pairs_sum=bstats["bucket_pairs_sum"],
                                   truth_edges=int(len(ti)),
                                   t_signature_s=round(t_sig, 2), t_band_s=round(t_band, 2),
                                   t_method_s=round(tA, 2), t_union_s=round(tuf, 3),
                                   **m)
                        K.append_row(K.RUNS, row)
                del ei, ej
            # ---------- B: all pairs in buckets + exact verification (tau-specific)
            if "B" in methods or "W" in methods:
                t0 = time.time()
                pi, pj, bstats = K.all_bucket_pairs(Kb, valid, n)
                tBcand = time.time() - t0
                t0 = time.time()
                jac = K.L.jaccard_pairs(off, flat, pi, pj)
                tBver = time.time() - t0
                print(f"[{w}] s{seed} B candidates {len(pi)} in {tBcand:.1f}s, "
                      f"verify {tBver:.1f}s, largest bucket {bstats['largest_bucket']}",
                      flush=True)
                for tau in taus:
                    if seed > seeds_for(w, tau):
                        continue
                    ti, tj, hnb = truth[tau]
                    ver = jac >= tau - 1e-12
                    ei, ej = pi[ver], pj[ver]
                    es = K.edge_stats(ti, tj, ei, ej, n)
                    if "B" in methods:
                        for rep in ("minid", "dt"):
                            key = f"{w}|t{tau}|B|{rep}|s{seed}"
                            if key in done:
                                continue
                            t0 = time.time()
                            root = (K.uf_minid(n, ei, ej) if rep == "minid"
                                    else K.uf_datatrove(n, ei, ej))
                            rem = rem_from_root(root, n, valid)
                            tuf = time.time() - t0
                            m = K.method_metrics(n, rem, ti, tj, hnb)
                            row = dict(base, key=key, workload=w, tau=tau, method="B",
                                       representative=rep, hash_seed=seed, n=n,
                                       valid_docs=int(valid.sum()),
                                       n_candidate_pairs=int(len(pi)),
                                       n_verified_edges=int(ver.sum()),
                                       largest_bucket=bstats["largest_bucket"],
                                       bucket_pairs_sum=bstats["bucket_pairs_sum"],
                                       truth_edges=int(len(ti)),
                                       t_signature_s=round(t_sig, 2), t_band_s=round(t_band, 2),
                                       t_candidates_s=round(tBcand, 2),
                                       t_verify_s=round(tBver, 2), t_union_s=round(tuf, 3),
                                       **m, **es)
                            K.append_row(K.RUNS, row)
                    if "W" in methods:
                        key = f"{w}|t{tau}|W|minid|s{seed}"
                        if key not in done:
                            t0 = time.time()
                            m, rem = K.witness_from_verified(n, ei, ej, valid, ti, tj, hnb)
                            tmis = time.time() - t0
                            row = dict(base, key=key, workload=w, tau=tau, method="W",
                                       representative="minid", hash_seed=seed, n=n,
                                       valid_docs=int(valid.sum()),
                                       n_verified_edges=int(ver.sum()),
                                       truth_edges=int(len(ti)),
                                       t_signature_s=round(t_sig, 2),
                                       t_mis_s=round(tmis, 2), **m, **es)
                            K.append_row(K.RUNS, row)
                del pi, pj, jac, ei, ej
            del Kb
        # ---------- C: exact self-join, independent of hash seed
        if "C" in methods:
            for tau in taus:
                ti, tj, hnb = truth[tau]
                for rep in ("minid", "dt"):
                    key = f"{w}|t{tau}|C|{rep}|s0"
                    if key in done:
                        continue
                    t0 = time.time()
                    z = np.load(K.RANKS_CACHE[w])
                    Rr, S, V = np.asarray(z["R"]), int(z["S"]), int(z["V"])
                    z.close()
                    res = K.EJ.exact_join(off, Rr, S, V, tau, K.THREADS)
                    ei = res["i"].astype(np.int64)
                    ej = res["j"].astype(np.int64)
                    tC = time.time() - t0
                    t0 = time.time()
                    root = (K.uf_minid(n, ei, ej) if rep == "minid"
                            else K.uf_datatrove(n, ei, ej))
                    rem = rem_from_root(root, n, valid)
                    tuf = time.time() - t0
                    m = K.method_metrics(n, rem, ti, tj, hnb)
                    row = dict(base, key=key, workload=w, tau=tau, method="C",
                               representative=rep, hash_seed=None, n=n,
                               valid_docs=int(valid.sum()),
                               n_verified_edges=int(len(ei)),
                               truth_edges=int(len(ti)),
                               index_entries=int(res.get("index_entries", -1)),
                               n_candidates=int(res.get("n_candidates", -1)),
                               t_method_s=round(tC, 2), t_union_s=round(tuf, 3),
                               **m)
                    K.append_row(K.RUNS, row)
        print(f"[{w}] done in {time.time()-t_w0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
