#!/usr/bin/env python3
"""Re-execute the LSH side for the remaining seeds (2 and 3) with a *standalone*
datatrove 14x8 draw (112 RandomState parameters, not the first 112 of a 117-hash draw).

The exact-join pairs saved by truth_rerun.py are seed-independent and reused.
Writes results/rerun.jsonl rows (kind = deleted_sets) and per-seed set dumps
.work/dedupjoin/rerun/sets_{tag}_s{seed}_14x8_t{tau}.npz.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CODE = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03")
sys.path.insert(0, CODE)
import common as C  # noqa: E402
import lsh as L  # noqa: E402

OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/rerun")


def append(path, row):
    row = dict(row)
    row.setdefault("ts_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    row.setdefault("provenance", "independently rerun (this run)")
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seeds", default="2,3")
    ap.add_argument("--taus", default="0.8,0.7")
    ap.add_argument("--rows", default=os.path.join(OUT, "rerun.jsonl"))
    a = ap.parse_args()
    C.set_threads(8)
    tag = f"{a.corpus}_{a.n}"
    n = a.n
    taus = [float(x) for x in a.taus.split(",")]

    z = np.load(os.path.join(WRK, f"{tag}.npz"), mmap_mode="r")
    off = np.asarray(z["off"])
    flat = z["flat"]
    sizes = np.diff(off)
    valid = sizes > 0

    exact = {}
    for tau in taus:
        p = np.load(os.path.join(WRK, f"pairs_{tag}_{tau}.npz"))
        exact[tau] = (p["i"].astype(np.int64), p["j"].astype(np.int64))
        has_nb = np.zeros(n, bool)
        has_nb[exact[tau][0]] = True
        has_nb[exact[tau][1]] = True
        exact[tau] = (exact[tau][0], exact[tau][1], has_nb)

    for seed in [int(s) for s in a.seeds.split(",")]:
        t0 = time.time()
        sig = L.signatures(off, flat, 112, seed, "dt")
        t_sig = time.time() - t0
        t0 = time.time()
        K = L.band_keys(sig, 14, 8)
        codes, ci, cj, tot = L.lsh_pairs(K, valid, 200_000_000)
        t_band = time.time() - t0
        del K
        pi, pj = codes // n, codes % n
        t0 = time.time()
        cjac = L.jaccard_pairs(off, flat, pi, pj)
        t_ver = time.time() - t0
        root_lsh = L.uf_components(n, ci, cj)
        rem_lsh = L.removed_set(root_lsh) & valid
        in_clu = np.zeros(n, bool)
        in_clu[ci] = True
        in_clu[cj] = True
        for tau in taus:
            ti, tj, has_nb = exact[tau]
            root_exact = L.uf_components(n, ti, tj)
            rem_exact = L.removed_set(root_exact)
            ver = cjac >= tau - 1e-12
            rem_lshv = L.removed_set(L.uf_components(n, pi[ver], pj[ver]))
            hit = L.truth_hits(sig, ti, tj, 14, 8).astype(bool)
            wrong = rem_lsh & ~has_nb
            sym = np.flatnonzero(rem_exact != rem_lshv)
            np.savez(os.path.join(WRK, f"sets_{tag}_s{seed}_14x8_t{tau}.npz"),
                     rem_exact=rem_exact, rem_lsh=rem_lsh, rem_lshv=rem_lshv,
                     root_exact=root_exact, root_lsh=root_lsh, has_nb=has_nb, in_clu=in_clu,
                     truth_hit=hit, truth_ver=ver, sym=sym, truth_i=ti, truth_j=tj)
            row = dict(kind="deleted_sets", corpus=a.corpus, n=n, tau=tau, seed=seed,
                       config="dt 14x8 (standalone 112-hash draw)",
                       lsh_removed=int(rem_lsh.sum()), lsh_wrongly_deleted=int(wrong.sum()),
                       exact_removed=int(rem_exact.sum()), lshv_removed=int(rem_lshv.sum()),
                       lshv_missed_removals=int(rem_exact.sum() - rem_lshv.sum()),
                       symmetric_difference=int(len(sym)),
                       n_truth_pairs=int(len(ti)), pair_recall=float(hit.mean()) if len(ti) else None,
                       n_cand_pairs=int(len(codes)), verified_pairs=int(ver.sum()),
                       raw_pair_precision=float(ver.mean()) if len(ver) else None,
                       lsh_missed_docs=int((has_nb & ~in_clu).sum()),
                       removed_overlap=int((rem_lsh & rem_exact).sum()),
                       t_signature_s=t_sig, t_banding_s=t_band, t_verify_all_s=t_ver,
                       peak_rss_mb=C.peak_rss_mb()[0])
            append(a.rows, row)
            print(json.dumps(row), flush=True)
        del sig, codes, ci, cj, pi, pj, cjac, root_lsh, rem_lsh, in_clu


if __name__ == "__main__":
    main()
