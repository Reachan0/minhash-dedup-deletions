#!/usr/bin/env python3
"""Signature-only verification: B with the exact Jaccard test replaced by the MinHash estimate
(the share of the 112 signature values two documents agree on), as in SEDD.

On the collapsed corpus of a rung (identical shingle sets share a signature, so their estimate is 1
and collapsing is exact), per seed and threshold: all within-bucket group pairs, the estimate and the
exact Jaccard of each, components of the pairs whose estimate reaches tau, min-id deletion lifted to
documents, and the same metrics as b1_rung.py. Also: pairs accepted by the estimate but below tau,
pairs at or above tau that the estimate rejects, unsupported share outside the largest group, and
the time to compute the estimate and the exact Jaccard on the same pairs.
Writes results/sig_verify.jsonl and, for seed 1 and tau 0.8, runs/<rung>/remB_s1.npy (B's mask).
Usage: env/bin/python sig_verify.py <rung N> <seeds, e.g. 1,2,3,4,5>
"""
import json
import os
import sys
import time

import numpy as np

W = os.environ.get("B2_W", "/path/to/scratch/dedupjoin")
sys.path[:0] = [f"{W}/dedupjoin-newsnap-selection/code/newsnap", f"{W}/dedupjoin-newsnap-selection/code/run"]
import snap_common as S  # noqa: E402

TAUS = [0.8, 0.75]


def main():
    name = f"rung_{int(sys.argv[1])}"
    seeds = [int(x) for x in sys.argv[2].split(",")]
    d = f"{W}/runs/{name}"
    S.SHINGLE_NPZ, S.CACHE, S.SNAP, S.RUNS = f"{d}/snapshot.npz", f"{d}/cache", d, f"{d}/quality.jsonl"
    import collapsed_lib as CL
    K = CL.K
    K.set_threads(K.THREADS)
    z = CL.load_collapsed()
    group, m, rep, nsh = z["group_of_doc"], z["group_size"], z["group_rep"], z["group_nshingle"]
    n, G = len(group), len(m)
    valid, valid_g = nsh[group] > 0, nsh > 0
    outside = group != int(np.argmax(m))
    truth, state = {}, {}
    for tau in TAUS:
        ti, tj, _ = CL.truth_u(tau)
        truth[tau] = (ti, tj)
        state[tau] = CL.exact_state(z, ti, tj)
    out = open(f"{W}/results/sig_verify.jsonl", "a")
    for seed in seeds:
        t0 = time.time()
        sig = K.L.signatures(z["off_u"], z["flat_u"], K.H, seed, "dt")
        t_sig = time.time() - t0
        kb_g = K.L.band_keys(sig, K.B, K.R)
        pi, pj, _ = K.all_bucket_pairs(kb_g, valid_g, G)
        del kb_g
        t0 = time.time()
        agree = np.empty(len(pi), np.int16)
        step = 2_000_000
        for s in range(0, len(pi), step):
            agree[s:s + step] = (sig[pi[s:s + step]] == sig[pj[s:s + step]]).sum(1)
        t_est = time.time() - t0
        t0 = time.time()
        jac = K.L.jaccard_pairs(z["off_u"], z["flat_u"], pi, pj)
        t_jac = time.time() - t0
        H = sig.shape[1]
        del sig
        for tau in TAUS:
            ti, tj = truth[tau]
            st = state[tau]
            hn = st["has_nb_doc"]
            for meth, acc in (("B", jac >= tau - 1e-12), ("B_sig", agree / H >= tau - 1e-12)):
                root = K.uf_minid(G, pi[acc], pj[acc])
                comp_min = np.full(G, np.iinfo(np.int64).max, np.int64)
                np.minimum.at(comp_min, root, rep)
                rem = valid & (np.arange(n) != comp_min[root[group]])
                met = CL.lift_metrics(rem, z, ti, tj, hn, n, valid)
                ro = rem & outside
                row = dict(kind="run", corpus=name, method=meth, tau=tau, seed=seed, n=n, **met,
                           missed=int((st["rem"] & ~rem).sum()),
                           unsupported_outside=int((ro & ~hn).sum()), deleted_outside=int(ro.sum()),
                           group_pairs=int(len(pi)), accepted_pairs=int(acc.sum()),
                           accepted_below_tau=int((acc & (jac < tau - 1e-12)).sum()),
                           rejected_at_tau=int((~acc & (jac >= tau - 1e-12)).sum()),
                           hashes=int(H), t_signature_s=round(t_sig, 1), t_estimate_s=round(t_est, 2),
                           t_jaccard_s=round(t_jac, 2))
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(json.dumps({k: row[k] for k in ("corpus", "method", "tau", "seed", "deleted", "unsupported",
                                                      "missed", "accepted_below_tau", "rejected_at_tau")}), flush=True)
                if meth == "B" and seed == 1 and tau == 0.8:
                    np.save(f"{d}/remB_s1.npy", rem)
        del pi, pj, agree, jac
    print(f"{name} SIG_DONE", flush=True)


if __name__ == "__main__":
    main()
