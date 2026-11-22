#!/usr/bin/env python3
"""SELECTION_PREREG.md section 4: 2,000-document evenly spaced diagnostic.

Rows floor(k*n/2000), k=0..1999.  On the sample only: shingles are sliced from
the frozen snapshot cache, then 112-permutation signatures (seed 1), 14x8 band
buckets and the candidate pair count / largest bucket are measured.  Used only
as a diagnostic; it does not change the frozen rule.

Writes newsnap/analysis/sampling_diag.json (and a row in sampling_diag.jsonl).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import snap_common as S  # noqa: E402
K = S.K

OUT = os.path.join(S.SNAP, "analysis")
NS = 2000
POOL_C4_PAIRS_PER_DOC = 0.19


def main():
    os.makedirs(OUT, exist_ok=True)
    K.set_threads(K.THREADS)
    t_all = time.perf_counter()
    t = time.perf_counter()
    off, flat, _ = S.load_snapshot()
    n = len(off) - 1
    idx = (np.arange(NS, dtype=np.int64) * n) // NS
    lo = off[idx]
    ln = off[idx + 1] - lo
    sizes = ln.astype(np.int64)
    sub_off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    sub_flat = np.empty(int(sizes.sum()), np.uint64)
    for k in range(NS):
        sub_flat[sub_off[k]:sub_off[k + 1]] = flat[lo[k]:lo[k] + ln[k]]
    t_sh = time.perf_counter() - t
    valid = sizes > 0

    t = time.perf_counter()
    sig = K.L.signatures(sub_off, sub_flat, K.H, 1, "dt")
    t_sig = time.perf_counter() - t
    t = time.perf_counter()
    Kb = K.band_keys(sig)
    t_band = time.perf_counter() - t
    t = time.perf_counter()
    pi, pj, bst = K.all_bucket_pairs(Kb, valid, NS)
    t_cand = time.perf_counter() - t
    t = time.perf_counter()
    jac = K.L.jaccard_pairs(sub_off, sub_flat, pi, pj)
    t_ver = time.perf_counter() - t
    n_ver = int((jac >= 0.8 - 1e-12).sum())
    docs_with = int(len(np.unique(np.concatenate([pi, pj])))) if len(pi) else 0
    out = dict(evidence_tier="rerun", ts_utc=S.now(), n_sample=NS, n= n,
               sample_indices="floor(k*n/2000), k=0..1999", valid_docs=int(valid.sum()),
               empty_docs=int((~valid).sum()), shingles_in_sample=int(sizes.sum()),
               pairs=int(len(pi)), pairs_per_doc=float(len(pi) / NS),
               unique_candidate_pairs=int(len(np.unique(pi * NS + pj))) if len(pi) else 0,
               docs_with_candidate=docs_with, largest_bucket=int(bst["largest_bucket"]),
               bucket_pairs_sum=int(bst["bucket_pairs_sum"]),
               verified_pairs_tau08=n_ver,
               pool_c4_pairs_per_doc=POOL_C4_PAIRS_PER_DOC,
               flag_bucket_expansion_exceeds_10x=bool(len(pi) / NS > 10 * POOL_C4_PAIRS_PER_DOC),
               t_slice_s=round(t_sh, 4), t_signature_s=round(t_sig, 4),
               t_band_s=round(t_band, 4), t_candidates_s=round(t_cand, 4),
               t_verify_s=round(t_ver, 4), wall_s=round(time.perf_counter() - t_all, 3),
               peak_rss_mb=round(K.peak_rss_mb(), 1), threads=K.THREADS, hw=K.hw_facts())
    json.dump(out, open(os.path.join(OUT, "sampling_diag.json"), "w"), indent=1)
    S.append_row(os.path.join(OUT, "sampling_diag.jsonl"),
                 dict(key="sample2000|s1", **out))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
