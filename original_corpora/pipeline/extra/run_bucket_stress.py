#!/usr/bin/env python3
"""Path-B candidate-expansion wall and mitigation, on a labelled synthetic set.

Synthetic construction (explicitly synthetic; no web text):
  * a group of K documents that all share an 800-shingle core plus a 100-shingle
    group signature and 100 shingles unique to the document.  Within a group every
    pair has exact Jaccard 900/1100 = 0.818; across the two groups 800/1200 = 0.667.
  * documents alternate between the two groups by id, so a bucket of size K contains
    two dense true sub-cliques of size K/2 and no cross-group true edge.
  * with the core's 112-permutation, 14x8 LSH all K documents land in one bucket
    (about 0.8^8 * 14 = 2.3 bands are dominated by the common core).

Modes
  wall        : K in a grid; enumerate ALL pairs in every bucket; record largest
                bucket, pairs before/after dedup, time and peak RSS; RLIMIT and a
                wall-clock timeout produce a first-failure row instead of an OOM kill.
  mitigation  : fixed K; compare full expansion with (a) a per-bucket cap and
                (b) NeMo-style anchor sampling (num_anchors = 1,2,4,8, 20 draws),
                scoring unsupported / missed / residual against the analytic truth.

Evidence tier: rerun (synthetic inputs generated and processed in this run).
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bigcode_lib as BC  # noqa: E402

K = BC.K
EXTRA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "extra")
OUT = os.path.join(EXTRA, "bucket_stress.jsonl")
MIT = os.path.join(EXTRA, "bucket_mitigation.jsonl")


def make_synth(n, seed=7, n_groups=2):
    """Return off, flat, group labels; exact Jaccard follows from construction.

    Docs in the same group share the 800-shingle core and the group's 100 shingles
    (J = 900/1100 = 0.818); docs in different groups share only the core
    (J = 800/1200 = 0.667).  Ids cycle through groups, so a cap on the first ids
    of a bucket samples every group rather than whole groups."""
    rng = np.random.RandomState(seed)
    base = rng.randint(1, np.iinfo(np.int64).max, size=800).astype(np.uint64)
    spec = [rng.randint(1, np.iinfo(np.int64).max, size=100).astype(np.uint64)
            for _ in range(n_groups)]
    groups = (np.arange(n) % n_groups).astype(np.int32)
    per_doc = 1000
    flat = np.empty(n * per_doc, np.uint64)
    for i in range(n):
        u = rng.randint(1, np.iinfo(np.int64).max, size=100).astype(np.uint64)
        # jaccard_pairs merges two sorted shingle lists; the production shingle
        # cache is sorted per document, so sort here as well.
        flat[i * per_doc:(i + 1) * per_doc] = np.sort(
            np.concatenate([base, spec[groups[i]], u]))
    off = np.arange(0, (n + 1) * per_doc, per_doc, dtype=np.int64)
    return off, flat, groups


def sig_for(off, flat):
    return K.L.signatures(off, flat, K.H, 1, "dt")


def truth_edges(groups):
    """All within-group pairs (complete sub-cliques)."""
    ei, ej = [], []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        ii, jj = np.triu_indices(len(idx), 1)
        ei.append(idx[ii].astype(np.int64))
        ej.append(idx[jj].astype(np.int64))
    return np.concatenate(ei), np.concatenate(ej)


def wall(args):
    lim = int(args.rlimit_gib * 1024 ** 3)
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    K.set_threads(args.workers)
    row = dict(mode="wall", n=args.n, rlimit_gib=args.rlimit_gib, workers=args.workers,
               evidence_tier="rerun", synthetic=True, ts_utc=K.now(), hw=K.hw_facts())
    rss0 = K.peak_rss_mb()
    t0 = time.perf_counter()
    status = "ok"
    try:
        off, flat, groups = make_synth(args.n, n_groups=args.groups)
        t1 = time.perf_counter()
        sig = sig_for(off, flat)
        t2 = time.perf_counter()
        Kb = K.band_keys(sig)
        t2b = time.perf_counter()
        del sig
        valid = np.ones(args.n, bool)
        # enumeration (all pairs inside every bucket, with band multiplicity) and
        # cross-band dedup are timed separately, as the cost note requires.
        codes, mult, largest = [], 0, 0
        for b, docs in K.band_groups(Kb, valid):
            s = len(docs)
            largest = max(largest, s)
            mult += s * (s - 1) // 2
            ii, jj = np.triu_indices(s, 1)
            codes.append(docs[ii].astype(np.int64) * np.int64(args.n) + docs[jj])
        t_enum = time.perf_counter()
        c = np.unique(np.concatenate(codes)) if codes else np.zeros(0, np.int64)
        t3 = time.perf_counter()
        row.update(largest_bucket=int(largest), n_pairs_mult=int(mult),
                   bucket_pairs_sum=int(mult), n_pairs_unique=int(len(c)),
                   t_synth_s=round(t1 - t0, 2), t_sig_s=round(t2 - t1, 2),
                   t_band_keys_s=round(t2b - t2, 2),
                   t_enum_s=round(t_enum - t2b, 2), t_dedup_s=round(t3 - t_enum, 2),
                   t_pairs_s=round(t3 - t2b, 2), t_band_s=round(t3 - t2, 2),
                   n_pairs_out=int(len(c)),
                   rss_before_pairs_mb=round(rss0, 1))
        del c, codes, Kb
    except MemoryError as e:
        status = "memory_error"
        row["error"] = repr(e)
    except Exception as e:  # noqa: BLE001
        status = f"error:{type(e).__name__}"
        row["error"] = repr(e)
    row["status"] = status
    row["wall_s"] = round(time.perf_counter() - t0, 2)
    row["peak_rss_mb"] = round(K.peak_rss_mb(), 1)
    K.append_row(OUT, row)
    print(json.dumps(row, sort_keys=True))


def _verify_and_score(n, pi, pj, off, flat, ti, tj, has_nb, rem_exact):
    jac = K.L.jaccard_pairs(off, flat, pi, pj)
    keep = jac >= 0.8 - 1e-12
    ei, ej = pi[keep], pj[keep]
    root = K.uf_minid(n, ei, ej)
    rem = K.removed_minid(root)
    m = K.method_metrics(n, rem, ti, tj, has_nb)
    # min-id convention: missed = exact deletions this method fails to make
    m["missed_truth_docs"] = int((rem_exact & ~rem).sum())
    m["n_candidate_pairs"] = int(len(pi))
    m["n_verified_edges"] = int(keep.sum())
    return m


def _unique_pairs(i, j, n):
    if len(i) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    c = np.unique(i.astype(np.int64) * n + j.astype(np.int64))
    return c // n, c % n


def mitigation(args):
    K.set_threads(args.workers)
    lim = int(args.rlimit_gib * 1024 ** 3)
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    n = args.n
    off, flat, groups = make_synth(n, n_groups=args.groups)
    sig = sig_for(off, flat)
    Kb = K.band_keys(sig)
    valid = np.ones(n, bool)
    pi, pj, st = K.all_bucket_pairs(Kb, valid, n)
    ti, tj = truth_edges(groups)
    has_nb = K.has_true_neighbour(n, ti, tj)
    rem_exact = K.removed_minid(K.uf_minid(n, ti, tj))

    base = dict(mode="mitigation", n=n, groups=int(groups.max()) + 1,
                synthetic=True, evidence_tier="rerun",
                ts_utc=K.now(), hw=K.hw_facts(), largest_bucket=st["largest_bucket"])
    rows = []

    t = time.perf_counter()
    m = _verify_and_score(n, pi, pj, off, flat, ti, tj, has_nb, rem_exact)
    rows.append(dict(base, method="full_expansion", **m,
                     wall_s=round(time.perf_counter() - t, 2)))

    # per-bucket cap: keep the first `cap` documents of each bucket, drop the rest
    cap = args.cap
    t = time.perf_counter()
    cpi, cpj = [], []
    for bi, d in BC.band_groups(sig, K.B, K.R):
        d = np.sort(d)[:cap]
        if len(d) >= 2:
            ii, jj = np.triu_indices(len(d), 1)
            cpi.append(d[ii].astype(np.int64))
            cpj.append(d[jj].astype(np.int64))
    if cpi:
        cpi, cpj = _unique_pairs(np.concatenate(cpi), np.concatenate(cpj), n)
    else:
        cpi = np.zeros(0, np.int64)
        cpj = np.zeros(0, np.int64)
    m = _verify_and_score(n, cpi, cpj, off, flat, ti, tj, has_nb, rem_exact)
    rows.append(dict(base, method=f"cap_{cap}", cap=cap, **m,
                     wall_s=round(time.perf_counter() - t, 2)))

    # NeMo-style anchor sampling: each non-anchor compared with `a` random anchors
    for a in (1, 2, 4, 8):
        acc = []
        t = time.perf_counter()
        for draw in range(args.anchor_draws):
            rng = np.random.RandomState(1000 + draw)
            anchors = rng.choice(n, size=a, replace=False)
            others = np.setdiff1d(np.arange(n), anchors)
            api = np.repeat(anchors, len(others))
            apj = np.tile(others, a)
            lo = np.minimum(api, apj)
            hi = np.maximum(api, apj)
            lo, hi = _unique_pairs(lo, hi, n)
            m = _verify_and_score(n, lo, hi, off, flat, ti, tj, has_nb, rem_exact)
            acc.append(m)
        row = dict(base, method=f"anchors_{a}", anchors=a, draws=args.anchor_draws,
                   wall_s=round(time.perf_counter() - t, 2))
        for key in ("deleted", "unsupported", "missed_truth_docs", "residual_duplicate_pairs",
                    "n_candidate_pairs", "n_verified_edges", "witness_coverage"):
            vals = [x.get(key) for x in acc if x.get(key) is not None]
            if vals:
                row[key + "_mean"] = float(np.mean(vals))
                row[key + "_min"] = float(np.min(vals))
                row[key + "_max"] = float(np.max(vals))
                row[key + "_p_nonzero"] = float(np.mean([v > 0 for v in vals]))
        rows.append(row)

    for r in rows:
        K.append_row(MIT, r)
        print(json.dumps({k: r.get(k) for k in ("method", "deleted_mean", "deleted",
                                                "unsupported", "missed_truth_docs_mean",
                                                "residual_duplicate_pairs_mean",
                                                "n_candidate_pairs", "n_verified_edges")},
                         ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["wall", "mitigation"])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--rlimit-gib", type=float, default=40.0)
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--anchor-draws", type=int, default=20)
    ap.add_argument("--groups", type=int, default=2,
                    help="within-bucket clusters (2 = two cliques; 400 = long tail)")
    args = ap.parse_args()
    (wall if args.mode == "wall" else mitigation)(args)


if __name__ == "__main__":
    main()
