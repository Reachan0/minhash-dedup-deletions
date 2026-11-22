#!/usr/bin/env python3
"""Run the BigCode MinHash estimator on a core workload and score it against truth C.

Usage:
  python3 run_bigcode.py --workload c4 --tau 0.8 --seed 42 [--configs "17x15,21x12,..."]

Writes one JSON line per (workload, tau, seed, config) to
`.work/dedupjoin-pvldb-v3/extra/baseline_rows.jsonl`.
Evidence tier: rerun (signatures computed from the frozen shingle caches in this run;
banding/union/metrics computed from those signatures and truth C).
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

EXTRA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "extra")
K = BC.K
OUT = os.path.join(EXTRA, "baseline_rows.jsonl")
SIGDIR = os.path.join(EXTRA, "sigs")
DEFAULT_CONFIGS = ["8x32", "13x19", "17x15", "21x12", "25x10", "32x8", "14x8", "40x6"]


def get_sig(workload, seed):
    os.makedirs(SIGDIR, exist_ok=True)
    p = os.path.join(SIGDIR, f"{workload}_bc_s{seed}_h256.npy")
    if os.path.exists(p):
        return np.load(p, mmap_mode="r"), p, False, None
    off, flat, _ = K.load_shingles(workload)
    t0 = time.perf_counter()
    sig = BC.signatures(off, flat, seed=seed, num_perm=BC.NUM_PERM)
    dt = time.perf_counter() - t0
    tmp = p + ".tmp.npy"
    np.save(tmp, sig)
    os.replace(tmp, p)
    return sig, p, True, dt


def truth(workload, tau):
    p = K.truth_path(workload, tau)
    z = np.load(p)
    ti, tj = np.asarray(z["i"]), np.asarray(z["j"])
    z.close()
    return ti, tj


def one_config(sig, n, B, R, tau, ti, tj, has_nb):
    row = dict(config=f"{B}x{R}", B=B, R=R, bands_rows=B * R)
    t0 = time.perf_counter()
    stats = BC.banding_stats(sig, B, R)
    t1 = time.perf_counter()
    pi, pj, pstats = BC.bucket_pairs(sig, B, R, n)
    t2 = time.perf_counter()
    root, _ = BC.bigcode_clusters(sig, B, R, n)
    t3 = time.perf_counter()
    rem = BC.removal_mask(root)
    m = K.method_metrics(n, rem, ti, tj, has_nb)
    es = K.edge_stats(ti, tj, pi, pj, n)
    t4 = time.perf_counter()
    row.update(stats)
    row.update(pstats)
    row.update(m)
    row.update(es)
    row.update(t_banding_stats_s=round(t1 - t0, 3), t_pairs_s=round(t2 - t1, 3),
               t_union_s=round(t3 - t2, 3), t_metrics_s=round(t4 - t3, 3),
               t_algo_s=round(t4 - t0, 3))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, choices=["c4", "fw", "mix", "wet"])
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", default="")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N docs only")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--rlimit-gib", type=float, default=44.0)
    args = ap.parse_args()
    if args.rlimit_gib:
        lim = int(args.rlimit_gib * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    K.set_threads(K.THREADS)
    configs = args.configs.split(",") if args.configs else DEFAULT_CONFIGS

    t_start = time.perf_counter()
    sig, sig_path, cold, dt_sig = get_sig(args.workload, args.seed)
    n = sig.shape[0]
    ti, tj = truth(args.workload, args.tau)
    keep_e = (ti < n) & (tj < n)
    ti, tj = ti[keep_e], tj[keep_e]
    has_nb = K.has_true_neighbour(n, ti, tj)
    base = dict(workload=args.workload, tau=args.tau, seed=args.seed, n=n,
                estimator="bigcode_minhash_256", num_perm=BC.NUM_PERM,
                shingles="core spaCy 5-gram shingles (same as A/B/C)",
                truth=f"core exact truth at tau={args.tau}",
                evidence_tier="rerun", ts_utc=K.now(), hw=K.hw_facts(),
                sig_path=os.path.relpath(sig_path, K.ROOT), sig_cold=cold,
                t_signature_s=round(dt_sig, 2) if dt_sig is not None else None,
                threads=K.THREADS)
    opt = BC.optimal_param(args.tau, BC.NUM_PERM)
    base["optimal_BR_at_tau"] = f"{opt[0]}x{opt[1]}"

    for cfg in configs:
        B, R = (int(x) for x in cfg.lower().split("x"))
        if B * R > BC.NUM_PERM:
            print(f"skip {cfg}: B*R>{BC.NUM_PERM}", file=sys.stderr)
            continue
        row = dict(base)
        try:
            row.update(one_config(sig, n, B, R, args.tau, ti, tj, has_nb))
            row["status"] = "ok"
        except MemoryError as e:
            row["status"] = "memory_error"
            row["error"] = repr(e)
        except Exception as e:  # noqa: BLE001
            row["status"] = f"error:{type(e).__name__}"
            row["error"] = repr(e)
        row["wall_s"] = round(time.perf_counter() - t_start, 2)
        row["peak_rss_mb"] = round(K.peak_rss_mb(), 1)
        K.append_row(OUT, row)
        print(json.dumps({k: row[k] for k in
                          ("workload", "tau", "seed", "config", "deleted", "unsupported",
                           "unsupported_share_del", "pair_recall", "residual_duplicate_pairs",
                           "witness_coverage", "status")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
