#!/usr/bin/env python3
"""Clean wall-time / peak-memory measurement for the decision case.

Runs each procedure in its own process so the peak RSS values are not mixed:
  lsh   : 112 signatures (seed 1) + 14x8 banding + exact Jaccard on candidate pairs
  exact : build_ranks (global order) + exact prefix-filter join at tau=0.8
Both start from the same shingle cache (memory-mapped) written by truth_rerun.py.

Writes results/decision_cost.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CODE = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03")
sys.path.insert(0, CODE)
import common as C  # noqa: E402
import exactjoin as EJ  # noqa: E402
import lsh as L  # noqa: E402

OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/rerun")


def run_phase(phase, corpus, n, tau):
    C.set_threads(8)
    t_load = time.time()
    z = np.load(os.path.join(WRK, f"{corpus}_{n}.npz"), mmap_mode="r")
    off = np.asarray(z["off"])
    flat = z["flat"]
    load_s = time.time() - t_load
    rss_after_load = C.peak_rss_mb()[0]
    if phase == "lsh":
        t0 = time.time()
        sig = L.signatures(off, flat, 112, 1, "dt")
        t_sig = time.time() - t0
        t0 = time.time()
        K = L.band_keys(sig, 14, 8)
        codes, ci, cj, _ = L.lsh_pairs(K, np.diff(off) > 0, 200_000_000)
        t_band = time.time() - t0
        del K
        pi, pj = codes // n, codes % n
        t0 = time.time()
        cjac = L.jaccard_pairs(off, flat, pi, pj)
        t_ver = time.time() - t0
        rem = L.removed_set(L.uf_components(n, ci, cj))
        t0 = time.time()
        ver = cjac >= tau - 1e-12
        rem_v = L.removed_set(L.uf_components(n, pi[ver], pj[ver]))
        t_comp = time.time() - t0
        return dict(phase=phase, load_s=load_s, t_signature_s=t_sig, t_banding_s=t_band,
                    t_verify_candidates_s=t_ver, t_components_s=t_comp,
                    total_s=t_sig + t_band + t_ver + t_comp,
                    peak_rss_mb=C.peak_rss_mb()[0], rss_after_load_mb=rss_after_load,
                    n_cand_pairs=int(len(codes)), verified_pairs=int(ver.sum()),
                    lsh_removed=int(rem.sum()), lshv_removed=int(rem_v.sum()))
    if phase == "exact":
        t0 = time.time()
        R, S, V = EJ.build_ranks(off, flat)
        t_rank = time.time() - t0
        t0 = time.time()
        res = EJ.exact_join(off, R, S, V, tau, 8)
        t_join = time.time() - t0
        rem = L.removed_set(L.uf_components(n, res["i"].astype(np.int64), res["j"].astype(np.int64)))
        return dict(phase=phase, load_s=load_s, t_ranks_s=t_rank, t_exact_join_s=t_join,
                    t_index_s=res["t_index"], t_probe_s=res["t_probe_verify"],
                    total_s=t_rank + t_join, peak_rss_mb=C.peak_rss_mb()[0],
                    rss_after_load_mb=rss_after_load, n_pairs=int(len(res["i"])),
                    index_entries=int(res["index_entries"]),
                    exact_removed=int(rem.sum()))
    raise ValueError(phase)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="c4")
    ap.add_argument("--n", type=int, default=700000)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--phase", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.phase:
        print(json.dumps(run_phase(a.phase, a.corpus, a.n, a.tau)))
        return
    out = dict(version=1, generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               provenance="independently rerun (this run)",
               cell=dict(corpus=a.corpus, n=a.n, tau=a.tau, seed=1, config="dt 14x8"),
               method="each procedure in a separate process, same memory-mapped shingle cache; "
                      "peak RSS via VmHWM includes the touched shingle pages")
    for ph in ("lsh", "exact"):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--corpus", a.corpus,
                            "--n", str(a.n), "--tau", str(a.tau), "--phase", ph],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            raise SystemExit(f"phase {ph} failed")
        out[ph] = json.loads(r.stdout.strip().splitlines()[-1])
        print(ph, json.dumps(out[ph]), flush=True)
    out["comparison"] = dict(
        wall_ratio_exact_over_lsh=out["exact"]["total_s"] / out["lsh"]["total_s"],
        peak_rss_ratio_exact_over_lsh=out["exact"]["peak_rss_mb"] / max(1e-9, out["lsh"]["peak_rss_mb"]),
        same_shingles=True)
    with open(os.path.join(OUT, "decision_cost.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/decision_cost.json")


if __name__ == "__main__":
    main()
