#!/usr/bin/env python3
"""Anchor-verified baseline on Snapshot-2025 (uncollapsed), seeds 1..5, tau 0.8 and 0.75.

Metrics against the exact truth (collapsed exact join, lifted): deleted, unsupported, missed exact
deletions, witness coverage and residual pairs; also the difference to full B (frozen masks).
Appends rows to results_post_reveal/operator_E.jsonl with method "anchor".
"""
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OP = os.path.dirname(HERE)
PKG = os.path.abspath(os.path.join(OP, "..", ".."))
sys.path[:0] = [OP, os.path.join(PKG, "code", "newsnap"), os.path.join(PKG, "code", "run")]
import collapsed_lib as CL  # noqa: E402
from anchor_b import anchor_b  # noqa: E402
from stream_b import candidate_keys  # noqa: E402

S = CL.S


def main():
    off, flat, _ = S.load_snapshot()
    valid = np.diff(off) > 0
    n = len(off) - 1
    z = CL.load_collapsed()
    truth = {t: CL.truth_u(t)[:2] for t in (0.8, 0.75)}
    state = {t: CL.exact_state(z, *truth[t]) for t in truth}
    out = open(os.path.join(PKG, "results_post_reveal", "operator_E.jsonl"), "a")
    for seed in range(1, 6):
        keys = candidate_keys(off, flat, seed)
        for tau in (0.8, 0.75):
            r = anchor_b(off, flat, keys, tau, valid=valid)
            rem = r["deleted"]
            st = state[tau]
            met = CL.lift_metrics(rem, z, *truth[tau], st["has_nb_doc"], n, valid)
            with np.load(os.path.join(S.CACHE, "masks", f"ver_s{seed}_t{tau}.npz")) as f:
                remB = f["rem"]
            row = dict(experiment="E", method="anchor", mode="anchor_no_collapse", seed=seed, tau=tau,
                       n_docs=n, **met, missed=int((st["rem"] & ~rem).sum()),
                       missed_vs_B=int((remB & ~rem).sum()), extra_vs_B=int((rem & ~remB).sum()),
                       verified_pairs=r["verified_pairs"], accepted_edges=r["accepted_edges"],
                       anchor_wall_s=r["wall_s"],
                       peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps(row), flush=True)
    print("ANCHOR_SNAPSHOT_DONE", flush=True)


if __name__ == "__main__":
    main()
