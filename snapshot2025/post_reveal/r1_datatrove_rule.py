#!/usr/bin/env python3
"""R1: path A under datatrove 0.6.0's representative rule (union by size, ties keep the first root)
on the CC-MAIN-2025-30 snapshot, next to the minimum-id rule used in the frozen runs.

For each seed and tau: deletion count, unsupported (no neighbour at J >= tau), unsupported under
the frozen protocol's definition (no retained neighbour = deleted - witness_covered), hijacked,
raw_missed, and the symmetric difference to the minimum-id deletion set.
Reads the pipeline outputs under newsnap/cache (collapsed corpus, exact truth, raw_s<seed>.npz).
Writes results_post_reveal/r1_datatrove_rule.jsonl.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "newsnap"), os.path.join(HERE, "..", "run")]
import collapsed_lib as CL  # noqa: E402

S, K = CL.S, CL.K
OUT = os.path.join(HERE, "..", "..", "results_post_reveal", "r1_datatrove_rule.jsonl")
TAUS = [0.8, 0.75]


def main():
    K.set_threads(K.THREADS)
    z = CL.load_collapsed()
    group, nsh = z["group_of_doc"], z["group_nshingle"]
    n = len(group)
    valid = nsh[group] > 0
    truth = {t: CL.truth_u(t)[:2] for t in TAUS}
    state = {t: CL.exact_state(z, *truth[t]) for t in TAUS}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = open(OUT, "a")
    for seed in range(1, 6):
        t0 = time.time()
        kb = CL.full_band_keys(CL.group_band_keys(z, seed), z)
        ei, ej, _ = K.chain_edges(kb, valid)
        del kb
        root_dt = K.uf_datatrove(n, ei, ej)
        rem_dt = (root_dt != np.arange(n)) & valid
        with np.load(os.path.join(S.CACHE, "masks", f"raw_s{seed}.npz")) as f:
            rem_min = f["rem"]
        for tau in TAUS:
            st = state[tau]
            for rule, rem in (("minid", rem_min), ("datatrove", rem_dt)):
                met = CL.lift_metrics(rem, z, *truth[tau], st["has_nb_doc"], n, valid)
                row = dict(key=f"snap|t{tau}|A|{rule}|s{seed}", method="A", representative=rule,
                           tau=tau, seed=seed, n=n, **met,
                           unsupported_retained=met["deleted"] - met["witness_covered"],
                           hijacked=int((rem & st["has_nb_doc"] & ~st["rem"]).sum()),
                           raw_missed=int((st["rem"] & ~rem).sum()),
                           symdiff_vs_minid=int((rem ^ rem_min).sum()),
                           n_chain_edges=int(len(ei)), wall_s=round(time.time() - t0, 1))
                out.write(json.dumps(row) + "\n")
                out.flush()
        print(f"seed {seed}: dt deleted {int(rem_dt.sum())}, minid {int(rem_min.sum())}, "
              f"symdiff {int((rem_dt ^ rem_min).sum())}", flush=True)
    print("R1_DONE", flush=True)


if __name__ == "__main__":
    main()
