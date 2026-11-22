#!/usr/bin/env python3
"""Run the no-collapse streaming-B snapshot equality campaign."""
import json
import os
import resource
import sys
import time

import numpy as np

OP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEL = os.path.abspath(os.path.join(OP, "..", ".."))
sys.path.insert(0, OP)
sys.path.insert(0, os.path.join(SEL, "code", "newsnap"))
from stream_b import candidate_keys, stream_b  # noqa: E402
import snap_common  # noqa: E402

OUT = os.path.join(SEL, "results_post_reveal", "operator_E.jsonl")
MASK = os.path.join(SEL, "newsnap", "cache", "masks")
LOG = os.path.abspath(os.path.join(SEL, "..", "logs", "e123.log"))


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    if not os.path.exists(LOG) or "E3_DONE" not in open(LOG, encoding="utf-8", errors="ignore").read():
        print(f"deferred until E3_DONE in {LOG}")
        return 2
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    off, flat, _ = snap_common.load_snapshot()
    for seed in range(1, 6):
        t0 = time.perf_counter()
        keys = candidate_keys(off, flat, seed)
        key_s = time.perf_counter() - t0
        for tau in (0.8, 0.75):
            t1 = time.perf_counter()
            out = stream_b(off, flat, keys, tau)
            expected = np.asarray(np.load(os.path.join(MASK, f"ver_s{seed}_t{tau}.npz"))["rem"])
            equal = bool(np.array_equal(out["deleted"], expected))
            row = dict(experiment="E", method="B", mode="stream_no_collapse",
                       seed=seed, tau=tau, n_docs=len(off) - 1,
                       equal_existing_mask=equal, differing_docs=int(np.sum(out["deleted"] != expected)),
                       key_wall_s=key_s, stream_wall_s=time.perf_counter() - t1,
                       wall_s=time.perf_counter() - t0, peak_rss_mb=rss_mb(),
                       enumerated_pairs=out["enumerated_pairs"], skipped_pairs=out["skipped_pairs"],
                       verified_pairs=out["verified_pairs"], accepted_edges=out["accepted_edges"],
                       unchecked_pairs=out["unchecked_pairs"])
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
