#!/usr/bin/env python3
"""R1 on the original corpora (C4, FineWeb, Mix, WET-2019) plus the anchor baseline and streaming B.

Uses the rebuilt pool caches (shingles, ranks, signatures, exact truth) through core_lib, unchanged.
For each workload, tau in {0.8, 0.75} and the seeds of the frozen runs:
  A under the minimum-id and datatrove (union-by-size) representative rules: deleted, unsupported,
  unsupported without a retained neighbour, hijacked, symmetric difference between the rules;
  anchor-verified baseline: deleted, unsupported, missed exact deletions, verified pairs;
  streaming B (seed 1): equality with the materialised B deletion set, verified pairs.
Run from the pool_rebuild root so that core_lib's ROOT resolves (see pool_tools/README).
Usage: env/bin/python r1_pool.py <dir of core_lib> <dir of code/operator> <out.jsonl>
"""
import json
import sys
import time

import numpy as np

sys.path[:0] = [sys.argv[1], sys.argv[2]]
import core_lib as K  # noqa: E402
from anchor_b import anchor_b  # noqa: E402
from stream_b import stream_b  # noqa: E402

OUT = sys.argv[3]
K.set_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "3")))


def seeds_for(w, tau):
    return 5 if (w in ("c4", "fw") and abs(tau - 0.8) < 1e-9) else 3


def main():
    out = open(OUT, "a")
    for w in ("c4", "fw", "mix", "wet"):
        off, flat = K.load_shingles(w)[:2]
        n = len(off) - 1
        valid = np.diff(off) > 0
        for tau in (0.8, 0.75):
            ti, tj, _ = K.compute_truth(w, tau, off)
            has_nb = K.has_true_neighbour(n, ti, tj)
            exact = (K.uf_minid(n, ti, tj) != np.arange(n)) & valid
            for seed in range(1, seeds_for(w, tau) + 1):
                sig, _ = K.signatures_cached(off, flat, w, seed)
                kb = K.band_keys(np.asarray(sig))
                ei, ej, _ = K.chain_edges(kb, valid)
                rems = {"minid": (K.uf_minid(n, ei, ej) != np.arange(n)) & valid,
                        "datatrove": (K.uf_datatrove(n, ei, ej) != np.arange(n)) & valid}
                for rule, rem in rems.items():
                    m = K.method_metrics(n, rem, ti, tj, has_nb)
                    out.write(json.dumps(dict(workload=w, tau=tau, seed=seed, method="A", representative=rule, **m,
                                              unsupported_retained=m["deleted"] - m["witness_covered"],
                                              hijacked=int((rem & has_nb & ~exact).sum()),
                                              symdiff_vs_minid=int((rem ^ rems["minid"]).sum()))) + "\n")
                t0 = time.time()
                r = anchor_b(off, flat, kb, tau, valid=valid)
                m = K.method_metrics(n, r["deleted"], ti, tj, has_nb)
                out.write(json.dumps(dict(workload=w, tau=tau, seed=seed, method="anchor", **m,
                                          missed=int((exact & ~r["deleted"]).sum()),
                                          verified_pairs=r["verified_pairs"], t_s=round(time.time() - t0, 2))) + "\n")
                if seed == 1:
                    t0 = time.time()
                    s = stream_b(off, flat, kb, tau, valid=valid)
                    t_stream = time.time() - t0
                    mb, _ = K.run_path_b(kb, valid, n, tau, ti, tj, has_nb, off, flat)
                    out.write(json.dumps(dict(workload=w, tau=tau, seed=seed, method="stream_B",
                                              deleted=int(s["deleted"].sum()), deleted_materialised_B=mb["deleted"],
                                              missed=int((exact & ~s["deleted"]).sum()),
                                              verified_pairs=s["verified_pairs"], enumerated_pairs=s["enumerated_pairs"],
                                              t_s=round(t_stream, 2))) + "\n")
                out.flush()
                print(w, tau, seed, "done", flush=True)
    print("R1_POOL_DONE", flush=True)


if __name__ == "__main__":
    main()
