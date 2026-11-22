#!/usr/bin/env python3
"""How many documents change keep/drop status when only the hash seed changes.

For each corpus, threshold and method (raw LSH, +verify, witness) over the five main
seeds, count documents deleted under some seeds but not all, with their token and
unique-shingle weight, split by whether the document has a true neighbour at J >= tau
(truth-at-risk) or none (unsupported).  The exact join is seed-independent by
construction and is reported as zero.

Writes results/method_flips.json.
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03"))
import main_compare as MC  # noqa: E402

OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/main")
SEEDS = [1, 2, 3, 4, 5]
TAUS = ["0.6", "0.7", "0.8", "0.9"]


def main():
    out = {"provenance": "independently rerun (this run)", "seeds": SEEDS, "corpora": {}}
    for corpus in ("c4", "fw", "mix", "wet"):
        if not os.path.exists(os.path.join(WRK, f"{corpus}_prepare.json")):
            continue
        n = MC.CORPORA[corpus]
        meta = pq.read_table(os.path.join(WRK, f"{corpus}_meta.parquet"),
                             columns=["nwords", "n_unique"]).to_pydict()
        nw = np.asarray(meta["nwords"], np.int64)
        uniq = np.asarray(meta["n_unique"], np.int64)
        if corpus == "wet":  # same token definition the pipeline uses
            nw = np.asarray(np.load(os.path.join(WRK, f"wet_{n}.npz"))["nwords"], np.int64)
        rec = {"n": int(n), "total_tokens": int(nw.sum()), "total_unique": int(uniq.sum()),
               "taus": {}}
        for tau in TAUS:
            t = np.load(os.path.join(WRK, f"{corpus}_truth_t{tau}.npz"))
            ti, tj = np.asarray(t["i"]), np.asarray(t["j"])
            has_nb = np.zeros(n, bool)
            has_nb[ti] = True
            has_nb[tj] = True
            cell = {}
            for meth in ("raw", "ver", "wit"):
                M = []
                for s in SEEDS:
                    z = np.load(os.path.join(WRK, f"{corpus}_s{s}_t{tau}_sets.npz"))
                    M.append(np.unpackbits(np.asarray(z[f"rem_{meth}"]))[:n].astype(bool))
                M = np.array(M)
                cnt = M.sum(axis=0)
                flip = (cnt > 0) & (cnt < len(SEEDS))
                fa = flip & has_nb
                fu = flip & ~has_nb
                cell[meth] = dict(
                    n_flip=int(flip.sum()),
                    flip_share_of_docs=float(flip.mean()),
                    flip_tokens=int(nw[flip].sum()),
                    flip_tokens_share=float(nw[flip].sum() / nw.sum()),
                    flip_unique=int(uniq[flip].sum()),
                    flip_unique_share=float(uniq[flip].sum() / uniq.sum()),
                    n_flip_at_risk=int(fa.sum()),
                    at_risk_tokens=int(nw[fa].sum()),
                    at_risk_unique=int(uniq[fa].sum()),
                    n_flip_unsupported=int(fu.sum()),
                    unsupported_tokens=int(nw[fu].sum()),
                    unsupported_unique=int(uniq[fu].sum()),
                    mean_deleted=float(M.sum(axis=1).mean()),
                    deleted_min=int(M.sum(axis=1).min()),
                    deleted_max=int(M.sum(axis=1).max()),
                )
            cell["exact"] = dict(n_flip=0, note="seed-independent by construction")
            rec["taus"][tau] = cell
        out["corpora"][corpus] = rec
        for tau in ("0.8",):
            c = rec["taus"][tau]
            print(corpus, tau, {m: (c[m]["n_flip"], c[m]["flip_tokens"], c[m]["flip_unique"])
                                for m in ("raw", "ver", "wit")})
    with open(os.path.join(OUT, "method_flips.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
