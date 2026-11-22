#!/usr/bin/env python3
"""Paired before/after check against an independent production pipeline.

The official datatrove 0.6.0 run on the 38,472 English WET documents of the snapshot
removed 1,183 documents (kept 37,289).  This script asks what those official deletions
look like under the exact Jaccard truth computed on the full 175,272-document snapshot:
how many official deletions have no neighbour at J >= tau anywhere in the snapshot, how
many the exact join would also delete, and how the raw and verified LSH variants of our
pipeline compare.  It also records the opposite direction: exact deletions the official
run kept.

Caveat recorded in the output: the official run sees only the English subset and uses
datatrove's NLTK tokenizer; our truth sees the whole snapshot and whitespace tokenisation
(parity on a 2,200-document sample: 0.16% token-count difference, identical kept set).

Usage: python3 paired_before_after.py --tau 0.8
"""
import argparse
import glob
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
PAIR = os.path.join(ROOT, ".work/dedupjoin/pairing_datatrove")


def load_ids(sub):
    ids = []
    for p in sorted(glob.glob(os.path.join(PAIR, sub, "*.jsonl"))):
        with open(p) as f:
            for line in f:
                ids.append(json.loads(line)["id"])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", default="0.8")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    tau = a.tau
    meta = pq.read_table(os.path.join(WRK, "wet_meta.parquet")).to_pydict()
    ids = meta["id"]
    idx = {v: i for i, v in enumerate(ids)}
    n = len(ids)
    off_removed = load_ids("removed")
    off_kept = load_ids("kept")
    assert len(off_removed) + len(off_kept) == 38472, (len(off_removed), len(off_kept))
    miss = [i for i in off_removed + off_kept if i not in idx]
    ri = np.array([idx[i] for i in off_removed if i in idx], np.int64)
    ki = np.array([idx[i] for i in off_kept if i in idx], np.int64)

    t = np.load(os.path.join(WRK, f"wet_truth_t{tau}.npz"))
    ti, tj = np.asarray(t["i"]), np.asarray(t["j"])
    rem_ex = np.asarray(t["removed"])
    z = np.load(os.path.join(WRK, f"wet_s{a.seed}_t{tau}_sets.npz"))
    n_ = int(z["n"])
    rem_raw = np.unpackbits(np.asarray(z["rem_raw"]))[:n_].astype(bool)
    rem_ver = np.unpackbits(np.asarray(z["rem_ver"]))[:n_].astype(bool)
    rem_wit = np.unpackbits(np.asarray(z["rem_wit"]))[:n_].astype(bool)

    deg = np.bincount(np.r_[ti, tj], minlength=n)
    has = deg > 0
    out = {
        "provenance": "independently rerun (this run)",
        "tau": float(tau), "seed": a.seed,
        "before_docs": 38472, "official_removed": int(len(ri)),
        "official_kept": int(len(ki)), "id_mismatch": len(miss),
        "official_removed_with_exact_neighbour": int(has[ri].sum()),
        "official_removed_unsupported_share": float(1 - has[ri].mean()),
        "official_removed_unsupported_docs": int((~has[ri]).sum()),
        "official_removed_also_exact_removed": int(rem_ex[ri].sum()),
        "official_removed_by_raw": int(rem_raw[ri].sum()),
        "official_removed_by_verify": int(rem_ver[ri].sum()),
        "official_removed_by_witness": int(rem_wit[ri].sum()),
        "exact_removed_in_english_subset": int(rem_ex[ri].sum() + rem_ex[ki].sum()),
        "exact_removed_that_official_kept": int(rem_ex[ki].sum()),
        "official_kept_by_verify": int((~rem_ver[ki]).sum()),
        "note": ("official run sees the 38,472 English documents only and uses datatrove's "
                 "NLTK tokenizer; exact truth is computed on the full 175,272-document "
                 "snapshot with whitespace tokenisation (parity sample: 0.16% token-count "
                 "difference, identical kept set)"),
    }
    with open(os.path.join(OUT, "paired_before_after.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
