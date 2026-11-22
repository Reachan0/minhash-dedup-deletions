#!/usr/bin/env python3
"""Corpus-level shares quoted alongside the per-document analyses.

Currently the share of documents with fewer than 100 shingles (the short-document
cut used in the nearest-neighbour subsection), computed from the rerun metadata.
Writes results/corpus_shares.json.
"""
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WRK = os.path.join(ROOT, ".work/dedupjoin/rerun")
OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
CELLS = [("c4", 700000), ("fw", 1000000)]


def main():
    cells = {}
    for corpus, n in CELLS:
        nsh = np.asarray(pq.read_table(os.path.join(WRK, f"{corpus}_{n}_meta.parquet"),
                                       columns=["nshingle"]).column("nshingle"))
        cells[f"{corpus}_{n}"] = dict(
            n=int(len(nsh)), frac_lt100_shingles=float((nsh < 100).mean()),
            median_nshingle=float(np.median(nsh)), mean_nshingle=float(nsh.mean()))
    doc = dict(version=1, generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               provenance="independently rerun (this run)", cells=cells)
    with open(os.path.join(OUT, "corpus_shares.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print(json.dumps(cells, indent=1))


if __name__ == "__main__":
    main()
