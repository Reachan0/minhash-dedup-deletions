#!/usr/bin/env python3
"""Merge the log-recomputed truth table with the independent re-execution into an
explicit "reproduces / differs" list (step requirement 1).

Inputs
  results/main_table_truth.json          log-recomputed (earlier run), tau=0.8
  .work/vldb2027-ai-db-ideas/experiment/logs/C03/round2/rows.jsonl   per-tau records
  results/rerun.jsonl                    independently rerun (this run), seed 1
  .work/data/C03/r2/cache/{corpus}_{n}_meta.parquet        earlier per-doc metadata
  .work/dedupjoin/rerun/{corpus}_{n}_meta.parquet          this run's per-doc metadata
Output
  results/truth_comparison.json
"""
import json
import os
import statistics
import sys
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
LOGS = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/logs/C03/round2")
CACHE = os.path.join(ROOT, ".work/data/C03/r2/cache")
RERUN_META = os.path.join(ROOT, ".work/dedupjoin/rerun")

CELLS = [("c4", 700000), ("fw", 1000000)]

# (log key, rerun key, kind)
METRICS = [
    ("lsh_removed", "lsh_removed", "int"),
    ("lsh_wrongly_deleted", "lsh_wrongly_deleted", "int"),
    ("exact_removed", "exact_removed", "int"),
    ("lshv_removed", "lshv_removed", "int"),
    ("lshv_missed_removals", "lshv_missed_removals", "int"),
    ("removed_overlap", "removed_overlap", "int"),
    ("lsh_missed_docs", "lsh_missed_docs", "int"),
    ("n_truth_pairs", "n_truth_pairs", "int"),
    ("recall", "pair_recall", "float"),
    ("raw_pair_precision", "raw_pair_precision", "float"),
]


def jload(p):
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def main():
    truth = json.load(open(os.path.join(OUT, "main_table_truth.json")))
    old_rows = jload(os.path.join(LOGS, "rows.jsonl"))
    rerun = jload(os.path.join(OUT, "rerun.jsonl"))

    old_lsh = {}
    for r in old_rows:
        if r.get("kind") == "lsh" and r.get("family") == "dt" and r.get("b") == 14 and r.get("r") == 8:
            old_lsh[(r["corpus"], r["n"], r["seed"])] = r
    new_sets = {}
    new_band = {}
    for r in rerun:
        if r.get("kind") == "deleted_sets":
            new_sets[(r["corpus"], r["n"], r["tau"], r["seed"])] = r
        elif r.get("kind") == "banding":
            new_band[(r["corpus"], r["n"])] = r

    comps = []
    for corpus, n in CELLS:
        o = old_lsh[(corpus, n, 1)]
        for tau in sorted({t for (c, nn, t, s) in new_sets if c == corpus and nn == n}):
            if str(tau) not in o["per_tau"]:
                continue
            p = o["per_tau"][str(tau)]
            q = new_sets[(corpus, n, tau, 1)]
            for lk, nk, kind in METRICS:
                if lk not in p or nk not in q or q[nk] is None:
                    continue
                a, b = p[lk], q[nk]
                if kind == "int":
                    a, b = int(a), int(b)
                    ok = a == b
                    delta = b - a
                else:
                    a, b = float(a), float(b)
                    ok = abs(a - b) <= max(1e-12, 1e-9 * max(abs(a), abs(b)))
                    delta = b - a
                comps.append(dict(corpus=corpus, n=n, tau=tau, seed=1, metric=lk,
                                  log_recomputed=a, rerun=b, delta=delta,
                                  status="reproduces" if ok else "DIFFERS"))
            # candidate pair count is tau-independent
            nb = new_band[(corpus, n)]
            if "n_cand_pairs" in o and nb["n_cand_pairs"] == o["n_cand_pairs"]:
                comps.append(dict(corpus=corpus, n=n, tau=tau, seed=1, metric="n_cand_pairs",
                                  log_recomputed=int(o["n_cand_pairs"]), rerun=int(nb["n_cand_pairs"]),
                                  delta=0, status="reproduces"))
            else:
                comps.append(dict(corpus=corpus, n=n, tau=tau, seed=1, metric="n_cand_pairs",
                                  log_recomputed=int(o.get("n_cand_pairs", -1)),
                                  rerun=int(nb["n_cand_pairs"]),
                                  delta=nb["n_cand_pairs"] - o.get("n_cand_pairs", -1),
                                  status="DIFFERS" if "n_cand_pairs" in o else "no_log_value"))

    # per-document metadata identity (shingles are deterministic functions of the text)
    meta_identity = []
    for corpus, n in CELLS:
        a = pq.read_table(os.path.join(CACHE, f"{corpus}_{n}_meta.parquet"),
                          columns=["id", "nchar", "nwords", "nshingle"]).to_pydict()
        b = pq.read_table(os.path.join(RERUN_META, f"{corpus}_{n}_meta.parquet"),
                          columns=["id", "nchar", "nwords", "nshingle"]).to_pydict()
        rec = dict(corpus=corpus, n=n, n_docs=len(b["id"]),
                   ids_identical=(a["id"] == b["id"]),
                   nchar_identical=bool(np.array_equal(a["nchar"], b["nchar"])),
                   nwords_identical=bool(np.array_equal(a["nwords"], b["nwords"])),
                   nshingle_identical=bool(np.array_equal(a["nshingle"], b["nshingle"])),
                   nchar_total=int(sum(b["nchar"])), nwords_total=int(sum(b["nwords"])),
                   nshingle_total=int(sum(b["nshingle"])),
                   provenance="independently rerun (this run) vs log-recomputed (earlier run)")
        rec["all_identical"] = all(rec[k] for k in
                                   ("ids_identical", "nchar_identical", "nwords_identical", "nshingle_identical"))
        meta_identity.append(rec)

    n_re = sum(c["status"] == "reproduces" for c in comps)
    n_diff = sum(c["status"] == "DIFFERS" for c in comps)
    doc = dict(version=1,
               generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               provenance="comparison of log-recomputed (earlier run) with independently rerun (this run)",
               cells_compared=sorted({(c["corpus"], c["n"], c["tau"]) for c in comps}),
               n_comparisons=len(comps), n_reproduces=n_re, n_differs=n_diff,
               comparisons=comps,
               differing=[c for c in comps if c["status"] != "reproduces"],
               per_document_meta_identity=meta_identity,
               headline_from_logs=truth["headline_check"],
               note=("seed 1 is re-executed from the raw files for both corpora; the 3-seed means in "
                     "main_table_truth.json are derived from the per-seed logs, whose seed-1 component "
                     "is independently reproduced here"))
    with open(os.path.join(OUT, "truth_comparison.json"), "w") as f:
        json.dump(doc, f, indent=2)
    for c in comps:
        print(f'{c["corpus"]:3s} n={c["n"]} tau={c["tau"]} {c["metric"]:24s} '
              f'log={c["log_recomputed"]!r:>22} rerun={c["rerun"]!r:>22} {c["status"]}')
    print("TOTAL", len(comps), "reproduces", n_re, "differs", n_diff)
    print("meta identity:", [(m["corpus"], m["all_identical"]) for m in meta_identity])


if __name__ == "__main__":
    main()
