#!/usr/bin/env python3
"""Recompute the main truth table from the frozen per-seed records of the earlier C03 run.

Reads (read-only):
  .work/vldb2027-ai-db-ideas/experiment/logs/C03/round2/rows.jsonl
  .work/data/C03/r2/cache/{corpus}_{n}_meta.parquet
  .work/data/C03/r2/cache/lsh_out/{corpus}_{n}_0.8_dt1_14x8.npz
Writes:
  output/dedupjoin-pvldb/results/main_table_truth.json

Provenance of every number: "log-recomputed (earlier run)".
The independent re-execution numbers are written by truth_rerun.py into
results/rerun.json and merged by compare_truth.py.
"""
import json
import os
import statistics
import sys
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LOGS = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/logs/C03/round2")
CACHE = os.path.join(ROOT, ".work/data/C03/r2/cache")
OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
PROV = "log-recomputed (earlier run)"

CELLS = [
    dict(corpus="c4", n=700000, n_seeds=[1, 2, 3],
         input_range="allenai/c4 en shards 00000-00001, JSON lines 0..699,999 "
                     "(shard 00000: lines 0..356,316; shard 00001: lines 0..343,683); "
                     "doc id c4-<shard:05d>-<line>"),
    dict(corpus="fw", n=1000000, n_seeds=[1, 2, 3],
         input_range="HuggingFaceFW/fineweb sample/10BT/000_00000.parquet rows 0..999,999 "
                     "(8 CC dumps mixed); doc id = parquet 'id' column"),
]


def load_rows():
    with open(os.path.join(LOGS, "rows.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def main():
    rows = load_rows()
    lsh = [r for r in rows if r.get("kind") == "lsh"]
    exact = [r for r in rows if r.get("kind") == "exact"]
    out = dict(version=1, generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               provenance=PROV, source=os.path.relpath(LOGS, ROOT) + "/rows.jsonl",
               config=dict(corpus_family="dt = datatrove hash family",
                           shingles="5-gram word shingles, xxh64, set semantics",
                           banding="14 bands x 8 rows = 112 hashes", tau=0.8,
                           keep_rule="keep minimum doc id per connected component",
                           delete_rule="delete every non-representative document"),
               cells=[], headline_check=[], token_share=[])

    metas = {}
    for c in CELLS:
        corpus, n = c["corpus"], c["n"]
        recs = []
        for r in lsh:
            if (r["corpus"] == corpus and r["n"] == n and r["family"] == "dt"
                    and r["b"] == 14 and r["r"] == 8):
                p = r["per_tau"]["0.8"]
                recs.append(dict(seed=r["seed"], n_valid_docs=r["n_valid_docs"],
                                 lsh_removed=p["lsh_removed"],
                                 wrongly_deleted=p["lsh_wrongly_deleted"],
                                 exact_removed=p["exact_removed"],
                                 lshv_removed=p["lshv_removed"],
                                 lshv_missed_removals=p["lshv_missed_removals"],
                                 removed_overlap=p["removed_overlap"],
                                 lsh_missed_docs=p["lsh_missed_docs"],
                                 n_truth_pairs=p["n_truth_pairs"],
                                 recall=p["recall"], pred_recall=p["pred_recall"],
                                 raw_pair_precision=p["raw_pair_precision"],
                                 n_cand_pairs=r["n_cand_pairs"],
                                 t_signature_s=r["t_signature_all_hashes"],
                                 t_band_cluster_s=r["t_band_cluster"],
                                 t_verify_s=r.get("t_verify_all_candidates"),
                                 peak_rss_mb_self=r["peak_rss_mb_self"]))
        recs.sort(key=lambda x: x["seed"])
        frac_deleted = [x["wrongly_deleted"] / max(1, x["lsh_removed"]) for x in recs]
        frac_input = [100.0 * x["wrongly_deleted"] / n for x in recs]
        cell = dict(corpus=corpus, n_input=n, config="dt 14x8", tau=0.8,
                    input_range=c["input_range"], per_seed=recs,
                    summary=dict(
                        lsh_removed_mean=mean([x["lsh_removed"] for x in recs]),
                        wrongly_deleted_mean=mean([x["wrongly_deleted"] for x in recs]),
                        exact_removed=recs[0]["exact_removed"],
                        misdelete_share_of_deleted=dict(mean=mean(frac_deleted), std=sd(frac_deleted),
                                                        min=min(frac_deleted), max=max(frac_deleted)),
                        misdelete_share_of_input_pct=dict(mean=mean(frac_input), std=sd(frac_input),
                                                          min=min(frac_input), max=max(frac_input)),
                        denominators=dict(deleted=f"lsh_removed per seed",
                                          input=f"n_input = {n}")),
                    provenance=PROV)
        out["cells"].append(cell)

        # token / character / shingle share using the seed-1 LSH dump + meta
        meta_p = os.path.join(CACHE, f"{corpus}_{n}_meta.parquet")
        dump_p = os.path.join(CACHE, "lsh_out", f"{corpus}_{n}_0.8_dt1_14x8.npz")
        if os.path.exists(meta_p) and os.path.exists(dump_p):
            t = pq.read_table(meta_p, columns=["nchar", "nwords", "nshingle"]).to_pydict()
            nchar = np.asarray(t["nchar"], np.int64)
            nwords = np.asarray(t["nwords"], np.int64)
            nsh = np.asarray(t["nshingle"], np.int64)
            z = np.load(dump_p)
            wrong = z["wrong"].astype(np.int64)
            rem = z["rem_lsh"].astype(np.int64)
            tot_w, tot_c, tot_s = int(nwords.sum()), int(nchar.sum()), int(nsh.sum())
            out["token_share"].append(dict(
                corpus=corpus, n=n, seed=1, tau=0.8,
                n_docs=len(nwords),
                wrongly_deleted_docs=int(len(wrong)),
                deleted_docs=int(len(rem)),
                nwords_total=tot_w, nchar_total=tot_c, nshingle_total=tot_s,
                nwords_wrongly_deleted=int(nwords[wrong].sum()),
                nwords_deleted=int(nwords[rem].sum()),
                token_share_wrongly_deleted_pct=100.0 * float(nwords[wrong].sum()) / tot_w,
                token_share_deleted_pct=100.0 * float(nwords[rem].sum()) / tot_w,
                char_share_wrongly_deleted_pct=100.0 * float(nchar[wrong].sum()) / tot_c,
                char_share_deleted_pct=100.0 * float(nchar[rem].sum()) / tot_c,
                shingle_share_wrongly_deleted_pct=100.0 * float(nsh[wrong].sum()) / tot_s,
                shingle_share_deleted_pct=100.0 * float(nsh[rem].sum()) / tot_s,
                median_nwords_wrongly_deleted=float(np.median(nwords[wrong])),
                median_nwords_all=float(np.median(nwords)),
                provenance=PROV))

    # headline check against the numbers quoted in the step order / Draft 0
    def cell(corpus):
        return next(c for c in out["cells"] if c["corpus"] == corpus)
    checks = [
        ("C4 misdelete share of deleted = 60.71%", 60.71,
         cell("c4")["summary"]["misdelete_share_of_deleted"]["mean"] * 100),
        ("FineWeb misdelete share of deleted = 70.76%", 70.76,
         cell("fw")["summary"]["misdelete_share_of_deleted"]["mean"] * 100),
        ("C4 misdelete share of input = 0.902%", 0.902,
         cell("c4")["summary"]["misdelete_share_of_input_pct"]["mean"]),
        ("FineWeb misdelete share of input = 0.077%", 0.077,
         cell("fw")["summary"]["misdelete_share_of_input_pct"]["mean"]),
        ("C4 seed1 raw LSH deletions = 10,347", 10347, cell("c4")["per_seed"][0]["lsh_removed"]),
        ("C4 seed1 exact deletions = 3,910", 3910, cell("c4")["per_seed"][0]["exact_removed"]),
        ("C4 seed1 LSH+verify deletions = 3,890", 3890, cell("c4")["per_seed"][0]["lshv_removed"]),
    ]
    for name, quoted, recomputed in checks:
        diff = recomputed - quoted
        out["headline_check"].append(dict(claim=name, quoted=quoted, recomputed=round(recomputed, 6),
                                          abs_diff=round(diff, 6),
                                          reproduces=abs(diff) <= max(0.005, abs(quoted) * 1e-4)))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "main_table_truth.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["headline_check"], indent=1))
    print("wrote", os.path.join(OUT, "main_table_truth.json"))


if __name__ == "__main__":
    main()
