#!/usr/bin/env python3
"""Three-seed summary for the shared-117 (earlier run) and standalone-112 (this run) designs.

Reads the earlier run's per-seed lsh rows and this run's deleted_sets rows, and writes
results/seed_sensitivity.json with, per cell and tau: per-seed values, the mean of
per-seed shares (the definition used for the paper's headline percentages), the pooled
ratio, and the mean counts.  Makes explicit which definition each number uses.
"""
import json
import os
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
OLD = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/logs/C03/round2/rows.jsonl")
CELLS = [("c4", 700000), ("fw", 1000000)]


def jload(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def main():
    old = {}
    for r in jload(OLD):
        if r.get("kind") == "lsh" and r.get("family") == "dt" and r.get("b") == 14 and r.get("r") == 8:
            old[(r["corpus"], r["n"], r["seed"])] = r
    new = {}
    for r in jload(os.path.join(OUT, "rerun.jsonl")):
        if r.get("kind") == "deleted_sets":
            new[(r["corpus"], r["n"], r["tau"], r["seed"])] = r

    doc = dict(version=1, generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               provenance="earlier run: log-recomputed (earlier run); this run: independently rerun (this run)",
               definitions=dict(
                   share_of_deleted_mean_of_seeds="mean over seeds of (wrongly_deleted / lsh_removed)",
                   share_of_deleted_pooled="sum(wrongly_deleted) / sum(lsh_removed)",
                   share_of_input_mean_of_seeds="mean over seeds of (wrongly_deleted / n)"),
               cells={})
    for corpus, n in CELLS:
        taus = sorted({t for (c, nn, t, s) in new if c == corpus and nn == n})
        out = {}
        for tau in taus:
            seeds_old = [s for (c, nn, s) in old if c == corpus and nn == n]
            so, sn = [], []
            for s in sorted(seeds_old):
                p = old[(corpus, n, s)]["per_tau"].get(str(tau))
                if not p:
                    continue
                so.append(dict(seed=s, lsh_removed=old[(corpus, n, s)]["lsh_removed"],
                               wrong=p["lsh_wrongly_deleted"], exact=p["exact_removed"],
                               lshv=p["lshv_removed"], recall=p["recall"],
                               cand=old[(corpus, n, s)]["n_cand_pairs"]))
            for s in sorted({k[3] for k in new if k[0] == corpus and k[1] == n and k[2] == tau}):
                q = new[(corpus, n, tau, s)]
                sn.append(dict(seed=s, lsh_removed=q["lsh_removed"],
                               wrong=q["lsh_wrongly_deleted"], exact=q["exact_removed"],
                               lshv=q["lshv_removed"], recall=q["pair_recall"],
                               cand=q["n_cand_pairs"]))
            def summ(rows):
                if not rows:
                    return None
                wr = [x["wrong"] / x["lsh_removed"] for x in rows]
                wi = [x["wrong"] / n for x in rows]
                return dict(n_seeds=len(rows),
                            share_of_deleted_mean_of_seeds=float(sum(wr) / len(wr)),
                            share_of_deleted_pooled=float(sum(x["wrong"] for x in rows) / sum(x["lsh_removed"] for x in rows)),
                            share_of_input_mean_of_seeds=float(sum(wi) / len(wi)),
                            mean_lsh_removed=float(sum(x["lsh_removed"] for x in rows) / len(rows)),
                            mean_wrong=float(sum(x["wrong"] for x in rows) / len(rows)),
                            mean_exact=float(sum(x["exact"] for x in rows) / len(rows)),
                            mean_lshv=float(sum(x["lshv"] for x in rows) / len(rows)),
                            mean_gap=float(sum(x["exact"] - x["lshv"] for x in rows) / len(rows)),
                            min_gap=int(min(x["exact"] - x["lshv"] for x in rows)),
                            max_gap=int(max(x["exact"] - x["lshv"] for x in rows)),
                            mean_recall=float(sum(x["recall"] for x in rows) / len(rows)),
                            per_seed=rows)
            out[str(tau)] = dict(shared_h117=summ(so), standalone_h112=summ(sn))
        doc["cells"][f"{corpus}_{n}"] = out
    with open(os.path.join(OUT, "seed_sensitivity.json"), "w") as f:
        json.dump(doc, f, indent=2)
    for cell, taus in doc["cells"].items():
        for tau, d in taus.items():
            for k in ("shared_h117", "standalone_h112"):
                if d[k]:
                    print(cell, tau, k, {kk: round(v, 6) if isinstance(v, float) else v
                                         for kk, v in d[k].items() if kk != "per_seed"})
    print("wrote results/seed_sensitivity.json")


if __name__ == "__main__":
    main()
