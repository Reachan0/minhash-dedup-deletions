#!/usr/bin/env python3
"""Recompute the pre-registered segment draw and compare it with the corpus.

PREREG.md section 1: the 100,000 WET paths were read in file order, shuffled
with random.Random(20261119).shuffle over indices 0..99,999, and segments were
consumed in that order until the English-document target was reached.  This
script re-runs that draw and checks that the manifest's 104 segments are
exactly ranks 0..103 of it, and that wet.paths.gz has the frozen sha256.
"""
import gzip
import hashlib
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BASE = os.path.join(ROOT, ".work", "dedupjoin-pvldb-v3", "newsnap")
OUT = HERE


def main():
    paths_gz = os.path.join(BASE, "wet.paths.gz")
    sha = hashlib.sha256(open(paths_gz, "rb").read()).hexdigest()
    with gzip.open(paths_gz, "rt") as f:
        paths = [ln.strip() for ln in f if ln.strip()]
    idx = list(range(len(paths)))
    random.Random(20261119).shuffle(idx)
    plan = json.load(open(os.path.join(BASE, "sampling_plan.json")))
    plan_paths = [o["path"] if isinstance(o, dict) else o for o in plan["order"]]
    man = json.load(open(os.path.join(BASE, "corpus", "manifest.json")))
    man_paths = [s["source_path"] for s in sorted(man["segments"], key=lambda x: x["rank"])]
    out = {
        "wet_paths_sha256": sha,
        "wet_paths_sha256_expected": "b00996a663900fbb38df54c3dd29be792bdc58cb488401ed00bc59056cf53338",
        "wet_paths_sha256_ok": sha == "b00996a663900fbb38df54c3dd29be792bdc58cb488401ed00bc59056cf53338",
        "paths_in_file": len(paths),
        "plan_order_len": len(plan_paths),
        "plan_seed": plan.get("seed"),
        "plan_equals_recomputed_shuffle": plan_paths == [paths[i] for i in idx],
        "manifest_segments": len(man_paths),
        "manifest_paths_equal_plan_first_104": man_paths == plan_paths[:104],
        "manifest_ranks_contiguous_0_103": [s["rank"] for s in man["segments"]] == list(range(104)),
        "note": "recomputed with random.Random(20261119).shuffle over 0..99,999",
    }
    with open(os.path.join(OUT, "sampling_plan_check.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    res = json.load(open(os.path.join(OUT, "input_verification.json"), encoding="utf-8"))
    res["segment_sampling"] = out
    with open(os.path.join(OUT, "input_verification.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
