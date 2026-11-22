#!/usr/bin/env python3
"""Content-audit release without document text: one row per sampled document with its design
fields and the label of each model in both labellings (closest member; retained document).
Usage: env/bin/python release_audit.py <out dir>"""
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

C = "/path/to/scratch/dedupjoin/dedupjoin-newsnap-selection/results_post_reveal/content_audit"
MODELS = ("gpt-6.2-sol", "qwen3.7-plus")


def labels(base, m):
    return {r["doc"]: r["label"] for r in map(json.loads, open(f"{base}/labels_{m}.jsonl")) if r.get("label")}


out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rows = [json.loads(l) for l in open(f"{C}/kept/samples.jsonl")]
first = {m: labels(C, m) for m in MODELS}
kept = {m: labels(f"{C}/kept", m) for m in MODELS}
cols = {k: [] for k in ("doc", "record_id", "stratum", "length_quintile", "weight", "words",
                        "retained_doc", "retained_jaccard", "closest_doc", "closest_jaccard")}
for m in MODELS:
    cols[f"label_retained_{m}"] = []
    cols[f"label_closest_{m}"] = []
for r in rows:
    cols["doc"].append(r["doc"])
    cols["record_id"].append(r["id"])
    cols["stratum"].append(r["stratum"])
    cols["length_quintile"].append(r["length_quintile"])
    cols["weight"].append(r["weight"])
    cols["words"].append(r["words"])
    cols["retained_doc"].append(r["counterpart"])
    cols["retained_jaccard"].append(r["counterpart_jaccard"])
    cols["closest_doc"].append(r.get("closest_member", r["counterpart"]))
    cols["closest_jaccard"].append(r.get("closest_member_jaccard", r["counterpart_jaccard"]))
    for m in MODELS:
        cols[f"label_retained_{m}"].append(kept[m].get(r["doc"]))
        cols[f"label_closest_{m}"].append(first[m].get(r["doc"]))
pq.write_table(pa.table(cols), f"{out}/content_audit_labels.parquet", compression="zstd")
print(len(rows), os.path.getsize(f"{out}/content_audit_labels.parquet"))
