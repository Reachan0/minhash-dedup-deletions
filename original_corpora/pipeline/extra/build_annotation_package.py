#!/usr/bin/env python3
"""Build the blind human-annotation package for the information-loss question.

Strata (seed-1 deletion sets, tau=0.8, core min-id convention):
  unsupported_deletion : removed by raw LSH but with no exact neighbour at tau
  supported_deletion   : removed by raw LSH and with at least one exact neighbour
  retained_control     : kept, and has at least one exact neighbour (a retained duplicate)
C4 and FineWeb are included.  The new snapshot has no deletion/neighbour sets yet
(independent-snapshot-eval is partial), so it cannot contribute a stratum; this is
stated in the package README rather than papered over.

Each pair presents the target document and a comparison document:
  * unsupported deletion -> the retained co-bucket document with the highest exact
    Jaccard (the document whose presence justified the deletion);
  * supported deletion / retained control -> a retained exact neighbour (J >= tau).
Document order inside the pair is randomised; the arm label is only in the withheld key.

Outputs (English texts, Chinese instructions):
  output/dedupjoin-pvldb-v3/human_annotation/{annotation_sheet.csv,key_withheld.csv,
  pairs.jsonl,guidelines_zh.md,README.md,statement_no_human_labels.md,
  agreement_analysis.py,MANIFEST.json}
"""
import csv
import hashlib
import json
import os
import random
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bigcode_lib as BC  # noqa: E402

K = BC.K
ROOT = K.ROOT
OUT = os.path.join(ROOT, "output", "dedupjoin-pvldb-v3", "human_annotation")
SEED = 20261119

SIZES = {"c4": {"unsupported_deletion": 200, "supported_deletion": 100,
                "retained_control": 100},
         "fw": {"unsupported_deletion": 100, "supported_deletion": 50,
                "retained_control": 50}}
WORK = {"c4": (700000, "c4"), "fw": (1000000, "fw")}


def load_ids(corpus, n):
    md = pq.read_table(K.META_CACHE[corpus], columns=["id"]).to_pydict()
    return md["id"][:n]


def doc_lengths(corpus, n):
    lens = np.zeros(n, np.int32)
    pos = 0
    for batch in K.iter_pool_prefix_texts(corpus, n, start=0, batch=4000):
        for t in batch:
            lens[pos] = len(t)
            pos += 1
    return lens


def fetch_texts(corpus, n, wanted):
    """Read the pool texts of workload positions in `wanted` (a set of ints)."""
    out = {}
    pos = 0
    for batch in K.iter_pool_prefix_texts(corpus, n, start=0, batch=4000):
        for t in batch:
            if pos in wanted:
                out[pos] = t
            pos += 1
    return out


def pair_adjacency(i, j, val, n):
    adj = {}
    for a, b, v in zip(i.tolist(), j.tolist(), np.asarray(val).tolist()):
        adj.setdefault(a, []).append((b, v))
        adj.setdefault(b, []).append((a, v))
    return adj


def main():
    rng = random.Random(SEED)
    os.makedirs(OUT, exist_ok=True)
    records = []
    for corpus, (n, cname) in WORK.items():
        z = np.load(os.path.join(ROOT, f".work/dedupjoin/rerun/sets_{cname}_{n}_s1_14x8_t0.8.npz"))
        rem, hb = z["rem_lsh"], z["has_nb"]
        ti, tj, tjac = z["truth_i"], z["truth_j"], z["truth_jac"]
        z.close()
        cand = np.load(os.path.join(ROOT, f".work/dedupjoin/rerun/cand_{cname}_{n}_s1_14x8.npz"))
        cadj = pair_adjacency(cand["i"], cand["j"], cand["jac"], n)
        cand.close()
        tadj = pair_adjacency(ti, tj, tjac, n)

        lens = doc_lengths(corpus, n)
        short = lens <= 6000
        strata = {
            "unsupported_deletion": np.flatnonzero(rem & ~hb & short),
            "supported_deletion": np.flatnonzero(rem & hb & short),
            "retained_control": np.flatnonzero(~rem & hb & short),
        }
        ids = load_ids(corpus, n)
        texts_pre = None  # fetched after selection
        picked = []
        for arm, pool in strata.items():
            need = SIZES[corpus][arm]
            order = list(pool)
            rng.shuffle(order)
            got = 0
            for t in order:
                if got >= need:
                    break
                t = int(t)
                partner, jv = None, None
                if arm == "unsupported_deletion":
                    opts = [(p, v) for p, v in cadj.get(t, []) if not rem[p]]
                    if not opts:
                        opts = cadj.get(t, [])
                    if opts:
                        partner, jv = max(opts, key=lambda x: x[1])
                else:
                    opts = [(p, v) for p, v in tadj.get(t, []) if not rem[p]]
                    if not opts:
                        opts = tadj.get(t, [])
                    if opts:
                        partner, jv = max(opts, key=lambda x: x[1])
                if partner is None or jv is None:
                    continue
                # identical pairs are uninformative for the information-loss question
                if jv >= 0.995 or lens[partner] > 6000:
                    continue
                picked.append(dict(corpus=corpus, n=n, arm=arm, target=t,
                                   partner=int(partner), j=float(jv)))
                got += 1
            if got < need:
                print(f"WARN {corpus}:{arm} only {got}/{need} after filters", file=sys.stderr)
        wanted = set()
        for p in picked:
            wanted.add(p["target"])
            wanted.add(p["partner"])
        texts = fetch_texts(corpus, n, wanted)
        for p in picked:
            if p["target"] not in texts or p["partner"] not in texts:
                continue
            p["target_id"] = ids[p["target"]]
            p["partner_id"] = ids[p["partner"]]
            p["target_text"] = texts[p["target"]]
            p["partner_text"] = texts[p["partner"]]
            records.append(p)

    rng.shuffle(records)
    rows, keyrows, jsonl = [], [], []
    for k, r in enumerate(records):
        pid = f"P{k + 1:04d}"
        flip = rng.random() < 0.5
        doc1 = r["partner_text"] if flip else r["target_text"]
        doc2 = r["target_text"] if flip else r["partner_text"]
        target_pos = "doc2" if flip else "doc1"
        rows.append(dict(pair_id=pid, doc1=doc1, doc2=doc2,
                         q1_info_in_doc1_missing_from_doc2="",
                         q2_info_in_doc2_missing_from_doc1="",
                         duplicate_verdict="", confidence_1to5="", notes=""))
        keyrows.append(dict(pair_id=pid, corpus=r["corpus"], arm=r["arm"],
                            target_pos=target_pos, target_id=r["target_id"],
                            partner_id=r["partner_id"], exact_jaccard=round(r["j"], 6),
                            target_removed_by="raw_minhash_seed1",
                            sampling_seed=SEED,
                            source=f".work/dedupjoin/rerun/sets_{r['corpus']}_{r['n']}_s1_14x8_t0.8.npz"))
        jsonl.append(dict(pair_id=pid, corpus=r["corpus"], arm=r["arm"],
                          target_pos=target_pos,
                          doc1_id=r["partner_id"] if flip else r["target_id"],
                          doc2_id=r["target_id"] if flip else r["partner_id"],
                          exact_jaccard=round(r["j"], 6), doc1=doc1, doc2=doc2))

    def wr(path, fieldnames, data):
        with open(os.path.join(OUT, path), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(data)

    wr("annotation_sheet.csv", list(rows[0].keys()), rows)
    wr("key_withheld.csv", list(keyrows[0].keys()), keyrows)
    with open(os.path.join(OUT, "pairs.jsonl"), "w", encoding="utf-8") as f:
        for r in jsonl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = {}
    for r in records:
        counts[f"{r['corpus']}:{r['arm']}"] = counts.get(f"{r['corpus']}:{r['arm']}", 0) + 1
    manifest = dict(generated_utc=K.now(), sampling_seed=SEED, n_pairs=len(records),
                    counts=counts,
                    files={p: hashlib.sha256(open(os.path.join(OUT, p), "rb").read()).hexdigest()
                           for p in ("annotation_sheet.csv", "key_withheld.csv", "pairs.jsonl")})
    with open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
