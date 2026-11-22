#!/usr/bin/env python3
"""R7: the official datatrove 0.6.0 MinHash pipeline on the English part of WET-2019, scored against
exact truth built on the same documents with datatrove's own tokenizer.

1. Runs dedupjoin-pvldb/code/pairing_wet_datatrove.py unchanged (official stages 1-4) on the
   extracted WET-2019 parquet; the English selection must reproduce the 38,472 documents of the
   earlier run (checked).
2. Shingles every input document with snap_common._shingle_batch_spacy (datatrove simplify_text,
   spaCy blank 'en' tokenizer, 5-grams, xxh64 sets), runs the exact join at tau = 0.8 on exactly
   these documents, and classifies the official removals: unsupported (no neighbour), hijacked
   (has a neighbour, the exact join keeps it), supported deletions; plus exact deletions kept.
Usage: python r7_official_matched.py --wet-dir <dir with wet-*.parquet> --work <dir> --repo <repo root>
       [--lang-score 0.8]
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wet-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--lang-score", default="0.8")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    script = os.path.join(a.repo, "dedupjoin-pvldb", "code", "pairing_wet_datatrove.py")
    if not os.path.exists(os.path.join(a.work, "out", "wet_datatrove_pairing.json")):
        subprocess.run([sys.executable, script, "--wet-dir", a.wet_dir, "--work", a.work,
                        "--out", os.path.join(a.work, "out"), "--lang-score", a.lang_score], check=True)
    summary = json.load(open(os.path.join(a.work, "out", "wet_datatrove_pairing.json")))

    def read_jsonl(pattern, fields):
        out = []
        for p in sorted(glob.glob(pattern)):
            for line in open(p):
                d = json.loads(line)
                out.append(tuple(d.get(f) for f in fields))
        return out

    docs = read_jsonl(os.path.join(a.work, "input", "*.jsonl"), ("id", "text"))
    removed = {i for (i,) in read_jsonl(os.path.join(a.work, "removed", "*.jsonl"), ("id",))}
    ids = [d[0] for d in docs]
    n = len(ids)
    sys.path[:0] = [os.path.join(a.repo, "dedupjoin-newsnap-selection", "code", "newsnap")]
    import snap_common as S
    sizes, flat, _ = S._shingle_batch_spacy([d[1] for d in docs])
    off = np.zeros(n + 1, np.int64)
    np.cumsum(sizes, out=off[1:])
    K = S.K
    R, Sv, V = K.EJ.build_ranks(off, flat)
    res = K.EJ.exact_join(off, R, Sv, V, 0.8, 12)
    ti, tj = res["i"].astype(np.int64), res["j"].astype(np.int64)
    has_nb = K.has_true_neighbour(n, ti, tj)
    root = K.uf_minid(n, ti, tj)
    k_exact = root != np.arange(n)
    rem = np.array([i in removed for i in ids])
    row = dict(lang_score=float(a.lang_score), documents=n, matches_earlier_run=(n == 38472),
               official_removed=int(rem.sum()), official_summary=summary,
               unsupported=int((rem & ~has_nb).sum()),
               unsupported_share=float((rem & ~has_nb).sum() / max(rem.sum(), 1)),
               hijacked=int((rem & has_nb & ~k_exact).sum()),
               supported_and_exact=int((rem & k_exact).sum()),
               exact_deletions=int(k_exact.sum()), exact_kept_by_official=int((k_exact & ~rem).sum()),
               exact_edges=int(len(ti)), tokenizer="datatrove spaCy blank en")
    json.dump(row, open(os.path.join(a.work, "r7_result.json"), "w"), indent=1)
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
