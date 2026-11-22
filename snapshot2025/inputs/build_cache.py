#!/usr/bin/env python3
"""Build the frozen snapshot shingle cache (PREREG.md section 3 definitions).

This is build_snapshot.py's cache stage without the tokenizer-sensitivity
comparison (not needed for this task).  Tokenisation is unchanged:

  simplify_text(TextNormConfig defaults) -> load_word_tokenizer("en")
  (spacy.blank("en") + sentencizer) -> contiguous 5-grams -> xxh64,
  deduplicated per document (a set).

Worker count is set to the 8 physical cores of this pod (the prior attempt on
the 12-thread machine reached 250k documents before the step deadline).
Worker count is a preprocessing resource choice; it does not change the
documents, tokenizer, shingles or hash.

Writes, under .work/dedupjoin-newsnap-selection/newsnap/cache/:
  snapshot_spacy.npz    off, flat, nwords
  snapshot_meta.parquet id, segment, url, nwords, nchar, nshingle
  build_report.json     timing, hardware, counts, comparison with manifest
"""
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(SEL, "code", "newsnap"))
import snap_common as S  # noqa: E402

WORKERS = int(os.environ.get("SNAP_WORKERS", "8"))


def rss_gib():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 3)


def main():
    os.makedirs(S.CACHE, exist_ok=True)
    t0 = time.time()
    off, flat, nwords, meta, info = S.shingle_all(workers=WORKERS)
    info["t_total_s"] = round(time.time() - t0, 1)
    info["hw"] = S.K.hw_facts()
    info["now"] = S.now()
    info["workers"] = WORKERS
    info["tokenizer"] = ("datatrove 0.6.0 load_word_tokenizer('en') -> "
                         "SpaCyTokenizer(spacy.blank('en')+sentencizer)")
    info["simplify"] = "datatrove.utils.text.simplify_text with TextNormConfig defaults"
    info["peak_rss_gib_after_shingling"] = rss_gib()
    np.savez(S.SHINGLE_NPZ, off=off, flat=flat, nwords=nwords)
    import pandas as pd
    pd.DataFrame(meta).to_parquet(S.META_PARQUET, index=False)

    nws = np.asarray(meta["nwords"], dtype=np.int64)
    man = json.load(open(os.path.join(S.CORPUS, "manifest.json")))
    info.update({
        "n_docs": int(len(nws)),
        "n_shingles_total": int(flat.size),
        "nwords_total": int(nws.sum()),
        "nwords_mean": float(nws.mean()),
        "nwords_p50": float(np.quantile(nws, 0.5)),
        "nwords_p10": float(np.quantile(nws, 0.1)),
        "nwords_p90": float(np.quantile(nws, 0.9)),
        "nwords_min": int(nws.min()),
        "nwords_max": int(nws.max()),
        "empty_docs": int((np.diff(off) == 0).sum()),
        "manifest_total_documents": man["total_documents"],
        "manifest_total_words": man["length_stats_words"]["total"],
        "manifest_mean_words": man["length_stats_words"]["mean"],
        "n_docs_match_manifest": int(len(nws)) == man["total_documents"],
        "nwords_total_match_manifest": int(nws.sum()) == man["length_stats_words"]["total"],
        "shingle_npz_bytes": os.path.getsize(S.SHINGLE_NPZ) if os.path.exists(S.SHINGLE_NPZ) else None,
        "peak_rss_gib_final": rss_gib(),
    })
    with open(os.path.join(S.CACHE, "build_report.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1, ensure_ascii=False)
    S.append_row(os.path.join(S.SNAP, "analysis", "build_report.jsonl"),
                 dict(key="build", evidence_tier="rerun", **info))
    print(json.dumps(info, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
