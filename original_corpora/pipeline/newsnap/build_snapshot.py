#!/usr/bin/env python3
"""Build the snapshot shingle cache and quantify the tokenisation choice.

Outputs (all under newsnap/cache/):
  snapshot_spacy.npz   off, flat, nwords  -- PREREG section 3 tokenisation
  snapshot_meta.parquet id, segment, url, nwords, nchar, nshingle
  tokenizer_compare.json  spacy vs project-whitespace shingle sets on ~20k docs
  build_report.json       timing, hardware, shard hashes re-verified
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import snap_common as S  # noqa: E402


def main():
    os.makedirs(S.CACHE, exist_ok=True)
    t0 = time.time()
    off, flat, nwords, meta, info = S.shingle_all()
    info["t_total_s"] = round(time.time() - t0, 1)
    info["hw"] = S.K.hw_facts()
    info["now"] = S.now()
    info["tokenizer"] = "datatrove 0.6.0 load_word_tokenizer('en') -> SpaCyTokenizer(spacy.blank('en')+sentencizer)"
    info["simplify"] = "datatrove.utils.text.simplify_text with TextNormConfig defaults"
    np.savez(S.SHINGLE_NPZ, off=off, flat=flat, nwords=nwords)
    import pandas as pd
    pd.DataFrame(meta).to_parquet(S.META_PARQUET, index=False)
    # length sanity vs PREREG
    nws = np.asarray(meta["nwords"], dtype=np.int64)
    info["nwords_mean"] = float(nws.mean())
    info["nwords_p50"] = float(np.quantile(nws, 0.5))
    info["empty_docs"] = int((np.diff(off) == 0).sum())
    S.append_row(os.path.join(S.SNAP, "corpus", "build_report.jsonl"),
                 dict(key="build", evidence_tier="rerun", **info))
    print("built", info, flush=True)

    # ---------------- tokenisation sensitivity: spacy vs project whitespace -------
    print("tokenizer comparison on every 26th document ...", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    batches = []
    k = 0
    for ids, segs, urls, texts, nw, nch in S.iter_text_batches(500):
        sel = [i for i, _ in enumerate(texts) if (k + i) % 26 == 0]
        if sel:
            batches.append([texts[i] for i in sel])
        k += len(texts)
    res = []
    with ProcessPoolExecutor(max_workers=S.WORKERS) as ex:
        for r in ex.map(_compare_batch, batches):
            res.extend(r)
    jac = np.array([r[0] for r in res])
    ident = np.array([r[1] for r in res])
    dw = np.array([r[2] for r in res], dtype=np.int64)
    out = dict(evidence_tier="rerun", now=S.now(), n_docs=int(len(res)),
               identical_shingle_set_share=float(ident.mean()),
               mean_shingle_jaccard_spacy_vs_whitespace=float(jac.mean()),
               p05=float(np.quantile(jac, 0.05)), p50=float(np.quantile(jac, 0.5)),
               p95=float(np.quantile(jac, 0.95)),
               docs_with_any_difference=int((~ident).sum()),
               mean_abs_shingle_count_diff=float(np.abs(dw).mean()),
               note="spacy splits 'cannot', 'gotta', and keeps/splits currency/symbol "
                    "characters that simplify_text leaves in place; whitespace does not",
               sampled_every=26, n_corpus_docs=int(off.shape[0] - 1))
    import json
    with open(os.path.join(S.CACHE, "tokenizer_compare.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1), flush=True)


def _compare_batch(texts):
    import numpy as np
    sp = S._shingle_batch_spacy(texts)
    wh = S._shingle_batch_whitespace(texts)
    out = []
    so = np.concatenate([[0], np.cumsum(sp[0])])
    wo = np.concatenate([[0], np.cumsum(wh[0])])
    for i in range(len(texts)):
        sa = set(sp[1][so[i]:so[i + 1]].tolist())
        sb = set(wh[1][wo[i]:wo[i + 1]].tolist())
        inter = len(sa & sb)
        uni = len(sa | sb)
        j = inter / uni if uni else 1.0
        out.append((j, sa == sb, int(len(sa)) - int(len(sb))))
    return out


if __name__ == "__main__":
    main()
