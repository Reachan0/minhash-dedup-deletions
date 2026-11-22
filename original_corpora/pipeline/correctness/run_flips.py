#!/usr/bin/env python3
"""Seed-to-seed flips and the expected symmetric difference of deletion sets.

For a document deleted in c of K runs, the probability that two independently
drawn seeds disagree is

    p_d = 2 c (K - c) / (K (K - 1))            (two distinct seeds, unordered)

so the expected symmetric difference of two runs is sum_d p_d.  We report it as
a document count, as a share of the input, and as a share of words (nwords) and
of unique shingles (nshingle).  The same numbers are split into all documents
and the subgroup with at least one true neighbour at J >= tau.

Sources (memory-bounded, one packed mask per seed at a time):
  raw A : .work/dedupjoin/main/<corpus>_var_s<seed>_t<tau>.npz  (100 c4 / 20 others)
  B, W  : .work/dedupjoin/main/<corpus>_s<seed>_t<tau>_sets.npz (5 c4,fw / 3 mix,wet)

The 1.14% figure inherited from Draft 2 is an nwords-weighted share; it is
labelled a *word* share here, not a document share.  nwords is the whitespace
token count of the frozen shingle cache, not a fixed tokenizer's training-token
count; converting to training tokens would require re-tokenising (not done).
"""
import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import correctness_lib as K

TAUS = [0.7, 0.8]
SET_TAUS = [0.6, 0.7, 0.8, 0.9]
METHODS_SETS = {"A": "rem_raw", "B": "rem_ver", "W": "rem_wit", "C": "rem_exact"}


def truth_has_neighbour(corpus, tau, n):
    p = os.path.join(K.WRK, f"{corpus}_truth_t{tau}.npz")
    if os.path.exists(p):
        z = np.load(p)
        ti, tj = np.asarray(z["i"], np.int64), np.asarray(z["j"], np.int64)
        z.close()
    else:
        p = os.path.join(K.CACHE, f"truth_{corpus}_t{tau}.npz")
        z = np.load(p)
        ti, tj = np.asarray(z["i"], np.int64), np.asarray(z["j"], np.int64)
        z.close()
    return K.has_true_neighbour(n, ti, tj), len(ti)


def meta_weights(corpus, n):
    t = pq.read_table(K.META_PARQUET[corpus], columns=["nshingle"])
    nsh = np.asarray(t.column("nshingle").to_numpy(), np.int64)
    assert len(nsh) == n, (corpus, len(nsh), n)
    return nsh


def flip_summary(counts, K_runs, nwords, nshingle, has_nb):
    n = len(counts)
    c = counts.astype(np.int64)
    flip = (c > 0) & (c < K_runs)
    p = np.zeros(n, np.float64)
    if K_runs > 1:
        p = (2.0 * c * (K_runs - c)) / (K_runs * (K_runs - 1))
    out = {}
    for name, mask in (("all", np.ones(n, bool)), ("true_neighbour", has_nb)):
        m = mask
        out[name] = dict(
            n_docs=int(m.sum()),
            n_flipped=int((flip & m).sum()),
            flip_share_docs=float((flip & m).sum() / m.sum()) if m.sum() else None,
            flip_words=int(nwords[flip & m].sum()),
            flip_word_share=float(nwords[flip & m].sum() / nwords[m].sum()) if nwords[m].sum() else None,
            flip_shingles=int(nshingle[flip & m].sum()),
            flip_shingle_share=float(nshingle[flip & m].sum() / nshingle[m].sum()) if nshingle[m].sum() else None,
            expected_symdiff=float(p[m].sum()),
            expected_symdiff_share_docs=float(p[m].sum() / m.sum()) if m.sum() else None,
            expected_symdiff_words=float((nwords * p)[m].sum()),
            expected_symdiff_word_share=float((nwords * p)[m].sum() / nwords[m].sum()) if nwords[m].sum() else None,
        )
    out["hist"] = {str(int(v)): int((c == v).sum())
                   for v in np.unique(c) if v != 0 and v != K_runs}
    out["always_deleted"] = int((c == K_runs).sum())
    out["mean_deleted_per_run"] = float(c.sum() / K_runs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(K.OUTDIR, "flips.jsonl"))
    args = ap.parse_args()
    rows = []
    seen = K.done_keys(args.out, key="key")

    for corpus in K.CORPORA:
        n = K.N_DOCS[corpus]
        off, nwords = K.load_offsets(corpus)
        nshingle = meta_weights(corpus, n)

        for tau in TAUS:
            has_nb, n_truth = truth_has_neighbour(corpus, tau, n)
            # ---------- A: raw LSH, the long seed series
            files = sorted(int(f.split("_var_s")[1].split("_")[0])
                           for f in os.listdir(K.WRK)
                           if f.startswith(f"{corpus}_var_s") and f.endswith(f"_t{tau}.npz"))
            if files:
                counts = np.zeros(n, np.uint16)
                for s in files:
                    z = np.load(os.path.join(K.WRK, f"{corpus}_var_s{s}_t{tau}.npz"))
                    rem = np.unpackbits(np.asarray(z["rem"]))[:n].astype(np.uint16)
                    z.close()
                    counts += rem
                key = f"{corpus}|t{tau}|A|raw|K{len(files)}"
                if key not in seen:
                    rec = dict(key=key, corpus=corpus, tau=tau, method="A", source="var",
                               n=n, K_runs=len(files), seeds=",".join(map(str, files)),
                               truth_edges=n_truth, **flip_summary(counts, len(files), nwords,
                                                                   nshingle, has_nb))
                    K.append_row(args.out, rec)
                    rows.append(rec)
                del counts

            # ---------- A/B/W/C from the per-seed mask sets
            sets = sorted(int(f.split("_s")[1].split("_")[0])
                          for f in os.listdir(K.WRK)
                          if f.startswith(f"{corpus}_s") and f.endswith(f"_t{tau}_sets.npz"))
            if not sets:
                continue
            for meth, field in METHODS_SETS.items():
                counts = np.zeros(n, np.uint16)
                for s in sets:
                    z = np.load(os.path.join(K.WRK, f"{corpus}_s{s}_t{tau}_sets.npz"))
                    rem = np.unpackbits(np.asarray(z[field]))[:n].astype(np.uint16)
                    z.close()
                    counts += rem
                key = f"{corpus}|t{tau}|{meth}|sets|K{len(sets)}"
                if key not in seen:
                    rec = dict(key=key, corpus=corpus, tau=tau, method=meth, source="sets",
                               n=n, K_runs=len(sets), seeds=",".join(map(str, sets)),
                               truth_edges=n_truth, **flip_summary(counts, len(sets), nwords,
                                                                   nshingle, has_nb))
                    K.append_row(args.out, rec)
                    rows.append(rec)
                del counts
        print(f"[{corpus}] flips done", flush=True)

    note = {
        "note": "nwords is the whitespace token count of the frozen shingle cache; "
                "the word shares here are not training-token shares under a fixed tokenizer.",
        "rows_written": len(rows),
    }
    with open(os.path.join(K.OUTDIR, "flips_notes.json"), "w", encoding="utf-8") as f:
        json.dump(note, f, indent=1, ensure_ascii=False)
    print(json.dumps(note, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
