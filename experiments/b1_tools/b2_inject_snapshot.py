#!/usr/bin/env python3
"""B2 on server: controlled skew at fixed size, on the filtered 2025 snapshot.

Base: Snapshot-2025 documents with at least 50 words (nwords = len(text.split()), the E3 filter),
which removes the 86,771-copy template group; 429,221 documents. A template is the base document
whose word count is closest to the base median (with >= 400 shingles). For each m, m base slots
other than the template, chosen uniformly at random (seed 2026), are replaced:
  exact_<m>: m identical copies of the template;
  near_<m> : m distinct near-duplicates, the template with 1% of its words replaced by words of the
             template (per-copy seed), so most pairs have J around 0.8.
Writes <B2_W>/b2/<variant>.npz (off, flat, nwords) and .json.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

W = os.environ.get("B2_W", "/path/to/home/dedupjoin-postreveal")
PKG = f"{W}/dedupjoin-newsnap-selection"
sys.path.insert(0, f"{PKG}/code/newsnap")
import snap_common as S  # noqa: E402


def shingle_many(texts, batch=2000):
    from concurrent.futures import ProcessPoolExecutor
    chunks = [texts[i:i + batch] for i in range(0, len(texts), batch)]
    with ProcessPoolExecutor(max_workers=12) as ex:
        parts = list(ex.map(S._shingle_batch_spacy, chunks))
    return [np.split(f, np.cumsum(s)[:-1]) for s, f, _ in parts]


def near_copies(text, m, rate=0.01):
    words = text.split()
    k = max(1, int(round(rate * len(words))))
    out = []
    for i in range(m):
        rng = np.random.default_rng(10_000 + i)
        w = list(words)
        for p in rng.choice(len(w), size=k, replace=False):
            w[p] = words[rng.integers(len(words))]
        out.append(" ".join(w))
    return out


def main():
    ms = [int(x) for x in sys.argv[1:]] or [1000, 10000, 100000]
    os.makedirs(f"{W}/b2", exist_ok=True)
    off, flat, nw_tok = S.load_snapshot()
    words, texts_idx = [], []
    for f in sorted(os.listdir(S.CORPUS)):
        if f.startswith("shard-"):
            words.extend(pq.read_table(os.path.join(S.CORPUS, f), columns=["nwords"])["nwords"].to_pylist())
    words = np.array(words)
    base = np.flatnonzero(words >= 50)
    sizes = (off[1:] - off[:-1])[base]
    b_off = np.zeros(len(base) + 1, np.int64)
    np.cumsum(sizes, out=b_off[1:])
    b_flat = np.concatenate([flat[off[d]:off[d + 1]] for d in base])
    b_nw = nw_tok[base]
    n = len(base)
    med = np.median(words[base])
    order = np.argsort(np.abs(words[base] - med))
    tpos = next(int(i) for i in order if sizes[i] >= 400)
    tdoc = int(base[tpos])
    pos = 0
    ttext = tid = None
    for f in sorted(os.listdir(S.CORPUS)):
        if not f.startswith("shard-"):
            continue
        t = pq.read_table(os.path.join(S.CORPUS, f), columns=["id", "text"])
        if pos <= tdoc < pos + t.num_rows:
            tid, ttext = t["id"][tdoc - pos].as_py(), t["text"][tdoc - pos].as_py()
            break
        pos += t.num_rows
    exact_set = shingle_many([ttext])[0][0]
    np.savez(f"{W}/b2/base.npz", off=b_off, flat=b_flat, nwords=b_nw)
    rng = np.random.default_rng(2026)
    perm = rng.permutation(np.setdiff1d(np.arange(n), [tpos]))
    for m in ms:
        slots = np.sort(perm[:m])
        for kind in ("exact", "near"):
            name = f"{kind}_{m}"
            if os.path.exists(f"{W}/b2/{name}.npz"):
                continue
            inj = [exact_set] * m if kind == "exact" else [s for p in shingle_many(near_copies(ttext, m)) for s in p]
            sets = [b_flat[b_off[i]:b_off[i + 1]] for i in range(n)]
            for s, x in zip(slots, inj):
                sets[s] = x
            sz = np.array([len(x) for x in sets], np.int64)
            o2 = np.zeros(n + 1, np.int64)
            np.cumsum(sz, out=o2[1:])
            np.savez(f"{W}/b2/{name}.npz", off=o2, flat=np.concatenate(sets), nwords=b_nw)
            json.dump(dict(variant=name, m=m, kind=kind, n=n, base="Snapshot-2025, >= 50 words",
                           template_id=tid, template_words=len(ttext.split()),
                           template_shingles=int(len(exact_set)), edit_rate=0.01 if kind == "near" else 0,
                           slots_sha256=hashlib.sha256(slots.astype(np.int64).tobytes()).hexdigest()),
                      open(f"{W}/b2/{name}.json", "w"), indent=1)
            print(f"{name}: n={n} shingles={int(sz.sum())}", flush=True)
    print("B2_INJECT_DONE", flush=True)


if __name__ == "__main__":
    main()
