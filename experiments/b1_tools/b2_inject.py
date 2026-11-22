#!/usr/bin/env python3
"""B2: controlled skew at a fixed corpus size.

Base: the 1M rung of B1 (smallest prefix of B1 segments with >= 1,000,000 English documents).
A template is a real document from outside the base whose word count is closest to the base
median (and has >= 400 shingles). For each variant, m document slots chosen uniformly at random
(seed 2026) are replaced by injected documents, so the corpus size stays fixed:
  exact_m  : m identical copies of the template;
  near_m   : m distinct near-duplicates, each the template with 1% of its words replaced by
             words drawn from the template itself (per-copy seed), so most pairs have J ~ 0.8.
Writes <W>/b2/<variant>.npz (off, flat, nwords) and <W>/b2/<variant>.json (slots, template, stats).
Usage: env/bin/python b2_inject.py [m ...]   (default 1000 10000 100000)
"""
import hashlib
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

W = "/path/to/scratch/dedupjoin"
sys.path.insert(0, f"{W}/dedupjoin-newsnap-selection/code/newsnap")
import snap_common as S  # noqa: E402


def rung_ranks(target):
    rows = [json.loads(l) for l in open(f"{W}/logs/segments.jsonl")]
    ok = sorted((r["rank"], r["english"]) for r in rows if r.get("status") == "ok")
    total, ranks = 0, []
    for i, (rank, eng) in enumerate(ok):
        if rank != f"{i:04d}":
            raise SystemExit(f"segments not contiguous at {rank}")
        ranks.append(rank)
        total += eng
        if total >= target:
            return ranks, total
    raise SystemExit(f"only {total} documents processed so far")


def load_base(ranks):
    sizes, flats, nws = [], [], []
    for r in ranks:
        with np.load(f"{W}/cache/{r}.npz") as z:
            sizes.append(z["sizes"]), flats.append(z["flat"]), nws.append(z["nwords"])
    sizes = np.concatenate(sizes).astype(np.int64)
    off = np.zeros(len(sizes) + 1, np.int64)
    np.cumsum(sizes, out=off[1:])
    return off, np.concatenate(flats), np.concatenate(nws)


def pick_template(base_ranks, median_words):
    last = int(base_ranks[-1])
    for r in range(last + 1, last + 50):
        p = f"{W}/eng/{r:04d}.parquet"
        if not os.path.exists(p):
            continue
        t = pq.read_table(p, columns=["id", "text", "nwords"]).to_pydict()
        order = np.argsort(np.abs(np.array(t["nwords"]) - median_words))
        for i in order[:50]:
            sizes, _, _ = S._shingle_batch_spacy([t["text"][i]])
            if sizes[0] >= 400:
                return t["id"][i], t["text"][i]
    raise SystemExit("no template found")


def near_copies(text, m, rate=0.01):
    words = text.split()
    k = max(1, int(round(rate * len(words))))
    out = []
    for i in range(m):
        rng = np.random.default_rng(10_000 + i)
        w = list(words)
        pos = rng.choice(len(w), size=k, replace=False)
        for p in pos:
            w[p] = words[rng.integers(len(words))]
        out.append(" ".join(w))
    return out


def shingle_many(texts, batch=2000):
    from concurrent.futures import ProcessPoolExecutor
    chunks = [texts[i:i + batch] for i in range(0, len(texts), batch)]
    with ProcessPoolExecutor(max_workers=14) as ex:
        parts = list(ex.map(S._shingle_batch_spacy, chunks))
    return [np.split(f, np.cumsum(s)[:-1]) for s, f, _ in parts]


def main():
    ms = [int(x) for x in sys.argv[1:]] or [1000, 10000, 100000]
    os.makedirs(f"{W}/b2", exist_ok=True)
    ranks, n = rung_ranks(1_000_000)
    off, flat, nwords = load_base(ranks)
    n = len(off) - 1
    tid, ttext = pick_template(ranks, float(np.median(nwords)))
    rng = np.random.default_rng(2026)
    perm = rng.permutation(n)
    exact_set = shingle_many([ttext])[0][0]
    for m in ms:
        slots = np.sort(perm[:m])
        for kind in ("exact", "near"):
            name = f"{kind}_{m}"
            if os.path.exists(f"{W}/b2/{name}.npz"):
                continue
            if kind == "exact":
                inj = [exact_set] * m
            else:
                inj = [s for part in shingle_many(near_copies(ttext, m)) for s in part]
            sets = [flat[off[i]:off[i + 1]] for i in range(n)]
            for s, x in zip(slots, inj):
                sets[s] = x
            sizes = np.array([len(x) for x in sets], np.int64)
            o2 = np.zeros(n + 1, np.int64)
            np.cumsum(sizes, out=o2[1:])
            np.savez(f"{W}/b2/{name}.npz", off=o2, flat=np.concatenate(sets), nwords=nwords)
            json.dump(dict(variant=name, m=m, kind=kind, n=n, base_ranks=[ranks[0], ranks[-1]],
                           template_id=tid, template_words=len(ttext.split()),
                           template_shingles=int(len(exact_set)), edit_rate=0.01 if kind == "near" else 0,
                           slots_seed=2026, slots_sha256=hashlib.sha256(slots.astype(np.int64).tobytes()).hexdigest()),
                      open(f"{W}/b2/{name}.json", "w"), indent=1)
            print(f"{name}: n={n} shingles={int(sizes.sum())}", flush=True)
    np.savez(f"{W}/b2/base_1m.npz", off=off, flat=flat, nwords=nwords)
    print("B2_INJECT_DONE", flush=True)


if __name__ == "__main__":
    main()
