#!/usr/bin/env python3
"""Shared paths and the snapshot shingle builder (step independent-snapshot-eval).

The snapshot (CC-MAIN-2025-30, 531,597 English WET docs) is shingled with the
datatrove 0.6.0 tokenisation fixed in newsnap/PREREG.md section 3:
simplify_text(TextNormConfig defaults) -> load_word_tokenizer("en")
(spacy.blank("en") + sentencizer) -> contiguous 5-grams -> xxhash64, deduplicated
per document (a set; min-hash is invariant to duplicates).

Everything downstream (signatures, banding, A/B/W/C, exact join) is the frozen
core harness in ../core/core_lib.py, unchanged.
"""
import importlib.metadata  # noqa: F401  (datatrove 0.6.0 touches importlib.metadata)
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SLUG = os.path.abspath(os.path.join(HERE, "..", ".."))
CODE = os.path.abspath(os.path.join(HERE, ".."))
CORE = os.path.join(CODE, "core")
sys.path.insert(0, CORE)

import core_lib as K  # noqa: E402

SNAP = os.path.join(SLUG, "newsnap")
CORPUS = os.path.join(SNAP, "corpus")
CACHE = os.path.join(SNAP, "cache")
RUNS = os.path.join(SNAP, "runs.jsonl")
BATCH = 2000
WORKERS = K.THREADS
TAUS = [0.8, 0.75]
SEEDS = [1, 2, 3, 4, 5]
SHINGLE_NPZ = os.path.join(CACHE, "snapshot_spacy.npz")
META_PARQUET = os.path.join(CACHE, "snapshot_meta.parquet")
RANKS = os.path.join(CACHE, "snapshot_ranks.npz")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def iter_text_batches(batch=BATCH):
    """Yield (ids, segments, urls, texts) in shard order (the frozen corpus order)."""
    import pyarrow.parquet as pq
    for sh, f in enumerate(sorted(os.listdir(CORPUS))):
        if not f.startswith("shard-") or not f.endswith(".parquet"):
            continue
        pf = pq.ParquetFile(os.path.join(CORPUS, f))
        for b in pf.iter_batches(batch_size=batch,
                                 columns=["id", "segment", "url", "text", "nwords", "nchar"]):
            d = b.to_pydict()
            yield d["id"], d["segment"], d["url"], d["text"], d["nwords"], d["nchar"]


# ---------------------------------------------------------------- tokenisation
_TOK = None


def _tok():
    global _TOK
    if _TOK is None:
        from datatrove.utils.word_tokenizers import load_word_tokenizer
        _TOK = load_word_tokenizer("en")
    return _TOK


def _shingle_batch_spacy(texts):
    """datatrove get_shingles semantics: xxh64(' '.join(5-gram tokens)); set per doc."""
    from datatrove.utils.text import simplify_text
    import xxhash
    tok = _tok()
    h = xxhash.xxh64_intdigest
    out, lens = [], []
    for t in texts:
        toks = tok.word_tokenize(simplify_text(t))
        if len(toks) < 5:
            out.append(np.zeros(0, np.uint64))
            lens.append(len(toks))
            continue
        arr = np.fromiter((h(" ".join(toks[i:i + 5]).encode("utf-8"))
                           for i in range(len(toks) - 4)), dtype=np.uint64,
                          count=len(toks) - 4)
        out.append(np.unique(arr))
        lens.append(len(toks))
    sizes = np.array([len(x) for x in out], dtype=np.int64)
    flat = np.concatenate(out) if out else np.zeros(0, np.uint64)
    return sizes, flat, np.array(lens, dtype=np.int32)


def _shingle_batch_whitespace(texts):
    """Project (C4/FineWeb) tokenisation: simplify_text then split on spaces."""
    sys.path.insert(0, os.path.join(CORE, "vendor"))
    import common as C
    out, lens = [], []
    for t in texts:
        s, nw = C.shingle_doc(t)
        out.append(s)
        lens.append(nw)
    sizes = np.array([len(x) for x in out], dtype=np.int64)
    flat = np.concatenate(out) if out else np.zeros(0, np.uint64)
    return sizes, flat, np.array(lens, dtype=np.int32)


def shingle_all(fn=_shingle_batch_spacy, workers=WORKERS, batch=BATCH, progress=25):
    """Parallel shingling of the whole snapshot; returns off, flat, nwords, meta, timing."""
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor

    t0 = time.perf_counter()
    parts, nwords, meta = [], [], {"id": [], "segment": [], "url": [],
                                   "nwords": [], "nchar": [], "nshingle": []}
    pend = deque()
    max_inflight = 2 * workers
    n_done = 0
    t_first = None

    def collect(fut):
        nonlocal n_done, t_first
        sizes, flat, nw = fut.result()
        parts.append((sizes, flat))
        nwords.append(nw)
        n_done += len(sizes)
        if t_first is None:
            t_first = time.perf_counter() - t0
        if n_done % (progress * batch) < batch:
            print(f"  shingled {n_done} docs, {time.perf_counter()-t0:.0f}s", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for ids, segs, urls, texts, nw, nch in iter_text_batches(batch):
            pend.append(ex.submit(fn, texts))
            meta["id"].extend(ids)
            meta["segment"].extend(segs)
            meta["url"].extend(urls)
            meta["nwords"].extend(nw)
            meta["nchar"].extend(nch)
            while len(pend) >= max_inflight:
                collect(pend.popleft())
        while pend:
            collect(pend.popleft())
    t_sh = time.perf_counter() - t0
    sizes = np.concatenate([p[0] for p in parts]).astype(np.int64)
    nw_all = np.concatenate(nwords).astype(np.int64)
    tot = int(sizes.sum())
    flat = np.empty(tot, np.uint64)
    i = 0
    while parts:
        _, f = parts.pop(0)
        flat[i:i + len(f)] = f
        i += len(f)
        del f
    off = np.zeros(len(sizes) + 1, np.int64)
    np.cumsum(sizes, out=off[1:])
    meta["nshingle"] = (off[1:] - off[:-1]).tolist()
    return off, flat, nw_all, meta, dict(t_shingle_s=round(t_sh, 1), workers=workers,
                                         n_docs=int(len(sizes)), n_shingles=tot,
                                         first_batch_s=None if t_first is None else round(t_first, 1))


def load_snapshot():
    z = np.load(SHINGLE_NPZ)
    off = np.asarray(z["off"], dtype=np.int64)
    flat = np.asarray(z["flat"], dtype=np.uint64)
    nwords = np.asarray(z["nwords"], dtype=np.int64)
    z.close()
    return off, flat, nwords


# ---------------------------------------------------------------- truth / ranks
def truth_path(tau):
    return os.path.join(CACHE, f"truth_snap_t{tau}.npz")


def build_ranks():
    """Cold rank build for the snapshot (int32, per-doc sorted, df-ascending order)."""
    import sys as _s
    _s.path.insert(0, os.path.join(CORE, "vendor"))
    import exactjoin as EJ
    off, flat, _ = load_snapshot()
    t0 = time.time()
    R, S, V = EJ.build_ranks(off, flat)
    dt = time.time() - t0
    os.makedirs(CACHE, exist_ok=True)
    np.savez(RANKS, R=R, S=np.int64(S), V=np.int64(V), t_build_s=np.float64(dt))
    return R, S, V, dt


def compute_truth(tau, threads=K.THREADS, off=None, flat=None):
    """Exact Jaccard graph at tau; cached. Evidence tier: rerun.

    Pass already-loaded `off`/`flat` to avoid holding a second copy of the
    shingle arrays (the snapshot flat array is ~4 GiB)."""
    p = truth_path(tau)
    if os.path.exists(p):
        z = np.load(p)
        out = (np.asarray(z["i"]), np.asarray(z["j"]), np.asarray(z["jac"]))
        z.close()
        return out
    R, S, V, _ = ranks()
    if off is None:
        off, flat, _ = load_snapshot()
    res = K.EJ.exact_join(off, R, S, V, tau, threads)
    i = res["i"].astype(np.int64)
    j = res["j"].astype(np.int64)
    jac = res["jac"].astype(np.float64)
    tmp = p + ".tmp.npz"
    np.savez(tmp, i=i, j=j, jac=jac)
    os.replace(tmp, p)
    return i, j, jac


def ranks():
    z = np.load(RANKS)
    R = np.asarray(z["R"])
    S, V = int(z["S"]), int(z["V"])
    z.close()
    return R, S, V, 0.0


def append_row(path, row):
    row = dict(row)
    row.setdefault("ts_utc", now())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def done_keys(path, key="key"):
    s = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                s.add(json.loads(line).get(key))
            except Exception:
                pass
    return s
