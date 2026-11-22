"""C03 DedupJoin - shared utilities: datatrove-compatible normalisation/shingling,
corpus loading, shingle cache (CSR), measured-row writer, peak RSS."""
import gzip
import json
import os
import re
import resource
import time
import unicodedata

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
RUN = os.path.join(ROOT, ".work", "vldb2027-ai-db-ideas", "experiment")
DATA = os.path.join(ROOT, ".work", "data", "C03")
LOGS = os.path.join(RUN, "logs", "C03")

FW_PARQUET = os.environ.get("C03_FW_PARQUET", os.path.join(DATA, "raw_fw", "000_00000.parquet"))
FW_SUBSET = os.environ.get("C03_FW_SUBSET", os.path.join(DATA, "fw_subset.parquet"))
C4_DIR = os.environ.get("C03_C4_DIR", os.path.join(DATA, "raw_c4"))
CACHE = os.environ.get("C03_CACHE", os.path.join(DATA, "cache"))

# ---- datatrove simplify_text (TextNormConfig defaults), copied semantics ----
PUNCTUATION = "!/—”:％１〈&(、━\\【#%「」，】；+^]~“《„';’{|∶´[=-`*．（–？！：$～«〉,><》)?）。…@_.\"}►»" + "".join(
    map(chr, (x for a, b in ((0, 9), (11, 13), (13, 32), (127, 160)) for x in range(a, b))))
PUNCTUATION_TRANS = str.maketrans(PUNCTUATION, " " * len(PUNCTUATION))
# python `re` \d == Unicode Nd, identical to regex \p{Nd}
NUMBERS_PATTERN = re.compile(r"\d+([.,،٫⎖⎗⎘]{1}\d+)?")
WHITESPACE_PATTERN = re.compile(r"\s+")


def simplify_text(text):
    text = text.lower()
    text = NUMBERS_PATTERN.sub("0", text)
    text = text.translate(PUNCTUATION_TRANS)
    text = WHITESPACE_PATTERN.sub(" ", text.strip())
    if not text.isascii():  # ASCII has no combining marks after NFD -> identical result
        text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return text.strip()


def shingle_doc(text, n=5):
    """datatrove get_shingles: xxhash64 of ' '.join(n-gram) over word tokens; we return the
    *set* (sorted unique uint64). Word tokenizer = whitespace split after simplify_text
    (datatrove uses an NLTK/spaCy word tokenizer; after punctuation removal they coincide
    for almost all English text -- documented deviation)."""
    import xxhash
    w = simplify_text(text).split(" ")
    if len(w) < n or w == [""]:
        return np.zeros(0, dtype=np.uint64), len(w) if w != [""] else 0
    h = xxhash.xxh64_intdigest
    arr = np.fromiter((h(" ".join(w[i:i + n]).encode("utf-8")) for i in range(len(w) - n + 1)),
                      dtype=np.uint64, count=len(w) - n + 1)
    return np.unique(arr), len(w)


def _shingle_chunk(texts):
    out, lens = [], []
    for t in texts:
        s, nw = shingle_doc(t)
        out.append(s)
        lens.append(nw)
    sizes = np.array([len(s) for s in out], dtype=np.int64)
    flat = np.concatenate(out) if out else np.zeros(0, np.uint64)
    return sizes, flat, np.array(lens, dtype=np.int32)


# ---- corpora ----
def load_corpus(corpus, n):
    """Returns (texts, meta dict of lists). fw = FineWeb sample-10BT first n rows of
    000_00000.parquet (rows mix 8 CC dumps); c4 = first n lines of c4 en train shards."""
    if corpus == "fw":
        import pyarrow.parquet as pq
        src = FW_SUBSET if os.path.exists(FW_SUBSET) else FW_PARQUET
        pf = pq.ParquetFile(src)
        cols = ["text", "id", "dump", "url"]
        texts, meta, got = [], {"id": [], "dump": [], "url": []}, 0
        for bi in pf.iter_batches(batch_size=10000, columns=cols):
            d = bi.to_pydict()
            k = min(n - got, len(d["text"]))
            texts += d["text"][:k]
            for c in meta:
                meta[c] += d[c][:k]
            got += k
            if got >= n:
                break
        if got < n:
            raise RuntimeError(f"fw has only {got} rows < {n}")
        return texts, meta
    if corpus == "c4":
        texts, meta = [], {"id": [], "dump": [], "url": []}
        shard = 0
        while len(texts) < n:
            p = os.path.join(C4_DIR, f"c4-train.{shard:05d}-of-01024.json.gz")
            if not os.path.exists(p):
                raise RuntimeError(f"missing {p}")
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    r = json.loads(line)
                    texts.append(r["text"])
                    meta["id"].append(f"c4-{shard:05d}-{i}")
                    meta["dump"].append(f"c4-{shard:05d}")
                    meta["url"].append(r.get("url", ""))
                    if len(texts) >= n:
                        break
            shard += 1
        return texts, meta
    if corpus.startswith("mix"):  # fw + c4 half/half (round 2)
        a, ma = load_corpus("fw", n // 2)
        b, mb = load_corpus("c4", n - n // 2)
        return a + b, {k: ma[k] + mb[k] for k in ma}
    raise ValueError(corpus)


def iter_corpus(corpus, n, batch=2000):
    """Streaming version of load_corpus: yields (texts, meta) batches; never holds the corpus."""
    if corpus == "fw":
        import pyarrow.parquet as pq
        src = FW_SUBSET if os.path.exists(FW_SUBSET) else FW_PARQUET
        pf = pq.ParquetFile(src)
        got = 0
        for bi in pf.iter_batches(batch_size=batch, columns=["text", "id", "dump", "url"]):
            d = bi.to_pydict()
            k = min(n - got, len(d["text"]))
            yield d["text"][:k], {c: d[c][:k] for c in ("id", "dump", "url")}
            got += k
            if got >= n:
                return
        raise RuntimeError(f"fw has only {got} rows < {n}")
    if corpus == "c4":
        got, shard = 0, 0
        while got < n:
            p = os.path.join(C4_DIR, f"c4-train.{shard:05d}-of-01024.json.gz")
            if not os.path.exists(p):
                raise RuntimeError(f"missing {p}")
            buf, mb = [], {"id": [], "dump": [], "url": []}
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    r = json.loads(line)
                    buf.append(r["text"])
                    mb["id"].append(f"c4-{shard:05d}-{i}")
                    mb["dump"].append(f"c4-{shard:05d}")
                    mb["url"].append(r.get("url", ""))
                    got += 1
                    if len(buf) == batch or got >= n:
                        yield buf, mb
                        buf, mb = [], {"id": [], "dump": [], "url": []}
                    if got >= n:
                        return
            if buf:
                yield buf, mb
            shard += 1
        return
    if corpus.startswith("mix"):
        yield from iter_corpus("fw", n // 2, batch)
        yield from iter_corpus("c4", n - n // 2, batch)
        return
    raise ValueError(corpus)


def _shingle_chunk_meta(texts):
    sizes, flat, nw = _shingle_chunk(texts)
    return sizes, flat, nw, np.array([len(t) for t in texts], np.int64), [t[:300] for t in texts]


def _shingle_chunk_nometa(texts):
    sizes, flat, nw = _shingle_chunk(texts)
    return sizes, flat, nw, np.array([len(t) for t in texts], np.int64), []


def shingle_stream(corpus, n, workers, max_inflight=None, concat=True, keep_meta=True):
    """Bounded-memory parallel shingling straight from the corpus files.
    Returns off, flat, nwords, meta(dict: id, dump, url, nchar, head).
    concat=False: returns off, list-of-chunk-arrays (no contiguous copy; the caller frees
    chunks as it consumes them), nwords, meta.  keep_meta=False skips url/id/head strings."""
    from concurrent.futures import ProcessPoolExecutor
    from collections import deque
    max_inflight = max_inflight or 2 * max(1, workers)
    parts, meta = [], {"id": [], "dump": [], "url": [], "nchar": [], "head": []}
    pend = deque()

    def collect(fut, mb):
        s, f, w, nc, hd = fut.result() if hasattr(fut, "result") else fut
        parts.append((s, f, w))
        if keep_meta:
            for c in ("id", "dump", "url"):
                meta[c] += mb[c]
        meta["nchar"].append(nc)
        meta["head"] += hd

    fn = _shingle_chunk_meta if keep_meta else _shingle_chunk_nometa
    if workers <= 1:
        for texts, mb in iter_corpus(corpus, n):
            collect(fn(texts), mb)
    else:
        with ProcessPoolExecutor(workers) as ex:
            for texts, mb in iter_corpus(corpus, n):
                pend.append((ex.submit(fn, texts), mb))
                while len(pend) >= max_inflight:
                    collect(*pend.popleft())
            while pend:
                collect(*pend.popleft())
    sizes = np.concatenate([p[0] for p in parts])
    nw = np.concatenate([p[2] for p in parts])
    if not concat:
        off = np.zeros(len(sizes) + 1, np.int64)
        np.cumsum(sizes, out=off[1:])
        meta["nchar"] = np.concatenate(meta["nchar"])
        return off, [p[1] for p in parts], nw, meta
    tot = int(sizes.sum())
    flat = np.empty(tot, np.uint64)
    i = 0
    while parts:  # fill and free chunk by chunk (peak ~ 1x flat + one chunk)
        s, f, w = parts.pop(0)
        flat[i:i + len(f)] = f
        i += len(f)
        del f
    off = np.zeros(len(sizes) + 1, np.int64)
    np.cumsum(sizes, out=off[1:])
    meta["nchar"] = np.concatenate(meta["nchar"])
    return off, flat, nw, meta


def build_shingles(texts, workers):
    """Parallel shingling. Returns offsets(int64, n+1), flat uint64, nwords int32."""
    from concurrent.futures import ProcessPoolExecutor
    chunk = 2000
    parts = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
    if workers <= 1:
        res = [_shingle_chunk(p) for p in parts]
    else:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_shingle_chunk, parts))
    sizes = np.concatenate([r[0] for r in res])
    flat = np.concatenate([r[1] for r in res])
    nw = np.concatenate([r[2] for r in res])
    off = np.zeros(len(sizes) + 1, np.int64)
    np.cumsum(sizes, out=off[1:])
    return off, flat, nw


def cache_path(corpus, n):
    return os.path.join(CACHE, f"{corpus}_{n}.npz")


def load_cache(corpus, n):
    z = np.load(cache_path(corpus, n), allow_pickle=False)
    return z["off"], z["flat"], z["nwords"], json.loads(str(z["info"]))


# ---- measurement helpers ----
def peak_rss_mb():
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ch_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return self_kb / 1024.0, ch_kb / 1024.0


def append_row(path, row):
    row = dict(row)
    row.setdefault("ts_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    row.setdefault("mode", "measured")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def done_keys(path):
    s = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    s.add(r.get("key"))
                except Exception:
                    pass
    return s


def set_threads(t):
    os.environ["OMP_NUM_THREADS"] = str(t)
    try:
        import numba
        numba.set_num_threads(t)
    except Exception:
        pass
