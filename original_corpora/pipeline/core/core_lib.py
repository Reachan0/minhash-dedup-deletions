#!/usr/bin/env python3
"""Core engineering library for DedupJoin v3 (step core-cost-benchmark).

One implementation of every stage, shared by the accuracy, timing, scaling and
cross-check drivers, so that A/B/W/C run on identical documents, tokenisation,
5-gram shingles, hash family, hardware and timing boundaries.

Frozen definitions: see `.work/dedupjoin-pvldb-v3/audit/glossary.md`.
Evidence tier of every row written by the drivers: `rerun`.
"""
import json
import os
import resource
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "vendor")
sys.path.insert(0, VENDOR)

import common as C  # noqa: E402  (Draft 2 shingle/normalisation semantics)
import exactjoin as EJ  # noqa: E402
import lsh as L  # noqa: E402

def _find_root(start):
    """Workspace root = nearest ancestor that holds the frozen .work/dedupjoin data.

    The published copy of this code lives under output/<slug>/experiment/code/core,
    one level deeper than the working copy, so a fixed number of ".." would break;
    searching for .work/dedupjoin makes both layouts resolve to the same root.
    """
    p = start
    for _ in range(8):
        if os.path.isdir(os.path.join(p, ".work", "dedupjoin")):
            return p
        p = os.path.dirname(p)
    return os.path.abspath(os.path.join(start, "..", "..", "..", ".."))


ROOT = _find_root(HERE)
SLUG = os.path.join(ROOT, ".work", "dedupjoin-pvldb-v3")
CORE = os.path.join(SLUG, "core")
CACHE = os.path.join(CORE, "cache")
RUNS = os.path.join(CORE, "runs.jsonl")
TIMING = os.path.join(CORE, "timing.jsonl")
SCALING = os.path.join(CORE, "bucket_scaling.jsonl")
REPRULES = os.path.join(CORE, "representative_rules.jsonl")
POOL = os.path.join(ROOT, ".work", "dedupjoin", "pool")

H, B, R = 112, 14, 8          # 112 permutations, 14 bands x 8 rows (datatrove default)
S_CURVE_075 = None            # design target: FineWeb 0.75; the S-curve is reported in analysis
TAUS = [0.8, 0.75]
THREADS = 12                  # fixed for every path; also a 1-thread variant is timed

# pool order is fw -> c4 -> wet (build_summary.json); local row ranges define the workloads
POOL_COUNTS = {"fw": 1048581, "c4": 2850538, "wet": 175272}
POOL_START = {"fw": 0, "c4": POOL_COUNTS["fw"], "wet": POOL_COUNTS["fw"] + POOL_COUNTS["c4"]}
# each source starts on a fresh 100k-row shard (source-boundary-safe pool build)
SHARD_BASE = {"fw": 0, "c4": 11, "wet": 40}

WORKLOADS = {
    # name: list of (pool source, local start, count)
    "c4":  [("c4", 0, 700000)],
    "fw":  [("fw", 0, 1000000)],
    "mix": [("fw", 0, 500000), ("c4", 0, 500000)],
    "wet": [("wet", 0, 175272)],
}

SHINGLE_CACHE = {
    "c4":  os.path.join(ROOT, ".work/dedupjoin/rerun/c4_700000.npz"),
    "fw":  os.path.join(ROOT, ".work/dedupjoin/rerun/fw_1000000.npz"),
    "mix": os.path.join(ROOT, ".work/dedupjoin/main/mix_1000000.npz"),
    "wet": os.path.join(ROOT, ".work/dedupjoin/main/wet_175272.npz"),
}
META_CACHE = {
    "c4":  os.path.join(ROOT, ".work/dedupjoin/rerun/c4_700000_meta.parquet"),
    "fw":  os.path.join(ROOT, ".work/dedupjoin/rerun/fw_1000000_meta.parquet"),
    "mix": os.path.join(ROOT, ".work/dedupjoin/main/mix_1000000_meta.parquet"),
    "wet": os.path.join(ROOT, ".work/dedupjoin/main/wet_175272_meta.parquet"),
}
RANKS_CACHE = {w: os.path.join(ROOT, ".work/dedupjoin/main", f"{w}_ranks.npz") for w in WORKLOADS}


# ------------------------------------------------------------------ bookkeeping
def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    s.add(json.loads(line).get(key))
                except Exception:
                    pass
    return s


def peak_rss_mb():
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024.0


def read_bytes():
    """Bytes actually fetched from storage by this process (page-cache probe)."""
    try:
        with open("/proc/self/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return None


def hw_facts():
    cpu = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    mem = None
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            mem = int(open(p).read().strip())
            break
        except Exception:
            continue
    try:
        nproc = len(os.sched_getaffinity(0))
    except Exception:
        nproc = os.cpu_count()
    online = None
    try:
        online = sum(1 for line in open("/proc/cpuinfo") if line.startswith("processor"))
    except Exception:
        pass
    return dict(cpu_model=cpu, nproc_affinity=nproc, cpus_online_proc_cpuinfo=online,
                omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
                cgroup_memory_max_bytes=mem, cgroup_cpu_max=open(
                    "/sys/fs/cgroup/cpu.max").read().strip() if os.path.exists(
                    "/sys/fs/cgroup/cpu.max") else None,
                threads=THREADS, python=sys.version.split()[0], numpy=np.__version__)


def set_threads(t):
    C.set_threads(t)
    os.environ["OMP_NUM_THREADS"] = str(t)


# ------------------------------------------------------------------ documents
def load_shingles(workload):
    """Return off (int64), flat (uint64), nwords. Reused in place from the frozen caches."""
    z = np.load(SHINGLE_CACHE[workload])
    off = np.asarray(z["off"], dtype=np.int64)
    flat = np.asarray(z["flat"], dtype=np.uint64)
    nwords = np.asarray(z["nwords"], dtype=np.int64)
    z.close()
    return off, flat, nwords


def load_ids(workload):
    md = pq.read_table(META_CACHE[workload], columns=["id"]).to_pydict()
    return md["id"]


def _pool_shard(source, local_row):
    shard = SHARD_BASE[source] + local_row // 100000
    p = os.path.join(POOL, source, f"shard-{shard:05d}.parquet")
    return p, local_row % 100000


def iter_pool_texts(workload, batch=4000):
    """Yield text batches in workload order, read directly from the frozen pool parquet."""
    for source, start, count in WORKLOADS[workload]:
        yield from iter_pool_prefix_texts(source, count, start=start, batch=batch)


def iter_pool_prefix_texts(source, count, start=0, batch=4000):
    """Yield text batches for a pool prefix (source local rows [start, start+count))."""
    got = 0
    while got < count:
        p, off = _pool_shard(source, start + got)
        pf = pq.ParquetFile(p)
        # read only the row groups that intersect [off, off+need)
        remaining = count - got
        acc = []
        base = 0
        for rg in range(pf.metadata.num_row_groups):
            nrg = pf.metadata.row_group(rg).num_rows
            lo, hi = base, base + nrg
            base = hi
            if hi <= off or lo >= off + remaining:
                continue
            t = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
            a = max(0, off - lo)
            b = min(len(t), off + remaining - lo)
            acc.extend(t[a:b])
        got += len(acc)
        for i in range(0, len(acc), batch):
            yield acc[i:i + batch]
        del acc


def _shingle_texts(it, workers):
    t0 = time.perf_counter()
    parts, nwords = [], []
    pend = deque()
    max_inflight = 2 * max(1, workers)

    def collect(fut):
        sizes, flat, nw = fut.result()
        parts.append((sizes, flat))
        nwords.append(nw)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for texts in it:
            pend.append(ex.submit(C._shingle_chunk, texts))
            while len(pend) >= max_inflight:
                collect(pend.popleft())
        while pend:
            collect(pend.popleft())
    t_sh = time.perf_counter() - t0
    sizes = np.concatenate([p[0] for p in parts]).astype(np.int64)
    nw = np.concatenate(nwords).astype(np.int64)
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
    del sizes
    return off, flat, nw, dict(t_read_shingle_s=t_sh, workers=workers)


def shingle_from_pool(workload, workers=THREADS, progress=None):
    """Cold path: read text from the frozen pool and build 5-gram shingle sets."""
    return _shingle_texts(iter_pool_texts(workload), workers)


def shingle_pool_prefix(source, count, workers=THREADS, start=0):
    """Cold path for a pool prefix, used by the bucket-scaling experiment."""
    return _shingle_texts(iter_pool_prefix_texts(source, count, start=start), workers)


# ------------------------------------------------------------------ hash family
def sig_path(workload, seed):
    return os.path.join(CACHE, "sig", f"{workload}_s{seed}.npy")


def signatures_cached(off, flat, workload, seed, threads=THREADS):
    """112 datatrove-compatible permutations (RandomState(seed), 2^61-1). Cached by seed."""
    p = sig_path(workload, seed)
    if os.path.exists(p):
        return np.load(p, mmap_mode="r"), True
    os.makedirs(os.path.dirname(p), exist_ok=True)
    sig = L.signatures(off, flat, H, seed, "dt")
    tmp = p + ".tmp.npy"
    np.save(tmp, sig)
    os.replace(tmp, p)
    return sig, False


def band_keys(sig):
    return L.band_keys(sig, B, R)


# ------------------------------------------------------------------ pair enumeration
def band_groups(K, valid):
    """Yield (band_index, sorted_doc_ids_in_group) for every group of >=2 docs sharing
    a band signature. Docs inside a group are in increasing id order (stable sort), which
    is datatrove's tie order (sort key = sig tuple, doc id last)."""
    idx_all = np.nonzero(valid)[0]
    for b in range(K.shape[0]):
        k = np.asarray(K[b, idx_all])
        o = np.argsort(k, kind="stable")
        ks = k[o]
        docs = idx_all[o]
        if len(ks) == 0:
            continue
        brk = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        ends = np.r_[brk[1:], len(ks)]
        for s, e in zip(brk, ends):
            if e - s >= 2:
                yield b, docs[s:e]


def chain_edges(K, valid):
    """Path A edges: consecutive documents inside every bucket (datatrove stage 2).
    Returns ei, ej in emission order, plus bucket statistics; never expands all pairs."""
    idx_all = np.nonzero(valid)[0]
    eis, ejs = [], []
    largest, sum_pairs = 0, 0
    for b in range(K.shape[0]):
        k = np.asarray(K[b, idx_all])
        o = np.argsort(k, kind="stable")
        ks = k[o]
        docs = idx_all[o]
        same = np.flatnonzero(ks[1:] == ks[:-1])
        eis.append(docs[same])
        ejs.append(docs[same + 1])
        if len(ks):
            brk = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
            ends = np.r_[brk[1:], len(ks)]
            sz = ends - brk
            largest = max(largest, int(sz.max()))
            sum_pairs += int((sz.astype(np.int64) * (sz - 1) // 2).sum())
    ei = np.concatenate(eis).astype(np.int64) if eis else np.zeros(0, np.int64)
    ej = np.concatenate(ejs).astype(np.int64) if ejs else np.zeros(0, np.int64)
    return ei, ej, dict(largest_bucket=largest, bucket_pairs_sum=sum_pairs,
                        n_chain_pairs=int(len(ei)))


def all_bucket_pairs(K, valid, n):
    """Path B candidate edges: all pairs inside every bucket, deduplicated across bands.
    Returns pi, pj (int64, i<j), n_pairs_mult (before dedup), largest_bucket, bucket_pairs_sum."""
    codes = []
    largest, sum_pairs = 0, 0
    for b, docs in band_groups(K, valid):
        s = len(docs)
        largest = max(largest, s)
        sum_pairs += s * (s - 1) // 2
        ii, jj = np.triu_indices(s, 1)
        codes.append(docs[ii].astype(np.int64) * np.int64(n) + docs[jj])
    if codes:
        c = np.unique(np.concatenate(codes))
    else:
        c = np.zeros(0, np.int64)
    pi = c // n
    pj = c % n
    return pi.astype(np.int64), pj.astype(np.int64), dict(
        n_pairs_mult=int(sum_pairs), largest_bucket=int(largest),
        bucket_pairs_sum=int(sum_pairs), n_pairs_unique=int(len(c)))


# ------------------------------------------------------------------ clustering
def uf_minid(n, ei, ej):
    return L.uf_components(n, np.asarray(ei, dtype=np.int64), np.asarray(ej, dtype=np.int64))


def removed_minid(root):
    return L.removed_set(root)


@np.errstate(all="ignore")
def _uf_by_size_kernel(n, ei, ej):
    import numba as nb

    @nb.njit(cache=True)
    def _uf(n, ei, ej):
        par = np.arange(n)
        sz = np.ones(n, np.int64)
        for k in range(ei.shape[0]):
            a = ei[k]
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            b = ej[k]
            while par[b] != b:
                par[b] = par[par[b]]
                b = par[b]
            if a != b:
                # datatrove stage 3: larger set wins; on a tie the first argument stays root
                if sz[a] < sz[b]:
                    a, b = b, a
                par[b] = a
                sz[a] += sz[b]
        root = np.empty(n, np.int64)
        for x in range(n):
            y = x
            while par[y] != y:
                y = par[y]
            root[x] = y
        return root

    return _uf(n, np.asarray(ei, np.int64), np.asarray(ej, np.int64))


def uf_datatrove(n, ei, ej):
    """datatrove 0.6.0 stage-3 union-by-size over the edges in emission order."""
    return _uf_by_size_kernel(n, ei, ej)


@np.errstate(all="ignore")
def greedy_mis(n, ei, ej):
    """Draft 2 witness rule: greedy maximal independent set in increasing doc id."""
    import numba as nb

    @nb.njit(cache=True)
    def _mis(n, ei, ej):
        deg = np.zeros(n, np.int64)
        for k in range(ei.shape[0]):
            deg[ei[k]] += 1
            deg[ej[k]] += 1
        off = np.zeros(n + 1, np.int64)
        for i in range(n):
            off[i + 1] = off[i] + deg[i]
        fill = off[:-1].copy()
        adj = np.empty(off[-1], np.int32)
        for k in range(ei.shape[0]):
            a, b = ei[k], ej[k]
            adj[fill[a]] = b
            fill[a] += 1
            adj[fill[b]] = a
            fill[b] += 1
        kept = np.zeros(n, np.uint8)
        for d in range(n):
            ok = 1
            for q in range(off[d], off[d + 1]):
                if kept[adj[q]]:
                    ok = 0
                    break
            kept[d] = ok
        return kept

    return _mis(n, ei.astype(np.int32), ej.astype(np.int32)).astype(bool)


# ------------------------------------------------------------------ truth / metrics
def truth_path(workload, tau):
    return os.path.join(CACHE, f"truth_{workload}_t{tau}.npz")


def compute_truth(workload, tau, off, threads=THREADS):
    """Exact Jaccard graph at tau using the frozen global rank caches."""
    p = truth_path(workload, tau)
    if os.path.exists(p):
        z = np.load(p)
        out = (np.asarray(z["i"]), np.asarray(z["j"]), np.asarray(z["jac"]))
        z.close()
        return out
    z = np.load(RANKS_CACHE[workload])
    Rr, S, V = np.asarray(z["R"]), int(z["S"]), int(z["V"])
    z.close()
    res = EJ.exact_join(off, Rr, S, V, tau, threads)
    i = res["i"].astype(np.int64)
    j = res["j"].astype(np.int64)
    jac = res["jac"].astype(np.float64)
    os.makedirs(CACHE, exist_ok=True)
    tmp = p + ".tmp.npz"
    np.savez(tmp, i=i, j=j, jac=jac)
    os.replace(tmp, p)
    return i, j, jac


def has_true_neighbour(n, ti, tj):
    h = np.zeros(n, bool)
    h[ti] = True
    h[tj] = True
    return h


def witness_coverage(n, removed, ti, tj):
    """Share of deleted docs with at least one retained direct true neighbour (J >= tau)."""
    if removed.sum() == 0:
        return 1.0, 0
    cnt = (np.bincount(ti, weights=(~removed[tj]).astype(np.float64), minlength=n)
           + np.bincount(tj, weights=(~removed[ti]).astype(np.float64), minlength=n))
    covered = removed & (cnt > 0)
    return float(covered.sum() / removed.sum()), int(covered.sum())


def method_metrics(n, removed, ti, tj, has_nb=None):
    removed = np.asarray(removed, dtype=bool)
    if has_nb is None:
        has_nb = has_true_neighbour(n, ti, tj)
    unsupported = removed & ~has_nb
    cov, cov_n = witness_coverage(n, removed, ti, tj)
    residual = int((~removed[ti] & ~removed[tj]).sum())
    return dict(
        deleted=int(removed.sum()),
        unsupported=int(unsupported.sum()),
        unsupported_share_del=float(unsupported.sum() / removed.sum()) if removed.sum() else None,
        unsupported_share_input=float(unsupported.sum() / n),
        witness_coverage=cov,
        witness_covered=int(cov_n),
        residual_duplicate_pairs=residual,
    )


def edge_stats(ti, tj, pi, pj, n=None):
    """Pair recall and candidate precision of a candidate edge set against the truth.
    Vectorised: pairs are encoded as i*n+j and matched with searchsorted (truth edge
    counts reach 1e7 on FineWeb, where a Python set of tuples does not fit the budget)."""
    n_truth = len(ti)
    if n_truth == 0:
        return dict(pair_recall=None, candidate_precision=None)
    if n is None:
        n = int(max(ti.max(initial=0), tj.max(initial=0),
                    pi.max(initial=0), pj.max(initial=0))) + 1
    tc = np.sort(np.asarray(ti, np.int64) * n + np.asarray(tj, np.int64))
    pc = np.unique(np.asarray(pi, np.int64) * n + np.asarray(pj, np.int64))
    pos = np.searchsorted(tc, pc)
    pos[pos >= len(tc)] = 0
    hit = int((tc[pos] == pc).sum()) if len(pc) else 0
    return dict(pair_recall=float(hit / n_truth) if n_truth else None,
                candidate_precision=float(hit / len(pc)) if len(pc) else None,
                candidate_edges_unique=int(len(pc)))


# ------------------------------------------------------------------ paths
def run_path_a(K, valid, n, ti, tj, has_nb):
    ei, ej, bstats = chain_edges(K, valid)
    root = uf_minid(n, ei, ej)
    rem = removed_minid(root) & valid
    m = method_metrics(n, rem, ti, tj, has_nb)
    m.update(bstats)
    m["n_candidate_edges"] = int(len(ei))
    m.update(edge_stats(ti, tj, ei, ej, n))
    return m, (ei, ej, root, rem)


def run_path_b(K, valid, n, tau, ti, tj, has_nb, off, flat):
    pi, pj, bstats = all_bucket_pairs(K, valid, n)
    jac = L.jaccard_pairs(off, flat, pi, pj)
    ver = jac >= tau - 1e-12
    ei, ej = pi[ver], pj[ver]
    root = uf_minid(n, ei, ej)
    rem = removed_minid(root) & valid
    m = method_metrics(n, rem, ti, tj, has_nb)
    m.update(bstats)
    m["n_verified_edges"] = int(ver.sum())
    m.update(edge_stats(ti, tj, ei, ej, n))
    return m, (pi, pj, jac, ver, ei, ej, root, rem)


def witness_from_verified(n, ei, ej, valid, ti, tj, has_nb):
    kept = greedy_mis(n, ei, ej)
    rem = (~kept) & valid
    m = method_metrics(n, rem, ti, tj, has_nb)
    return m, rem


def run_path_c(off, workload, tau, n, ti, tj, has_nb, threads=THREADS):
    z = np.load(RANKS_CACHE[workload])
    Rr, S, V = np.asarray(z["R"]), int(z["S"]), int(z["V"])
    z.close()
    res = EJ.exact_join(off, Rr, S, V, tau, threads)
    ei, ej = res["i"].astype(np.int64), res["j"].astype(np.int64)
    root = uf_minid(n, ei, ej)
    rem = removed_minid(root)
    m = method_metrics(n, rem, ti, tj, has_nb)
    m["n_truth_edges"] = int(res["n_pairs"] if "n_pairs" in res else len(ei))
    m["index_entries"] = int(res["index_entries"])
    m["n_candidates"] = int(res["n_candidates"])
    return m, (res, root, rem)


def workload_overlap_facts():
    """Static provenance facts reused from audit/provenance_overlap.json (no recomputation)."""
    p = os.path.join(SLUG, "audit", "provenance_overlap.json")
    try:
        return json.load(open(p))
    except Exception:
        return {}
