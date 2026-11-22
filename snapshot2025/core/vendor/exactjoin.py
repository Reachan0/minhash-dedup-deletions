"""C03 exact set-similarity self-join (AllPairs / PPJoin-style prefix filtering) on
datatrove-style 5-gram shingle sets.  Global token order = document frequency ascending.
Filters: prefix (probe prefix + PPJoin mid-prefix index), length, positional; then
merge-based verification.  Core loops in numba (nogil), parallelised with T Python threads.

Usage (one cell per process so that peak RSS is per cell):
  python exactjoin.py --corpus fw --n 200000 --tau 0.8 --threads 2 --workers 2
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

import numba as nb  # noqa: E402


# ------------------------------------------------------------------ ranks
@nb.njit(parallel=True, cache=True)
def _sort_segments(R, off):
    n = off.shape[0] - 1
    for d in nb.prange(n):
        R[off[d]:off[d + 1]] = np.sort(R[off[d]:off[d + 1]])


def build_ranks_naive(off, flat):
    """(v0, kept for the record) np.unique(return_inverse) -- ran out of an 8 GiB address
    space at 150k FineWeb docs."""
    uniq, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    order = np.lexsort((uniq, cnt))
    rank_of = np.empty(len(uniq), np.int32)
    rank_of[order] = np.arange(len(uniq), dtype=np.int32)
    S = int((cnt == 1).sum())
    V = len(uniq)
    del uniq, order, cnt
    R = rank_of[inv]
    del inv, rank_of
    _sort_segments(R, off)
    return R, S, V


def build_ranks(off, flat, chunk=8_000_000):
    """Low-memory global order. Singleton shingles (df==1, ~98% of FineWeb 5-gram shingles)
    can never be shared, so they get unique ranks = their global position (all < S=N);
    non-singletons get ranks S + position in (df asc, hash asc) order. Returns R (int32,
    per-doc sorted), S, V (rank upper bound). Equivalent order to build_ranks_naive up to
    the (irrelevant) relative order of singletons."""
    N = len(flat)
    s = np.sort(flat)
    dup = s[1:] == s[:-1]
    rep = s[1:][dup]
    del s, dup
    nsv, c = np.unique(rep, return_counts=True)
    del rep
    cnt = c + 1
    order = np.lexsort((nsv, cnt))
    rank_of = np.empty(len(nsv), np.int64)
    rank_of[order] = np.arange(len(nsv), dtype=np.int64)
    S = N
    V = N + len(nsv)
    assert V < 2 ** 31 - 1
    if N == 0 or len(nsv) == 0:
        # every shingle is a singleton (or there is no shingle at all): ranks are the
        # global positions, which are already increasing inside each document
        R = np.arange(N, dtype=np.int32)
        return R, S, V
    R = np.empty(N, np.int32)
    for i0 in range(0, N, chunk):
        x = flat[i0:i0 + chunk]
        pos = np.searchsorted(nsv, x)
        pos[pos == len(nsv)] = 0
        isn = nsv[pos] == x if len(nsv) else np.zeros(len(x), bool)
        R[i0:i0 + len(x)] = np.where(isn, S + rank_of[pos], np.arange(i0, i0 + len(x)))
    _sort_segments(R, off)
    return R, S, V


def build_ranks_parts(off, parts, nbuckets=16, free=True):
    """Same ranks as build_ranks, but from a list of chunk arrays without ever building the
    contiguous uint64 array or a full sorted copy: duplicates are found per radix bucket of
    the top 4 hash bits; R is filled chunk by chunk while each uint64 chunk is released.
    Peak ~ (uint64 shingles) + (one bucket); then ~2x int32 ranks at the final concat."""
    N = int(off[-1])
    reps = []
    for b in range(nbuckets):
        sel = np.concatenate([a[(a >> np.uint64(60)) == np.uint64(b)] for a in parts]) if parts else np.zeros(0, np.uint64)
        sel.sort()
        reps.append(sel[1:][sel[1:] == sel[:-1]])
        del sel
    rep = np.concatenate(reps)
    del reps
    nsv, c = np.unique(rep, return_counts=True)
    del rep
    cnt = c + 1
    order = np.lexsort((nsv, cnt))
    rank_of = np.empty(len(nsv), np.int64)
    rank_of[order] = np.arange(len(nsv), dtype=np.int64)
    S = N
    V = N + len(nsv)
    assert V < 2 ** 31 - 1
    if N == 0 or len(nsv) == 0:
        return np.arange(N, dtype=np.int32), S, V
    Rp = []
    i0 = 0
    for k in range(len(parts)):
        x = parts[k]
        pos = np.searchsorted(nsv, x)
        pos[pos == len(nsv)] = 0
        isn = nsv[pos] == x if len(nsv) else np.zeros(len(x), bool)
        Rp.append(np.where(isn, S + rank_of[pos], np.arange(i0, i0 + len(x))).astype(np.int32))
        i0 += len(x)
        if free:
            parts[k] = None
        del x, pos, isn
    R = np.concatenate(Rp) if Rp else np.zeros(0, np.int32)
    del Rp
    _sort_segments(R, off)
    return R, S, V


# ------------------------------------------------------------------ index
@nb.njit(cache=True)
def _ceil_div(a, b):
    return -((-a) // b)


@nb.njit(cache=True)
def build_index(R, off, ordd, S, V, num, den, allpairs):
    """Inverted index over non-singleton prefix tokens. Entries per token sorted by order
    position k (=> by size). index prefix: PPJoin mid-prefix l - ceil(2t/(1+t) l) + 1, or
    AllPairs probe prefix l - ceil(t l) + 1 when allpairs=1."""
    n = ordd.shape[0]
    cnt = np.zeros(V - S + 1, np.int64)
    for k in range(n):
        d = ordd[k]
        l = off[d + 1] - off[d]
        if l == 0:
            continue
        if allpairs:
            ip = l - _ceil_div(num * l, den) + 1
        else:
            ip = l - _ceil_div(2 * num * l, num + den) + 1
        if ip > l:
            ip = l
        for j in range(ip):
            t = R[off[d] + j]
            if t >= S:
                cnt[t - S + 1] += 1
    for i in range(1, cnt.shape[0]):
        cnt[i] += cnt[i - 1]
    tot = cnt[-1]
    ent_k = np.empty(tot, np.int32)
    ent_p = np.empty(tot, np.int32)
    fill = cnt[:-1].copy()
    for k in range(n):
        d = ordd[k]
        l = off[d + 1] - off[d]
        if l == 0:
            continue
        if allpairs:
            ip = l - _ceil_div(num * l, den) + 1
        else:
            ip = l - _ceil_div(2 * num * l, num + den) + 1
        if ip > l:
            ip = l
        for j in range(ip):
            t = R[off[d] + j]
            if t >= S:
                q = fill[t - S]
                ent_k[q] = k
                ent_p[q] = j
                fill[t - S] = q + 1
    return cnt, ent_k, ent_p


@nb.njit(cache=True)
def _overlap(R, a0, a1, b0, b1, need):
    o = 0
    i = a0
    j = b0
    while i < a1 and j < b1:
        # early termination: even if all remaining match we cannot reach `need`
        rem = min(a1 - i, b1 - j)
        if o + rem < need:
            return o
        if R[i] == R[j]:
            o += 1
            i += 1
            j += 1
        elif R[i] < R[j]:
            i += 1
        else:
            j += 1
    return o


@nb.njit(nogil=True, cache=True)
def probe_range(R, off, ordd, sizes_ord, S, cnt, ent_k, ent_p, num, den, k0, k1,
                use_len, use_pos):
    n = ordd.shape[0]
    A = np.zeros(n, np.int32)
    seen = np.zeros(n, np.uint8)
    touched = np.empty(n, np.int32)
    cap = 1024
    out_a = np.empty(cap, np.int32)
    out_b = np.empty(cap, np.int32)
    out_o = np.empty(cap, np.int32)
    no = 0
    n_cand = 0
    for k in range(k0, k1):
        d = ordd[k]
        lx = sizes_ord[k]
        if lx == 0:
            continue
        minlen = _ceil_div(num * lx, den)
        px = lx - minlen + 1
        nt = 0
        for j in range(px):
            t = R[off[d] + j]
            if t < S:
                continue
            s0 = cnt[t - S]
            s1 = cnt[t - S + 1]
            # entries sorted by k (so by size): binary search first with size >= minlen
            lo = s0
            if use_len:
                hi = s1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if sizes_ord[ent_k[mid]] < minlen:
                        lo = mid + 1
                    else:
                        hi = mid
            for e in range(lo, s1):
                ky = ent_k[e]
                if ky >= k:
                    break
                if seen[ky] == 2:  # pruned
                    continue
                ly = sizes_ord[ky]
                if seen[ky] == 0:
                    seen[ky] = 1
                    touched[nt] = ky
                    nt += 1
                if use_pos:
                    alpha = _ceil_div(num * (lx + ly), num + den)
                    rx = lx - j - 1
                    ry = ly - ent_p[e] - 1
                    ub = A[ky] + 1 + (rx if rx < ry else ry)
                    if ub < alpha:
                        seen[ky] = 2
                        continue
                A[ky] += 1
        for q in range(nt):
            ky = touched[q]
            if seen[ky] == 1 and A[ky] > 0:
                n_cand += 1
                ly = sizes_ord[ky]
                alpha = _ceil_div(num * (lx + ly), num + den)
                if use_len and ly < minlen:
                    pass
                else:
                    dy = ordd[ky]
                    o = _overlap(R, off[d], off[d + 1], off[dy], off[dy + 1], alpha)
                    if o >= alpha:
                        if no == cap:
                            cap *= 2
                            na = np.empty(cap, np.int32)
                            na[:no] = out_a
                            out_a = na
                            nb_ = np.empty(cap, np.int32)
                            nb_[:no] = out_b
                            out_b = nb_
                            nc = np.empty(cap, np.int32)
                            nc[:no] = out_o
                            out_o = nc
                        out_a[no] = dy
                        out_b[no] = d
                        out_o[no] = o
                        no += 1
            A[ky] = 0
            seen[ky] = 0
    return out_a[:no], out_b[:no], out_o[:no], n_cand


def exact_join(off, R, S, V, tau, threads, allpairs=False, use_len=True, use_pos=True,
               nchunks=None):
    fr = Fraction(str(tau)).limit_denominator(1000)
    num, den = fr.numerator, fr.denominator
    sizes = (off[1:] - off[:-1]).astype(np.int64)
    ordd = np.lexsort((np.arange(len(sizes)), sizes)).astype(np.int64)
    sizes_ord = sizes[ordd]
    t0 = time.time()
    cnt, ent_k, ent_p = build_index(R, off, ordd, S, V, num, den, 1 if allpairs else 0)
    t_index = time.time() - t0
    n = len(ordd)
    nchunks = nchunks or max(64, threads * 32)
    bounds = np.linspace(0, n, nchunks + 1).astype(np.int64)
    t0 = time.time()
    args = [(int(bounds[i]), int(bounds[i + 1])) for i in range(nchunks)]

    def run(ab):
        return probe_range(R, off, ordd, sizes_ord, S, cnt, ent_k, ent_p, num, den, ab[0], ab[1],
                           use_len, use_pos)
    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(run, args))
    t_probe = time.time() - t0
    a = np.concatenate([r[0] for r in res])
    b = np.concatenate([r[1] for r in res])
    o = np.concatenate([r[2] for r in res])
    ncand = int(sum(r[3] for r in res))
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    la = sizes[lo]
    lb = sizes[hi]
    oo = o.astype(np.float64)
    jac = oo / (la + lb - oo)
    return dict(i=lo.astype(np.int32), j=hi.astype(np.int32), jac=jac, index_entries=int(len(ent_k)),
                t_index=t_index, t_probe_verify=t_probe, n_candidates=ncand)


def get_shingles_stream(corpus, n, workers):
    """No cache: bounded-memory streaming shingling (load+shingle timed together).
    Returns chunk list (not contiguous) -> use build_ranks_parts."""
    t0 = time.time()
    off, flat, nw, meta = C.shingle_stream(corpus, n, workers, concat=False, keep_meta=False)
    t_sh = time.time() - t0
    info = dict(t_load=0.0, t_shingle=t_sh, shingle_workers=workers, n_docs=n, streamed=True,
                n_shingles=int(off[-1]), n_empty=int(((off[1:] - off[:-1]) == 0).sum()), from_cache=False)
    del meta
    return off, flat, nw, info


def get_shingles(corpus, n, workers, rows_path=None):
    """Load cache or build it (timed). Returns off, flat, nwords, info."""
    p = C.cache_path(corpus, n)
    if os.path.exists(p):
        off, flat, nw, info = C.load_cache(corpus, n)
        info["from_cache"] = True
        return off, flat, nw, info
    t0 = time.time()
    texts, meta = C.load_corpus(corpus, n)
    t_load = time.time() - t0
    t0 = time.time()
    off, flat, nw = C.build_shingles(texts, workers)
    t_sh = time.time() - t0
    info = dict(t_load=t_load, t_shingle=t_sh, shingle_workers=workers, n_docs=n,
                n_shingles=int(off[-1]), n_empty=int(((off[1:] - off[:-1]) == 0).sum()))
    os.makedirs(C.CACHE, exist_ok=True)
    tmp = p + ".tmp.npz"
    np.savez(tmp, off=off, flat=flat, nwords=nw, info=json.dumps(info))
    os.replace(tmp, p)
    import pyarrow as pa
    import pyarrow.parquet as pq
    lens = np.array([len(t) for t in texts], np.int64)
    pq.write_table(pa.table({"id": meta["id"], "dump": meta["dump"], "url": meta["url"],
                             "nchar": lens, "nwords": nw,
                             "nshingle": (off[1:] - off[:-1]),
                             "head": [t[:200] for t in texts]}),
                   os.path.join(C.CACHE, f"{corpus}_{n}_meta.parquet"), compression="zstd")
    del texts, meta
    info["from_cache"] = False
    return off, flat, nw, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="fw")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--variant", default="ppjoin", choices=["ppjoin", "allpairs", "nopos", "nolen", "prefixonly"])
    ap.add_argument("--rows", default=os.path.join(C.LOGS, "rows.jsonl"))
    ap.add_argument("--tag", default="r1v1")
    ap.add_argument("--save_pairs", type=int, default=1)
    ap.add_argument("--no_cache", type=int, default=0)
    a = ap.parse_args()
    C.set_threads(a.threads)
    key = f"{a.tag}|exact|{a.corpus}|{a.n}|{a.tau}|{a.variant}|t{a.threads}"
    if key in C.done_keys(a.rows):
        print("skip", key)
        return
    T0 = time.time()
    if a.no_cache:
        off, flat, nw, info = get_shingles_stream(a.corpus, a.n, a.workers)
    else:
        off, flat, nw, info = get_shingles(a.corpus, a.n, a.workers)
    t_get = time.time() - T0
    t0 = time.time()
    if isinstance(flat, list):
        R, S, V = build_ranks_parts(off, flat)
    else:
        R, S, V = build_ranks(off, flat)
    del flat
    t_rank = time.time() - t0
    allp = a.variant in ("allpairs", "prefixonly")
    use_pos = a.variant not in ("nopos", "prefixonly", "allpairs")
    use_len = a.variant not in ("nolen", "prefixonly")
    res = exact_join(off, R, S, V, a.tau, a.threads, allpairs=allp, use_len=use_len, use_pos=use_pos)
    t_join_total = time.time() - T0
    rss_self, rss_ch = C.peak_rss_mb()
    if a.save_pairs and a.variant == "ppjoin":
        np.savez(os.path.join(C.CACHE, f"pairs_{a.corpus}_{a.n}_{a.tau}.npz"), i=res["i"], j=res["j"], jac=res["jac"])
    shingle_time = info.get("t_load", 0) + info.get("t_shingle", 0)
    row = dict(key=key, kind="exact", tag=a.tag, corpus=a.corpus, n=a.n, tau=a.tau, variant=a.variant,
               threads=a.threads, shingle_workers=info.get("shingle_workers"), streamed=bool(info.get("streamed")),
               shingles_from_cache=info["from_cache"], t_load=info.get("t_load"), t_shingle=info.get("t_shingle"),
               t_cache_io=t_get if info["from_cache"] else None,
               t_rank=t_rank, t_index=res["t_index"], t_probe_verify=res["t_probe_verify"],
               t_join_only=t_rank + res["t_index"] + res["t_probe_verify"],
               t_end_to_end=shingle_time + t_rank + res["t_index"] + res["t_probe_verify"],
               t_wall_this_process=t_join_total,
               n_pairs=int(len(res["i"])), n_candidates=res["n_candidates"], index_entries=res["index_entries"],
               n_tokens_distinct=V, n_singleton_tokens=S, n_shingles=int(off[-1]), n_empty_docs=info.get("n_empty"),
               peak_rss_mb_self=rss_self, peak_rss_mb_children=rss_ch,
               peak_rss_mb_sum_upper=rss_self + rss_ch)
    C.append_row(a.rows, row)
    print(json.dumps(row))


if __name__ == "__main__":
    main()
