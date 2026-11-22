#!/usr/bin/env python3
"""Scale ladder for DedupJoin: one protocol, one thread count, one threshold set.

Workloads
  prefix:<N>   the first N documents of the pool in its write order (FineWeb, then C4,
               then raw WET).  Rungs 100k, 300k, 1M, 2M, 3M, 4M.
  strata:<B>   a 100k-document sample (all members where fewer exist) from one stratum of
               the pool manifest: length buckets L1..L5 and duplicate-density buckets
               D1/D4, plus a uniform random sample.  Same size, different regime.

Methods, all on identical input and the same 16 threads:
  raw      MinHash-LSH 14x8 (datatrove family, seed 1) -> co-bucket components -> keep min id
  verify   raw candidate edges filtered by exact 5-gram Jaccard -> components -> keep min id
  witness  verified edges -> greedy maximal independent set in id order (every deletion has
           a direct retained neighbour)
  exact    full exact Jaccard self-join (PPJoin-style prefix filter) -> components -> keep min id

Stages timed separately: shard I/O and composition, signature computation, banding,
global token ranking, candidate generation (band sort), bucket expansion (intra-bucket
pair enumeration), verification, clustering (union-find), exact index, exact probe,
witness selection.  Peak RSS is measured per stage with VmHWM after a clear_refs reset.

Usage:
  python3 scale_ladder.py run --workload prefix:100000 [--mem-cap-gib 58]
  python3 scale_ladder.py strata-list
  python3 scale_ladder.py run --workload strata:L4_500_1499
Rows are appended to output/dedupjoin-pvldb/results/scale_ladder_rows.jsonl.
"""
import argparse
import json
import os
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
C03 = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03")
sys.path.insert(0, C03)
import common as C  # noqa: E402
import exactjoin as EJ  # noqa: E402
import lsh as L  # noqa: E402

import numba as nb  # noqa: E402

_WARMED = False
SHARDS = os.path.join(ROOT, ".work/dedupjoin/ladder/shards")
STRATA = os.path.join(ROOT, ".work/dedupjoin/pool/strata.parquet")
OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
ROWS = os.path.join(OUT, "scale_ladder_rows.jsonl")
H, B, R = 112, 14, 8
TAUS = [0.6, 0.7, 0.8, 0.9]
THREADS = 16


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def append_row(row):
    """One row per workload: rerunning a workload replaces its earlier row."""
    row = dict(row)
    row.setdefault("ts_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    row.setdefault("provenance", "independently rerun (this run)")
    os.makedirs(OUT, exist_ok=True)
    keep = []
    if os.path.exists(ROWS):
        with open(ROWS) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    if json.loads(line).get("workload_id") != row["workload_id"]:
                        keep.append(line)
                except Exception:
                    keep.append(line)
    keep.append(json.dumps(row, ensure_ascii=False))
    tmp = ROWS + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(keep) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, ROWS)


def reset_peak():
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
    except Exception:
        pass


def peak_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        return None
    return None


class Stage:
    """Time a stage and record its own peak RSS (VmHWM reset beforehand)."""

    def __init__(self, name, stages, peaks):
        self.name, self.stages, self.peaks = name, stages, peaks

    def __enter__(self):
        reset_peak()
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.stages[self.name] = round(time.time() - self.t0, 3)
        self.peaks[self.name] = peak_mb()
        log(f"  {self.name}: {self.stages[self.name]}s peak={self.peaks[self.name]}MB")
        return False


# ------------------------------------------------------------------ shard layout
_SHARDS = None


def shard_entries():
    global _SHARDS
    if _SHARDS is not None:
        return _SHARDS
    out = []
    start = 0
    for src in ("fw", "c4", "wet"):
        for p in sorted(os.listdir(os.path.join(ROOT, ".work/dedupjoin/pool", src))):
            if not p.endswith(".parquet"):
                continue
            idx = int(p.split("-")[1].split(".")[0])
            n = pq.ParquetFile(os.path.join(ROOT, ".work/dedupjoin/pool", src, p)).metadata.num_rows
            out.append(dict(src=src, idx=idx, n=n, start=start, end=start + n))
            start += n
    _SHARDS = out
    return out


def _load_shard(src, idx):
    z = np.load(os.path.join(SHARDS, f"{src}-{idx:05d}.npz"))
    return np.asarray(z["off"]), np.asarray(z["flat"]), np.asarray(z["nwords"]), np.asarray(z["nchar"])


def compose_prefix(n, stages, peaks):
    """off (n+1), parts list of uint64 shingle arrays, nwords, nchar."""
    parts, offs, nws, ncs = [], [], [], []
    doc_base = 0
    sh_base = 0
    with Stage("io_shards", stages, peaks):
        for e in shard_entries():
            if doc_base >= n:
                break
            take = min(e["n"], n - doc_base)
            off, flat, nw, nc = _load_shard(e["src"], e["idx"])
            f = flat[:off[take]]
            o = np.asarray(off[:take + 1]).copy() + sh_base
            if parts:
                o = o[1:]
            parts.append(f)
            offs.append(o)
            nws.append(nw[:take])
            ncs.append(nc[:take])
            doc_base += take
            sh_base += len(f)
    off = np.concatenate(offs)
    assert len(off) - 1 == n, (len(off) - 1, n)
    return off, parts, np.concatenate(nws).astype(np.int64), np.concatenate(ncs).astype(np.int64)


def compose_selection(row_ids, stages, peaks):
    """Select arbitrary pool rows by global row_id (sorted, unique)."""
    parts, offs, nws, ncs = [], [], [], []
    sh_base = 0
    rid = np.asarray(row_ids, np.int64)
    with Stage("io_shards", stages, peaks):
        for e in shard_entries():
            lo, hi = np.searchsorted(rid, [e["start"], e["end"]])
            if hi <= lo:
                continue
            sel = rid[lo:hi] - e["start"]
            off, flat, nw, nc = _load_shard(e["src"], e["idx"])
            sizes = off[sel + 1] - off[sel]
            o = np.zeros(len(sel) + 1, np.int64)
            np.cumsum(sizes, out=o[1:])
            f = np.concatenate([flat[off[s]:off[s + 1]] for s in sel])
            o = o + sh_base
            if parts:
                o = o[1:]
            parts.append(f)
            offs.append(o)
            nws.append(nw[sel])
            ncs.append(nc[sel])
            sh_base += len(f)
    off = np.concatenate(offs)
    return off, parts, np.concatenate(nws).astype(np.int64), np.concatenate(ncs).astype(np.int64)


# ------------------------------------------------------------------ signatures / bands
def compute_signatures(off, kind, n, row_ids, stages, peaks):
    """112 datatrove permutations of seed 1, shard by shard (bounded temporary)."""
    sig = np.empty((n, H), np.uint64)
    with Stage("signature", stages, peaks):
        base = 0
        if kind == "prefix":
            for e in shard_entries():
                if base >= n:
                    break
                take = min(e["n"], n - base)
                o, f, _, _ = _load_shard(e["src"], e["idx"])
                oo = np.asarray(o[:take + 1])
                sig[base:base + take] = L.signatures(oo, f[:o[take]], H, 1, "dt")
                base += take
                del f, oo
        else:
            rid = np.asarray(row_ids, np.int64)
            for e in shard_entries():
                lo, hi = np.searchsorted(rid, [e["start"], e["end"]])
                if hi <= lo:
                    continue
                sel = rid[lo:hi] - e["start"]
                o, f, _, _ = _load_shard(e["src"], e["idx"])
                sizes = o[sel + 1] - o[sel]
                oo = np.zeros(len(sel) + 1, np.int64)
                np.cumsum(sizes, out=oo[1:])
                ff = np.concatenate([f[o[s]:o[s + 1]] for s in sel])
                sig[base:base + len(sel)] = L.signatures(oo, ff, H, 1, "dt")
                base += len(sel)
                del f, ff
    return sig


# ------------------------------------------------------------------ numba kernels
@nb.njit(parallel=True, cache=True)
def jac_ranks(off, R, pi, pj):
    m = pi.shape[0]
    out = np.empty(m, np.float64)
    for p in nb.prange(m):
        a0, a1 = off[pi[p]], off[pi[p] + 1]
        b0, b1 = off[pj[p]], off[pj[p] + 1]
        i, j, o = a0, b0, 0
        while i < a1 and j < b1:
            if R[i] == R[j]:
                o += 1
                i += 1
                j += 1
            elif R[i] < R[j]:
                i += 1
            else:
                j += 1
        u = (a1 - a0) + (b1 - b0) - o
        out[p] = o / u if u > 0 else 0.0
    return out


@nb.njit(cache=True)
def unique_per_doc(R, off, S):
    out = np.zeros(len(off) - 1, np.int64)
    for d in range(len(off) - 1):
        c = 0
        for i in range(off[d], off[d + 1]):
            if R[i] < S:
                c += 1
        out[d] = c
    return out


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


def greedy_mis(n, ei, ej):
    return _mis(n, ei.astype(np.int32), ej.astype(np.int32)).astype(bool)


# ------------------------------------------------------------------ candidate stages
def bucket_stage(K, valid, stages, peaks, max_pairs=400_000_000):
    """Datatrove-style co-bucket candidate generation, timed in two parts."""
    b, n = K.shape
    idx_all = np.nonzero(valid)[0]
    stats = dict(n_buckets_ge2=0, max_bucket=0, pairs_mult=0, n_bands=b, capped=False)
    with Stage("cand_sort", stages, peaks):
        order = [np.argsort(K[i, idx_all], kind="stable") for i in range(b)]
    codes, ci, cj = [], [], []
    with Stage("bucket_expand", stages, peaks):
        for i in range(b):
            k = K[i, idx_all]
            o = order[i]
            ks = k[o]
            docs = idx_all[o]
            brk = np.nonzero(np.diff(ks) != 0)[0] + 1
            starts = np.concatenate([[0], brk])
            ends = np.concatenate([brk, [len(ks)]])
            sz = ends - starts
            m = sz >= 2
            starts, ends, sz = starts[m], ends[m], sz[m]
            if len(sz) == 0:
                continue
            stats["n_buckets_ge2"] += int(len(sz))
            stats["max_bucket"] = max(stats["max_bucket"], int(sz.max()))
            stats["pairs_mult"] += int((sz * (sz - 1) // 2).sum())
            same = np.nonzero(np.diff(ks) == 0)[0]
            ci.append(docs[same])
            cj.append(docs[same + 1])
            if stats["pairs_mult"] > max_pairs:
                stats["capped"] = True
                continue  # chain edges still formed; all-pairs expansion stops
            for s, e in zip(starts, ends):
                g = np.sort(docs[s:e])
                ii, jj = np.triu_indices(len(g), 1)
                codes.append(g[ii].astype(np.int64) * n + g[jj])
        del order
    ci = np.concatenate(ci) if ci else np.zeros(0, np.int64)
    cj = np.concatenate(cj) if cj else np.zeros(0, np.int64)
    with Stage("cand_dedup", stages, peaks):
        codes = np.unique(np.concatenate(codes)) if codes else np.zeros(0, np.int64)
    pi = (codes // n).astype(np.int64)
    pj = (codes % n).astype(np.int64)
    stats["n_pairs_unique"] = int(len(codes))
    return pi, pj, ci, cj, stats


def exact_probe(off, R, ordd, sizes_ord, S, cnt, ent_k, ent_p, num, den, threads):
    """exactjoin.exact_join without the index build (the index is timed separately)."""
    n = len(ordd)
    nchunks = max(64, threads * 32)
    bounds = np.linspace(0, n, nchunks + 1).astype(np.int64)
    args = [(int(bounds[i]), int(bounds[i + 1])) for i in range(nchunks)]

    def run(ab):
        return EJ.probe_range(R, off, ordd, sizes_ord, S, cnt, ent_k, ent_p, num, den,
                              ab[0], ab[1], True, True)

    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(run, args))
    a = np.concatenate([r[0] for r in res])
    b = np.concatenate([r[1] for r in res])
    o = np.concatenate([r[2] for r in res])
    ncand = int(sum(r[3] for r in res))
    sizes = (off[1:] - off[:-1]).astype(np.int64)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    la = sizes[lo]
    lb = sizes[hi]
    oo = o.astype(np.float64)
    jac = oo / (la + lb - oo)
    return lo.astype(np.int32), hi.astype(np.int32), jac, int(len(ent_p)), ncand


# ------------------------------------------------------------------ warm-up
def warmup():
    """Compile every timed numba kernel on tiny inputs so no stage pays JIT time."""
    off = np.array([0, 4, 8, 12, 16], np.int64)
    flat = np.array([1, 2, 3, 4, 2, 5, 6, 7, 3, 5, 8, 9, 4, 6, 10, 11], np.uint64)
    EJ.build_ranks_parts(off, [flat.copy()])
    sig = L.signatures(off, flat, H, 1, "dt")
    L.band_keys(sig, B, R)
    pi = np.array([0, 1], np.int64)
    pj = np.array([1, 2], np.int64)
    rk = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], np.int32)
    jac_ranks(off, rk, pi, pj)
    unique_per_doc(rk, off, 12)
    _mis(4, np.array([0, 1], np.int32), np.array([1, 2], np.int32))
    sizes = (off[1:] - off[:-1]).astype(np.int64)
    ordd = np.lexsort((np.arange(len(sizes)), sizes)).astype(np.int64)
    sizes_ord = sizes[ordd]
    cnt, ent_k, ent_p = EJ.build_index(rk, off, ordd, 12, 16, 8, 10, 0)
    EJ.probe_range(rk, off, ordd, sizes_ord, 12, cnt, ent_k, ent_p, 8, 10, 0, 2, True, True)
    L.uf_components(4, np.array([0], np.int64), np.array([1], np.int64))


# ------------------------------------------------------------------ one workload
def run_workload(wid, kind, label, n, row_ids, stages, peaks, memcap_gib):
    t_start = time.time()
    if not _WARMED:
        warmup()
        globals()["_WARMED"] = True
    reset_peak()
    if memcap_gib:
        cap = int(memcap_gib * (1 << 30))
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    if kind == "prefix":
        off, parts, nwords, nchar = compose_prefix(n, stages, peaks)
    else:
        off, parts, nwords, nchar = compose_selection(row_ids, stages, peaks)
    n = len(off) - 1
    valid = (np.diff(off) > 0)
    nsh = int(off[-1])
    log(wid, "docs", n, "shingles", nsh, "valid", int(valid.sum()))
    with Stage("rank_build", stages, peaks):
        rk, S, V = EJ.build_ranks_parts(off, parts)
        del parts
    with Stage("unique_count", stages, peaks):
        uniq = unique_per_doc(rk, off, S)
    log(wid, "ranks S", S, "V", V, "V<2^31", V < 2 ** 31 - 1)

    sig = compute_signatures(off, kind, n, row_ids, stages, peaks)
    with Stage("banding", stages, peaks):
        K = L.band_keys(sig, B, R)
        del sig
    pi, pj, ci, cj, bstats = bucket_stage(K, valid, stages, peaks)
    del K
    with Stage("verify", stages, peaks):
        jac = jac_ranks(off, rk, pi, pj)
    with Stage("cluster_raw", stages, peaks):
        root_raw = L.uf_components(n, ci, cj)
        rem_raw = L.removed_set(root_raw) & valid
    del ci, cj, root_raw

    sizes = (off[1:] - off[:-1]).astype(np.int64)
    exact, per_tau = {}, {}
    for tau in TAUS:
        fr = Fraction(str(tau)).limit_denominator(1000)
        num, den = fr.numerator, fr.denominator
        with Stage(f"exact_index_t{tau}", stages, peaks):
            ordd = np.lexsort((np.arange(len(sizes)), sizes)).astype(np.int64)
            sizes_ord = sizes[ordd]
            cnt, ent_k, ent_p = EJ.build_index(rk, off, ordd, S, V, num, den, 0)
        with Stage(f"exact_probe_t{tau}", stages, peaks):
            ti, tj, jacE, n_index, n_cand = exact_probe(
                off, rk, ordd, sizes_ord, S, cnt, ent_k, ent_p, num, den, THREADS)
        with Stage(f"cluster_exact_t{tau}", stages, peaks):
            root_ex = L.uf_components(n, ti, tj)
            rem_ex = L.removed_set(root_ex) & valid
        exact[str(tau)] = dict(n_pairs=int(len(ti)), removed=int(rem_ex.sum()),
                               index_entries=n_index, n_candidates=n_cand,
                               t_index=stages[f"exact_index_t{tau}"],
                               t_probe=stages[f"exact_probe_t{tau}"])
        has_nb = np.zeros(n, bool)
        has_nb[ti] = True
        has_nb[tj] = True
        ver = jac >= tau - 1e-12
        with Stage(f"cluster_verify_t{tau}", stages, peaks):
            root_ver = L.uf_components(n, pi[ver], pj[ver])
            rem_ver = L.removed_set(root_ver)
        with Stage(f"witness_t{tau}", stages, peaks):
            kept = greedy_mis(n, pi[ver], pj[ver])
            rem_wit = (~kept) & valid
        wrong_raw = rem_raw & ~has_nb
        hijack = rem_raw & has_nb & ~rem_ex

        def residual(rem):
            return int((~rem[ti] & ~rem[tj]).sum())

        def content(rem):
            keep = ~rem
            return dict(tokens=int(nwords[keep].sum()), uniq=int(uniq[keep].sum()),
                        docs=int(keep.sum()))

        per_tau[str(tau)] = dict(
            n_truth_pairs=int(len(ti)), n_cand_pairs=int(len(pi)), n_verified=int(ver.sum()),
            raw_removed=int(rem_raw.sum()), raw_unsupported=int(wrong_raw.sum()),
            raw_hijacked=int(hijack.sum()),
            raw_unsupported_share_of_deleted=(float(wrong_raw.sum() / rem_raw.sum())
                                              if rem_raw.sum() else None),
            raw_unsupported_share_of_input=(float(wrong_raw.sum() / int(valid.sum()))
                                            if valid.sum() else None),
            exact_removed=int(rem_ex.sum()), ver_removed=int(rem_ver.sum()),
            ver_unsupported=int((rem_ver & ~has_nb).sum()),
            ver_missed=int((rem_ex & ~rem_ver).sum()),
            wit_removed=int(rem_wit.sum()), wit_missed=int((rem_ex & ~rem_wit).sum()),
            wit_unsupported=int((rem_wit & ~has_nb).sum()),
            raw_residual_pairs=residual(rem_raw), ver_residual_pairs=residual(rem_ver),
            wit_residual_pairs=residual(rem_wit), exact_residual_pairs=residual(rem_ex),
            raw_content=content(rem_raw), ver_content=content(rem_ver),
            wit_content=content(rem_wit), exact_content=content(rem_ex),
            symdiff_raw_exact=int((rem_raw != rem_ex).sum()),
            symdiff_ver_exact=int((rem_ver != rem_ex).sum()),
        )
        log(f"  tau {tau}: exact {exact[str(tau)]['removed']} "
            f"raw {per_tau[str(tau)]['raw_removed']} "
            f"unsupported {per_tau[str(tau)]['raw_unsupported']} "
            f"ver {per_tau[str(tau)]['ver_removed']} wit {per_tau[str(tau)]['wit_removed']}")
        del ti, tj, jacE, root_ex, rem_ex, has_nb, ver, root_ver, rem_ver, kept, rem_wit, cnt, ent_k, ent_p

    row = dict(workload_id=wid, kind=kind, label=label, n_docs=n,
               n_valid=int(valid.sum()), n_shingles=nsh,
               nwords_mean=float(nwords.mean()), nchar_mean=float(nchar.mean()),
               S=int(S), V=int(V), threads=THREADS, tau_list=TAUS,
               stages=stages, peak_rss_mb=peaks, bucket_stats=bstats,
               exact=exact, per_tau=per_tau, memcap_gib=memcap_gib)
    row["measured_docs_in_lsh_components"] = int(rem_raw.sum())
    row["measured_dup_density_14x8"] = float(rem_raw.sum() / max(1, int(valid.sum())))
    # per-stage peaks are the truth: VmHWM is reset before each stage, so the value at the
    # end of the run only reflects the last stage, not the run high-water mark
    row["total_peak_rss_mb"] = max([peak_mb()] + [v for v in peaks.values() if v])
    row["wall_s"] = round(time.time() - t_start, 1)
    append_row(row)
    log(wid, "done, total peak", row["total_peak_rss_mb"], "MB", "wall", row["wall_s"])
    return row


# ------------------------------------------------------------------ strata
def source_selection(src, n):
    """First n documents of one source, in pool order."""
    ids = []
    got = 0
    for e in shard_entries():
        if e["src"] != src:
            continue
        take = min(e["n"], n - got)
        if take <= 0:
            break
        ids.append(np.arange(e["start"], e["start"] + take, dtype=np.int64))
        got += take
    return np.concatenate(ids) if ids else np.zeros(0, np.int64)


def strata_selection(bucket, n=100000, seed=7):
    t = pq.read_table(STRATA, columns=["row_id", "length_bucket", "dup_bucket", "source"])
    d = t.to_pydict()
    rid = np.asarray(d["row_id"], np.int64)
    if bucket == "uniform":
        mask = np.ones(len(rid), bool)
    elif bucket.startswith("L"):
        mask = np.array([b == bucket for b in d["length_bucket"]])
    elif bucket.startswith("D"):
        mask = np.array([b == bucket for b in d["dup_bucket"]])
    else:
        raise ValueError(bucket)
    pool = rid[mask]
    if len(pool) > n:
        rng = np.random.RandomState(seed)
        pool = np.sort(rng.choice(pool, size=n, replace=False))
    return np.sort(pool), int(mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "strata-list", "check-shards"])
    ap.add_argument("--workload", default="")
    ap.add_argument("--mem-cap-gib", type=float, default=0)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    global THREADS
    THREADS = a.threads
    C.set_threads(a.threads)
    np.seterr(all="ignore")
    if a.mode == "strata-list":
        t = pq.read_table(STRATA, columns=["length_bucket", "dup_bucket"]).to_pydict()
        from collections import Counter
        print("length", Counter(t["length_bucket"]))
        print("dup", Counter(t["dup_bucket"]))
        return
    if a.mode == "check-shards":
        miss = [p for e in shard_entries()
                for p in [os.path.join(SHARDS, f"{e['src']}-{e['idx']:05d}.npz")]
                if not os.path.exists(p)]
        print("missing", len(miss), miss[:5])
        return
    wid = a.workload
    kind, label = wid.split(":", 1)
    stages, peaks = {}, {}
    if kind == "prefix":
        run_workload(wid, kind, label, int(label), None, stages, peaks, a.mem_cap_gib)
    elif kind == "strata":
        ids, avail = strata_selection(label)
        log("stratum", label, "available", avail, "selected", len(ids))
        run_workload(wid, kind, label, len(ids), ids, stages, peaks, a.mem_cap_gib)
    elif kind == "source":
        src, nn = label.split(":")
        ids = source_selection(src, int(nn))
        log("source", src, "requested", nn, "selected", len(ids))
        run_workload(wid, kind, label, len(ids), ids, stages, peaks, a.mem_cap_gib)
    else:
        raise ValueError(kind)


if __name__ == "__main__":
    main()
