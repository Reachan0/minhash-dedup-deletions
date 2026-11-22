#!/usr/bin/env python3
"""Shared, memory-bounded primitives for the correctness ledger (step
correctness-ledger-bounded).

Frozen definitions (also written to analysis/correctness_ledger.md):

  A  raw LSH: consecutive documents inside each 14x8 band bucket are chained,
     union-find, keep the minimum doc id per component.
  B  all pairs inside each band bucket, exact 5-gram Jaccard verification
     (keep the pair iff J >= tau), union-find, keep the minimum doc id.
  W  greedy maximal independent set on B's verified graph: iterate documents in
     a fixed order and keep a document iff no already-kept direct neighbour.
     (The same rule is also run on the exact truth graph for comparison.)
  C  exact self-join truth graph at J >= tau, union-find, keep the minimum id.

Memory contract
---------------
* The big shingle archives (`.work/dedupjoin/rerun/c4_700000.npz` 2.0 GB,
  `fw_1000000.npz` 4.1 GB, ...) are never loaded whole.  `flat.npy` is extracted
  once per corpus into a local cache and opened with `np.lib.format.open_memmap`;
  `off`/`nwords` are small members read directly.  The extraction is sequential
  (constant memory) and the temporary file is deleted when the corpus is done.
* Signature shards (`core/cache/sig/*.npy`) are opened with mmap_mode="r".
* One (corpus, seed) is processed at a time; nothing accumulates across seeds.
"""
import json
import os
import shutil
import sys
import time
import zipfile

import numba as nb
import numpy as np

# The 14x8 banding hash and the exact-pair Jaccard kernel are imported from the
# vendored copy of the core library so that this step reproduces the frozen runs
# bit for bit instead of re-implementing them.
def _vendor_dir(start):
    """Locate code/core/vendor in both the working and the published layout."""
    here = os.path.abspath(start)
    for cand in (os.path.join(os.path.dirname(here), "core", "vendor"),
                 os.path.join(here, "..", "..", "..", "..", "experiment", "code", "core", "vendor")):
        if os.path.isfile(os.path.join(cand, "lsh.py")):
            return cand
    raise RuntimeError("vendored lsh.py not found")


sys.path.insert(0, _vendor_dir(os.path.dirname(os.path.abspath(__file__))))
import lsh as PYLSH  # noqa: E402

band_keys = PYLSH.band_keys
jaccard_pairs = PYLSH.jaccard_pairs

H, B, R = 112, 14, 8
CORPORA = ["c4", "fw", "mix", "wet"]
N_DOCS = {"c4": 700000, "fw": 1000000, "mix": 1000000, "wet": 175272}


def _find_root(start):
    """Workspace root: nearest ancestor holding the frozen .work/dedupjoin data.

    Resolves both the working copy (experiment/code/correctness) and the
    published copy (output/<slug>/experiment/code/correctness).
    """
    p = start
    for _ in range(8):
        if os.path.isdir(os.path.join(p, ".work", "dedupjoin")):
            return p
        p = os.path.dirname(p)
    raise RuntimeError("workspace root with .work/dedupjoin not found")


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _find_root(HERE)
SLUG = os.path.join(ROOT, ".work", "dedupjoin-pvldb-v3")
CORE = os.path.join(SLUG, "core")
CACHE = os.path.join(CORE, "cache")
WRK = os.path.join(ROOT, ".work", "dedupjoin", "main")
RERUN = os.path.join(ROOT, ".work", "dedupjoin", "rerun")
OUTDIR = os.path.join(CORE, "correctness")
LOCAL_CACHE = os.path.join(OUTDIR, "cache")
RUNS = os.path.join(CORE, "runs.jsonl")

SHINGLE_NPZ = {
    "c4": os.path.join(RERUN, "c4_700000.npz"),
    "fw": os.path.join(RERUN, "fw_1000000.npz"),
    "mix": os.path.join(WRK, "mix_1000000.npz"),
    "wet": os.path.join(WRK, "wet_175272.npz"),
}
META_PARQUET = {
    "c4": os.path.join(RERUN, "c4_700000_meta.parquet"),
    "fw": os.path.join(RERUN, "fw_1000000_meta.parquet"),
    "mix": os.path.join(WRK, "mix_1000000_meta.parquet"),
    "wet": os.path.join(WRK, "wet_175272_meta.parquet"),
}
SEEDS = {"c4": [1, 2, 3, 4, 5], "fw": [1, 2, 3, 4, 5],
         "mix": [1, 2, 3], "wet": [1, 2, 3]}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def peak_rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def cur_rss_mb():
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)
    except Exception:
        return None


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
                    r = json.loads(line)
                    if r.get("status", "ok") == "ok":
                        s.add(r.get(key))
                except Exception:
                    pass
    return s


# ------------------------------------------------------------------ shingles
def extract_flat(corpus):
    """Extract flat.npy from the shingle npz once, sequentially, and return a
    read-only memmap.  Peak memory is one 1 MiB copy buffer."""
    os.makedirs(LOCAL_CACHE, exist_ok=True)
    dst = os.path.join(LOCAL_CACHE, f"{corpus}_flat.npy")
    if not os.path.exists(dst):
        src = SHINGLE_NPZ[corpus]
        tmp = dst + ".tmp"
        zf = zipfile.ZipFile(src)
        with zf.open("flat.npy") as fin, open(tmp, "wb") as fout:
            shutil.copyfileobj(fin, fout, 1024 * 1024)
        os.replace(tmp, dst)
    return np.lib.format.open_memmap(dst, mode="r")


def drop_flat(corpus):
    dst = os.path.join(LOCAL_CACHE, f"{corpus}_flat.npy")
    if os.path.exists(dst):
        os.remove(dst)


def load_offsets(corpus):
    z = np.load(SHINGLE_NPZ[corpus])
    off = np.asarray(z["off"], dtype=np.int64)
    nwords = np.asarray(z["nwords"], dtype=np.int64)
    z.close()
    return off, nwords


# ------------------------------------------------------------------ hashing
def sig_path(corpus, seed):
    return os.path.join(CACHE, "sig", f"{corpus}_s{seed}.npy")


def load_signatures(corpus, seed):
    return np.load(sig_path(corpus, seed), mmap_mode="r")


# ------------------------------------------------------------------ pairs
def iter_band_groups(K, valid):
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
    """A: consecutive documents inside every band bucket (datatrove stage 2)."""
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
    """B candidates: every pair inside every band bucket, unique, i < j."""
    codes = []
    largest, sum_pairs = 0, 0
    for _b, docs in iter_band_groups(K, valid):
        s = len(docs)
        largest = max(largest, s)
        sum_pairs += s * (s - 1) // 2
        ii, jj = np.triu_indices(s, 1)
        codes.append(docs[ii].astype(np.int64) * np.int64(n) + docs[jj])
    c = np.unique(np.concatenate(codes)) if codes else np.zeros(0, np.int64)
    pi = c // n
    pj = c % n
    return pi.astype(np.int64), pj.astype(np.int64), dict(
        n_pairs_mult=int(sum_pairs), largest_bucket=int(largest),
        bucket_pairs_sum=int(sum_pairs), n_pairs_unique=int(len(c)))


# ------------------------------------------------------------------ union find
@nb.njit(cache=True)
def uf_minid(n, ei, ej):
    """Union-find, root = minimum doc id of the component."""
    par = np.arange(n)
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
            if a < b:
                par[b] = a
            else:
                par[a] = b
    root = np.empty(n, np.int64)
    for x in range(n):
        y = x
        while par[y] != y:
            y = par[y]
        root[x] = y
    return root


def removed_minid(root, valid=None):
    rem = root != np.arange(len(root))
    if valid is not None:
        rem &= valid
    return rem


@nb.njit(cache=True)
def _csr(n, ei, ej):
    deg = np.zeros(n, np.int64)
    for k in range(ei.shape[0]):
        deg[ei[k]] += 1
        deg[ej[k]] += 1
    off = np.zeros(n + 1, np.int64)
    for i in range(n):
        off[i + 1] = off[i] + deg[i]
    fill = off[:-1].copy()
    adj = np.empty(off[-1], np.int64)
    for k in range(ei.shape[0]):
        a, b = ei[k], ej[k]
        adj[fill[a]] = b
        fill[a] += 1
        adj[fill[b]] = a
        fill[b] += 1
    return off, adj


@nb.njit(cache=True)
def _greedy_kept(n, off, adj, order):
    kept = np.zeros(n, np.uint8)
    for t in range(order.shape[0]):
        d = order[t]
        ok = 1
        for q in range(off[d], off[d + 1]):
            if kept[adj[q]]:
                ok = 0
                break
        kept[d] = ok
    return kept


def build_csr(n, ei, ej):
    return _csr(n, np.asarray(ei, np.int64), np.asarray(ej, np.int64))


def greedy_mis_from_csr(n, off, adj, order):
    return _greedy_kept(n, off, adj, np.asarray(order, np.int64)).astype(bool)


def greedy_mis(n, ei, ej, order=None):
    """Witness rule: greedy maximal independent set in `order` (doc ids)."""
    off, adj = build_csr(n, ei, ej)
    if order is None:
        order = np.arange(n, dtype=np.int64)
    return greedy_mis_from_csr(n, off, adj, order)


# ------------------------------------------------------------------ metrics
def has_true_neighbour(n, ti, tj):
    h = np.zeros(n, bool)
    h[ti] = True
    h[tj] = True
    return h


def witness_coverage(n, removed, ti, tj):
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


def pair_codes(i, j, n):
    i = np.asarray(i, np.int64)
    j = np.asarray(j, np.int64)
    lo = np.minimum(i, j)
    hi = np.maximum(i, j)
    return lo * np.int64(n) + hi


def edge_stats(ti, tj, pi, pj, n):
    """Pair recall (truth edges present in the candidate/verified set) and precision."""
    n_truth = len(ti)
    if n_truth == 0:
        return dict(pair_recall=None, candidate_precision=None,
                    candidate_edges_unique=0, truth_edges_detected=0)
    tc = np.sort(pair_codes(ti, tj, n))
    pc = np.unique(pair_codes(pi, pj, n))
    pos = np.searchsorted(tc, pc)
    pos[pos >= len(tc)] = 0
    hit = int((tc[pos] == pc).sum()) if len(pc) else 0
    return dict(
        pair_recall=float(hit / n_truth),
        candidate_precision=float(hit / len(pc)) if len(pc) else None,
        candidate_edges_unique=int(len(pc)),
        truth_edges_detected=hit,
    )


def detected_truth_mask(ti, tj, pi, pj, n):
    """Boolean per truth edge: is this exact edge in the candidate pair set?"""
    tc = pair_codes(ti, tj, n)
    pc = np.unique(pair_codes(pi, pj, n))
    pos = np.searchsorted(pc, tc)
    pos[pos >= len(pc)] = 0
    return (pc[pos] == tc) if len(pc) else np.zeros(len(tc), bool)


# ------------------------------------------------------------------ components
def exact_components(n, ti, tj):
    root = uf_minid(n, ti, tj)
    removed = removed_minid(root)
    return root, removed


def component_split(root_exact, root_ver, n):
    """Split every non-singleton exact component by B's verified connectivity.

    Returns
      n_components  exact components with >= 2 documents
      blocks        number of B-connectivity blocks inside them
      sum_k_minus_1 blocks - n_components  (== |K_C \\ K_B| under min-id)
      k_of_edge     k for each exact component that has at least one true edge
                    (order = np.unique(root_exact[ti]))
    """
    uniq, inv, cnt = np.unique(root_exact, return_inverse=True, return_counts=True)
    mask = cnt[inv] > 1
    if not mask.any():
        return dict(n_components=0, blocks=0, sum_k_minus_1=0,
                    k_of_edge=np.zeros(0, np.int64))
    keys = inv[mask].astype(np.int64) * np.int64(n) + np.asarray(root_ver, np.int64)[mask]
    uk = np.unique(keys)
    comp = uk // n
    ucomp, blocks_per_comp = np.unique(comp, return_counts=True)
    n_components = int(len(ucomp))
    blocks = int(len(uk))
    kk = np.zeros(len(uniq), np.int64)
    kk[ucomp] = blocks_per_comp
    return dict(n_components=n_components, blocks=blocks,
                sum_k_minus_1=int(blocks - n_components),
                k_of_edge=kk, uniq=uniq, inv=inv, cnt=cnt)


def component_edge_stats(split, root_exact, ti, detected):
    """Per exact component: true edges, detected (candidate) edges, missing edges, k.

    `detected` is the boolean per truth edge returned by detected_truth_mask.
    Also counts the two qualitative classes used in the Draft 2 counterexample
    check: components that stay connected although many edges were missed, and
    components that split although only one or two edges were missed.
    """
    ti = np.asarray(ti, np.int64)
    root_exact = np.asarray(root_exact, np.int64)
    ue = np.unique(root_exact[ti])
    eidx = np.searchsorted(ue, root_exact[ti])
    ecount = np.bincount(eidx, minlength=len(ue)).astype(np.int64)
    hit = np.bincount(eidx, weights=np.asarray(detected, np.float64),
                      minlength=len(ue)).astype(np.int64)
    missed = ecount - hit
    k = split["k_of_edge"][np.searchsorted(split["uniq"], ue)]
    return dict(
        ecount=ecount, hits=hit, missed=missed, k=k,
        connected_with_many_missed=int(((k == 1) & (missed >= 5)).sum()),
        missed_edges_in_those=int(missed[(k == 1) & (missed >= 5)].sum()),
        split_with_few_missed=int(((k > 1) & (missed <= 2)).sum()),
        split_total=int((k > 1).sum()),
    )


def order_permutations(n, k=3, base=101):
    return {f"perm{i + 1}": np.random.RandomState(base + 37 * i).permutation(n)
            for i in range(k)}
