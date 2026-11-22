#!/usr/bin/env python3
"""Same-condition method comparison for DedupJoin (this step).

Corpora (one process each, so peak RSS is per condition):
  c4    700,000 C4 documents (established cell; pool prefix)
  fw  1,000,000 FineWeb sample-10BT documents (pool prefix)
  mix 1,000,000 = 500,000 FineWeb + 500,000 C4 (round-2 mixed condition, from the caches)
  wet   175,272 raw Common Crawl WET documents (new independent snapshot)

Methods, all at tau in {0.6,0.7,0.8,0.9} and >=5 hash seeds where randomness exists:
  raw      MinHash-LSH banding (datatrove 14x8) -> connected components -> keep min id
  verify   candidate edges filtered by exact 5-gram Jaccard -> components -> keep min id
  exact    full exact Jaccard self-join (PPJoin-style prefix filter) -> components -> keep min id
  witness  verified candidate edges -> greedy maximal independent set in id order, so every
           deleted document is directly adjacent (J >= tau) to a retained document

Every per-seed row carries provenance; truth arrays and per-document masks are cached under
.work/dedupjoin/main/ so later analysis (figures, variance) does not recompute.

Usage:
  python3 main_compare.py prepare --corpus c4
  python3 main_compare.py run --corpus c4 --seeds 1,2,3,4,5
"""
import argparse
import glob
import json
import os
import sys
import time
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
C03 = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03")
sys.path.insert(0, C03)
import common as C  # noqa: E402
import exactjoin as EJ  # noqa: E402
import lsh as L  # noqa: E402

OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/main")
RERUN = os.path.join(ROOT, ".work/dedupjoin/rerun")
ROWS = os.path.join(OUT, "main_compare_rows.jsonl")
TAUS = [0.6, 0.7, 0.8, 0.9]
H, B, R = 112, 14, 8
CORPORA = {"c4": 700000, "fw": 1000000, "mix": 1000000, "wet": 175272}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def append_row(row):
    row = dict(row)
    row.setdefault("ts_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    row.setdefault("provenance", "independently rerun (this run)")
    os.makedirs(OUT, exist_ok=True)
    with open(ROWS, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def domains(urls):
    out = []
    for u in urls:
        try:
            h = urlparse(u).hostname or ""
        except Exception:
            h = ""
        if h.startswith("www."):
            h = h[4:]
        out.append(h)
    return out


# ------------------------------------------------------------------ shingles / meta
def load_shingles(corpus):
    """Returns off, flat, nwords, meta dict. Uses the established caches; builds WET."""
    n = CORPORA[corpus]
    if corpus in ("c4", "fw"):
        tag = f"{corpus}_{n}"
        z = np.load(os.path.join(RERUN, f"{tag}.npz"))
        off, flat, nw = np.asarray(z["off"]), np.asarray(z["flat"]), np.asarray(z["nwords"])
        m = pq.read_table(os.path.join(RERUN, f"{tag}_meta.parquet"))
        md = m.to_pydict()
        meta = {"id": md["id"], "url": md["url"], "nchar": md["nchar"],
                "nwords": md["nwords"], "nshingle": md["nshingle"]}
        del z, m
        return off, flat, nw.astype(np.int64), meta
    if corpus == "mix":
        cache = os.path.join(WRK, f"mix_{n}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            off, flat, nw = np.asarray(z["off"]), np.asarray(z["flat"]), np.asarray(z["nwords"])
            md = pq.read_table(os.path.join(WRK, f"mix_{n}_meta.parquet")).to_pydict()
            meta = {"id": md["id"], "url": md["url"], "nchar": md["nchar"],
                    "nwords": md["nwords"], "nshingle": md["nshingle"]}
            return off, flat, nw.astype(np.int64), meta
        # first 500k fw then first 500k c4, exactly the round-2 mixed definition
        offs, flats, nws = [], [], []
        metas = {k: [] for k in ("id", "url", "nchar", "nwords", "nshingle")}
        base = 0
        for src, k in (("fw", n // 2), ("c4", n - n // 2)):
            z = np.load(os.path.join(RERUN, f"{src}_{CORPORA[src]}.npz"))
            o = np.asarray(z["off"])
            f_ = np.asarray(z["flat"])[:o[k]]
            nm = np.asarray(z["nwords"])[:k]
            oo = np.asarray(o[:k + 1]).copy()
            if base:
                oo = oo[1:]  # drop the shared boundary: segment 1's end point
            offs.append(oo + base)
            flats.append(f_)
            nws.append(nm)
            base += int(oo[-1])
            md = pq.read_table(os.path.join(RERUN, f"{src}_{CORPORA[src]}_meta.parquet"),
                               columns=["id", "url", "nchar", "nwords", "nshingle"]).to_pydict()
            for key in metas:
                metas[key].extend(md[key][:k])
            del z, o, f_
        off = np.concatenate(offs)
        flat = np.concatenate(flats)
        nw = np.concatenate(nws)
        del offs, flats, nws
        meta = {k: metas[k] for k in metas}
        os.makedirs(WRK, exist_ok=True)
        np.savez(cache + ".tmp.npz", off=off, flat=flat, nwords=nw)
        os.replace(cache + ".tmp.npz", cache)
        pq.write_table(pa.table({k: meta[k] for k in ("id", "url", "nchar", "nwords", "nshingle")}),
                       os.path.join(WRK, f"mix_{n}_meta.parquet"), compression="zstd")
        return off, flat, nw.astype(np.int64), meta
    if corpus == "wet":
        cache = os.path.join(WRK, f"wet_{n}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            off, flat, nw = np.asarray(z["off"]), np.asarray(z["flat"]), np.asarray(z["nwords"])
            md = pq.read_table(os.path.join(WRK, f"wet_{n}_meta.parquet")).to_pydict()
            return off, flat, nw.astype(np.int64), md
        ids, urls, texts, nchars, nws = [], [], [], [], []
        for p in sorted(glob.glob(os.path.join(ROOT, ".work/dedupjoin/pool/wet/*.parquet"))):
            d = pq.read_table(p, columns=["id", "url", "text", "nchar", "nwords"]).to_pydict()
            ids.extend(d["id"]); urls.extend(d["url"]); texts.extend(d["text"])
            nchars.extend(d["nchar"]); nws.extend(d["nwords"])
        assert len(ids) == n, (len(ids), n)
        log("shingling WET", len(texts))
        t0 = time.time()
        offs = [np.zeros(1, np.int64)]
        flats, nword = [], []
        for t in texts:
            s, w = C.shingle_doc(t)
            flats.append(s)
            nword.append(w)
            offs.append(np.array([offs[-1][-1] + len(s)], np.int64))
        off = np.concatenate(offs)
        flat = np.concatenate(flats)
        nw = np.array(nword, np.int64)
        meta = {"id": ids, "url": urls, "nchar": np.array(nchars, np.int64),
                "nwords": np.array(nws, np.int64), "nshingle": np.diff(off)}
        os.makedirs(WRK, exist_ok=True)
        np.savez(cache + ".tmp.npz", off=off, flat=flat, nwords=nw)
        os.replace(cache + ".tmp.npz", cache)
        pq.write_table(pa.table({k: meta[k] for k in ("id", "url", "nchar", "nwords", "nshingle")}),
                       os.path.join(WRK, f"wet_{n}_meta.parquet"), compression="zstd")
        log("WET shingled in %.1fs, shingles=%d" % (time.time() - t0, len(flat)))
        return off, flat, nw, meta
    raise ValueError(corpus)


# ------------------------------------------------------------------ union-find helpers
@np.errstate(all="ignore")
def greedy_mis(n, ei, ej):
    """Greedy maximal independent set in id order on the undirected graph (ei, ej).
    Kept = MIS. Every non-kept vertex has a kept neighbour (dominating)."""
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
            adj[fill[a]] = b; fill[a] += 1
            adj[fill[b]] = a; fill[b] += 1
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


def component_stats(root, n):
    size = np.bincount(root, minlength=n)
    doc_size = size[root]
    comp = doc_size >= 2
    return dict(n_components_ge2=int((size >= 2).sum()),
                docs_in_components=int(comp.sum()),
                max_component=int(size.max()) if n else 0,
                sum_size2=int((size.astype(np.int64) ** 2).sum()),
                pairs=int((size.astype(np.int64) * (size - 1) // 2).sum()))


def unique_per_doc(R, S, off):
    """Number of shingles occurring exactly once in the corpus, per document."""
    n = len(off) - 1
    import numba as nb

    @nb.njit(cache=True)
    def _cnt(R, off, S):
        out = np.zeros(len(off) - 1, np.int64)
        for d in range(len(off) - 1):
            c = 0
            for i in range(off[d], off[d + 1]):
                if R[i] < S:
                    c += 1
            out[d] = c
        return out

    return _cnt(R, off, S)


# ------------------------------------------------------------------ prepare (truth)
def prepare(corpus, threads):
    n = CORPORA[corpus]
    os.makedirs(WRK, exist_ok=True)
    off, flat, nw, meta = load_shingles(corpus)
    log(corpus, "loaded", n, "docs", len(flat), "shingles")
    t0 = time.time()
    R, S, V = EJ.build_ranks(off, flat)
    t_rank = time.time() - t0
    log(corpus, "ranks %.1fs S=%d V=%d" % (t_rank, S, V))
    uniq = unique_per_doc(R, S, off)
    np.savez(os.path.join(WRK, f"{corpus}_ranks.npz"), R=R, S=S, V=V, uniq=uniq,
             nwords=nw, nshingle=np.diff(off))
    valid = (np.diff(off) > 0)
    truth = {}
    for tau in TAUS:
        t0 = time.time()
        res = EJ.exact_join(off, R, S, V, tau, threads)
        ti, tj, jac = res["i"].astype(np.int64), res["j"].astype(np.int64), res["jac"]
        root = L.uf_components(n, ti, tj)
        rem = L.removed_set(root)
        np.savez(os.path.join(WRK, f"{corpus}_truth_t{tau}.npz"), i=ti, j=tj, jac=jac,
                 root=root, removed=rem)
        cs = component_stats(root, n)
        truth[str(tau)] = dict(n_pairs=int(len(ti)), t_join_s=time.time() - t0,
                               index_entries=res["index_entries"],
                               n_candidates=res["n_candidates"], removed=int(rem.sum()),
                               **cs)
        log(corpus, "exact tau", tau, truth[str(tau)])
    # metadata columns needed later
    dom = domains(meta["url"])
    pq.write_table(pa.table({
        "id": meta["id"], "url": meta["url"], "domain": dom,
        "nchar": np.asarray(meta["nchar"], np.int64),
        "nwords": np.asarray(meta["nwords"], np.int64),
        "nshingle": np.asarray(meta["nshingle"], np.int64),
        "n_unique": uniq,
    }), os.path.join(WRK, f"{corpus}_meta.parquet"), compression="zstd")
    info = dict(corpus=corpus, n=n, n_shingles=int(len(flat)), n_empty=int((~valid).sum()),
                t_rank_s=t_rank, S=int(S), V=int(V), truth=truth,
                provenance="independently rerun (this run)")
    with open(os.path.join(WRK, f"{corpus}_prepare.json"), "w") as f:
        json.dump(info, f, indent=1)
    log(corpus, "prepare done")


# ------------------------------------------------------------------ run one seed
def run_corpus_seeds(corpus, seeds):
    n = CORPORA[corpus]
    off, flat, nw, meta = load_shingles(corpus)
    z = np.load(os.path.join(WRK, f"{corpus}_ranks.npz"))
    S = int(z["S"]); V = int(z["V"]); uniq = np.asarray(z["uniq"])
    valid = (np.diff(off) > 0)
    truth = {}
    for tau in TAUS:
        q = np.load(os.path.join(WRK, f"{corpus}_truth_t{tau}.npz"))
        truth[tau] = (np.asarray(q["i"]), np.asarray(q["j"]), np.asarray(q["jac"]),
                      np.asarray(q["root"]), np.asarray(q["removed"]))
    for seed in seeds:
        run_seed(corpus, seed, off, flat, nw, uniq, valid, truth)


def run_seed(corpus, seed, off, flat, nw, uniq, valid, truth):
    n = CORPORA[corpus]
    t0 = time.time()
    sig = L.signatures(off, flat, H, seed, "dt")
    t_sig = time.time() - t0
    t0 = time.time()
    t0a = time.time()
    K = L.band_keys(sig, B, R)
    t_band = time.time() - t0a
    codes, ci, cj, tot = L.lsh_pairs(K, valid, 400_000_000)
    del K
    pi, pj = (codes // n).astype(np.int64), (codes % n).astype(np.int64)
    t0b = time.time()
    cjac = L.jaccard_pairs(off, flat, pi, pj)
    t_jac = time.time() - t0b
    del codes
    # truth hit per pair (banding hit) for pair recall
    res = {"corpus": corpus, "seed": seed, "n_cand_pairs": int(len(pi)),
           "n_cand_pairs_with_mult": int(tot), "t_sig_s": t_sig, "t_band_s": t_band,
           "t_jaccard_s": t_jac, "peak_rss_mb": C.peak_rss_mb()[0], "per_tau": {}}
    root_raw = L.uf_components(n, ci, cj)
    rem_raw = L.removed_set(root_raw) & valid
    has_nb = np.zeros(n, bool)
    for tau in TAUS:
        ti, tj, jac, root_ex, rem_ex = truth[tau]
        has_nb[:] = False
        has_nb[ti] = True
        has_nb[tj] = True
        hit = L.truth_hits(sig, ti, tj, B, R).astype(bool)
        ver = cjac >= tau - 1e-12
        root_ver = L.uf_components(n, pi[ver], pj[ver])
        rem_ver = L.removed_set(root_ver)
        kept_w = greedy_mis(n, pi[ver], pj[ver])
        rem_wit = (~kept_w) & valid
        # stage counts
        n_truth = len(ti)
        n_ver = int(ver.sum())
        # wrong deletions: (a) unsupported = no true neighbour at all;
        # (b) hijacked = has a true neighbour and the exact join keeps it (it is its
        # component's representative), but a false candidate bridge merged it under a
        # lower-id document.
        wrong_raw = rem_raw & ~has_nb
        hijack = rem_raw & has_nb & ~rem_ex
        missed_ver = rem_ex & ~rem_ver
        # residual truth pairs in each retained set
        def residual(rem):
            return int((~rem[ti] & ~rem[tj]).sum())
        def content(rem):
            keep = ~rem
            return dict(tokens=int(nw[keep].sum()),
                        uniq=int(uniq[keep].sum()),
                        docs=int(keep.sum()))
        m = dict(
            n_truth_pairs=n_truth, n_cand_pairs=int(len(pi)), n_verified=n_ver,
            cand_precision=(n_ver / len(pi) if len(pi) else None),
            edge_recall=float(hit.mean()) if n_truth else None,
            raw_removed=int(rem_raw.sum()), raw_wrong=int(wrong_raw.sum()),
            raw_wrong_isolated=int(wrong_raw.sum()),
            raw_wrong_hijack=int(hijack.sum()),
            raw_wrong_total=int((wrong_raw | hijack).sum()),
            raw_missed=int((rem_ex & ~rem_raw).sum()),
            raw_overlap=int((rem_raw & rem_ex).sum()),
            ver_removed=int(rem_ver.sum()), ver_missed=int(missed_ver.sum()),
            ver_wrong=int((rem_ver & ~has_nb).sum()),
            wit_removed=int(rem_wit.sum()), wit_missed=int((rem_ex & ~rem_wit).sum()),
            wit_wrong=int((rem_wit & ~has_nb).sum()),
            exact_removed=int(rem_ex.sum()),
            raw_residual_pairs=residual(rem_raw), ver_residual_pairs=residual(rem_ver),
            wit_residual_pairs=residual(rem_wit), exact_residual_pairs=residual(rem_ex),
            raw_content=content(rem_raw), ver_content=content(rem_ver),
            wit_content=content(rem_wit), exact_content=content(rem_ex),
            n_symdiff_raw_exact=int((rem_raw != rem_ex).sum()),
            n_symdiff_ver_exact=int((rem_ver != rem_ex).sum()),
        )
        # deletion recall of exact-removed docs
        m["raw_deletion_recall"] = (float((rem_raw & rem_ex).sum() / rem_ex.sum())
                                    if rem_ex.sum() else None)
        m["ver_deletion_recall"] = (float((rem_ver & rem_ex).sum() / rem_ex.sum())
                                    if rem_ex.sum() else None)
        m["wit_deletion_recall"] = (float((rem_wit & rem_ex).sum() / rem_ex.sum())
                                    if rem_ex.sum() else None)
        # cluster-weighted edge recall: mean over exact components (size>=2) of the
        # fraction of that component's truth edges that banded together
        comp = root_ex[ti]
        m["cluster_weighted_edge_recall"] = cluster_weighted(comp, hit)
        res["per_tau"][str(tau)] = m
        np.savez_compressed(os.path.join(WRK, f"{corpus}_s{seed}_t{tau}_sets.npz"),
                            rem_raw=np.packbits(rem_raw), rem_ver=np.packbits(rem_ver),
                            rem_wit=np.packbits(rem_wit), rem_exact=np.packbits(rem_ex),
                            root_raw=root_raw.astype(np.int32),
                            root_ver=root_ver.astype(np.int32), n=n)
        del ti, tj, jac, root_ex, rem_ex, hit, ver, root_ver, rem_ver, kept_w, rem_wit
    append_row(res)
    log(corpus, "seed", seed, "cand", res["n_cand_pairs"], "done")
    return res


def cluster_weighted(comp, hit):
    """Mean over components of per-component edge hit rate."""
    if len(comp) == 0:
        return None
    order = np.argsort(comp, kind="stable")
    c = comp[order]
    h = hit[order].astype(np.float64)
    brk = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    sums = np.add.reduceat(h, brk)
    cnts = np.diff(np.r_[brk, len(c)])
    return float((sums / cnts).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "run", "seedargs"])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--threads", type=int, default=15)
    a = ap.parse_args()
    C.set_threads(a.threads)
    np.seterr(all="ignore")
    if a.mode == "prepare":
        prepare(a.corpus, a.threads)
    else:
        run_corpus_seeds(a.corpus, [int(x) for x in a.seeds.split(",")])


if __name__ == "__main__":
    main()
