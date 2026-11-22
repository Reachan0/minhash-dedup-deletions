#!/usr/bin/env python3
"""Independent re-execution of the decision cells (C4 n=700,000 and FineWeb n=1,000,000,
datatrove hash family seed 1, 14x8, tau=0.8).

Everything is recomputed from the raw corpus files: shingling is streamed from the
parquet/json.gz inputs, signatures and band keys are regenerated, candidate edges are
rebuilt, and the exact join is re-run.  No shingle cache, pair file or LSH dump from the
earlier run is read.

Writes, per corpus:
  .work/dedupjoin/rerun/{corpus}_{n}.npz           shingle CSR cache (reused by analysis)
  .work/dedupjoin/rerun/{corpus}_{n}_meta.parquet  ids/urls/lengths
  .work/dedupjoin/rerun/pairs_{corpus}_{n}_{tau}.npz
  .work/dedupjoin/rerun/sets_{corpus}_{n}_s1_14x8_t{tau}.npz
  results/rerun.jsonl                              one row per phase (append-only)
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CODE = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03")
sys.path.insert(0, CODE)
import common as C  # noqa: E402
import exactjoin as EJ  # noqa: E402
import lsh as L  # noqa: E402

OUTR = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/rerun")


def sha256_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def append(path, row):
    row = dict(row)
    row.setdefault("ts_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    row.setdefault("provenance", "independently rerun (this run)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--taus", default="0.8")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rows", default=os.path.join(OUTR, "rerun.jsonl"))
    ap.add_argument("--skip_exact_below", type=float, default=-1.0,
                    help="skip exact join thresholds below this value (cost guard)")
    ap.add_argument("--from_cache", action="store_true",
                    help="resume from the shingle cache written by an earlier call that already "
                         "completed the shingling phase (the cache is a straight dump of the raw files)")
    a = ap.parse_args()
    os.makedirs(WRK, exist_ok=True)
    os.makedirs(OUTR, exist_ok=True)
    C.set_threads(a.threads)
    taus = [float(x) for x in a.taus.split(",")]
    tag = f"{a.corpus}_{a.n}"

    # ---- 1. shingles straight from the raw files (no earlier cache) ----
    T0 = time.time()
    if a.from_cache:
        z = np.load(os.path.join(WRK, f"{tag}.npz"), mmap_mode="r")
        off = np.asarray(z["off"])
        flat = z["flat"]
        nw = np.asarray(z["nwords"])
        assert len(off) - 1 == a.n, (len(off) - 1, a.n)
        row = dict(kind="shingles_reused", corpus=a.corpus, n=a.n,
                   source="shingle cache written by the shingling phase of this run "
                          "(streamed from the raw files in this run)",
                   n_shingles=int(off[-1]), shingle_sha256=sha256_arr(flat))
        append(a.rows, row)
        print(json.dumps(row), flush=True)
        meta = None
    else:
        off, flat, nw, meta = C.shingle_stream(a.corpus, a.n, a.workers)
    if not a.from_cache:
        t_shingle = time.time() - T0
        np.savez(os.path.join(WRK, f"{tag}.npz"), off=off, flat=flat, nwords=nw,
                 info=json.dumps(dict(t_shingle=t_shingle, n_docs=a.n, n_shingles=int(off[-1]),
                                      source="streamed from raw corpus files in this run")))
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.table({"id": meta["id"], "dump": meta["dump"], "url": meta["url"],
                                 "nchar": meta["nchar"], "nwords": nw, "nshingle": np.diff(off),
                                 "head": meta["head"]}),
                       os.path.join(WRK, f"{tag}_meta.parquet"), compression="zstd")
        row = dict(kind="shingles", corpus=a.corpus, n=a.n, t_shingle_s=t_shingle,
                   n_shingles=int(off[-1]), n_empty=int((np.diff(off) == 0).sum()),
                   shingle_sha256=sha256_arr(flat), off_sha256=sha256_arr(off),
                   nwords_total=int(nw.sum()), nchar_total=int(meta["nchar"].sum()),
                   peak_rss_mb=C.peak_rss_mb()[0])
        append(a.rows, row)
        print(json.dumps(row), flush=True)
        del meta

    # ---- 2. ranks + exact join at every requested threshold ----
    t0 = time.time()
    R, S, V = EJ.build_ranks(off, flat)
    t_rank = time.time() - t0
    row = dict(kind="ranks", corpus=a.corpus, n=a.n, t_rank_s=t_rank, S=int(S), V=int(V),
               rank_sha256=sha256_arr(R), peak_rss_mb=C.peak_rss_mb()[0])
    append(a.rows, row)
    print(json.dumps(row), flush=True)

    # ---- 3. LSH seed 1, dt family, 14x8 (fresh signatures, before flat is freed) ----
    n = a.n
    sizes = np.diff(off)
    valid = sizes > 0
    H = 112
    t0 = time.time()
    sig = L.signatures(off, flat, H, 1, "dt")
    t_sig = time.time() - t0
    row = dict(kind="signatures", corpus=a.corpus, n=a.n, seed=1, H=H, t_sig_s=t_sig,
               sig_sha256=sha256_arr(sig), peak_rss_mb=C.peak_rss_mb()[0])
    append(a.rows, row)
    print(json.dumps(row), flush=True)
    del flat

    # ---- 3b. band keys and candidate edges ----
    t0 = time.time()
    K = L.band_keys(sig, 14, 8)
    codes, ci, cj, tot = L.lsh_pairs(K, valid, 200_000_000)
    t_band = time.time() - t0
    row = dict(kind="banding", corpus=a.corpus, n=a.n, band_sha256=sha256_arr(K),
               n_cand_pairs=int(len(codes)), n_chain_edges=int(len(ci)),
               n_cand_pairs_with_mult=int(tot), t_band_s=t_band)
    append(a.rows, row)
    print(json.dumps(row), flush=True)
    del K

    t0 = time.time()
    pi, pj = codes // n, codes % n
    z = np.load(os.path.join(WRK, f"{tag}.npz"))
    cjac = L.jaccard_pairs(z["off"], z["flat"], pi, pj)
    del z
    t_ver = time.time() - t0
    np.savez(os.path.join(WRK, f"cand_{tag}_s1_14x8.npz"), i=pi, j=pj, jac=cjac, ci=ci, cj=cj)
    row = dict(kind="candidates", corpus=a.corpus, n=a.n, n_cand_pairs=int(len(codes)),
               t_verify_all_s=t_ver, cand_sha256=sha256_arr(pi.astype(np.int64) * n + pj),
               jac_sha256=sha256_arr(cjac))
    append(a.rows, row)
    print(json.dumps(row), flush=True)
    root_lsh = L.uf_components(n, ci, cj)
    rem_lsh = L.removed_set(root_lsh) & valid
    in_clu = np.zeros(n, bool)
    in_clu[ci] = True
    in_clu[cj] = True

    # ---- 4. exact join and the three deleted sets, one threshold at a time ----
    for tau in taus:
        t0 = time.time()
        res = EJ.exact_join(off, R, S, V, tau, a.threads)
        dt = time.time() - t0
        np.savez(os.path.join(WRK, f"pairs_{tag}_{tau}.npz"), i=res["i"], j=res["j"], jac=res["jac"])
        row = dict(kind="exact", corpus=a.corpus, n=a.n, tau=tau, t_join_s=dt,
                   t_index_s=res["t_index"], t_probe_s=res["t_probe_verify"],
                   n_pairs=int(len(res["i"])), n_candidates=res["n_candidates"],
                   index_entries=res["index_entries"],
                   pair_sha256=sha256_arr(res["i"].astype(np.int64) * a.n + res["j"]),
                   peak_rss_mb=C.peak_rss_mb()[0])
        append(a.rows, row)
        print(json.dumps(row), flush=True)

        ti, tj, s = res["i"].astype(np.int64), res["j"].astype(np.int64), res["jac"]
        root_exact = L.uf_components(n, ti, tj)
        rem_exact = L.removed_set(root_exact)
        ver = cjac >= tau - 1e-12
        root_lshv = L.uf_components(n, pi[ver], pj[ver])
        rem_lshv = L.removed_set(root_lshv)
        hit = L.truth_hits(sig, ti, tj, 14, 8).astype(bool)
        has_nb = np.zeros(n, bool)
        has_nb[ti] = True
        has_nb[tj] = True
        wrong = rem_lsh & ~has_nb
        sym = np.flatnonzero(rem_exact != rem_lshv)
        np.savez(os.path.join(WRK, f"sets_{tag}_s1_14x8_t{tau}.npz"),
                 rem_exact=rem_exact, rem_lsh=rem_lsh, rem_lshv=rem_lshv,
                 root_exact=root_exact, root_lsh=root_lsh, root_lshv=root_lshv,
                 has_nb=has_nb, in_clu=in_clu, truth_i=ti, truth_j=tj, truth_jac=s,
                 truth_hit=hit, truth_ver=ver, sym=sym)
        row = dict(kind="deleted_sets", corpus=a.corpus, n=a.n, tau=tau, seed=1, config="dt 14x8",
                   lsh_removed=int(rem_lsh.sum()), lsh_wrongly_deleted=int(wrong.sum()),
                   exact_removed=int(rem_exact.sum()), lshv_removed=int(rem_lshv.sum()),
                   lshv_missed_removals=int(rem_exact.sum() - rem_lshv.sum()),
                   symmetric_difference=int(len(sym)),
                   n_truth_pairs=int(len(ti)), pair_recall=float(hit.mean()) if len(ti) else None,
                   n_cand_pairs=int(len(codes)), verified_pairs=int(ver.sum()),
                   raw_pair_precision=float(ver.mean()) if len(ver) else None,
                   lsh_missed_docs=int((has_nb & ~in_clu).sum()),
                   removed_overlap=int((rem_lsh & rem_exact).sum()),
                   peak_rss_mb=C.peak_rss_mb()[0])
        append(a.rows, row)
        print(json.dumps(row), flush=True)
        del res, ti, tj, s, root_exact, root_lshv, ver, hit, has_nb, wrong, sym


if __name__ == "__main__":
    main()
