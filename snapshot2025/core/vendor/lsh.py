"""C03 MinHash-LSH vs exact ground truth.
 - signatures: family 'dt'  = datatrove get_signature exactly: ((x*a + b) mod 2^64) mod (2^61-1),
                                a,b from np.random.RandomState(seed) like datatrove;
               family 'mix' = ideal-ish: splitmix64(x xor salt_k) (ablation for S-curve deviation)
 - banding (b, r): two docs collide in a band iff all r minhash values equal (datatrove criterion)
 - pair level: recall of exact truth pairs (J>=tau), S-curve prediction mean(1-(1-s^r)^b),
   raw-LSH precision (all co-bucketed pairs), LSH+exact verification
 - doc level: union-find clusters (datatrove stage 3), keep min id per cluster, removed sets
   vs exact union-find; wrongly deleted = removed docs that have no J>=tau neighbour at all;
   missed = docs with a J>=tau neighbour that LSH leaves in a singleton.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import numba as nb

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

P61 = np.uint64((1 << 61) - 1)


@nb.njit(cache=True)
def splitmix(x):
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@nb.njit(parallel=True, cache=True)
def sig_dt(off, flat, a, b, P):
    n = off.shape[0] - 1
    H = a.shape[0]
    out = np.full((n, H), np.uint64(0xFFFFFFFFFFFFFFFF), np.uint64)
    for d in nb.prange(n):
        for i in range(off[d], off[d + 1]):
            x = flat[i]
            for h in range(H):
                v = (x * a[h] + b[h]) % P
                if v < out[d, h]:
                    out[d, h] = v
    return out


@nb.njit(parallel=True, cache=True)
def sig_mix(off, flat, salts):
    n = off.shape[0] - 1
    H = salts.shape[0]
    out = np.full((n, H), np.uint64(0xFFFFFFFFFFFFFFFF), np.uint64)
    for d in nb.prange(n):
        for i in range(off[d], off[d + 1]):
            x = flat[i]
            for h in range(H):
                v = splitmix(x ^ salts[h])
                if v < out[d, h]:
                    out[d, h] = v
    return out


def signatures(off, flat, H, seed, family):
    if family == "dt":
        gen = np.random.RandomState(seed)
        a = gen.randint(1, P61, dtype=np.uint64, size=(1, H))[0]
        b = gen.randint(0, P61, dtype=np.uint64, size=(1, H))[0]
        return sig_dt(off, flat, a, b, P61)
    gen = np.random.RandomState(10_000 + seed)
    salts = gen.randint(0, np.iinfo(np.int64).max, dtype=np.int64, size=H).astype(np.uint64)
    return sig_mix(off, flat, salts)


@nb.njit(cache=True)
def band_keys(sig, b, r):
    n = sig.shape[0]
    K = np.empty((b, n), np.uint64)
    for d in range(n):
        for i in range(b):
            h = np.uint64(0x12345678ABCDEF)
            for j in range(r):
                h = splitmix(h ^ sig[d, i * r + j])
            K[i, d] = h
    return K


@nb.njit(cache=True)
def truth_hits(sig, ti, tj, b, r):
    m = ti.shape[0]
    hit = np.zeros(m, np.uint8)
    for p in range(m):
        x = ti[p]
        y = tj[p]
        for i in range(b):
            eq = True
            for j in range(r):
                if sig[x, i * r + j] != sig[y, i * r + j]:
                    eq = False
                    break
            if eq:
                hit[p] = 1
                break
    return hit


@nb.njit(cache=True)
def _find(par, x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


@nb.njit(cache=True)
def uf_components(n, ei, ej):
    par = np.arange(n)
    for k in range(ei.shape[0]):
        a = _find(par, ei[k])
        c = _find(par, ej[k])
        if a != c:
            if a < c:
                par[c] = a
            else:
                par[a] = c
    root = np.empty(n, np.int64)
    for x in range(n):
        root[x] = _find(par, x)
    return root


def removed_set(root):
    """keep min id per cluster (root is min id by construction) -> removed mask"""
    return root != np.arange(len(root))


def lsh_pairs(K, valid, max_pairs):
    """All co-bucketed pairs (i<j), as sorted unique int64 codes i*n+j. Returns codes,
    and group-chain edges (datatrove style consecutive pairs) for union-find."""
    b, n = K.shape
    codes = []
    chain_i, chain_j = [], []
    tot = 0
    idx_all = np.nonzero(valid)[0]
    for i in range(b):
        k = K[i, idx_all]
        o = np.argsort(k, kind="stable")
        ks = k[o]
        docs = idx_all[o]
        brk = np.nonzero(np.diff(ks) != 0)[0] + 1
        starts = np.concatenate([[0], brk])
        ends = np.concatenate([brk, [len(ks)]])
        sz = ends - starts
        m = sz >= 2
        starts, ends, sz = starts[m], ends[m], sz[m]
        # chain edges
        same = np.nonzero(np.diff(ks) == 0)[0]
        chain_i.append(docs[same])
        chain_j.append(docs[same + 1])
        tot += int((sz * (sz - 1) // 2).sum())
        if tot > max_pairs:
            codes = None
            continue
        if codes is not None:
            for s, e in zip(starts, ends):
                g = np.sort(docs[s:e])
                ii, jj = np.triu_indices(len(g), 1)
                codes.append(g[ii].astype(np.int64) * n + g[jj])
    ci = np.concatenate(chain_i) if chain_i else np.zeros(0, np.int64)
    cj = np.concatenate(chain_j) if chain_j else np.zeros(0, np.int64)
    if codes is not None:
        codes = np.unique(np.concatenate(codes)) if codes else np.zeros(0, np.int64)
    return codes, ci, cj, tot


@nb.njit(parallel=True, cache=True)
def jaccard_pairs(off, flat, pi, pj):
    m = pi.shape[0]
    out = np.empty(m, np.float64)
    for p in nb.prange(m):
        a0, a1 = off[pi[p]], off[pi[p] + 1]
        b0, b1 = off[pj[p]], off[pj[p] + 1]
        i, j, o = a0, b0, 0
        while i < a1 and j < b1:
            if flat[i] == flat[j]:
                o += 1
                i += 1
                j += 1
            elif flat[i] < flat[j]:
                i += 1
            else:
                j += 1
        u = (a1 - a0) + (b1 - b0) - o
        out[p] = o / u if u > 0 else 0.0
    return out


def scurve(a, off, flat, TI, TJ, TJAC, taus, cfgs, H):
    """S-curve test: restrict to docs that appear in truth pairs, many hash seeds; the
    expectation over seeds of pair recall equals mean(1-(1-s^r)^b) for an ideal min-wise
    family, so mean deviation +- SE across seeds isolates hash-family / implementation bias
    from the (strong) within-cluster correlation of pairs."""
    if len(TI) == 0:  # no truth pairs at all (tiny corpora): record and skip
        for fam in a.families.split(","):
            C.append_row(a.rows, dict(key=f"{a.tag}|scurve|{a.corpus}|{a.n}|{fam}|{a.scurve_seeds}", kind="scurve",
                                      tag=a.tag, corpus=a.corpus, n=a.n, family=fam, n_seeds=a.scurve_seeds,
                                      n_docs_in_pairs=0, res={}, note="no truth pairs"))
        return
    docs = np.unique(np.concatenate([TI, TJ]))
    remap = np.full(len(off) - 1, -1, np.int64)
    remap[docs] = np.arange(len(docs))
    sz = off[docs + 1] - off[docs]
    soff = np.zeros(len(docs) + 1, np.int64)
    np.cumsum(sz, out=soff[1:])
    sflat = np.concatenate([flat[off[d]:off[d + 1]] for d in docs])
    ti, tj = remap[TI], remap[TJ]
    K = a.scurve_seeds
    for fam in a.families.split(","):
        key = f"{a.tag}|scurve|{a.corpus}|{a.n}|{fam}|{K}"
        if key in C.done_keys(a.rows):
            continue
        t0 = time.time()
        rec = {(c, t): [] for c in cfgs for t in taus}
        for seed in range(1, K + 1):
            sig = signatures(soff, sflat, H, seed, fam)
            for (b, r) in cfgs:
                hit_all = truth_hits(sig, ti, tj, b, r).astype(bool)
                for t in taus:
                    m = TJAC >= t - 1e-12
                    rec[((b, r), t)].append(hit_all[m].mean() if m.sum() else np.nan)
        res = {}
        for (b, r) in cfgs:
            for t in taus:
                m = TJAC >= t - 1e-12
                s = TJAC[m]
                pred = float((1 - (1 - s ** r) ** b).mean()) if m.sum() else float("nan")
                v = np.array(rec[((b, r), t)])
                res[f"{b}x{r}|{t}"] = dict(n_truth_pairs=int(m.sum()), recall_mean=float(v.mean()),
                                           recall_std=float(v.std(ddof=0)), pred=pred,
                                           dev_pts=100 * (float(v.mean()) - pred),
                                           dev_se_pts=100 * float(v.std(ddof=0)) / np.sqrt(len(v)),
                                           recall_min=float(v.min()), recall_p05=float(np.quantile(v, 0.05)))
        row = dict(key=key, kind="scurve", tag=a.tag, corpus=a.corpus, n=a.n, family=fam, n_seeds=K,
                   n_docs_in_pairs=int(len(docs)), t_s=time.time() - t0, res=res)
        C.append_row(a.rows, row)
        print(key, json.dumps(res)[:3000])


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scurve_seeds", type=int, default=0)
    ap.add_argument("--scurve_only", type=int, default=0)
    ap.add_argument("--corpus", default="fw")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--taus", default="0.7,0.8,0.9")
    ap.add_argument("--configs", default="14x8,20x5,10x10,9x13")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--families", default="dt,mix")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--max_pairs", type=int, default=80_000_000)
    ap.add_argument("--rows", default=os.path.join(C.LOGS, "rows.jsonl"))
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--dump_dir", default=os.path.join(C.CACHE, "lsh_out"))
    return ap


def main():
    a = get_parser().parse_args()
    C.set_threads(a.threads)
    taus = [float(x) for x in a.taus.split(",")]
    off, flat, nw, info = C.load_cache(a.corpus, a.n)
    pf = os.path.join(C.CACHE, f"pairs_{a.corpus}_{a.n}_{min(taus)}.npz")
    z = np.load(pf)
    run_eval(a, off, flat, z["i"].astype(np.int64), z["j"].astype(np.int64), z["jac"])


def run_eval(a, off, flat, TI, TJ, TJAC):
    taus = [float(x) for x in a.taus.split(",")]
    cfgs = [tuple(int(v) for v in c.split("x")) for c in a.configs.split(",")]
    H = max(b * r for b, r in cfgs)
    done = C.done_keys(a.rows)
    n = len(off) - 1
    sizes = np.diff(off)
    valid = sizes > 0
    os.makedirs(a.dump_dir, exist_ok=True)
    if a.scurve_seeds:
        scurve(a, off, flat, TI, TJ, TJAC, taus, cfgs, H)
        if a.scurve_only:
            return
    # exact doc-level reference per tau
    ex = {}
    for tau in taus:
        m = TJAC >= tau - 1e-12
        root = uf_components(n, TI[m], TJ[m])
        has_nb = np.zeros(n, bool)
        has_nb[TI[m]] = True
        has_nb[TJ[m]] = True
        ex[tau] = dict(mask=m, removed=removed_set(root), has_nb=has_nb)
    for fam in a.families.split(","):
        for seed in [int(s) for s in a.seeds.split(",")]:
            pending = [c for c in cfgs if f"{a.tag}|lsh|{a.corpus}|{a.n}|{fam}|{seed}|{c[0]}x{c[1]}" not in done]
            if not pending:
                continue
            t0 = time.time()
            sig = signatures(off, flat, H, seed, fam)
            t_sig = time.time() - t0
            for (b, r) in pending:
                key = f"{a.tag}|lsh|{a.corpus}|{a.n}|{fam}|{seed}|{b}x{r}"
                t0 = time.time()
                K = band_keys(sig, b, r)
                codes, ci, cj, tot = lsh_pairs(K, valid, a.max_pairs)
                root_lsh = uf_components(n, ci, cj)
                t_lsh = time.time() - t0
                rem_lsh = removed_set(root_lsh) & valid
                in_clu_lsh = np.zeros(n, bool)
                in_clu_lsh[ci] = True
                in_clu_lsh[cj] = True
                row = dict(key=key, kind="lsh", tag=a.tag, corpus=a.corpus, n=a.n, family=fam, seed=seed, b=b, r=r,
                           num_hashes=b * r, threads=a.threads, t_signature_all_hashes=t_sig, H=H,
                           t_band_cluster=t_lsh, n_cand_pairs=(int(len(codes)) if codes is not None else None),
                           n_cand_pairs_with_mult=tot, n_valid_docs=int(valid.sum()),
                           lsh_removed=int(rem_lsh.sum()))
                cj_codes_jac = None
                if codes is not None and len(codes) == 0:
                    cj_codes_jac = np.zeros(0, np.float64)
                if codes is not None and len(codes):
                    pi, pj = codes // n, codes % n
                    t0 = time.time()
                    cjac = jaccard_pairs(off, flat, pi, pj)
                    row["t_verify_all_candidates"] = time.time() - t0
                    cj_codes_jac = cjac
                per_tau = {}
                for tau in taus:
                    e = ex[tau]
                    m = e["mask"]
                    ti, tj, s = TI[m], TJ[m], TJAC[m]
                    hit = truth_hits(sig, ti, tj, b, r).astype(bool)
                    pred = 1 - (1 - s ** r) ** b
                    nt = len(s)
                    rec = float(hit.mean()) if nt else float("nan")
                    prd = float(pred.mean()) if nt else float("nan")
                    # binomial-style SE of (measured - predicted): var = sum p(1-p) / N^2 (Poisson-binomial)
                    se = float(np.sqrt((pred * (1 - pred)).sum()) / nt) if nt else float("nan")
                    d = dict(n_truth_pairs=nt, recall=rec, pred_recall=prd, dev_pts=100 * (rec - prd), dev_se_pts=100 * se,
                             truth_jac_mean=float(s.mean()) if nt else None,
                             frac_truth_jac_ge_0_99=float((s >= 0.99).mean()) if nt else None)
                    # binned
                    bins = [tau, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0001]
                    bins = sorted(set([x for x in bins if x >= tau]))
                    bl = []
                    for lo, hi in zip(bins[:-1], bins[1:]):
                        mm = (s >= lo) & (s < hi)
                        if mm.sum():
                            bl.append([lo, hi, int(mm.sum()), float(hit[mm].mean()), float(pred[mm].mean())])
                    d["bins"] = bl
                    # raw LSH pair precision and LSH+verify
                    if cj_codes_jac is not None:
                        ver = cj_codes_jac >= tau - 1e-12
                        d["raw_pair_precision"] = float(ver.mean()) if len(ver) else None
                        d["verified_pairs"] = int(ver.sum())
                        d["verified_recall_check"] = int(ver.sum()) / nt if nt else None
                        pi, pj = codes // n, codes % n
                        root_v = uf_components(n, pi[ver], pj[ver])
                        rem_v = removed_set(root_v)
                        d["lshv_removed"] = int(rem_v.sum())
                        d["lshv_missed_removals"] = int(e["removed"].sum() - rem_v.sum())
                    # doc-level
                    hn = e["has_nb"]
                    d["exact_removed"] = int(e["removed"].sum())
                    d["exact_docs_with_nb"] = int(hn.sum())
                    d["lsh_removed"] = int(rem_lsh.sum())
                    wrong = rem_lsh & ~hn
                    d["lsh_wrongly_deleted"] = int(wrong.sum())  # removed though no J>=tau neighbour at all
                    d["lsh_missed_docs"] = int((hn & ~in_clu_lsh).sum())  # has neighbour but LSH singleton
                    d["removed_overlap"] = int((rem_lsh & e["removed"]).sum())
                    per_tau[str(tau)] = d
                    if fam == "dt" and seed == 1 and b == 14:
                        np.savez(os.path.join(a.dump_dir, f"{a.corpus}_{a.n}_{tau}_dt1_14x8.npz"),
                                 ti=ti, tj=tj, s=s, hit=hit, wrong=np.nonzero(wrong)[0],
                                 missed=np.nonzero(hn & ~in_clu_lsh)[0], rem_lsh=np.nonzero(rem_lsh)[0],
                                 root_lsh=root_lsh)
                if cj_codes_jac is not None and fam == "dt" and seed == 1 and b == 14:
                    np.savez(os.path.join(a.dump_dir, f"{a.corpus}_{a.n}_cand_dt1_14x8.npz"), codes=codes, jac=cj_codes_jac)
                row["per_tau"] = per_tau
                rs, rc = C.peak_rss_mb()
                row["peak_rss_mb_self"] = rs
                C.append_row(a.rows, row)
                print(key, json.dumps({t: {k: v for k, v in d.items() if k != "bins"} for t, d in per_tau.items()}))
            del sig


if __name__ == "__main__":
    main()
