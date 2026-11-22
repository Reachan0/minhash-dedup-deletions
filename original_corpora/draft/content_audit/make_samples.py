#!/usr/bin/env python3
"""Build the stratified sample for the content audit.

Design (pre-registered in notes/probe_gate.md before labels were produced):

  c4_union120       160 documents sampled uniformly from the union over 100 seeds of raw
                    LSH deletions that have no exact neighbour at tau=0.8.  Conditional on
                    a seed's deletion set this is a uniform sample of that set, so the same
                    labels estimate a per-seed lost-information share for all 100 seeds.
  c4_agreed_deleted  20 documents deleted by every one of the five main seeds and all four
                    methods (controls; expected repost/template).
  c4_agreed_kept     20 documents kept by every seed and method (controls; expected unique).
  wet_disputed       40 documents from K_raw \ K_exact, WET raw crawl, seed 1, tau=0.8.
  fw_disputed        20 documents from K_raw \ K_exact, FineWeb 1M, seed 1, tau=0.8.

For every sampled document the script stores: corpus index, document id, url, domain,
word/shingle counts, the per-seed raw deletion membership (100 seeds), method decisions
(raw/verified/witness/exact), the exact nearest-neighbour Jaccard over the full input
(J >= 0.3 for C4/FineWeb from the stored pair lists; exact inverted-index query for WET),
and the text.  The nearest-neighbour text is stored when max J >= 0.4 so the labeller can
compare facts.

Outputs (relative to output/dedupjoin-pvldb/):
  results/content_audit/samples.jsonl          one row per sampled document
  results/content_audit/strata_counts.json     population counts and design
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
WRK = os.path.join(ROOT, ".work/dedupjoin")
MAIN = os.path.join(WRK, "main")
RERUN = os.path.join(WRK, "rerun")
POOL = os.path.join(WRK, "pool")
OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results/content_audit")
N_C4 = 700000
N_FW = 1000000
N_WET = 175272


def unpack(path, key, n):
    z = np.load(path)
    return np.unpackbits(z[key], bitorder="big")[:n].astype(bool)


def c4_arrays():
    # exact truth (seed-independent) from the independent rerun
    r = np.load(os.path.join(RERUN, "sets_c4_700000_s1_14x8_t0.8.npz"))
    has = r["has_nb"].astype(bool)
    # method decisions from the five-main-seed run (np.packbits default = big endian);
    # the seed-1 exact set here is identical to the rerun truth (verified: 3910/3910)
    z1 = np.load(os.path.join(MAIN, "c4_s1_t0.8_sets.npz"))
    m1 = {k: np.unpackbits(z1[k], bitorder="big")[:N_C4].astype(bool)
          for k in ("rem_raw", "rem_ver", "rem_wit", "rem_exact")}
    ex = m1["rem_exact"]
    raw1, ver1, wit1 = m1["rem_raw"], m1["rem_ver"], m1["rem_wit"]
    assert ex.sum() == r["rem_exact"].sum() and (ex == r["rem_exact"]).all()
    # 100 variance seeds
    var = {}
    for f in sorted(glob.glob(os.path.join(MAIN, "c4_var_s*_t0.8.npz"))):
        s = int(re.search(r"_s(\d+)_", f).group(1))
        var[s] = unpack(f, "rem", N_C4)
    # five main seeds, four methods
    main = {}
    for s in range(1, 6):
        z = np.load(os.path.join(MAIN, f"c4_s{s}_t0.8_sets.npz"))
        main[s] = {k: np.unpackbits(z[k], bitorder="big")[:N_C4].astype(bool)
                   for k in ("rem_raw", "rem_ver", "rem_wit", "rem_exact")}
    return dict(has=has, ex=ex, raw1=raw1, ver1=ver1, wit1=wit1, var=var, main=main)


def pair_max(path, n):
    """Per-document max Jaccard and its argmax doc from a stored exact pair list."""
    z = np.load(path)
    i, j, s = z["i"].astype(np.int64), z["j"].astype(np.int64), z["jac"].astype(np.float64)
    mx = np.zeros(n, np.float64)
    arg = np.full(n, -1, np.int64)
    # two passes over the pair list; 871k rows, cheap
    order = np.argsort(s)
    for k in order:
        a, b, v = i[k], j[k], s[k]
        if v > mx[a]:
            mx[a], arg[a] = v, b
        if v > mx[b]:
            mx[b], arg[b] = v, a
    return mx, arg


def build_index(off, flat):
    t0 = time.time()
    uniq = np.unique(flat)
    inv = np.searchsorted(uniq, flat)
    n = len(off) - 1
    sizes = np.diff(off)
    rows = np.repeat(np.arange(n, dtype=np.int64), sizes)
    order = np.argsort(inv, kind="stable")
    postings = rows[order].astype(np.int32)
    counts = np.bincount(inv, minlength=len(uniq))
    del rows, order, inv
    indptr = np.zeros(len(uniq) + 1, np.int64)
    np.cumsum(counts, out=indptr[1:])
    del counts
    return dict(uniq=uniq, indptr=indptr, postings=postings, build_s=time.time() - t0)


def query(idx, off, flat, sizes, q, n):
    sid = np.searchsorted(idx["uniq"], flat[off[q]:off[q + 1]])
    if len(sid) == 0:
        return 0.0, -1
    segs = [idx["postings"][idx["indptr"][s]:idx["indptr"][s + 1]] for s in sid]
    cand = np.concatenate(segs) if segs else np.zeros(0, np.int32)
    if cand.size == 0:
        return 0.0, -1
    cnt = np.bincount(cand, minlength=n)
    nonz = np.nonzero(cnt)[0]
    o = cnt[nonz].astype(np.float64)
    jac = o / (float(len(sid)) + sizes[nonz] - o)
    jac[nonz == q] = -1.0
    k = int(np.argmax(jac))
    return float(jac[k]), int(nonz[k])


def fetch_text(corpus, ids_needed, index_meta):
    """Return id -> text for the requested ids, streaming the pool shards of one corpus."""
    want = set(ids_needed)
    out = {}
    files = sorted(glob.glob(os.path.join(POOL, corpus, "*.parquet")))
    for f in files:
        t = pq.read_table(f, columns=["id", "text"])
        ids = t.column("id").to_pylist()
        hit = [k for k, x in enumerate(ids) if x in want]
        if not hit:
            continue
        texts = t.column("text")
        for k in hit:
            out[ids[k]] = texts[k].as_py()
        t = None
        print(f"  {corpus} {os.path.basename(f)}: {len(hit)} hits (total {len(out)})", flush=True)
        if len(out) == len(want):
            break
    return out


def load_meta(path):
    t = pq.read_table(path)
    d = {c: t.column(c).to_pylist() for c in t.schema.names}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--n-c4", type=int, default=160)
    ap.add_argument("--n-wet", type=int, default=40)
    ap.add_argument("--n-fw", type=int, default=20)
    ap.add_argument("--n-control", type=int, default=20)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(OUT, exist_ok=True)

    print("loading C4 decisions", flush=True)
    A = c4_arrays()
    has, ex = A["has"], A["ex"]
    U = np.zeros(N_C4, bool)
    for s, b in A["var"].items():
        U |= b
    U_noNB = U & ~has
    D1 = A["raw1"] & ~ex
    print(f"C4 union {int(U.sum())} noNB {int(U_noNB.sum())} D1 {int(D1.sum())} "
          f"D1_noNB {int((D1 & ~has).sum())}", flush=True)

    # controls
    all_del = np.ones(N_C4, bool)
    all_keep = np.zeros(N_C4, bool)
    for s, m in A["main"].items():
        all_del &= m["rem_raw"] & m["rem_ver"] & m["rem_wit"] & m["rem_exact"]
        keep = ~(m["rem_raw"] | m["rem_ver"] | m["rem_wit"] | m["rem_exact"])
        all_keep |= keep
    all_keep &= ~U  # kept by all seeds as well
    print(f"controls all_del {int(all_del.sum())} all_keep {int(all_keep.sum())}", flush=True)

    sample = []
    idx = rng.choice(np.nonzero(U_noNB)[0], size=a.n_c4, replace=False)
    for i in idx:
        sample.append(dict(corpus="c4", index=int(i), stratum="c4_union_no_nb"))
    for i in rng.choice(np.nonzero(all_del)[0], size=a.n_control, replace=False):
        sample.append(dict(corpus="c4", index=int(i), stratum="c4_agreed_deleted"))
    for i in rng.choice(np.nonzero(all_keep)[0], size=a.n_control, replace=False):
        sample.append(dict(corpus="c4", index=int(i), stratum="c4_agreed_kept"))

    # WET / FineWeb disputed, seed 1
    z = np.load(os.path.join(MAIN, "wet_s1_t0.8_sets.npz"))
    wraw = np.unpackbits(z["rem_raw"], bitorder="big")[:N_WET].astype(bool)
    wex = np.unpackbits(z["rem_exact"], bitorder="big")[:N_WET].astype(bool)
    wver = np.unpackbits(z["rem_ver"], bitorder="big")[:N_WET].astype(bool)
    wwit = np.unpackbits(z["rem_wit"], bitorder="big")[:N_WET].astype(bool)
    wd = np.nonzero(wraw & ~wex)[0]
    for i in rng.choice(wd, size=min(a.n_wet, len(wd)), replace=False):
        sample.append(dict(corpus="wet", index=int(i), stratum="wet_disputed"))

    z = np.load(os.path.join(MAIN, "fw_s1_t0.8_sets.npz"))
    fraw = np.unpackbits(z["rem_raw"], bitorder="big")[:N_FW].astype(bool)
    fex = np.unpackbits(z["rem_exact"], bitorder="big")[:N_FW].astype(bool)
    fver = np.unpackbits(z["rem_ver"], bitorder="big")[:N_FW].astype(bool)
    fwit = np.unpackbits(z["rem_wit"], bitorder="big")[:N_FW].astype(bool)
    fd = np.nonzero(fraw & ~fex)[0]
    for i in rng.choice(fd, size=min(a.n_fw, len(fd)), replace=False):
        sample.append(dict(corpus="fw", index=int(i), stratum="fw_disputed"))

    # metadata + exact nearest neighbour
    meta = {
        "c4": load_meta(os.path.join(MAIN, "c4_meta.parquet")),
        "wet": load_meta(os.path.join(MAIN, "wet_meta.parquet")),
        "fw": load_meta(os.path.join(RERUN, "fw_1000000_meta.parquet")),
    }
    nn = {}
    for corpus, path, n in (("c4", os.path.join(RERUN, "pairs_c4_700000_0.3.npz"), N_C4),
                            ("fw", os.path.join(RERUN, "pairs_fw_1000000_0.3.npz"), N_FW)):
        print(f"max-J from {os.path.basename(path)}", flush=True)
        mx, arg = pair_max(path, n)
        nn[corpus] = (mx, arg)
    print("WET inverted index", flush=True)
    zw = np.load(os.path.join(MAIN, "wet_175272.npz"), mmap_mode="r")
    off = np.asarray(zw["off"])
    flat = zw["flat"]
    sizes_w = np.diff(off).astype(np.float64)
    widx = build_index(off, flat)
    print("  built in", round(widx["build_s"], 1), "s", flush=True)

    # enrich rows
    for r in sample:
        c, i = r["corpus"], r["index"]
        m = meta[c]
        r["id"] = m["id"][i]
        r["url"] = m.get("url", [None] * (i + 1))[i]
        r["domain"] = m.get("domain", [None] * (i + 1))[i]
        r["nwords"] = int(m["nwords"][i])
        r["nshingle"] = int(m["nshingle"][i])
        r["nchar"] = int(m.get("nchar", [0] * (i + 1))[i] or 0)
        if c == "c4":
            r["removed_by_raw_seed"] = [s for s, b in A["var"].items() if b[i]]
            r["method_s1"] = dict(raw=bool(A["raw1"][i]), ver=bool(A["ver1"][i]),
                                  wit=bool(A["wit1"][i]), exact=bool(A["ex"][i]))
            r["has_nb"] = bool(has[i])
            mx, arg = nn["c4"]
            r["max_j"], r["max_j_doc"] = float(mx[i]), int(arg[i])
        elif c == "fw":
            r["method_s1"] = dict(raw=bool(fraw[i]), ver=bool(fver[i]),
                                  wit=bool(fwit[i]), exact=bool(fex[i]))
            mx, arg = nn["fw"]
            r["max_j"], r["max_j_doc"] = float(mx[i]), int(arg[i])
        else:
            r["method_s1"] = dict(raw=bool(wraw[i]), ver=bool(wver[i]),
                                  wit=bool(wwit[i]), exact=bool(wex[i]))
            b, bid = query(widx, off, flat, sizes_w, i, N_WET)
            r["max_j"], r["max_j_doc"] = float(b), int(bid)
    del widx, zw

    # text fetch
    texts = {}
    for corpus in ("c4", "wet", "fw"):
        ids = [r["id"] for r in sample if r["corpus"] == corpus]
        print(f"fetching {len(ids)} {corpus} texts", flush=True)
        texts.update(fetch_text(corpus, ids, None))
    # neighbour texts
    nb_ids = {"c4": [], "wet": [], "fw": []}
    for r in sample:
        if r["max_j"] >= 0.4 and r["max_j_doc"] >= 0:
            nb_ids[r["corpus"]].append(meta[r["corpus"]]["id"][r["max_j_doc"]])
    nb_texts = {}
    for corpus in ("c4", "wet", "fw"):
        if nb_ids[corpus]:
            print(f"fetching {len(nb_ids[corpus])} {corpus} neighbour texts", flush=True)
            nb_texts.update(fetch_text(corpus, nb_ids[corpus], None))

    missing = [r["id"] for r in sample if r["id"] not in texts]
    if missing:
        print(f"WARNING {len(missing)} texts missing", flush=True)

    with open(os.path.join(OUT, "samples.jsonl"), "w") as f:
        for r in sample:
            r["text"] = texts.get(r["id"], "")
            r["n_words_text"] = len(r["text"].split())
            nb = None
            if r["max_j"] >= 0.4 and r["max_j_doc"] >= 0:
                nid = meta[r["corpus"]]["id"][r["max_j_doc"]]
                nb = dict(id=nid, j=r["max_j"], text=nb_texts.get(nid, ""))
            r["neighbor"] = nb
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = dict(
        design="uniform samples per stratum; C4 union is uniform over the union of 100 raw-LSH seeds restricted to no-exact-neighbour documents",
        seed=a.seed,
        c4=dict(n_docs=N_C4, union_100_seeds=int(U.sum()), union_no_nb=int(U_noNB.sum()),
                d1=int(D1.sum()), d1_no_nb=int((D1 & ~has).sum()),
                agreed_deleted=int(all_del.sum()), agreed_kept=int(all_keep.sum())),
        wet=dict(n_docs=N_WET, disputed=int(wd.size)),
        fw=dict(n_docs=N_FW, disputed=int(fd.size)),
        sampled={s: sum(1 for r in sample if r["stratum"] == s)
                 for s in sorted({r["stratum"] for r in sample})},
        total=len(sample))
    with open(os.path.join(OUT, "strata_counts.json"), "w") as f:
        json.dump(counts, f, indent=2)
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    sys.exit(main())
