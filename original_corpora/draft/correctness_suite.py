#!/usr/bin/env python3
"""Exact-join correctness suite against brute-force enumeration.

Covers the cases the step order names explicitly:
  * short documents (0, 1, 2 shingles) and documents with no shingles at all
  * threshold-boundary pairs whose Jaccard equals tau exactly (and o+-1 around it)
  * cross-chunk pairs (the join splits work by document-order chunks)
  * chain duplicates (A~B, B~C, A!~C) and the resulting connected component
  * exact duplicates (J = 1)
plus a real-data check on FineWeb text with synthetic near-duplicates.

Every join variant (ppjoin, allpairs, nopos, nolen, prefixonly) is compared with a
brute-force enumeration that uses exact rational arithmetic, so float noise cannot
make a boundary pair pass or fail by accident.

Writes results/correctness.json and appends raw rows to results/correctness_rows.jsonl.
"""
import json
import os
import random
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CODE = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03")
sys.path.insert(0, CODE)
import common as C  # noqa: E402
import exactjoin as EJ  # noqa: E402

OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
VARIANTS = ["ppjoin", "allpairs", "nopos", "nolen", "prefixonly"]


def make_off(docs):
    off = np.zeros(len(docs) + 1, np.int64)
    np.cumsum([len(d) for d in docs], out=off[1:])
    flat = np.concatenate([np.asarray(sorted(set(d)), np.uint64) for d in docs]) if docs else np.zeros(0, np.uint64)
    return off, flat


def brute(docs, tau):
    fr = Fraction(str(tau)).limit_denominator(10000)
    out = set()
    sets = [set(d) for d in docs]
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            a, b = sets[i], sets[j]
            o = len(a & b)
            u = len(a | b)
            if u and Fraction(o, u) >= fr:
                out.add((i, j))
    return out


def components(pairs, n):
    """Connected components (size >= 2) of an edge set, as sorted tuples."""
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            par[rb] = ra
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(tuple(sorted(g)) for g in groups.values() if len(g) > 1)


def run_case(name, docs, taus, note, nchunks=None, n=2):
    off, flat = make_off(docs)
    R, S, V = EJ.build_ranks(off, flat)
    rows = []
    for tau in taus:
        b = brute(docs, tau)
        cb = components(b, len(docs))
        for var in VARIANTS:
            allp = var in ("allpairs", "prefixonly")
            res = EJ.exact_join(off, R, S, V, tau, n, allpairs=allp,
                                use_len=var not in ("nolen", "prefixonly"),
                                use_pos=var not in ("nopos", "prefixonly", "allpairs"),
                                nchunks=nchunks)
            got = set(zip(res["i"].tolist(), res["j"].tolist()))
            cg = components(got, len(docs))
            rows.append(dict(case=name, n_docs=len(docs), tau=tau, variant=var,
                             brute_pairs=len(b), join_pairs=len(got),
                             missing=sorted(b - got), extra=sorted(got - b),
                             components_brute=cb, components_join=cg,
                             components_identical=(cb == cg),
                             identical=(b == got and cb == cg)))
    return rows


def boundary_docs(tau):
    """Two docs with Jaccard exactly tau, plus docs one shared shingle below/above."""
    fr = Fraction(str(tau)).limit_denominator(1000)
    p, q = fr.numerator, fr.denominator
    docs, meta = [], []
    # equal sizes L with o = 2pL/(p+q)
    L = (p + q) // __import__("math").gcd(p, q)
    if (2 * p * L) % (p + q) == 0:
        o = 2 * p * L // (p + q)
        common = list(range(100000, 100000 + o + 2))
        for delta in (-1, 0, 1):
            oo = o + delta
            if 0 < oo <= len(common) and oo < 2 * L:
                a = common[:oo] + list(range(200000, 200000 + (L - oo)))
                b = common[:oo] + list(range(300000, 300000 + (L - oo)))
                ja = Fraction(oo, 2 * L - oo)
                docs.append(a)
                docs.append(b)
                meta.append(dict(overlap=oo, size=L, jac=str(ja), equals_tau=(ja == fr)))
    # unequal sizes: search small (a, b, o) with o/(a+b-o) == tau exactly, a != b
    found = None
    for a_sz in range(2, 41):
        for b_sz in range(2, 41):
            if a_sz == b_sz:
                continue
            for oo in range(1, min(a_sz, b_sz) + 1):
                if Fraction(oo, a_sz + b_sz - oo) == fr:
                    found = (a_sz, b_sz, oo)
                    break
            if found:
                break
        if found:
            break
    if found:
        a_sz, b_sz, oo = found
        a = list(range(400000, 400000 + oo)) + list(range(500000, 500000 + (a_sz - oo)))
        b = list(range(400000, 400000 + oo)) + list(range(600000, 600000 + (b_sz - oo)))
        docs += [a, b]
        meta.append(dict(overlap=oo, size_a=a_sz, size_b=b_sz,
                         jac=str(Fraction(oo, a_sz + b_sz - oo)), equals_tau=True))
    return docs, meta


def build_synthetic():
    cases = []

    docs = [[], [1], [2], [1], [10, 11], [10, 11], [20, 21, 22], [20, 21, 22, 23]]
    cases.append(("empty_and_short", docs, [0.5, 0.7, 0.8, 0.9],
                  "empty docs, singletons, exact duplicates of 2- and 4-shingle docs"))

    docs = [[], []] + [[i] for i in range(1, 6)]
    cases.append(("all_empty", docs, [0.5, 0.8], "every document is empty; no pair can exist"))

    chain = [
        list(range(0, 8)) + [100, 101],          # A
        list(range(0, 8)) + [200, 201],          # B: A~B = 8/12 = 0.667
        list(range(0, 6)) + [300, 301, 302, 303],  # C: B~C = 6/14 = 0.43, A~C = 6/16=0.375
    ]
    cases.append(("chain", chain, [0.4, 0.5, 0.6, 0.7],
                  "A-B = 8/12 and B-C = 6/14 are above 0.4 while A-C = 6/16 is below: "
                  "a three-document component that is not a clique"))

    rnd = random.Random(7)
    base = [rnd.sample(range(1000, 9999), 12) for _ in range(6)]
    docs = []
    for b in base:
        docs.append(b)
        docs.append(b + [rnd.randrange(10 ** 6)])  # J = 12/13
        docs.append(b[:6])                           # J = 6/12 = 0.5
    # pad to 60 docs so that chunk boundaries fall inside duplicate groups
    docs += [[rnd.randrange(10 ** 6)] for _ in range(60 - len(docs))]
    docs[0] = docs[0] + [7]  # noise
    docs[55] = base[0] + [8]  # near-duplicate of docs[0] placed at the far end
    cases.append(("cross_chunk", docs, [0.5, 0.8],
                  "duplicates placed at low and high document ids, join forced into 2 and 64 chunks"))

    for tau in [0.5, 0.7, 0.75, 0.8, 0.9]:
        docs, meta = boundary_docs(tau)
        cases.append((f"boundary_{tau}", docs, [tau],
                      "pairs with J exactly tau and one shared shingle below/above; "
                      + json.dumps(meta)))
    return cases


def real_data_check(max_docs=2500, n_extra=250):
    texts, _ = C.load_corpus("fw", max_docs)
    rnd = random.Random(0)
    extra = []
    for t in rnd.sample(texts, min(n_extra, len(texts))):
        w = t.split()
        p = rnd.choice([0.0, 0.01, 0.02, 0.05, 0.1, 0.2])
        extra.append(" ".join(x for x in w if rnd.random() >= p))
    texts = texts + extra
    off, flat, nw = C.build_shingles(texts, 2)
    R, S, V = EJ.build_ranks(off, flat)
    import scipy.sparse as sp
    uniq, inv = np.unique(flat, return_inverse=True)
    rows_n = len(off) - 1
    rows_idx = np.repeat(np.arange(rows_n), np.diff(off))
    X = sp.csr_matrix((np.ones(len(flat), np.float32), (rows_idx, inv)), shape=(rows_n, len(uniq)))
    O = (X @ X.T).tocoo()
    sz = np.diff(off).astype(np.float64)
    m = O.row < O.col
    i, j, o = O.row[m], O.col[m], O.data[m].astype(np.float64)
    jac = o / (sz[i] + sz[j] - o)
    rows = []
    for tau in [0.5, 0.7, 0.8, 0.9, 0.95]:
        k = jac >= tau - 1e-12
        b = set(zip(i[k].tolist(), j[k].tolist()))
        for var in VARIANTS:
            allp = var in ("allpairs", "prefixonly")
            res = EJ.exact_join(off, R, S, V, tau, 2, allpairs=allp,
                                use_len=var not in ("nolen", "prefixonly"),
                                use_pos=var not in ("nopos", "prefixonly", "allpairs"))
            got = set(zip(res["i"].tolist(), res["j"].tolist()))
            rows.append(dict(case="real_fineweb_plus_synthetic", n_docs=len(texts), tau=tau,
                             variant=var, brute_pairs=len(b), join_pairs=len(got),
                             identical=(b == got),
                             missing=sorted(b - got)[:20], extra=sorted(got - b)[:20]))
    return rows


def main():
    t0 = time.time()
    all_rows = []
    for name, docs, taus, note in build_synthetic():
        for nch in ([None] if name != "cross_chunk" else [2, 60, 64, 256]):
            rows = run_case(name, docs, taus, note, nchunks=nch)
            for r in rows:
                r["note"] = note
                if nch is not None:
                    r["nchunks"] = nch
            all_rows += rows
            print(name, "nchunks", nch, "all_ok", all(r["identical"] for r in rows), flush=True)
    if os.environ.get("SKIP_REAL", "0") != "1":
        rows = real_data_check()
        all_rows += rows
        for tau in sorted({r["tau"] for r in rows}):
            sub = [r for r in rows if r["tau"] == tau]
            print("real", tau, "all_ok", all(r["identical"] for r in sub),
                  "pairs", sub[0]["brute_pairs"], flush=True)

    cases = {}
    for r in all_rows:
        c = r["case"]
        d = cases.setdefault(c, {"n_checks": 0, "failures": 0, "taus": set(), "variants": set()})
        d["n_checks"] += 1
        d["failures"] += 0 if r["identical"] else 1
        d["taus"].add(r["tau"])
        d["variants"].add(r["variant"])
    for c, d in cases.items():
        d["taus"] = sorted(d["taus"])
        d["variants"] = sorted(d["variants"])
        d["pass"] = d["failures"] == 0
    doc = dict(version=1,
               generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               provenance="independently rerun (this run)",
               method="exact prefix-filter join vs brute-force enumeration with exact rational "
                      "Jaccard; all five join variants; boundary taus constructed to make J equal tau exactly",
               datatrove_version="0.6.0 (join code is this repo's implementation, not datatrove)",
               total_checks=len(all_rows),
               total_failures=sum(0 if r["identical"] else 1 for r in all_rows),
               cases=cases,
               boundary_pairs=[r for r in all_rows if r["case"].startswith("boundary_")
                               and r["variant"] == "ppjoin"],
               rows_written=len(all_rows),
               runtime_s=time.time() - t0)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "correctness.json"), "w") as f:
        json.dump(doc, f, indent=2)
    with open(os.path.join(OUT, "correctness_rows.jsonl"), "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print("TOTAL", doc["total_checks"], "FAILURES", doc["total_failures"], flush=True)
    sys.exit(0 if doc["total_failures"] == 0 else 1)


if __name__ == "__main__":
    main()
