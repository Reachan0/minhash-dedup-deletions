#!/usr/bin/env python3
"""Close the gap between the earlier run's 14x8 numbers and the standalone re-run.

The earlier analysis drew one signature matrix with H = max(b*r over all banding
configurations) = 117 and re-banded prefixes of it.  The official datatrove path draws
exactly 14*8 = 112 hashes.  `np.random.RandomState.randint(..., dtype=uint64)` is not
prefix-stable in `size` (its output for size=H is not the first H columns of size=H'>H),
so the shared 14x8 cell is a different banding, not the same banding with 5 extra hashes.

This script recomputes, from this run's shingle cache:
  * H=112 standalone signatures -> band keys -> candidate pairs -> chain components
  * H=117 signatures, first 112 columns -> same
and compares both with (a) the earlier run's logged values and (b) this run's rerun row.
It also counts how many of the 112 columns differ between the two draws.

Usage:
  python shared_h117_check.py --corpus c4 --n 700000
  python shared_h117_check.py --corpus all       # runs both corpora in subprocesses
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CODE = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03")
sys.path.insert(0, CODE)
import common as C  # noqa: E402
import lsh as L  # noqa: E402

OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/rerun")
LOGS = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/logs/C03/round2")
CELLS = {"c4": 700000, "fw": 1000000}


def old_lsh_row(corpus, n):
    for line in open(os.path.join(LOGS, "rows.jsonl")):
        r = json.loads(line)
        if (r.get("kind") == "lsh" and r.get("family") == "dt" and r.get("b") == 14
                and r.get("r") == 8 and r.get("corpus") == corpus and r.get("n") == n
                and r.get("seed") == 1):
            return r
    raise KeyError((corpus, n))


def rerun_row(corpus, n, tau):
    for line in open(os.path.join(OUT, "rerun.jsonl")):
        r = json.loads(line)
        if (r.get("kind") == "deleted_sets" and r.get("corpus") == corpus
                and r.get("n") == n and r.get("tau") == tau and r.get("seed") == 1):
            return r
    return None


def one(corpus, n, tau=0.8):
    t0 = time.time()
    C.set_threads(8)
    z = np.load(os.path.join(WRK, f"{corpus}_{n}.npz"), mmap_mode="r")
    off = np.asarray(z["off"])
    flat = z["flat"]
    valid = np.diff(off) > 0
    nd = len(off) - 1

    sig112 = L.signatures(off, flat, 112, 1, "dt")
    t_s112 = time.time() - t0
    t1 = time.time()
    sig117 = L.signatures(off, flat, 117, 1, "dt")
    t_s117 = time.time() - t1

    # is the H=117 draw a prefix extension of the H=112 draw?
    pref = sig117[:, :112]
    col_equal = []
    for h in range(112):
        col_equal.append(float((pref[:, h] == sig112[:, h]).mean()))
    n_pref_cols_equal = int(sum(1 for c in col_equal if c == 1.0))
    n_docs_any_diff = int((pref != sig112).any(axis=1).sum())

    K112 = L.band_keys(sig112, 14, 8)
    K117 = L.band_keys(np.ascontiguousarray(pref), 14, 8)
    del sig112, sig117, pref

    out = {}
    for name, K in (("standalone_h112", K112), ("shared_h117_prefix", K117)):
        codes, ci, cj, tot = L.lsh_pairs(K, valid, 200_000_000)
        rem = L.removed_set(L.uf_components(nd, ci, cj))
        p = np.load(os.path.join(WRK, f"pairs_{corpus}_{n}_{tau}.npz"))
        has = np.zeros(nd, bool)
        has[p["i"]] = True
        has[p["j"]] = True
        wrong = rem & ~has
        out[name] = dict(n_cand_pairs=int(len(codes)), n_cand_pairs_with_mult=int(tot),
                         n_chain_edges=int(len(ci)), lsh_removed=int(rem.sum()),
                         lsh_wrongly_deleted=int(wrong.sum()),
                         wrong_share_of_deleted=float(wrong.sum() / max(1, rem.sum())),
                         wrong_share_of_input=float(wrong.sum() / nd))
        del K, codes, ci, cj, rem, has, wrong
    del K112, K117

    old = old_lsh_row(corpus, n)
    op = old["per_tau"][str(tau)]
    rr = rerun_row(corpus, n, tau)
    oldv = dict(n_cand_pairs=int(old["n_cand_pairs"]),
                n_cand_pairs_with_mult=int(old.get("n_cand_pairs_with_mult")),
                lsh_removed=int(old["lsh_removed"]),
                lsh_wrongly_deleted=int(op["lsh_wrongly_deleted"]),
                wrong_share_of_deleted=float(op["lsh_wrongly_deleted"] / old["lsh_removed"]),
                wrong_share_of_input=float(op["lsh_wrongly_deleted"] / n))
    rrv = None
    if rr:
        rrv = dict(n_cand_pairs=int(rr["n_cand_pairs"]), lsh_removed=int(rr["lsh_removed"]),
                   lsh_wrongly_deleted=int(rr["lsh_wrongly_deleted"]),
                   wrong_share_of_deleted=float(rr["lsh_wrongly_deleted"] / rr["lsh_removed"]),
                   wrong_share_of_input=float(rr["lsh_wrongly_deleted"] / n))

    def same(a, b, keys):
        return all(a[k] == b[k] for k in keys)

    keys = ["n_cand_pairs", "n_cand_pairs_with_mult", "lsh_removed", "lsh_wrongly_deleted"]
    doc = dict(corpus=corpus, n=n, tau=tau, seed=1,
               provenance="independently rerun (this run) from the shingle cache; "
                          "earlier values read from the round2 lsh log row",
               t_signatures_h112_s=t_s112, t_signatures_h117_s=t_s117,
               signature_prefix_stability=dict(
                   n_columns_compared=112, n_columns_identical=n_pref_cols_equal,
                   n_docs_with_any_difference=n_docs_any_diff,
                   fraction_docs_with_any_difference=n_docs_any_diff / nd,
                   columns_first_10_equal=[round(c, 4) for c in col_equal[:10]]),
               standalone_h112=out["standalone_h112"],
               shared_h117_prefix=out["shared_h117_prefix"],
               earlier_logged_seed1=oldv,
               this_run_rerun_row=rrv,
               checks=dict(
                   shared_h117_reproduces_earlier_log=same(out["shared_h117_prefix"], oldv, keys),
                   standalone_h112_reproduces_this_run_rerun=(rrv is not None and same(
                       out["standalone_h112"], rrv, ["n_cand_pairs", "lsh_removed", "lsh_wrongly_deleted"])),
                   shared_equals_standalone=(same(out["shared_h117_prefix"], out["standalone_h112"], keys))))
    print(json.dumps(doc, indent=1), flush=True)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="all")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        print("@@" + json.dumps(one(a.corpus, a.n)))
        return
    corpora = list(CELLS) if a.corpus == "all" else [a.corpus]
    docs = {}
    for corpus in corpora:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--corpus", corpus,
                            "--n", str(CELLS[corpus]), "--one"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
            raise SystemExit(f"phase {corpus} failed")
        line = [x for x in r.stdout.splitlines() if x.startswith("@@")][-1]
        docs[corpus] = json.loads(line[2:])
        print(corpus, json.dumps(docs[corpus]["checks"]), flush=True)
    doc = dict(version=1, generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               provenance="independently rerun (this run)",
               question="did the earlier 14x8 cell draw its 112 hashes the way datatrove does?",
               answer=("No. The earlier run drew H=117 (the maximum b*r over the banding grid) and "
                       "used the first 112 columns. NumPy's RandomState.randint with dtype=uint64 is "
                       "not prefix-stable in array width, so those 112 columns differ from a "
                       "standalone 112-hash draw; the resulting band keys, candidate pairs and "
                       "components therefore differ even though the shingles and the exact join are "
                       "identical. The shared-H design is internally consistent (all banding "
                       "configurations of the earlier grid share one draw) but its 14x8 cell is not "
                       "datatrove's standalone 14x8 at the same seed."),
               cells=docs)
    with open(os.path.join(OUT, "shared_h117.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote results/shared_h117.json")


if __name__ == "__main__":
    main()
