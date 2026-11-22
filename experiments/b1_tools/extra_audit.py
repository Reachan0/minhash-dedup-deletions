"""Two audits on Snapshot-2025 (the 0.53M rung on AutoDL), per hash seed and threshold.

1. Representative rule: A's components fix how many documents are deleted; the rule only picks the
   survivor. Over all rules, the unsupported count lies between choosing an unsupported survivor
   wherever a component has one and choosing a supported survivor wherever a component has one.
2. First-seen streaming deletion (LSHBloom-style, datasketch query-then-insert): documents in corpus
   order; a document is deleted if any band key was seen before. Variant `kept`: only retained
   documents insert their keys. Variant `all`: every document inserts its keys. No clustering,
   no verification.
Writes extra_audit.jsonl next to this script."""
import json
import os
import sys
import time

import numpy as np

W = "/path/to/scratch/dedupjoin"
RUNG = f"{W}/runs/rung_531597"
sys.path[:0] = [f"{W}/dedupjoin-newsnap-selection/code/newsnap", f"{W}/dedupjoin-newsnap-selection/code/run"]
import snap_common as S  # noqa: E402

S.SHINGLE_NPZ, S.CACHE, S.SNAP, S.RUNS = f"{RUNG}/snapshot.npz", f"{RUNG}/cache", RUNG, f"{RUNG}/quality.jsonl"
import collapsed_lib as CL  # noqa: E402

K = CL.K
K.set_threads(K.THREADS)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extra_audit.jsonl")


def first_seen(kb, valid, insert_kept_only):
    nb, n = kb.shape
    seen = [set() for _ in range(nb)]
    dele = np.zeros(n, bool)
    cols = [kb[b].tolist() for b in range(nb)]
    for d in range(n):
        if not valid[d]:
            continue
        hit = False
        for b in range(nb):
            if cols[b][d] in seen[b]:
                hit = True
                break
        dele[d] = hit
        if hit and insert_kept_only:
            continue
        for b in range(nb):
            seen[b].add(cols[b][d])
    return dele


def first_seen_all(kb, valid):
    # deleted iff the document is not the first valid document of its bucket in some band
    nb, n = kb.shape
    idx = np.flatnonzero(valid)
    dele = np.zeros(n, bool)
    for b in range(nb):
        k = kb[b, idx]
        o = np.lexsort((idx, k))
        ks = k[o]
        first = np.ones(len(o), bool)
        first[1:] = ks[1:] != ks[:-1]
        dele[idx[o[~first]]] = True
    return dele


def main():
    z = CL.load_collapsed()
    off, _, _ = S.load_snapshot()
    n = len(off) - 1
    valid = np.diff(off) > 0
    group = z["group_of_doc"]
    outside = group != int(np.argmax(z["group_size"]))
    has_nb = {}
    for tau in (0.8, 0.75):
        ti, tj, _ = CL.truth_u(tau)
        has_nb[tau] = CL.exact_state(z, ti, tj)["has_nb_doc"]
    out = open(OUT, "a")
    for seed in (1, 2, 3, 4, 5):
        t0 = time.time()
        kb = CL.full_band_keys(CL.group_band_keys(z, seed), z)
        ei, ej, _ = K.chain_edges(kb, valid)
        root = K.uf_minid(n, ei, ej)
        remA = (root != np.arange(n)) & valid
        del ei, ej
        dk = first_seen(kb, valid, True)
        da = first_seen_all(kb, valid)
        dk_check = None
        for tau in (0.8, 0.75):
            hn = has_nb[tau]
            for scope, m in (("all", valid), ("outside", valid & outside)):
                u = np.bincount(root[m & ~hn], minlength=n)
                s = np.bincount(root[m & hn], minlength=n)
                multi = (u + s) >= 2
                lo = int(np.where(multi & (u > 0), u - 1, 0).sum())
                hi = int(np.where(multi, np.where(s > 0, u, np.maximum(u - 1, 0)), 0).sum())
                rec = dict(kind="rep_range", seed=seed, tau=tau, scope=scope,
                           deleted=int((remA & m).sum()), unsup_minid=int((remA & m & ~hn).sum()),
                           unsup_lo=lo, unsup_hi=hi, components_mixed=int((multi & (u > 0) & (s > 0)).sum()))
                out.write(json.dumps(rec) + "\n")
                for name, dm in (("first_seen_kept", dk), ("first_seen_all", da), ("A_minid", remA)):
                    rec = dict(kind="first_seen", variant=name, seed=seed, tau=tau, scope=scope,
                               deleted=int((dm & m).sum()), unsupported=int((dm & m & ~hn).sum()))
                    out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"seed {seed} done in {time.time() - t0:.0f}s", flush=True)
    print("EXTRA_DONE", flush=True)


if __name__ == "__main__":
    main()
