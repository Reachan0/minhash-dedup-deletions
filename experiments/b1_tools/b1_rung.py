#!/usr/bin/env python3
"""B1/B2 analysis of one corpus (a B1 rung or a B2 variant) with the frozen method definitions.

Input: a shingle cache npz (off, flat, nwords). For a B1 rung it is built here from the smallest
prefix of B1 segments with >= N English documents.
Per seed s in 1..5 and tau in {0.8, 0.75}:
  A (min-id and datatrove representative rules), B, W, C deletion sets and their metrics
  (deleted, unsupported = no neighbour at J >= tau, witness coverage, unsupported_retained =
  no retained neighbour, residual pairs, missed exact deletions), pair recall of the candidate
  stage, and the cost breakdown (shingles, largest bucket, sum of within-bucket pairs, distinct
  candidate pairs, exact checks, verified edges, exact edges, stage times, peak RSS).
Across seeds: documents whose keep/delete status ever changes, and the mean pairwise symmetric
difference of deletion sets, per method.
Exact-copy collapse is checked by comparing shingle arrays inside each digest group.
Usage: env/bin/python b1_rung.py rung <N> | b1_rung.py file <name> <path.npz>
"""
import json
import os
import resource
import sys
import time

import numpy as np

W = os.environ.get("B2_W", "/path/to/scratch/dedupjoin")
sys.path[:0] = [f"{W}/dedupjoin-newsnap-selection/code/newsnap", f"{W}/dedupjoin-newsnap-selection/code/run"]
import snap_common as S  # noqa: E402

TAUS = [0.8, 0.75]
SEEDS = [1, 2, 3, 4, 5]


def peak_gib():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2)


def build_rung(target, out):
    rows = sorted((json.loads(l) for l in open(f"{W}/logs/segments.jsonl")), key=lambda r: r.get("rank", ""))
    sizes, flats, nws, total, ranks = [], [], [], 0, []
    for i, r in enumerate(x for x in rows if x.get("status") == "ok"):
        if r["rank"] != f"{i:04d}":
            raise SystemExit(f"segments not contiguous at {r['rank']}")
        with np.load(f"{W}/cache/{r['rank']}.npz") as z:
            sizes.append(z["sizes"]), flats.append(z["flat"]), nws.append(z["nwords"])
        ranks.append(r["rank"])
        total += r["english"]
        if total >= target:
            break
    else:
        raise SystemExit(f"only {total} documents processed so far")
    sizes = np.concatenate(sizes).astype(np.int64)
    off = np.zeros(len(sizes) + 1, np.int64)
    np.cumsum(sizes, out=off[1:])
    np.savez(out, off=off, flat=np.concatenate(flats), nwords=np.concatenate(nws))
    return dict(segments=len(ranks), first_rank=ranks[0], last_rank=ranks[-1])


def collapse_collisions(z, off, flat):
    """Compare every grouped document's shingle array with its group representative's."""
    group, rep = z["group_of_doc"], z["group_rep"]
    bad = 0
    for d in np.flatnonzero(z["group_size"][group] > 1):
        r = rep[group[d]]
        if d != r and not np.array_equal(flat[off[d]:off[d + 1]], flat[off[r]:off[r + 1]]):
            bad += 1
    return bad


def main():
    mode = sys.argv[1]
    if mode == "rung":
        target = int(sys.argv[2])
        name = f"rung_{target}"
    else:
        name, src = sys.argv[2], sys.argv[3]
    d = f"{W}/runs/{name}"
    os.makedirs(f"{d}/cache/masks", exist_ok=True)
    os.makedirs(f"{W}/results", exist_ok=True)
    info = {}
    npz = f"{d}/snapshot.npz"
    if mode == "rung" and not os.path.exists(npz):
        info = build_rung(target, npz)
    elif mode == "file":
        npz = src
    S.SHINGLE_NPZ, S.CACHE, S.SNAP, S.RUNS = npz, f"{d}/cache", d, f"{d}/quality.jsonl"
    import collapsed_lib as CL
    if os.environ.get("EJ64") == "1":
        # more than 2^31 shingle positions: 64-bit-offset exact join (code/operator/exactjoin64.py)
        sys.path.insert(0, f"{W}/dedupjoin-newsnap-selection/code/operator")
        import exactjoin64
        CL.EJ = exactjoin64
    K = CL.K
    K.set_threads(K.THREADS)
    out = open(f"{W}/results/{name}.jsonl", "a")
    t0 = time.time()
    cinfo = CL.build_collapsed()
    t_collapse = time.time() - t0
    z = CL.load_collapsed()
    off, flat, _ = S.load_snapshot()
    collisions = collapse_collisions(z, off, flat)
    n_shingles = int(off[-1])
    del off, flat
    t0 = time.time()
    CL.build_ranks_u()
    t_ranks = time.time() - t0
    group, m, rep, nsh = z["group_of_doc"], z["group_size"], z["group_rep"], z["group_nshingle"]
    n, G = len(group), len(m)
    valid, valid_g = nsh[group] > 0, nsh > 0
    m64 = m.astype(np.int64)
    within = int((m64 * (m64 - 1) // 2)[valid_g].sum())
    truth, state, t_truth = {}, {}, {}
    for tau in TAUS:
        t1 = time.time()
        ti, tj, _ = CL.truth_u(tau)
        t_truth[tau] = round(time.time() - t1, 1)
        truth[tau] = (ti, tj)
        state[tau] = CL.exact_state(z, ti, tj)
    head = dict(corpus=name, n=n, groups=G, n_shingles=n_shingles, collapse_collisions=collisions,
                t_collapse_s=round(t_collapse, 1), t_ranks_s=round(t_ranks, 1), **info, **cinfo)
    out.write(json.dumps(dict(kind="corpus", **head, t_truth_s=t_truth, peak_gib=peak_gib())) + "\n")
    masks = {}
    for seed in SEEDS:
        t1 = time.time()
        kb_g = CL.group_band_keys(z, seed)
        t_sig = time.time() - t1
        kb = CL.full_band_keys(kb_g, z)
        t1 = time.time()
        ei, ej, bst = K.chain_edges(kb, valid)
        t_chain = time.time() - t1
        del kb
        rootA = K.uf_minid(n, ei, ej)
        rootD = K.uf_datatrove(n, ei, ej)
        remA = (rootA != np.arange(n)) & valid
        remD = (rootD != np.arange(n)) & valid
        n_chain = int(len(ei))
        del ei, ej, rootA, rootD
        t1 = time.time()
        pi, pj, bst_g = K.all_bucket_pairs(kb_g, valid_g, G)
        t_cand = time.time() - t1
        t1 = time.time()
        jac = K.L.jaccard_pairs(z["off_u"], z["flat_u"], pi, pj)
        t_ver = time.time() - t1
        doc_cand = int((m64[pi] * m64[pj]).sum()) + within
        del kb_g
        for tau in TAUS:
            ti, tj = truth[tau]
            st = state[tau]
            true_pairs = within + int((m64[ti] * m64[tj]).sum())
            ver = jac >= tau - 1e-12
            vi, vj = pi[ver], pj[ver]
            found_pairs = within + int((m64[vi] * m64[vj]).sum())
            rootB = K.uf_minid(G, vi, vj)
            comp_min = np.full(G, np.iinfo(np.int64).max, np.int64)
            np.minimum.at(comp_min, rootB, rep)
            remB = valid & (np.arange(n) != comp_min[rootB[group]])
            kept = K.greedy_mis(G, vi, vj)
            remW = valid & ((~kept[group]) | (np.arange(n) != rep[group]))
            cost = dict(largest_bucket=int(bst["largest_bucket"]), bucket_pairs_sum=int(bst["bucket_pairs_sum"]),
                        chain_edges=n_chain, group_candidate_pairs=int(len(pi)), doc_candidate_pairs=doc_cand,
                        verified_group_edges=int(ver.sum()), exact_group_edges=int(len(ti)),
                        exact_doc_pairs=true_pairs, t_signature_s=round(t_sig, 1),
                        t_chain_s=round(t_chain, 1), t_candidates_s=round(t_cand, 1), t_verify_s=round(t_ver, 1))
            for meth, rem in (("A", remA), ("A_dt", remD), ("B", remB), ("W", remW), ("C", st["rem"])):
                met = CL.lift_metrics(rem, z, ti, tj, st["has_nb_doc"], n, valid)
                row = dict(kind="run", corpus=name, method=meth, tau=tau, seed=seed, n=n, **met,
                           unsupported_retained=met["deleted"] - met["witness_covered"],
                           missed=int((st["rem"] & ~rem).sum()),
                           hijacked=int((rem & st["has_nb_doc"] & ~st["rem"]).sum()),
                           pair_recall=found_pairs / true_pairs if true_pairs else None,
                           **cost, peak_gib=peak_gib())
                out.write(json.dumps(row) + "\n")
                masks.setdefault((meth, tau), []).append(rem)
            out.flush()
        del pi, pj, jac
        print(f"{name} seed {seed} done {time.time() - t0:.0f}s", flush=True)
    for (meth, tau), ms in masks.items():
        if meth == "C":
            continue
        stack = np.stack(ms)
        ever = int((stack.any(0) & ~stack.all(0)).sum())
        sd = [int((ms[a] ^ ms[b]).sum()) for a in range(len(ms)) for b in range(a + 1, len(ms))]
        out.write(json.dumps(dict(kind="stability", corpus=name, method=meth, tau=tau, seeds=len(ms),
                                  ever_flip=ever, mean_pairwise_symdiff=float(np.mean(sd)))) + "\n")
    print(f"{name} RUNG_DONE", flush=True)


if __name__ == "__main__":
    main()
