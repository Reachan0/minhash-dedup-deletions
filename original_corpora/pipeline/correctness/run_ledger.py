#!/usr/bin/env python3
"""Per-run correctness ledger for A / B / W / C (one row per corpus,tau,seed).

Everything is recomputed from the frozen caches with the memory-bounded path in
`correctness_lib.py`; the rows are appended and flushed one by one, so a kill
loses at most the cell in flight.  Re-running skips rows already marked ok.

What is checked, in three layers:

  candidate layer : candidate pairs, candidate pair recall, precision
  edge layer      : exact truth edges, truth edges missed by candidate generation
  deletion layer  : deletion ID sets (not just counts) for A, B, W, C;
                    raw_missed / verified_missed / witness_missed / unsupported /
                    hijacked / residual true-duplicate pairs / witness coverage

plus the component assertion: inside every exact connected component keep only
B's verified edges; if the component splits into k blocks, B keeps k-1 more
documents than C, and sum(k-1) == |K_exact \\ K_verified| under minimum-id
representatives.  A failed assertion marks the row `assert_fail`; it is never
skipped silently.

Usage:
  python3 run_ledger.py --corpus c4
  python3 run_ledger.py --corpus fw --taus 0.8
"""
import argparse
import gc
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import correctness_lib as K

TAUS_DEFAULT = [0.8, 0.75]
MAIN_TRUTH = os.path.join(K.WRK, "{corpus}_truth_t{tau}.npz")
CORE_TRUTH = os.path.join(K.CACHE, "truth_{corpus}_t{tau}.npz")


def load_runs():
    out = {}
    if os.path.exists(K.RUNS):
        with open(K.RUNS, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    out[r["key"]] = r
                except Exception:
                    pass
    return out


def load_truth(corpus, tau, n):
    """Exact truth graph.  Returns (ti, tj, root_exact, removed_exact)."""
    main = MAIN_TRUTH.format(corpus=corpus, tau=tau)
    if os.path.exists(main):
        z = np.load(main)
        ti = np.asarray(z["i"], np.int64)
        tj = np.asarray(z["j"], np.int64)
        root = np.asarray(z["root"], np.int64)
        rem = np.asarray(z["removed"], bool)
        z.close()
        # confirm the stored root is the minimum-id labelling
        r2, rem2 = K.exact_components(n, ti, tj)
        if not np.array_equal(root, r2):
            raise SystemExit(f"stored truth root differs from min-id UF: {corpus} tau={tau}")
        return ti, tj, root, rem
    core = CORE_TRUTH.format(corpus=corpus, tau=tau)
    if not os.path.exists(core):
        raise SystemExit(f"no truth graph for {corpus} tau={tau}")
    z = np.load(core)
    ti = np.asarray(z["i"], np.int64)
    tj = np.asarray(z["j"], np.int64)
    z.close()
    root, rem = K.exact_components(n, ti, tj)
    return ti, tj, root, rem


def w_block(n, ti, tj, ei, ej, valid, orders, has_nb, rem_exact):
    """Run W (greedy MIS) on the truth graph and on B's verified graph under the
    same order set; compares retained IDs, residual pairs and witness coverage.

    Separates the policy cost (W on the full truth graph vs C) from the
    missed-candidate effect (W on B's verified graph vs W on the truth graph).
    """
    csr_true = K.build_csr(n, ti, tj)
    csr_ver = K.build_csr(n, ei, ej)
    out, masks = {}, {}
    for name, order in orders.items():
        rec = {}
        for gname, csr in (("truth_graph", csr_true), ("verified_graph", csr_ver)):
            kept = K.greedy_mis_from_csr(n, csr[0], csr[1], order)
            rem = (~kept) & valid
            masks[(name, gname)] = rem
            rec[gname] = K.method_metrics(n, rem, ti, tj, has_nb)
        # missed-candidate effect, exact deleted-ID set comparison
        mt, mv = masks[(name, "truth_graph")], masks[(name, "verified_graph")]
        rec["truth_vs_verified"] = dict(
            deleted_ids_symdiff=int((mt ^ mv).sum()),
            truth_deleted_verified_kept=int((mt & ~mv).sum()),
            verified_deleted_truth_kept=int((mv & ~mt).sum()),
        )
        # policy cost against C, in both graphs
        for gname in ("truth_graph", "verified_graph"):
            m = masks[(name, gname)]
            rec[gname]["vs_C"] = dict(
                missed=int((rem_exact & ~m).sum()),
                extra=int((m & ~rem_exact).sum()),
                symdiff=int((m ^ rem_exact).sum()),
            )
        out[name] = rec
    # deleted-ID symmetric difference against the min-id order (exact set compare)
    for name in orders:
        for gname in ("truth_graph", "verified_graph"):
            out[name][gname]["deleted_ids_symdiff_vs_minid"] = int(
                (masks[(name, gname)] ^ masks[("minid", gname)]).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=K.CORPORA)
    ap.add_argument("--taus", default="0.8,0.75")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--out", default=os.path.join(K.OUTDIR, "ledger.jsonl"))
    args = ap.parse_args()
    corpus = args.corpus
    taus = [float(t) for t in args.taus.split(",") if t]
    seeds = ([int(s) for s in args.seeds.split(",") if s]
             if args.seeds else K.SEEDS[corpus])
    n = K.N_DOCS[corpus]

    off, nwords = K.load_offsets(corpus)
    assert len(off) - 1 == n
    valid = np.diff(off) > 0
    truths = {}
    for tau in taus:
        ti, tj, root, rem = load_truth(corpus, tau, n)
        truths[tau] = dict(ti=ti, tj=tj, root=root, rem=rem,
                           hnb=K.has_true_neighbour(n, ti, tj),
                           n_truth=len(ti))
        print(f"[{corpus}] truth tau={tau}: {len(ti)} edges, {int(rem.sum())} exact deletions",
              flush=True)

    flat = K.extract_flat(corpus)
    print(f"[{corpus}] flat shingles mmap: {flat.shape[0]} entries", flush=True)
    runs = load_runs()
    done = K.done_keys(args.out)
    orders = {"minid": np.arange(n, dtype=np.int64)}
    orders.update(K.order_permutations(n, 3))

    for seed in seeds:
        sig = K.load_signatures(corpus, seed)
        Kb = K.band_keys(sig, K.B, K.R)
        del sig
        pi, pj, bstats = K.all_bucket_pairs(Kb, valid, n)
        jac = K.jaccard_pairs(off, flat, pi, pj)
        eiA, ejA, astats = K.chain_edges(Kb, valid)
        print(f"[{corpus}] s{seed}: {len(pi)} candidate pairs, {len(eiA)} chain edges",
              flush=True)
        rootA = K.uf_minid(n, eiA, ejA)
        KA = K.removed_minid(rootA, valid)
        del eiA, ejA, rootA; gc.collect()

        for tau in taus:
            key = f"{corpus}|t{tau}|B|minid|s{seed}"
            if key in done:
                print(f"[{corpus}] s{seed} tau={tau}: already done, skip", flush=True)
                continue
            T = truths[tau]
            ti, tj, hnb = T["ti"], T["tj"], T["hnb"]
            ver = jac >= tau - 1e-12
            ei, ej = pi[ver], pj[ver]
            rootB = K.uf_minid(n, ei, ej)
            KB = K.removed_minid(rootB, valid)

            # candidate / edge layers
            det = K.detected_truth_mask(ti, tj, pi, pj, n)
            detected = int(det.sum())
            verified_not_truth = int(len(ei) - detected)
            cand_stats = K.edge_stats(ti, tj, pi, pj, n)
            ver_stats = K.edge_stats(ti, tj, ei, ej, n)

            # deletion layer, ID sets
            mA = K.method_metrics(n, KA, ti, tj, hnb)
            mB = K.method_metrics(n, KB, ti, tj, hnb)
            KC = T["rem"]
            mC = K.method_metrics(n, KC, ti, tj, hnb)
            raw_missed = int((KC & ~KA).sum())
            verified_missed = int((KC & ~KB).sum())

            # W (min-id + three permutations) on truth and verified graphs
            wres = w_block(n, ti, tj, ei, ej, valid, orders, hnb, KC)
            kept_w = K.greedy_mis_from_csr(n, *K.build_csr(n, ei, ej), orders["minid"])
            KW = (~kept_w) & valid
            mW = K.method_metrics(n, KW, ti, tj, hnb)
            # hijacked: deleted although it has a true neighbour and C keeps it
            for mm, rem in ((mA, KA), (mB, KB), (mW, KW), (mC, KC)):
                hj = int((rem & hnb & ~KC).sum())
                mm["hijacked"] = hj
                mm["hijacked_share_del"] = float(hj / rem.sum()) if rem.sum() else None
                mm["hijacked_share_input"] = float(hj / n)
            witness_missed = int((KC & ~KW).sum())

            # component split / assertion
            split = K.component_split(T["root"], rootB, n)
            cstats = K.component_edge_stats(split, T["root"], ti, det)
            comp_classes = {k: cstats[k] for k in (
                "connected_with_many_missed", "missed_edges_in_those",
                "split_with_few_missed", "split_total")}
            assert_ok = (split["sum_k_minus_1"] == verified_missed)

            # cross-check against the frozen accuracy campaign
            cross = {}
            rB = runs.get(key)
            if rB is not None:
                cross["deleted"] = int(rB["deleted"]) == int(mB["deleted"])
                cross["n_verified_edges"] = int(rB["n_verified_edges"]) == int(len(ei))
                cross["pair_recall"] = abs(float(rB["pair_recall"]) - float(ver_stats["pair_recall"])) < 1e-12
                cross["unsupported"] = int(rB["unsupported"]) == int(mB["unsupported"])
                cross["residual"] = int(rB["residual_duplicate_pairs"]) == int(mB["residual_duplicate_pairs"])
                cross["coverage"] = abs(float(rB["witness_coverage"]) - float(mB["witness_coverage"])) < 1e-12
            rA = runs.get(f"{corpus}|t{tau}|A|minid|s{seed}")
            if rA is not None:
                cross["A_deleted"] = int(rA["deleted"]) == int(mA["deleted"])
            rC = runs.get(f"{corpus}|t{tau}|C|minid|s0")
            if rC is not None:
                cross["C_deleted"] = int(rC["deleted"]) == int(mC["deleted"])

            # τ=0.8 per-seed masks from the seed study, when available
            sets_path = os.path.join(K.WRK, f"{corpus}_s{seed}_t{tau}_sets.npz")
            if os.path.exists(sets_path):
                z = np.load(sets_path)
                ex = np.unpackbits(np.asarray(z["rem_exact"]))[:n].astype(bool)
                vr = np.unpackbits(np.asarray(z["rem_ver"]))[:n].astype(bool)
                rw = np.unpackbits(np.asarray(z["rem_raw"]))[:n].astype(bool)
                z.close()
                cross["sets_exact"] = bool(np.array_equal(ex, KC))
                cross["sets_verified"] = bool(np.array_equal(vr, KB))
                cross["sets_raw"] = bool(np.array_equal(rw, KA))

            failures = []
            if not assert_ok:
                failures.append(f"sum(k-1)={split['sum_k_minus_1']} != verified_missed={verified_missed}")
            for kk, vv in cross.items():
                if not vv:
                    failures.append(f"crosscheck {kk} mismatch")

            row = dict(
                key=key, corpus=corpus, tau=tau, seed=seed, method="B",
                representative="minid", n=int(n), valid_docs=int(valid.sum()),
                status="ok" if not failures else "assert_fail",
                failures=failures,
                candidate_layer=dict(
                    n_candidate_pairs=int(len(pi)),
                    n_verified_edges=int(len(ei)),
                    candidate_pair_recall=cand_stats["pair_recall"],
                    candidate_precision=cand_stats["candidate_precision"],
                    verified_pair_recall=ver_stats["pair_recall"],
                    verified_precision=ver_stats["candidate_precision"],
                    verified_edges_not_in_truth=verified_not_truth,
                    largest_bucket=int(bstats["largest_bucket"]),
                    bucket_pairs_sum=int(bstats["bucket_pairs_sum"]),
                ),
                edge_layer=dict(
                    truth_edges=int(T["n_truth"]),
                    truth_edges_detected_by_candidates=detected,
                    missed_truth_edges=int(T["n_truth"] - detected),
                    miss_share_of_truth=float((T["n_truth"] - detected) / T["n_truth"]),
                ),
                deletion_layer=dict(
                    A=mA, B=mB, W=mW, C=mC,
                    raw_missed=raw_missed, verified_missed=verified_missed,
                    witness_missed=witness_missed,
                    symdiff_A_C=int((KA ^ KC).sum()),
                    symdiff_B_C=int((KB ^ KC).sum()),
                    symdiff_W_C=int((KW ^ KC).sum()),
                    B_subset_C=bool(not (KB & ~KC).any()),
                    W_subset_C=bool(not (KW & ~KC).any()),
                ),
                components=dict(
                    n_components=split["n_components"],
                    blocks_B=split["blocks"],
                    sum_k_minus_1=split["sum_k_minus_1"],
                    verified_missed=verified_missed,
                    assertion_ok=bool(assert_ok),
                    **comp_classes,
                ),
                witness_orders=wres,
                crosscheck=cross,
                peak_rss_mb=round(K.peak_rss_mb(), 1),
                rss_mb=round(K.cur_rss_mb() or 0.0, 1),
            )
            K.append_row(args.out, row)
            print(f"[{corpus}] s{seed} tau={tau}: status={row['status']} "
                  f"B={mB['deleted']} C={mC['deleted']} verified_missed={verified_missed} "
                  f"sum(k-1)={split['sum_k_minus_1']} rss={row['rss_mb']}MB "
                  f"peak={row['peak_rss_mb']}MB", flush=True)
            del ver, ei, ej, rootB, KB, split, wres, kept_w, KW
            gc.collect()
        del Kb, pi, pj, jac, KA
        gc.collect()

    del flat
    gc.collect()
    K.drop_flat(corpus)
    print(f"[{corpus}] done; peak RSS {K.peak_rss_mb():.1f} MB", flush=True)


if __name__ == "__main__":
    main()
