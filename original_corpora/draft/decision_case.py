#!/usr/bin/env python3
"""Decision case: C4 n=700,000, tau=0.8, datatrove hash family, seed 1, 14x8.

Compares the three deleted sets produced by the independent re-execution
  A. raw LSH            (banding only)
  B. LSH + candidate verification (exact Jaccard on co-bucketed pairs >= tau)
  C. full exact join
and reports, for the keep/drop difference, the *symmetric difference* of the two
deleted-document sets (not the net count), the documents involved, their tokens /
characters / shingles / unique shingles, and the measured wall time and memory of
the two procedures (from results/decision_cost.json if present, else phase rows).

It also reconstructs the same decision case under the earlier run's shared 117-hash
design from that run's saved dumps (raw LSH deletions, candidate codes, candidate
Jaccards), so the audited 20-document gap and the standalone 24-document gap are
reported side by side from the same data.

Writes results/decision_case.json.
"""
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CODE = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/src/C03")
sys.path.insert(0, CODE)
import lsh as L  # noqa: E402

OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/rerun")
OLD_DUMP = os.path.join(ROOT, ".work/data/C03/r2/cache/lsh_out")
OLD_LOGS = os.path.join(ROOT, ".work/vldb2027-ai-db-ideas/experiment/logs/C03/round2")
OLD_CACHE = os.path.join(ROOT, ".work/data/C03/r2/cache")


def jload(p):
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def set_stats(mask, meta, off=None, flat=None):
    ids = np.nonzero(mask)[0]
    st = dict(n_docs=int(len(ids)),
              nwords=int(meta["nwords"][ids].sum()),
              nchar=int(meta["nchar"][ids].sum()),
              nshingle=int(meta["nshingle"][ids].sum()),
              median_nwords=float(np.median(meta["nwords"][ids])) if len(ids) else None,
              median_nshingle=float(np.median(meta["nshingle"][ids])) if len(ids) else None)
    if flat is not None and off is not None and len(ids):
        parts = [flat[off[i]:off[i + 1]] for i in ids]
        st["unique_shingles"] = int(len(np.unique(np.concatenate(parts))))
    return st, ids


def old_row(kind, **kw):
    for line in open(os.path.join(OLD_LOGS, "rows.jsonl")):
        r = json.loads(line)
        if r.get("kind") != kind:
            continue
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def audited_shared_design(n, tau, meta_t, meta, off, flat, rem_exact, root_exact,
                          standalone_gap_ids):
    """Rebuild the earlier run's shared-117 decision case from its saved dumps."""
    z8 = np.load(os.path.join(OLD_DUMP, f"c4_{n}_{tau}_dt1_14x8.npz"))
    zc = np.load(os.path.join(OLD_DUMP, f"c4_{n}_cand_dt1_14x8.npz"))
    rem_lsh = np.zeros(n, bool)
    rem_lsh[z8["rem_lsh"].astype(np.int64)] = True
    codes = zc["codes"].astype(np.int64)
    ver = zc["jac"] >= tau - 1e-12
    root = L.uf_components(n, codes[ver] // n, codes[ver] % n)
    rem_v = L.removed_set(root)
    sym = rem_exact ^ rem_v
    st, ids = set_stats(sym, meta, off, flat)
    wrong = np.zeros(n, bool)
    wrong[z8["wrong"].astype(np.int64)] = True
    st_wrong, _ = set_stats(wrong, meta, off, flat)
    detail = []
    for i in ids:
        nb = np.nonzero((root_exact == root_exact[i]) & ~rem_exact)[0]
        detail.append(dict(doc_id=int(i), doc_key=str(meta_t["id"][i]),
                           url=str(meta_t["url"][i])[:200], dump=str(meta_t["dump"][i]),
                           nwords=int(meta["nwords"][i]), nchar=int(meta["nchar"][i]),
                           nshingle=int(meta["nshingle"][i]),
                           kept_representative=int(nb[0]) if len(nb) else None,
                           head=str(meta_t["head"][i])[:160]))
    lsh_row = old_row("lsh", family="dt", b=14, r=8, corpus="c4", n=n, seed=1)
    ex_row = old_row("exact", corpus="c4", n=n, tau=tau, variant="ppjoin")
    timings = None
    if lsh_row and ex_row:
        lp = dict(t_signature_all_hashes_s=lsh_row["t_signature_all_hashes"],
                  t_band_cluster_s=lsh_row["t_band_cluster"],
                  t_verify_all_candidates_s=lsh_row["t_verify_all_candidates"],
                  peak_rss_mb=lsh_row["peak_rss_mb_self"],
                  H=lsh_row["H"],
                  t_ranks_s=ex_row["t_rank"], t_index_s=ex_row["t_index"],
                  t_probe_verify_s=ex_row["t_probe_verify"],
                  peak_rss_mb_exact=ex_row["peak_rss_mb_self"],
                  provenance="log-recomputed (earlier run)")
        lp["lsh_pipeline_s"] = (lp["t_signature_all_hashes_s"] + lp["t_band_cluster_s"]
                                + lp["t_verify_all_candidates_s"])
        lp["exact_pipeline_s"] = lp["t_ranks_s"] + lp["t_index_s"] + lp["t_probe_verify_s"]
        lp["ratio_exact_over_lsh"] = lp["exact_pipeline_s"] / lp["lsh_pipeline_s"]
        timings = lp
    gap_ids = set(int(x) for x in ids)
    return dict(
        design="earlier run: one 117-hash draw, first 112 columns banded 14x8 (shared across the banding grid)",
        source_dumps="earlier run's lsh_out dumps (raw-LSH deletions, co-bucketed codes, candidate Jaccards)",
        deleted_sets=dict(raw_lsh=int(rem_lsh.sum()), lsh_verify=int(rem_v.sum()),
                          exact=int(rem_exact.sum())),
        subset_identity=dict(lshv_subset_of_exact=bool((rem_v & ~rem_exact).sum() == 0),
                             lshv_not_exact=int((rem_v & ~rem_exact).sum())),
        gap=dict(pair="lsh_verify vs exact", symmetric_difference=int(sym.sum()),
                 only_in_exact=int((rem_exact & ~rem_v).sum()),
                 only_in_lshv=int((rem_v & ~rem_exact).sum()),
                 net=int(rem_exact.sum()) - int(rem_v.sum()), docs=st,
                 doc_ids=ids.tolist()),
        gap_docs=dict(docs=st, detail=detail),
        raw_lsh_wrongly_deleted_docs=st_wrong,
        overlap_with_standalone_gap=dict(
            standalone_n=len(standalone_gap_ids),
            shared=int(len(gap_ids & set(int(x) for x in standalone_gap_ids))),
            only_shared_design=sorted(gap_ids - set(int(x) for x in standalone_gap_ids)),
            only_standalone=sorted(set(int(x) for x in standalone_gap_ids) - gap_ids)),
        timings=timings)


def main():
    corpus, n, tau, seed = "c4", 700000, 0.8, 1
    tag = f"{corpus}_{n}"
    z = np.load(os.path.join(WRK, f"sets_{tag}_s1_14x8_t{tau}.npz"))
    rem_lsh = z["rem_lsh"].astype(bool)
    rem_lshv = z["rem_lshv"].astype(bool)
    rem_exact = z["rem_exact"].astype(bool)
    root_exact = z["root_exact"]
    meta_t = pq.read_table(os.path.join(WRK, f"{tag}_meta.parquet"),
                           columns=["id", "dump", "url", "nchar", "nwords", "nshingle", "head"]).to_pydict()
    meta = {k: np.asarray(meta_t[k]) for k in ("nchar", "nwords", "nshingle")}

    # subset identity: LSH+verify deletions are a subset of exact deletions
    lshv_not_exact = int((rem_lshv & ~rem_exact).sum())
    assert lshv_not_exact == 0, lshv_not_exact

    # the three pairwise differences
    def pair(a, b, name_a, name_b):
        sym = a ^ b
        d = dict(pair=f"{name_a} vs {name_b}",
                 n_a=int(a.sum()), n_b=int(b.sum()), net=abs(int(a.sum()) - int(b.sum())),
                 symmetric_difference=int(sym.sum()),
                 only_in_a=int((a & ~b).sum()), only_in_b=int((b & ~a).sum()))
        st, ids = set_stats(sym, meta)
        d.update(docs=st)
        d["doc_ids"] = ids.tolist()
        return d, ids

    gaps = {}
    gaps["lsh_vs_lshv"], _ = pair(rem_lsh, rem_lshv, "raw_lsh", "lsh_verify")
    gaps["lshv_vs_exact"], ids_gap = pair(rem_lshv, rem_exact, "lsh_verify", "exact")
    gaps["lsh_vs_exact"], _ = pair(rem_lsh, rem_exact, "raw_lsh", "exact")

    # the documents that differ between LSH+verify and the exact join
    zc = np.load(os.path.join(WRK, f"{tag}.npz"), mmap_mode="r")
    off = np.asarray(zc["off"])
    flat = zc["flat"]
    st_gap, _ = set_stats(rem_exact & ~rem_lshv, meta, off, flat)
    has_nb = z["has_nb"].astype(bool)
    standalone_comp = dict(
        unsupported=int((rem_lsh & ~has_nb).sum()),
        representative_only=int((rem_lsh & has_nb & ~rem_exact).sum()),
        deleted_by_both=int((rem_lsh & rem_exact).sum()),
        total=int(rem_lsh.sum()))
    gap_docs = []
    for i in ids_gap:
        nb = np.nonzero((root_exact == root_exact[i]) & ~rem_exact)[0]  # the kept representative
        gap_docs.append(dict(doc_id=int(i), doc_key=str(meta_t["id"][i]), url=str(meta_t["url"][i])[:200],
                             dump=str(meta_t["dump"][i]),
                             nwords=int(meta["nwords"][i]), nchar=int(meta["nchar"][i]),
                             nshingle=int(meta["nshingle"][i]),
                             kept_representative=int(nb[0]) if len(nb) else None,
                             head=str(meta_t["head"][i])[:160]))

    # timings: from the independent re-execution phase rows
    rows = jload(os.path.join(OUT, "rerun.jsonl"))
    phase = {}
    for r in rows:
        if r.get("corpus") == corpus and r.get("n") == n:
            if r.get("kind") == "signatures" and r.get("seed") == seed:
                phase["t_signature_s"] = r["t_sig_s"]
            elif r.get("kind") == "banding":
                phase["t_banding_s"] = r["t_band_s"]
            elif r.get("kind") == "candidates":
                phase["t_verify_candidates_s"] = r["t_verify_all_s"]
            elif r.get("kind") == "ranks":
                phase["t_ranks_s"] = r["t_rank_s"]
            elif r.get("kind") == "exact" and r.get("tau") == tau:
                phase["t_exact_join_s"] = r["t_join_s"]
                phase["exact_index_entries"] = r.get("index_entries")
                phase["exact_candidates"] = r.get("n_candidates")
            elif r.get("kind") == "deleted_sets" and r.get("tau") == tau:
                phase["n_cand_pairs"] = r.get("n_cand_pairs")
                phase["peak_rss_mb"] = r.get("peak_rss_mb")
                phase["verified_pairs"] = r.get("verified_pairs")
    cost = dict(phases=phase)
    lsh_total = sum(phase.get(k, 0.0) for k in ("t_signature_s", "t_banding_s", "t_verify_candidates_s"))
    exact_total = phase.get("t_ranks_s", 0.0) + phase.get("t_exact_join_s", 0.0)
    cost.update(lsh_pipeline_s=lsh_total, exact_pipeline_s=exact_total,
                ratio_exact_over_lsh=(exact_total / lsh_total) if lsh_total else None,
                note="both pipelines start from the same shingles; exact needs the global rank order "
                     "(build_ranks) plus the join, LSH needs signatures + banding + candidate verification")
    cp = os.path.join(OUT, "decision_cost.json")
    if os.path.exists(cp):
        cost["dedicated_measurement"] = json.load(open(cp))

    audited = audited_shared_design(n, tau, meta_t, meta, off, flat, rem_exact, root_exact,
                                    ids_gap)

    doc = dict(version=1, generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               provenance="independently rerun (this run)",
               cell=dict(corpus=corpus, n=n, tau=tau, seed=seed, config="dt 14x8 (standalone 112-hash draw)"),
               deleted_sets=dict(raw_lsh=int(rem_lsh.sum()), lsh_verify=int(rem_lshv.sum()),
                                 exact=int(rem_exact.sum())),
               subset_identity=dict(lshv_subset_of_exact=lshv_not_exact == 0,
                                    lshv_not_exact=lshv_not_exact),
               gaps=gaps,
               gap_docs=dict(docs=st_gap, detail=gap_docs),
               standalone_composition=standalone_comp,
               audited_shared_design=audited,
               cost=cost,
               correction_note=("Draft 0 reported a 20-document gap (3890 vs 3910) because its 14x8 "
                                "configuration read the first 112 of a shared 117-hash draw; under the "
                                "standalone datatrove 14x8 draw the gap is 24 documents (3886 vs 3910). "
                                "Both numbers describe the same subset relation, and the exact join is "
                                "unchanged at 3910."))
    with open(os.path.join(OUT, "decision_case.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print(json.dumps({k: {kk: v[kk] for kk in ("n_a", "n_b", "net", "symmetric_difference", "only_in_a", "only_in_b")}
                      for k, v in gaps.items()}, indent=1))
    print(json.dumps(st_gap, indent=1))
    print(json.dumps(cost, indent=1))
    print("gap doc ids:", ids_gap.tolist())
    print("audited design:", json.dumps({k: audited[k] for k in
                                         ("deleted_sets", "subset_identity", "timings")}, indent=1))
    print("audited gap:", json.dumps({k: audited["gap"][k] for k in
                                      ("symmetric_difference", "only_in_exact", "only_in_lshv", "docs")}, indent=1))
    print("overlap:", json.dumps({k: v for k, v in audited["overlap_with_standalone_gap"].items()
                                  if k != "only_standalone"})[:800])


if __name__ == "__main__":
    main()
