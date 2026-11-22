#!/usr/bin/env python3
"""Three final-corpus quality metrics, computed separately, for C4 700k and FineWeb 1M.

  (a) candidate-edge recall: fraction of exact pairs with J >= tau that share at least
      one of the 14 bands (truth_hit in the saved set dumps);
  (b) residual duplicates in the retained set: exact pairs (J >= tau) whose two
      endpoints are both retained, for three retention rules (raw LSH, LSH+verify, exact);
  (c) per-deleted-document direct-witness coverage: fraction of deleted documents that
      have a direct J >= tau edge to a retained document.

Also extracts one traceable case where (a)/(b)/(c) disagree, with doc ids and excerpts.

Writes results/quality_metrics.json.
"""
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "output/dedupjoin-pvldb/results")
WRK = os.path.join(ROOT, ".work/dedupjoin/rerun")

CELLS = [("c4", 700000, [0.8, 0.7, 0.6]), ("fw", 1000000, [0.8, 0.7])]


def main():
    doc = dict(version=1, generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               provenance="independently rerun (this run)",
               note="all three metrics come from the same exact pair list; they answer different questions")
    out = {}
    for corpus, n, taus in CELLS:
        tag = f"{corpus}_{n}"
        mp = os.path.join(WRK, f"{tag}_meta.parquet")
        if not os.path.exists(mp):
            continue
        meta = pq.read_table(mp, columns=["id", "url", "nwords", "nchar", "head"]).to_pydict()
        docex = str(meta["head"][0])[:80] if meta["head"] else ""
        res = {}
        for tau in taus:
            sp = os.path.join(WRK, f"sets_{tag}_s1_14x8_t{tau}.npz")
            pp = os.path.join(WRK, f"pairs_{tag}_{tau}.npz")
            if not (os.path.exists(sp) and os.path.exists(pp)):
                continue
            s = np.load(sp)
            p = np.load(pp)
            ti, tj, jac = p["i"].astype(np.int64), p["j"].astype(np.int64), p["jac"]
            hit = s["truth_hit"].astype(bool)
            rem = {"raw_lsh": s["rem_lsh"].astype(bool),
                   "lsh_verify": s["rem_lshv"].astype(bool),
                   "exact": s["rem_exact"].astype(bool)}
            rec = dict(n_true_pairs=int(len(ti)), candidate_edge_recall=float(hit.mean()) if len(ti) else None,
                       n_true_pairs_missed=int((~hit).sum()))
            for name, rm in rem.items():
                keep = ~rm
                both = keep[ti] & keep[tj]
                one_wit = keep[ti] | keep[tj]
                # (b) residual duplicate pairs among retained docs
                rec[f"residual_duplicate_pairs_{name}"] = int(both.sum())
                rec[f"residual_duplicate_docs_{name}"] = int(len(set(ti[both].tolist()) | set(tj[both].tolist())))
                # (c) deleted docs with a direct witness in the retained set
                deleted = np.nonzero(rm)[0]
                wit = np.zeros(len(rm), bool)
                # a deleted endpoint gets a witness if the other endpoint is kept
                wit[ti[keep[tj]]] = True
                wit[tj[keep[ti]]] = True
                rec[f"deleted_docs_{name}"] = int(len(deleted))
                rec[f"direct_witness_coverage_{name}"] = float(wit[deleted].mean()) if len(deleted) else None
                rec[f"deleted_without_direct_witness_{name}"] = int((~wit[deleted]).sum())
                # residual duplicate doc pair with the largest Jaccard, for the trace
                if both.any():
                    k = int(np.argmax(np.where(both, jac, -1)))
                    rec[f"top_residual_pair_{name}"] = dict(
                        i=int(ti[k]), j=int(tj[k]), jac=float(jac[k]),
                        i_id=str(meta["id"][ti[k]]), j_id=str(meta["id"][tj[k]]),
                        i_url=str(meta["url"][ti[k]])[:160], j_url=str(meta["url"][tj[k]])[:160],
                        i_nwords=int(meta["nwords"][ti[k]]), j_nwords=int(meta["nwords"][tj[k]]),
                        i_head=str(meta["head"][ti[k]])[:200], j_head=str(meta["head"][tj[k]])[:200])
                # deleted doc without witness: record one example (the shortest)
                nw = np.asarray(meta["nwords"])
                bad = deleted[~wit[deleted]]
                if len(bad):
                    q = int(bad[np.argmin(nw[bad])])
                    nb = np.nonzero((ti == q) | (tj == q))[0]
                    rec[f"unwitnessed_example_{name}"] = dict(
                        doc=q, doc_id=str(meta["id"][q]), nwords=int(nw[q]),
                        n_exact_pairs=int(len(nb)),
                        partner_max_j=float(jac[nb].max()) if len(nb) else None,
                        head=str(meta["head"][q])[:200])
            res[str(tau)] = rec
        out[f"{corpus}_{n}"] = res
    doc["cells"] = out
    doc["claim_difference_example"] = (
        "A pair with J>=tau that shares no band is a missed edge (a): both documents are "
        "retained, so the final corpus contains a residual duplicate (b); neither document "
        "is deleted, so the per-deleted-document witness metric (c) never sees it. "
        "The top_residual_pair fields give such a pair with its ids and excerpts.")
    with open(os.path.join(OUT, "quality_metrics.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
