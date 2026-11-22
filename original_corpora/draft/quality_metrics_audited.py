#!/usr/bin/env python3
"""The three final-corpus metrics on the audited (shared 117-hash draw) cells.

Same definitions as quality_metrics.py, but the deletion sets are the earlier run's
14x8 cells (shared draw): rem_lsh from its dump, rem_v recomputed from its cached
candidate edges with exact verification, rem_exact from this run's exact join
(draw-independent). Writes results/quality_metrics_audited.json.
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
CELLS = [("c4", 700000, [0.8, 0.7]), ("fw", 1000000, [0.8, 0.7])]


def main():
    doc = dict(version=1, generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               provenance="earlier run: deletion and candidate edges; independently rerun: exact pairs and exact set",
               design="audited shared 117-hash draw, first 112 columns, 14x8, seed 1")
    out = {}
    for corpus, n, taus in CELLS:
        tag = f"{corpus}_{n}"
        meta = pq.read_table(os.path.join(WRK, f"{tag}_meta.parquet"),
                             columns=["id", "url", "nwords", "nchar", "head"]).to_pydict()
        zc = np.load(os.path.join(OLD_DUMP, f"{tag}_cand_dt1_14x8.npz"))
        ci, cj, cjac = zc["codes"] // n, zc["codes"] % n, zc["jac"]
        res = {}
        for tau in taus:
            z = np.load(os.path.join(OLD_DUMP, f"{tag}_{tau}_dt1_14x8.npz"))
            p = np.load(os.path.join(WRK, f"pairs_{tag}_{tau}.npz"))
            ti, tj, jac = p["i"].astype(np.int64), p["j"].astype(np.int64), p["jac"]
            assert np.array_equal(ti, z["ti"]) and np.array_equal(tj, z["tj"])
            st = np.load(os.path.join(WRK, f"sets_{tag}_s1_14x8_t{tau}.npz"))
            rem_lsh = np.zeros(n, bool)
            rem_lsh[z["rem_lsh"]] = True
            ver = cjac >= tau - 1e-12
            rem_v = L.removed_set(L.uf_components(n, ci[ver], cj[ver]))
            rem = {"raw_lsh": rem_lsh, "lsh_verify": rem_v,
                   "exact": st["rem_exact"].astype(bool)}
            rec = dict(n_true_pairs=int(len(ti)),
                       candidate_edge_recall=float(z["hit"].mean()),
                       n_true_pairs_missed=int((~z["hit"]).sum()))
            for name, rm in rem.items():
                keep = ~rm
                both = keep[ti] & keep[tj]
                rec[f"residual_duplicate_pairs_{name}"] = int(both.sum())
                rec[f"residual_duplicate_docs_{name}"] = int(
                    len(set(ti[both].tolist()) | set(tj[both].tolist())))
                deleted = np.nonzero(rm)[0]
                wit = np.zeros(n, bool)
                wit[ti[keep[tj]]] = True
                wit[tj[keep[ti]]] = True
                rec[f"deleted_docs_{name}"] = int(len(deleted))
                rec[f"direct_witness_coverage_{name}"] = float(wit[deleted].mean())
                rec[f"deleted_without_direct_witness_{name}"] = int((~wit[deleted]).sum())
                if both.any():
                    k = int(np.argmax(np.where(both, jac, -1)))
                    rec[f"top_residual_pair_{name}"] = dict(
                        i=int(ti[k]), j=int(tj[k]), jac=float(jac[k]),
                        i_id=str(meta["id"][ti[k]]), j_id=str(meta["id"][tj[k]]),
                        i_url=str(meta["url"][ti[k]])[:160],
                        j_url=str(meta["url"][tj[k]])[:160],
                        i_nwords=int(meta["nwords"][ti[k]]),
                        j_nwords=int(meta["nwords"][tj[k]]),
                        i_head=str(meta["head"][ti[k]])[:200],
                        j_head=str(meta["head"][tj[k]])[:200])
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
        out[tag] = res
    doc["cells"] = out
    with open(os.path.join(OUT, "quality_metrics_audited.json"), "w") as f:
        json.dump(doc, f, indent=2)
    for c, v in out.items():
        for t, r in v.items():
            print(c, t, "recall", round(r["candidate_edge_recall"], 4),
                  "resid", r["residual_duplicate_pairs_raw_lsh"],
                  r["residual_duplicate_pairs_lsh_verify"], r["residual_duplicate_pairs_exact"],
                  "wit", round(r["direct_witness_coverage_raw_lsh"], 4),
                  round(r["direct_witness_coverage_lsh_verify"], 4),
                  round(r["direct_witness_coverage_exact"], 4))


if __name__ == "__main__":
    main()
