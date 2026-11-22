#!/usr/bin/env python3
"""Per-document release files for Snapshot-2025 (the 0.53M rung), no document text.

documents.parquet: one row per document in corpus order
  doc            index in the corpus (0-based)
  record_id      the record identifier of the WET extract (column "id" of the corpus shards)
  set_id         identifier of its distinct shingle set (documents with identical sets share it)
  n_shingles     number of distinct 5-gram shingles
  nb_080, nb_075 has a neighbour at exact J >= 0.8 / 0.75 (the soundness label)
  C_080          deleted by the exact join at tau 0.8 (minimum-identifier representative)
  A, A_dt, B, W  deleted at tau 0.8, one bit per hash seed 1..5 (bit s-1 set if deleted in seed s)
reference_edges_t0.80.parquet, reference_edges_t0.75.parquet: the exact reference between distinct
  shingle sets, columns set_i < set_j and jaccard; documents with the same set_id are duplicates at J = 1.
Usage: env/bin/python release_snapshot.py <out dir>
"""
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

W = "/path/to/scratch/dedupjoin"
RUNG = f"{W}/runs/rung_531597"
sys.path[:0] = [f"{W}/dedupjoin-newsnap-selection/code/newsnap", f"{W}/dedupjoin-newsnap-selection/code/run"]
import snap_common as S  # noqa: E402

S.SHINGLE_NPZ, S.CACHE, S.SNAP, S.RUNS = f"{RUNG}/snapshot.npz", f"{RUNG}/cache", RUNG, f"{RUNG}/quality.jsonl"


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    import collapsed_lib as CL
    K = CL.K
    K.set_threads(K.THREADS)
    z = CL.load_collapsed()
    group, m, rep, nsh = z["group_of_doc"], z["group_size"], z["group_rep"], z["group_nshingle"]
    n, G = len(group), len(m)
    valid, valid_g = nsh[group] > 0, nsh > 0
    st, truth = {}, {}
    for tau in (0.8, 0.75):
        ti, tj, tj_jac = CL.truth_u(tau)
        truth[tau] = (ti, tj, tj_jac)
        st[tau] = CL.exact_state(z, ti, tj)
    bits = {k: np.zeros(n, np.uint8) for k in ("A", "A_dt", "B", "W")}
    for seed in (1, 2, 3, 4, 5):
        kb_g = CL.group_band_keys(z, seed)
        ei, ej, _ = K.chain_edges(CL.full_band_keys(kb_g, z), valid)
        remA = (K.uf_minid(n, ei, ej) != np.arange(n)) & valid
        remD = (K.uf_datatrove(n, ei, ej) != np.arange(n)) & valid
        del ei, ej
        pi, pj, _ = K.all_bucket_pairs(kb_g, valid_g, G)
        jac = K.L.jaccard_pairs(z["off_u"], z["flat_u"], pi, pj)
        ver = jac >= 0.8 - 1e-12
        vi, vj = pi[ver], pj[ver]
        rootB = K.uf_minid(G, vi, vj)
        comp_min = np.full(G, np.iinfo(np.int64).max, np.int64)
        np.minimum.at(comp_min, rootB, rep)
        remB = valid & (np.arange(n) != comp_min[rootB[group]])
        kept = K.greedy_mis(G, vi, vj)
        remW = valid & ((~kept[group]) | (np.arange(n) != rep[group]))
        for k, r in (("A", remA), ("A_dt", remD), ("B", remB), ("W", remW)):
            bits[k] |= (r.astype(np.uint8) << np.uint8(seed - 1))
        print(f"seed {seed}: A {int(remA.sum())} B {int(remB.sum())} W {int(remW.sum())}", flush=True)
    ids = []
    for f in sorted(os.listdir(f"{RUNG}/corpus_links")):
        if f.startswith("shard-"):
            ids.extend(pq.read_table(f"{RUNG}/corpus_links/{f}", columns=["id"])["id"].to_pylist())
    assert len(ids) == n, (len(ids), n)
    t = pa.table({"doc": np.arange(n, dtype=np.int32), "record_id": ids, "set_id": group.astype(np.int32),
                  "n_shingles": nsh[group].astype(np.int32), "nb_080": st[0.8]["has_nb_doc"],
                  "nb_075": st[0.75]["has_nb_doc"], "C_080": st[0.8]["rem"],
                  **{k: v for k, v in bits.items()}})
    pq.write_table(t, f"{out}/documents.parquet", compression="zstd")
    for tau, lab in ((0.8, "0.80"), (0.75, "0.75")):
        ti, tj, jj = truth[tau]
        pq.write_table(pa.table({"set_i": ti.astype(np.int32), "set_j": tj.astype(np.int32),
                                 "jaccard": np.asarray(jj, np.float32)}), f"{out}/reference_edges_t{lab}.parquet",
                       compression="zstd")
    for f in sorted(os.listdir(out)):
        print(f, os.path.getsize(f"{out}/{f}"), flush=True)
    print("RELEASE_DONE", flush=True)


if __name__ == "__main__":
    main()
