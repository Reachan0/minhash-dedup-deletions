"""Duplicate-density / length / domain strata for the pool, plus the sampling check.

Inputs
  <pool>/meta/index-*.parquet          per-document metadata written by build_pool.py
  <pool>/doc_clusters.parquet         LSH cluster size per document (pool_cluster.py)
  <pool>/cluster_stats.json           aggregate cluster statistics

Outputs
  <pool>/strata.parquet               labels the scale step subsamples by
  <results>/pool_stats.json           characterisation numbers used by the paper
  <results>/sampling_check.json       proof that the earlier C4/FineWeb/Mix samples are
                                      exactly recoverable from the pool (same IDs, same order)
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


LEN_EDGES = [0, 50, 150, 500, 1500, 10 ** 9]
LEN_LABELS = ["L1_lt50", "L2_50_149", "L3_150_499", "L4_500_1499", "L5_ge1500"]
DUP_LABELS = ["D1_isolated", "D2_2_3", "D3_4_15", "D4_ge16"]
DOM_LABELS = ["H1_unique", "H2_2_9", "H3_10_99", "H4_ge100"]


def bucket(v, edges, labels):
    return np.asarray(pd.cut(v, bins=edges, labels=labels, right=False))


def sampling_check(index):
    """Compare the pool's first N ids with the earlier meta caches."""
    out = {}
    fw = index[index.source == "fw"].sort_values("row_id")
    c4 = index[index.source == "c4"].sort_values("row_id")
    for n, path, pick in [
        (1_000_000, ".work/data/C03/r2/cache/fw_1000000_meta.parquet", fw),
        (700_000, ".work/data/C03/r2/cache/c4_700000_meta.parquet", c4),
    ]:
        if not os.path.exists(path):
            out[path] = "cache missing"
            continue
        old = pq.read_table(path, columns=["id"]).column("id").to_pylist()[:n]
        new = pick["id"].to_list()[:n]
        out[os.path.basename(path)] = {
            "n": n, "ids_identical": old == new,
            "first_mismatch": next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), None),
        }
    # mixed setting: first 500k fw then first 500k c4 (round-2 definition)
    path = ".work/data/C03/r2/cache/mix_1000000_meta.parquet"
    if os.path.exists(path):
        old = pq.read_table(path, columns=["id"]).column("id").to_pylist()
        new = fw["id"].to_list()[:500_000] + c4["id"].to_list()[:500_000]
        out["mix_1000000_meta.parquet"] = {
            "n": 1_000_000, "ids_identical": old == new,
            "first_mismatch": next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), None),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--indices", default="")
    ap.add_argument("--clusters", default="")
    args = ap.parse_args()
    os.makedirs(args.results, exist_ok=True)
    idx_glob = args.indices or os.path.join(args.pool, "meta", "index-*.parquet")
    cols = ["row_id", "id", "source", "shard", "local_idx", "url", "domain",
            "nchar", "nwords", "raw_hash", "norm_hash"]
    frames = [pq.read_table(f, columns=cols).to_pandas() for f in sorted(glob.glob(idx_glob))]
    df = pd.concat(frames, ignore_index=True).sort_values("row_id").reset_index(drop=True)
    cl = pd.read_parquet(args.clusters or os.path.join(args.pool, "doc_clusters.parquet"))
    cl = cl.sort_values("row_id").reset_index(drop=True)
    assert len(cl) == len(df), (len(cl), len(df))
    df["lsh_size"] = cl["lsh_size"].values
    df["exact_raw_size"] = cl["exact_raw_size"].values

    df["length_bucket"] = bucket(df["nwords"].values, LEN_EDGES, LEN_LABELS)
    df["dup_bucket"] = np.where(df.lsh_size == 1, "D1_isolated",
                                np.where(df.lsh_size <= 3, "D2_2_3",
                                         np.where(df.lsh_size <= 15, "D3_4_15", "D4_ge16")))
    dc = df.domain.value_counts()
    df["domain_count"] = df.domain.map(dc).fillna(0).astype(int)
    df["domain_bucket"] = np.where(df.domain_count == 1, "H1_unique",
                                   np.where(df.domain_count <= 9, "H2_2_9",
                                            np.where(df.domain_count <= 99, "H3_10_99", "H4_ge100")))
    df[["row_id", "id", "source", "length_bucket", "dup_bucket", "domain_bucket",
        "nwords", "lsh_size", "exact_raw_size", "domain_count"]].to_parquet(
        os.path.join(args.pool, "strata.parquet"), compression="zstd")

    stats = {"provenance": "independently rerun (this run)", "n_docs": int(len(df)),
             "dup_density_lsh_ge2": round(float((df.lsh_size >= 2).mean()), 6),
             "sources": {}}
    for src, g in df.groupby("source"):
        stats["sources"][src] = {
            "n": int(len(g)),
            "nwords_mean": round(float(g.nwords.mean()), 2),
            "nwords_p50": int(g.nwords.median()),
            "nwords_p90": int(g.nwords.quantile(0.9)),
            "dup_density_lsh_ge2": round(float((g.lsh_size >= 2).mean()), 6),
            "dup_density_exact_raw": round(float((g.exact_raw_size >= 2).mean()), 6),
            "dup_density_exact_norm": round(float(g.norm_hash.duplicated(keep=False).mean()), 6),
            "largest_lsh_cluster": int(g.lsh_size.max()),
            "docs_in_largest": int((g.lsh_size == g.lsh_size.max()).sum()),
            "distinct_domains": int(g.domain.nunique()),
            "top20_domain_share": round(float(g.domain.value_counts().head(20).sum() / len(g)), 6),
        }
        stats["sources"][src]["by_length"] = {
            b: {"n": int((g.length_bucket == b).sum()),
                "dup_density": round(float((g.loc[g.length_bucket == b, "lsh_size"] >= 2).mean()), 4)
                if (g.length_bucket == b).any() else None}
            for b in LEN_LABELS}
        stats["sources"][src]["by_dup_bucket"] = {
            b: int((g.dup_bucket == b).sum()) for b in DUP_LABELS}
        stats["sources"][src]["top_domains"] = [
            {"domain": d, "docs": int(c),
             "dup_density": round(float((g.loc[g.domain == d, "lsh_size"] >= 2).mean()), 3)}
            for d, c in g.domain.value_counts().head(10).items()]

    # cross-source strata table: the subsampling menu for the scale step
    strata = (df.groupby(["source", "length_bucket", "dup_bucket"])
                .agg(n=("row_id", "size"), mean_nwords=("nwords", "mean"),
                     mean_lsh_size=("lsh_size", "mean"))
                .reset_index())
    stats["strata_cells"] = strata.to_dict(orient="records")
    stats["strata_cells_note"] = ("length_bucket x dup_bucket x source; the scale step can "
                                  "draw any target mix from these cells via strata.parquet")
    stats["sampling_check"] = sampling_check(df)
    with open(os.path.join(args.results, "pool_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "strata_cells"}, indent=2)[:4000])


if __name__ == "__main__":
    main()
