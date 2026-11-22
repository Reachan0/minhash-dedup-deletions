"""Cross-source duplication in the pool.

Uses the connected components from pool_cluster.py to answer two questions the
mixed-setting experiments depend on:
  1. how much of the FineWeb sample re-appears inside the C4 shards we built
     (near-duplicate overlap after LSH), and
  2. how many byte- and whitespace-normalised-identical documents are shared
     between sources.
Writes <results>/pool_cross_source.json.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--bands-dir", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    root = np.load(os.path.join(args.bands_dir, "cluster_root.npy"))
    dc = pd.read_parquet(os.path.join(args.bands_dir, "doc_clusters.parquet"),
                         columns=["row_id", "source", "lsh_size"])
    dc = dc.sort_values("row_id").reset_index(drop=True)
    assert len(dc) == len(root)
    dc["root"] = root

    # ---- LSH cluster composition ----
    multi = dc[dc.lsh_size >= 2].copy()
    g = multi.groupby("root")["source"]
    comp = g.agg(lambda s: "|".join(sorted(set(s))))
    sizes = g.size()
    out = {"provenance": "independently rerun (this run)",
           "n_docs": int(len(dc)), "n_lsh_clusters_ge2": int(len(comp))}
    combos = comp.value_counts()
    out["clusters_by_source_composition"] = {k: int(v) for k, v in combos.items()}
    dc["combo"] = dc["root"].map(comp).fillna("")
    out["docs_by_source_composition"] = {
        k: int(v) for k, v in dc.loc[dc.combo != "", "combo"].value_counts().items()}

    # share of each source's docs that sit in a cluster containing another source
    multi["combo"] = multi["root"].map(comp)
    multi["cross"] = multi["combo"].str.contains(r"\|")
    per_src = {}
    for src, d in dc.groupby("source"):
        c = multi[(multi.source == src) & multi.cross]
        per_src[src] = {
            "n": int(len(d)),
            "docs_in_lsh_cluster_ge2": int((d.lsh_size >= 2).sum()),
            "docs_in_cross_source_cluster": int(len(c)),
            "fraction_in_cross_source_cluster": round(len(c) / len(d), 6),
            "partner_sources": {
                k: int(v) for k, v in
                multi[(multi.source == src) & multi.cross]["combo"].value_counts().items()},
        }
    out["per_source"] = per_src

    # ---- exact hashes across sources ----
    idx = pd.concat([pq.read_table(p, columns=["source", "raw_hash", "norm_hash"]).to_pandas()
                     for p in sorted(__import__("glob").glob(
                         os.path.join(args.pool, "meta", "index-*.parquet")))],
                    ignore_index=True)
    exact = {}
    for col in ("raw_hash", "norm_hash"):
        n = idx.groupby(col)["source"].nunique()
        shared = set(n[n >= 2].index)
        sub = idx[idx[col].isin(shared)]
        pairs = sub.groupby(col)["source"].agg(lambda s: "|".join(sorted(set(s))))
        docs = sub.groupby("source").size()
        exact[col] = {
            "n_hashes_shared_across_sources": int(len(shared)),
            "docs_in_shared_hashes": {k: int(v) for k, v in docs.items()},
            "source_pairs": {k: int(v) for k, v in pairs.value_counts().items()},
        }
    out["exact"] = exact
    out["method"] = ("LSH = 14x8 datatrove banding, 112 hashes, connected components; "
                     "raw_hash = xxh64(text), norm_hash = xxh64(' '.join(text.lower().split()))")

    os.makedirs(args.results, exist_ok=True)
    p = os.path.join(args.results, "pool_cross_source.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_source",)}, indent=2)[:3000])


if __name__ == "__main__":
    main()
