"""Cluster the whole pool with datatrove-style 14x8 LSH band keys.

For every band, documents sharing a band key are ordered by key; consecutive
documents inside a group are joined (the datatrove union-find construction).
The resulting connected components are the LSH duplicate clusters.

Outputs:
  <out>/cluster_root.npy   int32[N]   component representative (minimum row id)
  <out>/cluster_size.npy   int32[N]   component size indexed by root id (0 for non-roots)
  <out>/cluster_stats.json            per-source duplicate density, size histogram,
                                      largest clusters with representative documents
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq


def load_band_keys(band_dir):
    files = sorted(glob.glob(os.path.join(band_dir, "bands-*.npy")))
    parts = [np.load(f) for f in files]
    return np.concatenate(parts, axis=0), files


def band_edges(keys, n):
    """Consecutive-pair edges per (band, key) group. Returns uint32 (u, v)."""
    us, vs = [], []
    for b in range(keys.shape[1]):
        k = keys[:, b]
        order = np.argsort(k, kind="stable")
        sk = k[order]
        starts = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
        lens = np.diff(np.r_[starts, n])
        keep = lens >= 2
        starts = starts[keep]
        lens = lens[keep]
        if starts.size == 0:
            continue
        counts = lens - 1
        total = int(counts.sum())
        run_id = np.repeat(np.arange(starts.size), counts)
        pos = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        s = starts[run_id] + pos
        us.append(order[s].astype(np.uint32))
        vs.append(order[s + 1].astype(np.uint32))
    if not us:
        z = np.zeros(0, np.uint32)
        return z, z
    return np.concatenate(us), np.concatenate(vs)


from numba import njit


@njit(cache=True)
def _find(par, x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


@njit(cache=True)
def uf(n, ei, ej):
    par = np.arange(n)
    for k in range(ei.shape[0]):
        a = _find(par, ei[k])
        c = _find(par, ej[k])
        if a != c:
            if a < c:
                par[c] = a
            else:
                par[a] = c
    root = np.empty(n, np.int32)
    for x in range(n):
        root[x] = _find(par, x)
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    keys, files = load_band_keys(args.bands)
    n = keys.shape[0]
    print(f"loaded {len(files)} chunks: N={n}", flush=True)
    u, v = band_edges(keys, n)
    print(f"edges: {len(u):,} ({time.time()-t0:.0f}s)", flush=True)
    root = uf(n, u, v)
    size = np.bincount(root, minlength=n).astype(np.int32)   # per root id
    doc_size = size[root]                                    # per document
    np.save(os.path.join(args.out, "cluster_root.npy"), root)
    np.save(os.path.join(args.out, "cluster_size.npy"), size)
    print(f"clusters done ({time.time()-t0:.0f}s); docs in clusters>=2: "
          f"{(doc_size >= 2).sum():,}", flush=True)

    # per-source aggregates from the pool index
    meta_cols = ["row_id", "id", "source", "domain", "nwords", "raw_hash", "norm_hash"]
    idx = []
    for f in sorted(glob.glob(os.path.join(args.pool, "meta", "index-*.parquet"))):
        idx.append(pq.read_table(f, columns=meta_cols).to_pandas())
    import pandas as pd
    df = pd.concat(idx, ignore_index=True)
    assert len(df) == n, (len(df), n)
    df = df.sort_values("row_id").reset_index(drop=True)
    assert (df["row_id"].values == np.arange(n)).all()
    df["lsh_size"] = doc_size
    raw_codes = pd.factorize(df["raw_hash"])[0]
    norm_codes = pd.factorize(df["norm_hash"])[0]
    df["exact_raw_size"] = np.bincount(raw_codes)[raw_codes]
    df["exact_norm_size"] = np.bincount(norm_codes)[norm_codes]

    stats = {"n_docs": int(n), "n_edges": int(len(u)), "seconds": round(time.time() - t0, 1),
             "provenance": "independently rerun (this run)",
             "sources": {}, "cluster_size_hist": {}, "largest": []}
    for src, g in df.groupby("source"):
        stats["sources"][src] = {
            "n": int(len(g)),
            "mean_nwords": float(g["nwords"].mean()),
            "p50_nwords": float(g["nwords"].median()),
            "p90_nwords": float(g["nwords"].quantile(0.9)),
            "dup_density_lsh": float((g["lsh_size"] >= 2).mean()),
            "exact_raw_density": float((g["exact_raw_size"] >= 2).mean()),
            "exact_norm_density": float((g["exact_norm_size"] >= 2).mean()),
            "largest_lsh_cluster": int(g["lsh_size"].max()),
            "largest_exact_raw_cluster": int(g["exact_raw_size"].max()),
            "distinct_domains": int(g["domain"].nunique()),
            "top_domain_share": float(g["domain"].value_counts().head(20).sum() / len(g)),
        }
    hist = np.bincount(size)
    for k in list(range(1, 11)) + [16, 32, 64, 128, 512, 2048]:
        stats["cluster_size_hist"][str(k)] = int(hist[k]) if k < len(hist) else 0
    stats["cluster_size_hist"]["max"] = int(size.max())
    # largest clusters with representatives
    roots, counts = np.unique(root, return_counts=True)
    top = roots[np.argsort(-counts)[:10]]
    for r in top:
        members = df.iloc[np.flatnonzero(root == r)[:3]]
        stats["largest"].append({
            "size": int((root == r).sum()),
            "examples": [{"id": m["id"], "source": m["source"], "domain": m["domain"],
                          "nwords": int(m["nwords"])} for _, m in members.iterrows()],
        })
    df[["row_id", "id", "source", "domain", "nwords", "lsh_size",
        "exact_raw_size", "exact_norm_size"]].to_parquet(
        os.path.join(args.out, "doc_clusters.parquet"), compression="zstd")
    with open(os.path.join(args.out, "cluster_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({"n": n, "seconds": stats["seconds"],
                      "dup_density": {s: v["dup_density_lsh"] for s, v in stats["sources"].items()}}))


if __name__ == "__main__":
    main()
