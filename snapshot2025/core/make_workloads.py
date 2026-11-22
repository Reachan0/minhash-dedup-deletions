#!/usr/bin/env python3
"""Freeze the four workloads as pool row_id prefixes and record their ids.

Writes core/workloads.json:
  - pool ranges, n, id sha256, first/last id
  - verification that the reused shingle caches have exactly the pool-prefix ids
  - Mix construction/overlap facts from audit/provenance_overlap.json
"""
import hashlib
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_lib as K


def pool_ids(workload):
    ids = []
    for source, start, count in K.WORKLOADS[workload]:
        got = 0
        while got < count:
            p, off = K._pool_shard(source, start + got)
            pf = pq.ParquetFile(p)
            need = count - got
            base = 0
            seg = []
            for rg in range(pf.metadata.num_row_groups):
                nrg = pf.metadata.row_group(rg).num_rows
                lo, hi = base, base + nrg
                base = hi
                if hi <= off or lo >= off + need:
                    continue
                col = pf.read_row_group(rg, columns=["id"]).column("id").to_pylist()
                a = max(0, off - lo)
                b = min(len(col), off + need - lo)
                seg.extend(col[a:b])
            ids.extend(seg)
            got += len(seg)
    return ids


def main():
    os.makedirs(K.CORE, exist_ok=True)
    out = dict(generated_utc=K.now(), pool_order="fw -> c4 -> wet",
               pool_counts=K.POOL_COUNTS, workloads={})
    overlap = K.workload_overlap_facts()
    for w in K.WORKLOADS:
        ids_pool = pool_ids(w)
        ids_cache = K.load_ids(w)
        n = len(ids_pool)
        same = ids_pool == ids_cache
        sha = hashlib.sha256("\n".join(ids_pool).encode()).hexdigest()
        ranges = []
        for source, start, count in K.WORKLOADS[w]:
            ranges.append(dict(source=source, local_start=start, count=count,
                               pool_row_id_start=K.POOL_START[source] + start))
        rec = dict(n=n, pool_ranges=ranges, id_sha256=sha,
                   first_id=ids_pool[0], last_id=ids_pool[-1],
                   cache_meta=os.path.relpath(K.META_CACHE[w], K.ROOT),
                   cache_ids_identical=same,
                   cache_n=len(ids_cache))
        out["workloads"][w] = rec
        print(w, n, "ids match cache:", same, sha[:16])
    # Mix is a concatenation, not an independent corpus (audit provenance)
    if "mix" in out["workloads"]:
        out["workloads"]["mix"]["construction"] = (
            "first 500,000 FineWeb rows followed by first 500,000 C4 rows; "
            "100% overlap with the FW and C4 prefixes; NOT an independent corpus")
        out["workloads"]["mix"]["overlap_with_fw_c4"] = {
            "fw_prefix_docs": 500000, "c4_prefix_docs": 500000,
            "share_of_mix": 1.0}
    out["audit_provenance_overlap"] = {
        "source": "audit/provenance_overlap.json",
        "mix_prefix_identical": overlap.get("mix_prefix_identical"),
        "wet_snapshot": overlap.get("wet_snapshot")}
    p = os.path.join(K.CORE, "workloads.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
