"""Shared timed path for literal and collapsed plans."""
import hashlib
import statistics
import time

from driver import Stages, require
from reference import maskname


def input_path(c):
    return c.work / "filter50.npz" if c.experiment == "E3" else c.snapshot


def pack(off, flat, docs):
    """Copy selected CSR rows in their given order without a per-shingle index array."""
    import numpy as np
    ends = off[docs + 1]
    starts = off[docs]
    offsets = np.r_[np.int64(0), (ends - starts).cumsum()]
    values = np.empty(offsets[-1], dtype=flat.dtype)
    for lo, hi, dest in zip(starts, ends, offsets[:-1], strict=True):
        values[dest:dest + hi - lo] = flat[lo:hi]
    return offsets, values


def group_map(c, off, flat, key):
    import numpy as np
    import xxhash
    if key == "shingle_set":
        hashes = np.fromiter((xxhash.xxh64_intdigest(memoryview(flat[a:b])) if b > a else 0
                              for a, b in zip(off[:-1], off[1:], strict=True)),
                             np.uint64, len(off) - 1)
    else:
        hashes = np.load(c.hashes)
        if c.experiment == "E3":
            hashes = hashes[np.load(c.work / "source_rows.npy")]
    require(len(hashes) == len(off) - 1, "Collapse keys do not align with the shingle cache")
    # For both keys, group ids must follow first occurrence for the frozen greedy MIS.
    # Give every empty document an independent representative, as collapsed_lib does.
    _, first, inv = np.unique(hashes, return_index=True, return_inverse=True)
    docrep = first[inv]
    empty = np.flatnonzero(off[:-1] == off[1:])
    docrep[empty] = empty
    rep, group = np.unique(docrep, return_inverse=True)
    return rep, group


def load_map(path):
    import numpy as np
    with np.load(path) as f:
        return f["rep"], f["group"]


def cache_paths(c, key):
    return c.work / f"map-{key}.npz", c.work / f"sig-{key}-s1.npy"


def release_cache(c, key):
    for path in cache_paths(c, key):
        path.unlink(missing_ok=True)


def prepare(c, cl, collapse_key):
    import numpy as np
    cl.S.SHINGLE_NPZ = str(input_path(c))
    off, flat, _ = cl.S.load_snapshot()
    map_path, sig_path = cache_paths(c, collapse_key)
    if collapse_key:
        rep, group = group_map(c, off, flat, collapse_key)
        np.savez(map_path, rep=rep, group=group)
        off, flat = pack(off, flat, rep)
    np.save(sig_path, cl.K.L.signatures(off, flat, cl.K.H, 1, "dt"))
    return [dict(n_signature_rows=len(off) - 1)]


def timed(c, cl, method, tau, mode, repeat, collapse_key, seed=1):
    import numpy as np
    k = cl.K
    path = input_path(c)
    cl.S.SHINGLE_NPZ = str(path)
    map_path, sig_path = cache_paths(c, collapse_key)
    mask = c.dir / f"{method}-{tau}-{mode}-{repeat}-{collapse_key}-s{seed}.npy"
    timer = Stages()
    rb0 = k.read_bytes()
    off, flat, _ = timer.run("load_shingles", cl.S.load_snapshot)
    rb1 = k.read_bytes()
    valid = np.diff(off) > 0
    n = len(valid)
    if collapse_key:
        if mode == "incremental":
            rep, group = timer.run("load_group_map", load_map, map_path)
        else:
            rep, group = timer.run("grouping", group_map, c, off, flat, collapse_key)
        nodes_valid = valid[rep]
    else:
        nodes_valid = valid
    nodes = len(nodes_valid)
    if mode == "incremental":
        require(seed == 1, "Only seed 1 signatures are cached for timing")
        sig = timer.run("load_signatures", np.load, sig_path)
    else:
        if collapse_key:
            ou, fu = timer.run("pack_representatives", pack, off, flat, rep)
        else:
            ou, fu = off, flat
        sig = timer.run("signature", k.L.signatures, ou, fu, k.H, seed, "dt")
        del ou, fu
    bands = timer.run("banding", k.band_keys, sig)
    del sig
    if method == "A":
        i, j, stats = timer.run("chain_edges", k.chain_edges, bands, valid)
    else:
        pi, pj, stats = timer.run("bucket_pairs", k.all_bucket_pairs, bands, nodes_valid, nodes)
        def verify():
            # Evaluate on representative slices of the already-loaded document shingles.
            a, b = (rep[pi], rep[pj]) if collapse_key else (pi, pj)
            jac = k.L.jaccard_pairs(off, flat, a, b)
            keep = jac >= tau - 1e-12
            return pi[keep], pj[keep]
        i, j = timer.run("verify", verify)
        del pi, pj
    del bands
    if method == "W":
        kept = timer.run("greedy_mis", k.greedy_mis, nodes, i, j)
        def deletion_mask():
            if collapse_key:
                return valid & (~kept[group] | (np.arange(n) != rep[group]))
            return valid & ~kept
    else:
        root = timer.run("union_find", k.uf_minid, nodes, i, j)
        def deletion_mask():
            representative = rep[root[group]] if collapse_key else root
            return valid & (np.arange(n) != representative)
    rem = timer.run("lift" if collapse_key else "document_mask", deletion_mask)
    timer.run("write_mask", np.save, mask, rem)
    wall = time.perf_counter() - timer.start
    # Acceptance is outside the timed boundary.
    ref = None
    if c.experiment == "E2":
        ref = c.accepted() / maskname(method, tau, seed)
    elif c.smoke and seed == 1:
        ref = c.work / "subset_masks" / maskname(method, tau, seed)
    if ref is not None:
        print("STAGE mask_acceptance", flush=True)
        with np.load(ref) as f:
            reference = f["rem"]
        require(np.array_equal(rem, reference),
                f"Mask mismatch against {ref}: {int(np.count_nonzero(rem != reference))} documents")
        if not c.smoke and tau == .8 and method in ("B", "W"):
            require(rem.sum() == {"B": 113625, "W": 112798}[method], "Deletion count mismatch")
    read_bytes = rb1 - rb0 if rb0 is not None and rb1 is not None else None
    return [dict(wall_s=wall, load_shingles_s=timer.times["stage_load_shingles_s"],
                 load_shingles_read_bytes=read_bytes, shingle_cache_bytes=path.stat().st_size,
                 page_cache="uncontrolled; no flush; see load_shingles_read_bytes",
                 mask=str(mask), mask_sha256=hashlib.sha256(memoryview(rem)).hexdigest(),
                 reference_mask=str(ref) if ref else None, n=n, n_nodes=nodes, valid_docs=int(valid.sum()),
                 deleted=int(rem.sum()), edges=len(i), peak_self_rss_mb=k.peak_rss_mb(),
                 **stats, **timer.times)]


def repetitions(c, method, tau, key, *, allow_memory_failure=False):
    rows = []
    for mode, repeats in (("pathcold", 3), ("incremental", 5)):
        for repeat in range(1, (1 if c.smoke else repeats) + 1):
            r = c.job("timed", method=method, seed=1, tau=tau, mode=mode, repeat=repeat,
                      collapse_key=key, allow_memory_failure=allow_memory_failure)[0]
            rows.append(r)
    return rows


def timing_summary(c, rows):
    result = {}
    for mode in ("pathcold", "incremental"):
        group = [r for r in rows if r["mode"] == mode]
        if any(r["status"] != "ok" for r in group):
            result[mode] = dict(status="failed", keys=[r["key"] for r in group])
            continue
        stage_keys = [k for k in group[0] if k.startswith("stage_")]
        result[mode] = dict(status="ok", repeats=len(group),
                            median_wall_s=statistics.median(r["wall_s"] for r in group),
                            peak_tree_rss_mb=max(r["peak_tree_rss_mb"] for r in group),
                            stage_medians_s={k: statistics.median(r[k] for r in group) for k in stage_keys})
    return result
