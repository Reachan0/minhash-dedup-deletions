"""Build shingle caches (npz off/flat/nwords), meta parquet (id) and rank caches (R,S,V)
at the paths core_lib expects. mix is sliced from the fw and c4 caches (per-document shingling).
Usage: build_caches.py CORE_DIR WORKERS"""
import sys, os, time, json
sys.path.insert(0, sys.argv[1]); W = int(sys.argv[2])
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
import core_lib as K, make_workloads as M
K.set_threads(W)

def save(w, off, flat, nw, ids):
    for p in (K.SHINGLE_CACHE[w], K.META_CACHE[w], K.RANKS_CACHE[w]):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez(K.SHINGLE_CACHE[w], off=off, flat=flat, nwords=nw)
    pq.write_table(pa.table({"id": ids}), K.META_CACHE[w])
    t = time.time(); R, S, V = K.EJ.build_ranks(off, flat)
    np.savez(K.RANKS_CACHE[w], R=R, S=np.int64(S), V=np.int64(V))
    print(f"[{w}] saved n={len(off)-1} flat={len(flat)} ranks {time.time()-t:.0f}s rss={K.peak_rss_mb():.0f}MB", flush=True)

store = {}
for w in ["wet", "c4", "fw", "mix"]:
    t0 = time.time()
    if w == "mix":
        o1, f1, n1 = store["fw"]; o2, f2, n2 = store["c4"]
        a = 500000
        off = np.concatenate([o1[:a + 1], o2[1:a + 1] - o2[0] + o1[a]])
        flat = np.concatenate([f1[:o1[a]], f2[:o2[a]]])
        nw = np.concatenate([n1[:a], n2[:a]])
    else:
        off, flat, nw, info = K.shingle_from_pool(w, workers=W)
        print(f"[{w}] shingled {time.time()-t0:.0f}s {info}", flush=True)
    ids = M.pool_ids(w)
    assert len(ids) == len(off) - 1
    save(w, off, flat, nw, ids)
    if w in ("fw", "c4"): store[w] = (off, flat, nw)
    elif w == "wet": pass
    print(f"[{w}] total {time.time()-t0:.0f}s", flush=True)
    del off, flat, nw
