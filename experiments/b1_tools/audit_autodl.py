"""Run the content audit (c_content_audit.py) on AutoDL against the 0.53M rung, which is
Snapshot-2025 rebuilt from the same 104 WET segments. Points snap_common at the rung, links the
segment parquet files as the corpus, and rebuilds A's seed-1 mask if it is missing.
Usage: env/bin/python audit_autodl.py sample | kept | label <model> | analyze"""
import os
import sys

import numpy as np

W = "/path/to/scratch/dedupjoin"
RUNG = f"{W}/runs/rung_531597"
CORPUS = f"{W}/runs/rung_531597/corpus_links"
sys.path[:0] = [f"{W}/dedupjoin-newsnap-selection/code/newsnap", f"{W}/dedupjoin-newsnap-selection/code/run",
                f"{W}/dedupjoin-newsnap-selection/code/post_reveal"]
import snap_common as S  # noqa: E402

S.SHINGLE_NPZ, S.CACHE, S.SNAP, S.RUNS = f"{RUNG}/snapshot.npz", f"{RUNG}/cache", RUNG, f"{RUNG}/quality.jsonl"
S.CORPUS = CORPUS
if not os.path.isdir(CORPUS):
    os.makedirs(CORPUS)
    for i in range(104):
        os.symlink(f"{W}/eng/{i:04d}.parquet", f"{CORPUS}/shard-{i:04d}.parquet")
import collapsed_lib as CL  # noqa: E402
import c_content_audit as CA  # noqa: E402

mask = f"{S.CACHE}/masks/raw_s1.npz"
if sys.argv[1] in ("sample", "kept") and not os.path.exists(mask):
    K = CL.K
    K.set_threads(K.THREADS)
    z = CL.load_collapsed()
    off, _, _ = S.load_snapshot()
    n = len(off) - 1
    valid = np.diff(off) > 0
    kb = CL.full_band_keys(CL.group_band_keys(z, 1), z)
    ei, ej, _ = K.chain_edges(kb, valid)
    root = K.uf_minid(n, ei, ej)
    rem = (root != np.arange(n)) & valid
    os.makedirs(os.path.dirname(mask), exist_ok=True)
    np.savez(mask, root=root, rem=rem)
    print("A seed 1: deleted", int(rem.sum()), flush=True)
{"sample": CA.sample, "kept": CA.kept, "label": lambda: CA.label(sys.argv[2]), "analyze": CA.analyze}[sys.argv[1]]()
