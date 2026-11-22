"""Rebuild the B2 injected variants on AutoDL (same seeds as on server) and rerun capped B with
two star passes. Usage: env/bin/python b2_autodl.py inject | runs"""
import os
import sys

W = "/path/to/scratch/dedupjoin"
RUNG = f"{W}/runs/rung_531597"
os.environ["B2_W"] = W
sys.path[:0] = [f"{W}/dedupjoin-newsnap-selection/code/newsnap", f"{W}/tools"]
import snap_common as S  # noqa: E402

S.SHINGLE_NPZ, S.CACHE, S.SNAP, S.RUNS = f"{RUNG}/snapshot.npz", f"{RUNG}/cache", RUNG, f"{RUNG}/quality.jsonl"
S.CORPUS = f"{RUNG}/corpus_links"

if sys.argv[1] == "inject":
    import b2_inject_snapshot as B
    sys.argv = sys.argv[:1]
    B.main()
else:
    import b2_runs as R
    for v in ["base", "exact_1000", "exact_10000", "exact_100000", "near_1000", "near_10000", "near_100000"]:
        R.run(v, "stream", out="b2_runs_autodl.jsonl")
        for q in (100, 1000):
            R.run(v, "cap", q, cap_prefix=False, out="b2_cap2.jsonl")
    R.run("exact_100000", "cap", 1000, cap_prefix=True, out="b2_cap2.jsonl")
    print("B2_CAP2_DONE", flush=True)
