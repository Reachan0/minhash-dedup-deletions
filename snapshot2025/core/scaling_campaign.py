#!/usr/bin/env python3
"""Grow the path-B bucket-expansion experiment until a run fails or the memory cap hits.

Each point runs in a fresh process (run_bucket_scaling.py) and appends core/bucket_scaling.jsonl.
A failed point is recorded and stops that source; the previous point is the last safe size.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_lib as K

PY = sys.executable
DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_bucket_scaling.py")
PLAN = {
    "wet": [50000, 100000, 175272],
    "fw": [250000, 500000, 750000, 1000000],
    "c4": [100000, 250000, 500000, 700000, 1400000, 2100000, 2850538],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="wet,fw,c4")
    ap.add_argument("--rlimit-gib", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=K.THREADS)
    args = ap.parse_args()
    done = K.done_keys(K.SCALING, "key")
    for src in [s for s in args.sources.split(",") if s]:
        stop = False
        for n in PLAN[src]:
            key = f"{src}|n{n}"
            if stop:
                K.append_row(K.SCALING, dict(key=key, source=src, n=n,
                                             status="skipped_after_failure",
                                             evidence_tier="rerun", ts_utc=K.now()))
                continue
            if key in done:
                # stop if an earlier recorded point failed
                continue
            cmd = [PY, DRIVER, "--source", src, "--n", str(n),
                   "--workers", str(args.workers), "--rlimit-gib", str(args.rlimit_gib)]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                K.append_row(K.SCALING, dict(key=key, source=src, n=n,
                                             status=f"process_rc{p.returncode}",
                                             stderr=p.stderr[-2000:],
                                             evidence_tier="rerun", ts_utc=K.now()))
                print(f"FAILED {key} rc={p.returncode}", flush=True)
                stop = True
            else:
                print("ok", key, flush=True)
                row = json.loads(p.stdout.strip().splitlines()[-1])
                if row.get("status") != "ok":
                    stop = True


if __name__ == "__main__":
    main()
