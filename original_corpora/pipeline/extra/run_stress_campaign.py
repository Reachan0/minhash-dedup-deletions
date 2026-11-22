#!/usr/bin/env python3
"""Drive the synthetic bucket-stress experiments, one fresh process per point.

wall        : K grid 5k..60k; stop at the first failure (memory limit, timeout).
mitigation  : K=4000; full expansion vs per-bucket cap vs NeMo-style anchors.
A timed-out or OOM-killed point is recorded explicitly (last success / first failure),
so the shape of the wall does not depend on the pod's own OOM killer.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bigcode_lib as BC  # noqa: E402

K = BC.K
EXTRA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "extra")
DRIVER = os.path.join(HERE, "run_bucket_stress.py")
OUT = os.path.join(EXTRA, "bucket_stress.jsonl")
MIT = os.path.join(EXTRA, "bucket_mitigation.jsonl")
WALL_GRID = [5000, 10000, 20000, 30000, 40000, 50000, 60000]


def run(cmd, timeout):
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout", round(time.perf_counter() - t0, 1), ""
    if p.returncode != 0:
        return None, f"process_rc{p.returncode}", round(time.perf_counter() - t0, 1), p.stderr[-2000:]
    line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    try:
        return json.loads(line), None, round(time.perf_counter() - t0, 1), ""
    except Exception as e:  # noqa: BLE001
        return None, f"bad_json:{type(e).__name__}", round(time.perf_counter() - t0, 1), line[-2000:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["wall", "mitigation"], required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--rlimit-gib", type=float, default=40.0)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--cap", type=int, default=500)
    args = ap.parse_args()
    py = sys.executable

    if args.mode == "wall":
        for n in WALL_GRID:
            row = dict(mode="wall", n=n, workers=args.workers,
                       rlimit_gib=args.rlimit_gib, evidence_tier="rerun",
                       synthetic=True, ts_utc=K.now(), hw=K.hw_facts())
            out, err, wall, tail = run([py, DRIVER, "wall", "--n", str(n),
                                        "--workers", str(args.workers),
                                        "--rlimit-gib", str(args.rlimit_gib)],
                                       args.timeout)
            if err:
                row.update(status=err, wall_s=wall, stderr_tail=tail)
                K.append_row(OUT, row)
                print(json.dumps({k: row[k] for k in ("n", "status", "wall_s")}), flush=True)
                break
            print(json.dumps({k: out.get(k) for k in
                              ("n", "status", "largest_bucket", "n_pairs_mult",
                               "n_pairs_unique", "t_pairs_s", "peak_rss_mb")}), flush=True)
    else:
        out, err, wall, tail = run([py, DRIVER, "mitigation", "--n", str(args.n),
                                    "--workers", str(args.workers),
                                    "--rlimit-gib", str(args.rlimit_gib),
                                    "--cap", str(args.cap)], args.timeout)
        if err:
            K.append_row(MIT, dict(mode="mitigation", n=args.n, status=err,
                                   wall_s=wall, stderr_tail=tail,
                                   evidence_tier="rerun", ts_utc=K.now()))
            print(err, flush=True)


if __name__ == "__main__":
    main()
