#!/usr/bin/env python3
"""Orchestrate the timing campaign: one fresh OS process per timed configuration.

Plan (documented in the delivery README):
  incremental  algorithm only, shared preprocessing loaded from disk
               5 repeats: C4 700k and FineWeb 1M at tau=0.8; 3 repeats elsewhere
  cold         text -> shingles -> signatures/bands or ranks -> method
               2 repeats for A, B, C on C4/FineWeb at tau=0.8; 1 repeat elsewhere
               W cold is composed: cold B (shares every stage) + incremental MIS
               tau=0.75 cold is composed: cold preprocessing at tau=0.8 (tau-independent)
               + incremental algorithm at tau=0.75
  1 thread     incremental for every workload/tau/method (1 repeat), plus one cold A
               run per workload at tau=0.8 to scale the preprocessing itself

Resumable: keys already in core/timing.jsonl are skipped.  Runs are sequential; peak
RSS per configuration is recorded by run_timing.py, and a memory guard pauses before
starting a run if cgroup usage is above the guard.

Usage: python3 timing_campaign.py [--modes cold,inc,inc1] [--workloads c4,fw,mix,wet]
                                  [--taus 0.8,0.75] [--limit N] [--tag main]
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_lib as K

PY = sys.executable
DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_timing.py")
MEM_GUARD_GIB = 45.0


def cgroup_used_gib():
    try:
        return int(open("/sys/fs/cgroup/memory.current").read()) / 1024 ** 3
    except Exception:
        return None


def build_plan(workloads, taus, modes, tag):
    plan = []
    for w in workloads:
        for tau in taus:
            reps_inc = 5 if (w in ("c4", "fw") and abs(tau - 0.8) < 1e-9) else 3
            reps_cold = 3 if (w in ("c4", "fw") and abs(tau - 0.8) < 1e-9) else 2
            for m in ("A", "B", "W", "C"):
                if "inc" in modes:
                    for r in range(reps_inc):
                        plan.append(("incremental", w, tau, m, 12, r))
                if "inc1" in modes:
                    plan.append(("incremental", w, tau, m, 1, 0))
                if "cold" in modes and m != "W":
                    # tau=0.75 cold preprocessing is identical to tau=0.8 (composed)
                    if abs(tau - 0.8) < 1e-9:
                        for r in range(reps_cold):
                            plan.append(("cold", w, tau, m, 12, r))
                if "cold1" in modes and abs(tau - 0.8) < 1e-9 and m == "A" and w in ("c4", "wet"):
                    plan.append(("cold", w, tau, m, 1, 0))
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", default="c4,fw,mix,wet")
    ap.add_argument("--taus", default="0.8,0.75")
    ap.add_argument("--modes", default="inc")
    ap.add_argument("--limit", type=int, default=10 ** 9)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    workloads = [x for x in args.workloads.split(",") if x]
    taus = [float(x) for x in args.taus.split(",") if x]
    modes = set(args.modes.split(","))
    plan = build_plan(workloads, taus, modes, args.tag)
    done = set()
    if os.path.exists(K.TIMING):
        for line in open(K.TIMING):
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass
    n_run = 0
    for mode, w, tau, m, thr, rep in plan:
        key = f"{args.tag}|{w}|t{tau}|{m}|{mode}|thr{thr}|r{rep}"
        if key in done:
            continue
        if n_run >= args.limit:
            break
        for _ in range(30):
            used = cgroup_used_gib()
            if used is None or used < MEM_GUARD_GIB:
                break
            print(f"memory guard: cgroup {used:.1f} GiB, waiting", flush=True)
            time.sleep(20)
        cmd = [PY, DRIVER, "--workload", w, "--tau", str(tau), "--method", m,
               "--mode", mode, "--threads", str(thr), "--repeat", str(rep),
               "--tag", args.tag]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        n_run += 1
        if p.returncode != 0:
            K.append_row(os.path.join(K.CORE, "timing_errors.jsonl"),
                         dict(key=key, returncode=p.returncode,
                              stderr=p.stderr[-3000:], cmd=cmd))
            print(f"FAILED {key} rc={p.returncode}: {p.stderr.strip().splitlines()[-1:]}",
                  flush=True)
        else:
            print(f"ok {key} {time.time()-t0:.1f}s", flush=True)
    print(f"campaign finished: {n_run} runs", flush=True)


if __name__ == "__main__":
    main()
