#!/usr/bin/env python3
"""Run one command, log its output, and record wall time plus peak *process-tree*
RSS sampled from /proc (parent + all descendants every 0.5 s).

Usage: run_timed_tree.py --label <name> --log <file> --out <jsonl> -- cmd...
Each invocation appends one JSON line; failures are recorded with the exit code
and are never deleted. The measured process itself also reports RUSAGE_SELF
(peak_rss_mb) inside its own row; this sidecar adds the tree-scope maximum.
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time


def tree_rss_kb(root):
    rss = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as f:
                parts = f.read().split()
            ppid = int(parts[3])
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss[int(pid)] = (ppid, int(line.split()[1]))
                        break
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    kids = {root}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _) in rss.items():
            if ppid in kids and pid not in kids:
                kids.add(pid)
                changed = True
    return sum(rss[p][1] for p in kids if p in rss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--failed", default=None, help="append non-zero exits here")
    ap.add_argument("--mem-gb", type=float, default=None, help="RLIMIT_AS in GiB")
    ap.add_argument("--timeout", type=float, default=None, help="wall seconds")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pre = None
    if a.mem_gb:
        lim = int(a.mem_gb * (1 << 30))
        pre = lambda: resource.setrlimit(resource.RLIMIT_AS, (lim, lim))  # noqa: E731
    t0 = time.perf_counter()
    timed_out = False
    with open(a.log, "w", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=pre)
        peak = 0
        while p.poll() is None:
            try:
                peak = max(peak, tree_rss_kb(p.pid))
            except Exception:
                pass
            if a.timeout and time.perf_counter() - t0 > a.timeout:
                timed_out = True
                p.terminate()
                time.sleep(3)
                if p.poll() is None:
                    p.kill()
                break
            time.sleep(0.5)
        rc = p.wait()
    wall = time.perf_counter() - t0
    row = dict(label=a.label, cmd=cmd, exit_code=rc, timed_out=timed_out,
               wall_s=round(wall, 3),
               peak_tree_rss_mb=round(peak / 1024.0, 1), log=a.log,
               ts_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if rc != 0 and a.failed:
        try:
            tail = open(a.log, encoding="utf-8", errors="replace").read()[-3000:]
        except Exception:
            tail = ""
        with open(a.failed, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(row, failure_tail=tail)) + "\n")
            f.flush()
            os.fsync(f.fileno())
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(row), flush=True)
    sys.exit(0 if rc == 0 else rc)


if __name__ == "__main__":
    main()
