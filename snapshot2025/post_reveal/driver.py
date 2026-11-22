"""Campaign bookkeeping and isolated access to the frozen modules."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
PROJECT = CODE.parent
W = PROJECT.parent
RESULTS = PROJECT / "results_post_reveal"
TAUS = (0.8, 0.75)
KEYS = ("shingle_set", "raw_text")
LIMIT = 24 * 1024**3

# Set before importing any extension or frozen module, including in subprocesses.
sys.dont_write_bytecode = True
os.environ.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="12", NUMBA_NUM_THREADS="12",
                  OPENBLAS_NUM_THREADS="12", MKL_NUM_THREADS="12", BLIS_NUM_THREADS="12",
                  NUMEXPR_NUM_THREADS="12", CUDA_VISIBLE_DEVICES="")
sys.path[:0] = [str(CODE / "newsnap"), str(CODE / "run")]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


class Campaign:
    def __init__(self, experiment, directory, smoke):
        self.experiment = experiment
        self.dir = Path(directory)
        self.smoke = smoke
        self.source = self.dir / "fixture" if smoke else PROJECT / "newsnap"
        self.snapshot = self.source / "cache/snapshot_spacy.npz"
        self.hashes = (self.source if smoke else PROJECT / "inputs") / "doc_text_xxh64.npy"
        self.work = self.dir / "work"
        self.work.mkdir(exist_ok=True)
        self.logs = self.dir / "logs"
        self.logs.mkdir(exist_ok=True)
        self.rows = []
        self.report = []
        self.gate = self.dir / "gate.json" if smoke else RESULTS / "accepted.json"

    def fingerprint(self):
        files = [self.snapshot, self.hashes, *sorted((self.source / "corpus").glob("shard-*.parquet"))]
        return {str(p): [p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino] for p in files}

    def accepted(self):
        require(self.gate.exists(), "Run E1 first to reproduce the frozen quality rows and masks")
        gate = read(self.gate)
        require(gate["inputs"] == self.fingerprint(), "Inputs changed after acceptance. Rerun E1.")
        return Path(gate["masks"])

    def configure(self):
        # env_paths.prepared() has an import-time mkdir. Redirect before importing CL.
        os.environ["NUMBA_CACHE_DIR"] = str((self.dir.parent if self.smoke else RESULTS) / "numba")
        import numba
        import pyarrow as pa
        import snap_common as S
        pa.set_cpu_count(12)
        pa.set_io_thread_count(12)
        S.CACHE = str(self.work)
        S.SNAP = str(self.work)
        S.RUNS = str(self.work / "quality.jsonl")
        S.CORPUS = str(self.source / "corpus")
        S.SHINGLE_NPZ = str(self.snapshot)
        import collapsed_lib as CL
        CL.K.set_threads(12)
        require(numba.get_num_threads() == 12, "Numba did not enable the frozen 12 threads")
        return CL

    def record(self, row):
        import snap_common as S
        row = dict(experiment=self.experiment, method=None, tau=None, seed=None,
                   mode=None, repeat=0, collapse_key=None, smoke=self.smoke,
                   threads=12, ts_utc=S.now(), hw=S.K.hw_facts()) | row
        row["key"] = "|".join(map(str, (self.dir.name, self.experiment, row["mode"],
                                        row["method"], row["tau"], row["seed"],
                                        row["collapse_key"], row["repeat"])))
        S.append_row(str(RESULTS / "post_reveal.jsonl"), row)
        self.rows.append(row)
        return row

    def job(self, action, *, allow_memory_failure=False, **params):
        """Run exactly one worker. Keep log, exit status and 0.5-s process-tree samples."""
        tag = f"{len(self.rows):03d}-{action}"
        log = self.logs / (tag + ".log")
        result = self.logs / (tag + ".json")
        args = dict(action=action, result=str(result), **params)
        cmd = [sys.executable, "-B", str(CODE / "run/run_timed_tree.py"),
               "--label", tag, "--out", str(self.logs / "tree.jsonl"), "--log", str(log),
               "--mem-gb", "24", "--", sys.executable, "-B", sys.argv[0],
               "--worker", str(self.dir), json.dumps(args)]
        if self.smoke:
            cmd.append("--smoke")
        print(f"{self.experiment} {action} {params}", flush=True)
        run = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        tree = json.loads(run.stdout.strip().splitlines()[-1])
        base = dict(process_wall_s=tree["wall_s"], peak_tree_rss_mb=tree["peak_tree_rss_mb"],
                    log=str(log), exit_code=tree["exit_code"], rlimit_as_gib=24,
                    mode=action) | {k: v for k, v in params.items() if k != "samples"}
        if run.returncode:
            lines = log.read_text(errors="replace").splitlines()
            stage = next((s[6:] for s in reversed(lines) if s.startswith("STAGE ")), action)
            memory = any("MemoryError" in s or "bad_alloc" in s or "Cannot allocate memory" in s
                         for s in lines[-20:])
            partial = {}
            for line in lines:
                if line.startswith("ELAPSED "):
                    name, seconds = line.split()[1:3]
                    partial[f"stage_{name}_s"] = float(seconds)
            signal = -tree["exit_code"] if tree["exit_code"] < 0 else None
            status = "memory_failure" if memory else "signal_failure" if signal else "failure"
            failure = self.record(base | partial | dict(status=status, signal=signal,
                                               wall_s=tree["wall_s"], failed_stage=stage,
                                               detail="\n".join(lines[-8:])))
            if allow_memory_failure and (memory or signal):
                return [failure]
            raise RuntimeError(f"{action} failed at {stage}. See {log}\n{failure['detail']}")
        return [self.record(base | dict(status="ok", wall_s=tree["wall_s"]) | row)
                for row in read(result)]

    def summary(self, members, **data):
        return self.record(dict(status="ok", mode="summary", wall_s=None, peak_tree_rss_mb=None,
                                source_keys=[r["key"] for r in members]) | data)


def entry(experiment, campaign, workers):
    ap = argparse.ArgumentParser(description=campaign.__doc__)
    ap.add_argument("--smoke", action="store_true", help="run on a new 3000-document fixture")
    ap.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    ns = ap.parse_args()
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = min([LIMIT] + [v for v in (soft, hard) if v != resource.RLIM_INFINITY])
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    RESULTS.mkdir(exist_ok=True)
    if ns.worker:
        directory, arg = ns.worker
        c = Campaign(experiment, directory, ns.smoke)
        cl = c.configure()
        params = json.loads(arg)
        result = Path(params.pop("result"))
        action = params.pop("action")
        start = time.perf_counter()
        rows = workers[action](c, cl, **params)
        elapsed = time.perf_counter() - start
        for row in rows:
            if "wall_s" not in row:
                row["wall_s"] = elapsed
                row[f"stage_{action}_s"] = elapsed
                row["measurement_scope"] = "shared worker" if len(rows) > 1 else "worker"
        save(result, rows)
        return

    parent = W / "smoke" if ns.smoke else RESULTS / "campaigns"
    parent.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"{experiment}-", dir=parent))
    c = Campaign(experiment, directory, ns.smoke)
    c.configure()
    if experiment == "E1" and not ns.smoke:
        c.gate.unlink(missing_ok=True)
    if ns.smoke:
        c.job("fixture")
    from importlib.metadata import version
    versions = [f"{name}=={version(name)}" for name in
                ("numpy", "pyarrow", "datatrove", "spacy", "numba", "xxhash")]
    (c.dir / "environment.txt").write_text("\n".join([sys.version, *versions]) + "\n")
    campaign(c)
    fragment = "\n".join(c.report) + "\n"
    (c.dir / "report.md").write_text(fragment)
    if not c.smoke:
        # Commands run sequentially, in E1, E2, E3 order.
        parts = RESULTS / "reports"
        parts.mkdir(exist_ok=True)
        (parts / f"{experiment}.md").write_text(fragment)
        report = RESULTS / "POST_REVEAL_RESULTS.md"
        report.write_text("\n".join(p.read_text() for p in sorted(parts.glob("E*.md"))))
    print(f"PASS {experiment} ({'synthetic' if c.smoke else 'snapshot'}), {len(c.rows)} rows, {c.dir}")
    print(fragment)


class Stages:
    def __init__(self):
        self.times = {}
        self.start = time.perf_counter()

    def run(self, name, fn, *args):
        print("STAGE " + name, flush=True)
        t = time.perf_counter()
        try:
            return fn(*args)
        finally:
            elapsed = time.perf_counter() - t
            self.times[f"stage_{name}_s"] = elapsed
            print(f"ELAPSED {name} {elapsed}", flush=True)
