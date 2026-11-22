#!/usr/bin/env python3
"""Publish this step's artifacts into output/dedupjoin-pvldb-v3/experiment/.

Copies data, analysis documents, scripts and logs; writes a manifest with sha256,
source path and the command that produced each data file.
"""
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
EXTRA = os.path.join(ROOT, ".work", "dedupjoin-pvldb-v3", "extra")
OUT = os.path.join(ROOT, "output", "dedupjoin-pvldb-v3", "experiment")

FILES = {
    "data/baseline_rows.jsonl": (os.path.join(EXTRA, "baseline_rows.jsonl"),
                                 "code/extra/run_bigcode.py --workload {c4,fw} --tau {0.8,0.75} --seed 42"),
    "data/baseline_comparison.json": (os.path.join(EXTRA, "baseline_comparison.json"),
                                      "code/extra/analyze_baseline.py"),
    "data/bucket_stress.jsonl": (os.path.join(EXTRA, "bucket_stress.jsonl"),
                                 "code/extra/run_stress_campaign.py --mode wall"),
    "data/bucket_mitigation.jsonl": (os.path.join(EXTRA, "bucket_mitigation.jsonl"),
                                     "code/extra/run_bucket_stress.py mitigation --n 4000 --groups {2,400} --cap 500"),
    "data/gpu_decision.json": (os.path.join(EXTRA, "gpu_decision.json"),
                               "code/extra/make_gpu_memo.py"),
    "analysis/baseline_bigcode.md": (os.path.join(OUT, "analysis", "baseline_bigcode.md"),
                                     "written from baseline_rows.jsonl / baseline_comparison.json"),
    "analysis/bucket_scaling_v3.md": (os.path.join(OUT, "analysis", "bucket_scaling_v3.md"),
                                      "written from bucket_stress.jsonl / core/bucket_scaling.jsonl"),
    "data/baseline_comparison.md": (os.path.join(EXTRA, "baseline_comparison.md"),
                                    "code/extra/analyze_baseline.py"),
    "analysis/gpu_decision.md": (os.path.join(EXTRA, "gpu_decision.md"),
                                 "written from gpu_decision.json"),
}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    os.makedirs(os.path.join(OUT, "code", "extra"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "analysis"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    manifest = {"generated_utc": __import__("time").strftime(
                    "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
                "data": {}, "code": {}, "logs": {}}
    for rel, (src, cmd) in FILES.items():
        if not os.path.exists(src):
            raise SystemExit(f"missing {src}")
        dst = os.path.join(OUT, rel)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
        manifest["data"][rel] = dict(sha256=sha(dst), bytes=os.path.getsize(dst),
                                     produced_by=cmd)

    for name in ("bigcode_lib.py", "run_bigcode.py", "analyze_baseline.py",
                 "run_bucket_stress.py", "run_stress_campaign.py", "make_gpu_memo.py",
                 "build_annotation_package.py", "publish_extra.py",
                 "extra_bigcode_minhash.py"):
        src = os.path.join(HERE, name)
        if name == "extra_bigcode_minhash.py":
            src = os.path.join(os.path.dirname(HERE), name)
        dst = os.path.join(OUT, "code", "extra", name)
        shutil.copy2(src, dst)
        manifest["code"][f"code/extra/{name}"] = dict(
            sha256=sha(dst),
            note=("upstream BigCode text-dedup minhash_dedup.py snapshot; author Chenghao Mou; "
                  "kept for semantics audit" if name == "extra_bigcode_minhash.py" else ""))

    for name in ("bigcode_c4_t0.8_s42.log", "bigcode_rest.log", "analyze_baseline.log",
                 "stress_wall.log", "stress_wall_split.log"):
        src = os.path.join(EXTRA, "logs", name)
        if os.path.exists(src):
            dst = os.path.join(OUT, "logs", name)
            shutil.copy2(src, dst)
            manifest["logs"][f"logs/{name}"] = dict(sha256=sha(dst),
                                                    bytes=os.path.getsize(dst))

    with open(os.path.join(OUT, "publication_manifest_extra.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(json.dumps(manifest, ensure_ascii=False, indent=1)[:1200])


if __name__ == "__main__":
    main()
