#!/usr/bin/env python3
"""Publish this step's code, data, logs and analysis into output/<slug>/experiment/."""
import hashlib
import json
import os
import shutil

CORE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))
CODE = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
SLUG = os.path.abspath(os.path.join(CORE, ".."))
ROOT = os.path.abspath(os.path.join(SLUG, "..", ".."))
OUT = os.path.join(ROOT, "output", "dedupjoin-pvldb-v3", "experiment")


def cp(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    manifest = []
    # code
    code_files = [f for f in os.listdir(CODE) if f.endswith((".py", ".sh"))]
    for f in sorted(code_files):
        manifest.append(cp(os.path.join(CODE, f),
                           os.path.join(OUT, "code", "core", f)))
    for sub in ("diagnostics", "vendor", "reference"):
        for f in sorted(os.listdir(os.path.join(CODE, sub))):
            if f.endswith((".py", ".md")):
                manifest.append(cp(os.path.join(CODE, sub, f),
                                   os.path.join(OUT, "code", "core", sub, f)))
    manifest.append(cp(os.path.join(CODE, "figures", "make_figures.py"),
                       os.path.join(OUT, "code", "core", "figures", "make_figures.py")))
    manifest.append(cp(os.path.join(CODE, "SHA256SUMS"),
                       os.path.join(OUT, "code", "core", "SHA256SUMS")))
    manifest.append(cp(os.path.join(SLUG, "code", "COPY_MANIFEST.md"),
                       os.path.join(OUT, "code", "COPY_MANIFEST.md")))
    # data
    for f in ("runs.jsonl", "timing.jsonl", "bucket_scaling.jsonl",
              "bucket_scaling.failed.jsonl", "main_decision_table.csv",
              "main_decision_table.json", "main_decision_table.tex",
              "set_metrics.json", "representative_rules.jsonl", "workloads.json",
              "crosscheck_bruteforce.json", "crosscheck_datatrove.json"):
        manifest.append(cp(os.path.join(CORE, f), os.path.join(OUT, "data", f)))
    # logs
    for f in sorted(os.listdir(CORE)):
        if f.endswith(".log"):
            manifest.append(cp(os.path.join(CORE, f), os.path.join(OUT, "logs", f)))
    # analysis
    for f in ("decision_map.md",):
        manifest.append(cp(os.path.join(CORE, f), os.path.join(OUT, "analysis", f)))
    manifest.append(cp(os.path.join(CORE, "figures", "figure_contract.md"),
                       os.path.join(OUT, "analysis", "figure_contract.md")))
    # checksums of everything published
    rel = []
    for p in manifest:
        rel.append((os.path.relpath(p, OUT), sha(p)))
    with open(os.path.join(OUT, "SHA256SUMS"), "w") as f:
        for r, h in rel:
            f.write(f"{h}  {r}\n")
    json.dump({"published": [{"path": r, "sha256": h} for r, h in rel]},
              open(os.path.join(OUT, "PUBLISH_MANIFEST.json"), "w"), indent=1)
    print(f"published {len(rel)} files under {OUT}")


if __name__ == "__main__":
    main()
