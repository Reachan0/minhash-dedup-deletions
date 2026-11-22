#!/usr/bin/env python3
"""Verify the on-disk C4 / FineWeb samples against the Hugging Face repository metadata
(LFS sha256 = lfs.oid, file size) and the data contract, and write
results/data_manifest.json with provenance labels.

Sources (data contract, experiment/data_contract.md):
  FineWeb  sample-10BT 000_00000.parquet   HF: HuggingFaceFW/fineweb
  C4 en    c4-train.00000-of-01024.json.gz HF: allenai/c4
  C4 en    c4-train.00001-of-01024.json.gz HF: allenai/c4
"""
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "output", "dedupjoin-pvldb", "results")
DATA_R2 = os.path.join(ROOT, ".work", "data", "C03", "r2")

SOURCES = [
    dict(name="fineweb_sample10bt_000_00000", repo="HuggingFaceFW/fineweb",
         path="sample/10BT/000_00000.parquet",
         local=os.path.join(DATA_R2, "raw_fw", "000_00000.parquet"),
         used_as="FineWeb corpus (first n rows), n up to 1,000,000"),
    dict(name="c4_en_train_00000", repo="allenai/c4",
         path="en/c4-train.00000-of-01024.json.gz",
         local=os.path.join(DATA_R2, "raw_c4", "c4-train.00000-of-01024.json.gz"),
         used_as="C4 corpus shard 0 (356,317 lines)"),
    dict(name="c4_en_train_00001", repo="allenai/c4",
         path="en/c4-train.00001-of-01024.json.gz",
         local=os.path.join(DATA_R2, "raw_c4", "c4-train.00001-of-01024.json.gz"),
         used_as="C4 corpus shard 1 (lines 356,317..699,999 for n=700,000)"),
]


def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hf_api(url):
    r = subprocess.run(["curl", "-s", "-m", "60", url], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        return json.loads(r.stdout)
    except Exception:
        return None


def hf_rev(repo):
    d = hf_api(f"https://huggingface.co/api/datasets/{repo}")
    return (d or {}).get("sha")


def hf_tree(repo, directory=""):
    u = f"https://huggingface.co/api/datasets/{repo}/tree/main"
    if directory:
        u += f"/{directory}"
    return hf_api(u) or []


def main():
    items = []
    for s in SOURCES:
        it = dict(s)
        it["exists"] = os.path.exists(s["local"])
        if not it["exists"]:
            it["status"] = "MISSING"
            items.append(it)
            continue
        it["size_bytes"] = os.path.getsize(s["local"])
        it["sha256_local"] = sha256_file(s["local"])
        d = s["path"].rsplit("/", 1)[0]
        tree = hf_tree(s["repo"], d)
        ent = next((e for e in tree if e.get("path") == s["path"]), None)
        it["repo_revision_head"] = hf_rev(s["repo"])
        if ent:
            it["sha256_hf_lfs"] = (ent.get("lfs") or {}).get("oid")
            it["size_bytes_hf"] = ent.get("size")
            it["hf_commit_oid"] = ent.get("oid")
            it["sha256_match"] = it["sha256_local"] == it["sha256_hf_lfs"]
            it["size_match"] = it["size_bytes"] == it["size_bytes_hf"]
            it["status"] = "verified" if (it["sha256_match"] and it["size_match"]) else "MISMATCH"
        else:
            it["status"] = "hf_metadata_unavailable"
        items.append(it)
        print(it["name"], it["status"], it["size_bytes"], it["sha256_local"], flush=True)
    doc = dict(version=1,
               generated_at=__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
               source_contract=".work/vldb2027-ai-db-ideas/experiment/data_contract.md",
               provenance="independently rerun (this run): local sha256 computed now; compared with the "
                          "Hugging Face LFS object id (sha256) of the same path",
               items=items)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "data_manifest.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote", os.path.join(OUT, "data_manifest.json"))
    if any(i.get("status") not in ("verified", "hf_metadata_unavailable") for i in items):
        sys.exit(1)


if __name__ == "__main__":
    main()
