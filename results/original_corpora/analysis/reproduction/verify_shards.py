#!/usr/bin/env python3
"""Recompute sha256 of every pool file listed in the data manifest and compare."""
import hashlib, json, os, sys
manifest = json.load(open(sys.argv[1]))
root = sys.argv[2]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
bad = []; ok = 0; nbytes = 0
for item in manifest["pool"]["files"]:
    rel = item["path"]
    p = os.path.join(root, rel)
    if not os.path.exists(p):
        bad.append((rel, "missing")); continue
    got = sha(p); nbytes += os.path.getsize(p)
    if got == item["sha256"]:
        ok += 1
    else:
        bad.append((rel, got, item["sha256"]))
print(json.dumps({"checked": ok, "mismatch": bad, "bytes": nbytes}, indent=1))
