#!/usr/bin/env python3
"""Record the exact datatrove build used by parity_harness.py: distribution version,
installer, the installed minhash.py hash, and interpreter/library versions. There is no
VCS checkout (installed wheel), so the file hash is the pin.

Updates results/parity.json in place.
"""
import hashlib
import json
import os
import sys

import importlib.metadata as md

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
P = os.path.join(ROOT, "output/dedupjoin-pvldb/results/parity.json")


def main():
    d = md.distribution("datatrove")
    files = [f for f in (d.files or []) if str(f).endswith("datatrove/pipeline/dedup/minhash.py")]
    src = os.path.join(d.locate_file(files[0])) if files else None
    sha = hashlib.sha256(open(src, "rb").read()).hexdigest() if src and os.path.exists(src) else None
    import numpy
    import xxhash
    doc = json.load(open(P))
    doc["datatrove_pin"] = dict(
        distribution="datatrove", version=d.version, installer=d.read_text("INSTALLER") or "pip",
        direct_url=d.read_text("direct_url.json"), vcs_commit=None,
        vcs_note="installed from a wheel, not a git checkout; the installed minhash.py hash is the pin",
        minhash_py_sha256=sha, Python=sys.version.split()[0],
        numpy=numpy.__version__, xxhash=xxhash.VERSION,
        provenance="independently rerun (this run)")
    with open(P, "w") as f:
        json.dump(doc, f, indent=2)
    print(json.dumps(doc["datatrove_pin"], indent=1))


if __name__ == "__main__":
    main()
