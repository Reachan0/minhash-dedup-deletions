#!/usr/bin/env python3
"""B1 segment plan: the frozen seeded order of CC-MAIN-2025-30 WET segments, extended past rank 103.

Downloads wet.paths.gz, checks its sha256 against PREREG.md, shuffles indices 0..N-1 with
random.Random(20261119) exactly as the frozen sampling plan did, and writes ranks 0..K-1.
Ranks 0..103 must equal the frozen manifest's segments (the reveal snapshot is the first rung).
Usage: python3 b1_plan.py <workdir> <K> <frozen_manifest.json>
"""
import gzip, hashlib, json, random, sys, urllib.request

W, K, MAN = sys.argv[1], int(sys.argv[2]), sys.argv[3]
URL = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-30/wet.paths.gz"
SHA = "b00996a663900fbb38df54c3dd29be792bdc58cb488401ed00bc59056cf53338"
raw = urllib.request.urlopen(URL, timeout=120).read()
assert hashlib.sha256(raw).hexdigest() == SHA, "wet.paths.gz sha256 differs from PREREG.md"
paths = gzip.decompress(raw).decode().split()
idx = list(range(len(paths)))
random.Random(20261119).shuffle(idx)
man = sorted(json.load(open(MAN))["segments"], key=lambda s: s["rank"])
for s in man:
    assert idx[s["rank"]] == s["index"] and paths[idx[s["rank"]]] == s["source_path"], f"rank {s['rank']} differs"
frozen = {s["rank"]: s for s in man}
with open(f"{W}/b1_segments.tsv", "w") as f:
    for r in range(K):
        p = paths[idx[r]]
        name = p.rsplit("/", 1)[1].replace(".warc.wet.gz", "")
        sha = frozen[r]["source_sha256"] if r in frozen else "-"
        f.write(f"{r:04d}\t{idx[r]}\t{name}\thttps://data.commoncrawl.org/{p}\t{sha}\n")
print(f"segments={len(paths)} plan_ranks={K} frozen_prefix_ok=104")
