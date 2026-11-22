#!/usr/bin/env python3
"""Extract English documents from downloaded CC-MAIN-2025-30 WET segments, in parallel.

For every segment whose download is verified (wet/<rank>.warc.wet.gz.ok):
  1. run the project's original extractor, dedupjoin-pvldb/code/wet_extract.py, unmodified
     (one WARC conversion record = one document, fastText lid.176 language id);
  2. keep rows with lang == 'en' and lang_score >= 0.8 (the frozen PREREG rule);
  3. check the English count against the frozen manifest;
  4. write eng/<rank>.parquet and delete the all-language intermediate.
Loops until all 104 segments are done. Usage: env/bin/python extract_driver.py [workers]
"""
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow.compute as pc
import pyarrow.parquet as pq

W = "/path/to/home/dedupjoin-postreveal"
PY = os.path.join(W, "env", "bin", "python")
EXTRACTOR = os.path.join(W, "dedupjoin-pvldb", "code", "wet_extract.py")
LID = os.path.join(W, "tools", "lid.176.bin")
MANIFEST = os.path.join(W, "dedupjoin-pvldb-v3", "experiment", "data", "newsnap", "manifest.json")


def one(seg):
    rank = f"{seg['rank']:03d}"
    t0 = time.time()
    tmp = os.path.join(W, "seg_all", rank)
    os.makedirs(tmp, exist_ok=True)
    spec = os.path.join(tmp, "segments.json")
    json.dump([{"name": seg["segment"], "path": os.path.join(W, "wet", f"{rank}.warc.wet.gz")}],
              open(spec, "w"))
    r = subprocess.run([PY, EXTRACTOR, "--out", tmp, "--segments", spec, "--lang-model", LID],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return dict(rank=rank, status="extract_failed", stderr=r.stderr[-2000:])
    t = pq.read_table(os.path.join(tmp, f"wet-{seg['segment']}.parquet"))
    stats = [json.loads(l) for l in open(os.path.join(tmp, "extract_stats.jsonl"))][0]
    keep = pc.and_(pc.equal(t["lang"], "en"), pc.greater_equal(t["lang_score"], 0.8))
    e = t.filter(keep)
    out = os.path.join(W, "eng", f"{rank}.parquet")
    pq.write_table(e, out + ".tmp", compression="zstd")
    os.replace(out + ".tmp", out)
    res = dict(rank=rank, segment=seg["segment"], status="ok",
               records=stats["records"], records_manifest=seg["records"],
               english=e.num_rows, english_manifest=seg["english_docs"],
               counts_match=(e.num_rows == seg["english_docs"] and stats["records"] == seg["records"]),
               secs=round(time.time() - t0, 1))
    shutil.rmtree(tmp)
    return res


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    segs = sorted(json.load(open(MANIFEST))["segments"], key=lambda s: s["rank"])
    log = open(os.path.join(W, "logs", "extract_status.jsonl"), "a")
    done = {f.split(".")[0] for f in os.listdir(os.path.join(W, "eng")) if f.endswith(".parquet")}
    pending = [s for s in segs if f"{s['rank']:03d}" not in done]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {}
        while pending or futs:
            ready = [s for s in pending
                     if os.path.exists(os.path.join(W, "wet", f"{s['rank']:03d}.warc.wet.gz.ok"))]
            for s in ready[: max(0, workers - len(futs))]:
                futs[ex.submit(one, s)] = s
                pending.remove(s)
            if not futs:
                time.sleep(10)
                continue
            fin = [f for f in futs if f.done()]
            if not fin:
                time.sleep(2)
                continue
            for f in fin:
                futs.pop(f)
                try:
                    res = f.result()
                except Exception as exc:  # noqa: BLE001
                    res = dict(status="exception", error=repr(exc))
                log.write(json.dumps(res) + "\n")
                log.flush()
                print(json.dumps(res), flush=True)
    print("EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
