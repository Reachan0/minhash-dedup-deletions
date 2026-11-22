#!/usr/bin/env python3
"""B1: per-segment extraction and shingling on AutoDL.

For each downloaded segment (wet/<rank>.ok), in parallel:
  1. run the original extractor dedupjoin-pvldb/code/wet_extract.py unmodified;
  2. keep lang == 'en' and lang_score >= 0.8 (frozen rule) -> eng/<rank>.parquet;
  3. shingle with the frozen definition (snap_common._shingle_batch_spacy: datatrove
     simplify_text + spaCy blank 'en' tokenizer, 5-grams, xxh64, set per document)
     -> cache/<rank>.npz with sizes, flat, nwords.
Ranks 0..103 are also checked against the frozen manifest's English counts.
Usage: env/bin/python b1_segment_worker.py <workers>
"""
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

W = "/path/to/scratch/dedupjoin"
PY = f"{W}/env/bin/python"
EXTRACTOR = f"{W}/dedupjoin-pvldb/code/wet_extract.py"
LID = f"{W}/tools/lid.176.bin"
MANIFEST = f"{W}/dedupjoin-pvldb-v3/experiment/data/newsnap/manifest.json"
sys.path.insert(0, f"{W}/dedupjoin-newsnap-selection/code/newsnap")


def one(rank, name):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import snap_common as S
    t0 = time.time()
    tmp = f"{W}/seg_all/{rank}"
    os.makedirs(tmp, exist_ok=True)
    spec = f"{tmp}/segments.json"
    json.dump([{"name": name, "path": f"{W}/wet/{rank}.warc.wet.gz"}], open(spec, "w"))
    r = subprocess.run([PY, EXTRACTOR, "--out", tmp, "--segments", spec, "--lang-model", LID],
                       capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
    if r.returncode:
        return dict(rank=rank, status="extract_failed", stderr=r.stderr[-1500:])
    t = pq.read_table(f"{tmp}/wet-{name}.parquet")
    records = json.loads(open(f"{tmp}/extract_stats.jsonl").readline())["records"]
    e = t.filter(pc.and_(pc.equal(t["lang"], "en"), pc.greater_equal(t["lang_score"], 0.8)))
    pq.write_table(e, f"{W}/eng/{rank}.parquet.tmp", compression="zstd")
    os.replace(f"{W}/eng/{rank}.parquet.tmp", f"{W}/eng/{rank}.parquet")
    shutil.rmtree(tmp)
    t1 = time.time()
    sizes, flat, nw = S._shingle_batch_spacy(e["text"].to_pylist())
    np.savez(f"{W}/cache/{rank}.npz.tmp.npz", sizes=sizes, flat=flat, nwords=nw)
    os.replace(f"{W}/cache/{rank}.npz.tmp.npz", f"{W}/cache/{rank}.npz")
    return dict(rank=rank, segment=name, status="ok", records=records, english=e.num_rows,
                shingles=int(flat.size), extract_s=round(t1 - t0, 1), shingle_s=round(time.time() - t1, 1))


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    os.makedirs(f"{W}/cache", exist_ok=True)
    plan = [l.rstrip("\n").split("\t") for l in open(f"{W}/b1_segments.tsv")]
    frozen = {f"{s['rank']:04d}": s for s in json.load(open(MANIFEST))["segments"]}
    done = {f[:-4] for f in os.listdir(f"{W}/cache") if f.endswith(".npz") and ".tmp" not in f}
    pending = [(r, name) for r, _, name, _, _ in plan if r not in done]
    log = open(f"{W}/logs/segments.jsonl", "a")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {}
        while pending or futs:
            ready = [p for p in pending if os.path.exists(f"{W}/wet/{p[0]}.ok")]
            for p in ready[: max(0, 2 * workers - len(futs))]:
                futs[ex.submit(one, *p)] = p
                pending.remove(p)
            fin = [f for f in futs if f.done()]
            if not fin:
                time.sleep(5)
                continue
            for f in fin:
                futs.pop(f)
                try:
                    res = f.result()
                except Exception as exc:  # report and continue with the other segments
                    res = dict(status="exception", error=repr(exc))
                fz = frozen.get(res.get("rank"))
                if fz is not None and res.get("status") == "ok":
                    res["frozen_counts_match"] = (res["english"] == fz["english_docs"]
                                                  and res["records"] == fz["records"])
                log.write(json.dumps(res) + "\n")
                log.flush()
    print("B1_SEGMENTS_DONE", flush=True)


if __name__ == "__main__":
    main()
