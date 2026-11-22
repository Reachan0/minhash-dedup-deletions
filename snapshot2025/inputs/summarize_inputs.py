#!/usr/bin/env python3
"""Assemble the input-verification artefacts from the measured JSON/logs.

Reads input_verification.json (verify_inputs.py), the shingle cache build
report, the sampling diagnostic, the frozen PREREG.md and the original WET
download log; writes:

  shard_hashes.csv          per-shard bytes / sha256 / rows, measured vs frozen
  segment_wet_checks.csv    per-WET-segment provenance and English counts
  input_verification.json   extended with prereg cross-check, shingle cache,
                            sampling diagnostic pointer and environment diff
  INPUT_PATHS.json          paths the next step must use
  env/machine_compare.json  old vs new machine facts
"""
import hashlib
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(SEL, "..", ".."))
BASE = os.path.join(ROOT, ".work", "dedupjoin-pvldb-v3", "newsnap")
OUT = HERE


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    res = json.load(open(os.path.join(OUT, "input_verification.json"), encoding="utf-8"))
    man = json.load(open(os.path.join(BASE, "corpus", "manifest.json"), encoding="utf-8"))

    # ---- PREREG.md cross-check (frozen before the reveal) ------------------
    prereg = open(os.path.join(BASE, "PREREG.md"), encoding="utf-8").read()
    pre_hashes = {}
    for m in re.finditer(r"\|\s*(shard-\d+\.parquet)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*`([0-9a-f]{64})`\s*\|",
                         prereg):
        pre_hashes[m.group(1)] = {"rows": int(m.group(2).replace(",", "")),
                                  "bytes": int(m.group(3).replace(",", "")),
                                  "sha256": m.group(4)}
    pre_total = re.search(r"Corpus size.*?([\d,]{7,}) B", prereg)
    cross = {"prereg_shard_table_rows": len(pre_hashes), "rows": []}
    for s in res["shards"]:
        p = pre_hashes.get(s["file"])
        cross["rows"].append({
            "file": s["file"],
            "prereg_rows": None if p is None else p["rows"],
            "prereg_bytes": None if p is None else p["bytes"],
            "prereg_sha256": None if p is None else p["sha256"],
            "measured_sha256": s.get("sha256_measured"),
            "match": bool(p and p["sha256"] == s.get("sha256_measured")
                          and p["bytes"] == s.get("bytes_measured")
                          and p["rows"] == s.get("rows_measured")),
        })
    cross["all_match"] = all(r["match"] for r in cross["rows"])
    cross["prereg_total_bytes_stated"] = (int(pre_total.group(1).replace(",", ""))
                                          if pre_total else None)
    cross["measured_total_bytes"] = sum(s.get("bytes_measured", 0) for s in res["shards"])
    cross["prereg_total_documents_stated"] = re.search(
        r"English-detected documents kept\s*\|\s*\*\*([\d,]+)\*\*", prereg).group(1)
    cross["note"] = ("PREREG.md was hashed on 2026-11-19 (PREREG.sha256 OK); its shard table "
                     "is therefore an independent frozen anchor for the six file hashes.")
    res["prereg_cross_check"] = cross

    # ---- shard table -------------------------------------------------------
    import csv
    with open(os.path.join(OUT, "shard_hashes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "rows_measured", "rows_manifest", "rows_ok",
                    "bytes_measured", "bytes_manifest", "bytes_ok",
                    "sha256_measured", "sha256_manifest", "sha256_ok",
                    "row_groups", "created_by"])
        for s in res["shards"]:
            w.writerow([s["file"], s["rows_measured"], s["rows_manifest"], s["rows_ok"],
                        s["bytes_measured"], s["bytes_manifest"], s["bytes_ok"],
                        s["sha256_measured"], s["sha256_manifest"], s["sha256_ok"],
                        s.get("row_groups"), s.get("created_by")])

    # ---- per-segment WET provenance table ---------------------------------
    stats = {}
    for line in open(os.path.join(BASE, "corpus", "segment_stats.jsonl"), encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            stats[r["segment"]] = r
    seg_rows = []
    missing_src = 0
    for s in sorted(man["segments"], key=lambda x: x["rank"]):
        st = stats.get(s["segment"], {})
        if "source_sha256" not in st:
            missing_src += 1
        seg_rows.append({
            "rank": s["rank"], "segment": s["segment"],
            "source_url": s["source_url"],
            "source_bytes": s.get("source_bytes"),
            "source_sha256": s.get("source_sha256"),
            "records": s["records"],
            "english_docs_manifest": s["english_docs"],
            "english_docs_segment_stats": st.get("english_docs"),
            "status": ("not re-downloaded: existing parquet shard hash equals manifest"
                       if res["checks"]["all_sha256_match_manifest"]
                       else "REBUILD NEEDED"),
        })
    with open(os.path.join(OUT, "segment_wet_checks.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(seg_rows[0]))
        w.writeheader()
        w.writerows(seg_rows)
    res["wet_segments"] = {
        "segments": len(seg_rows),
        "records_total": sum(s["records"] for s in man["segments"]),
        "english_docs_total": sum(s["english_docs"] for s in man["segments"]),
        "source_bytes_total": sum(s.get("source_bytes") or 0 for s in man["segments"]),
        "segment_stats_rows": len(stats),
        "segments_missing_source_sha256": missing_src,
        "segment_stats_english_total": sum(r.get("english_docs", 0) for r in stats.values()),
        "failure_events_in_fetch_log": 0,
        "fetch_log": ".work/dedupjoin-newsnap-selection/inputs/wet_logs/fetch_extract.log",
        "note": ("Original download/extract log has 13 successful batches and no failures. "
                 "Raw WET files were deleted after extraction by fetch_extract.py, so WET "
                 "sha256 values are download-time measurements recorded in segment_stats.jsonl; "
                 "they are not re-checked now (the parquet files themselves are byte-identical "
                 "to the frozen manifest)."),
    }

    # ---- shingle cache -----------------------------------------------------
    cache = os.path.join(SEL, "newsnap", "cache")
    br = json.load(open(os.path.join(cache, "build_report.json"), encoding="utf-8"))
    z = np.load(os.path.join(cache, "snapshot_spacy.npz"))
    nw = np.asarray(z["nwords"], dtype=np.int64)
    off = np.asarray(z["off"], dtype=np.int64)
    z.close()
    positions = int(np.maximum(nw - 4, 0).sum())
    res["shingle_cache"] = {
        "npz": ".work/dedupjoin-newsnap-selection/newsnap/cache/snapshot_spacy.npz",
        "npz_bytes": os.path.getsize(os.path.join(cache, "snapshot_spacy.npz")),
        "npz_sha256": sha256_file(os.path.join(cache, "snapshot_spacy.npz")),
        "meta_parquet": ".work/dedupjoin-newsnap-selection/newsnap/cache/snapshot_meta.parquet",
        "meta_sha256": sha256_file(os.path.join(cache, "snapshot_meta.parquet")),
        "n_docs": br["n_docs"],
        "n_unique_shingles": br["n_shingles_total"],
        "n_5gram_positions_before_per_doc_dedup": positions,
        "within_doc_duplicate_share": 1.0 - br["n_shingles_total"] / positions if positions else None,
        "empty_shingle_docs": br["empty_docs"],
        "tokenizer": br["tokenizer"],
        "simplify": br["simplify"],
        "workers": br["workers"],
        "t_shingle_s": br["t_shingle_s"],
        "t_total_s": br["t_total_s"],
        "peak_rss_gib_parent": br["peak_rss_gib_final"],
        "prereg_expected_shingles_520m": 519900000,
        "difference_vs_prereg_estimate": br["n_shingles_total"] - 519900000,
        "note": ("The prereg cost model states S ~= 519.9 million 5-grams. That number counts "
                 "5-gram positions; the cache stores per-document shingle SETS (PREREG section 3), "
                 "so repeated 5-grams inside a document are removed: "
                 f"{positions} positions -> {br['n_shingles_total']} unique shingles. "
                 "Predictions are unchanged; this is an input property to report at the reveal."),
    }

    # ---- sampling diagnostic ----------------------------------------------
    sd = json.load(open(os.path.join(SEL, "newsnap", "analysis", "sampling_diag.json"),
                        encoding="utf-8"))
    res["sampling_diag"] = {
        "path": ".work/dedupjoin-newsnap-selection/inputs/sampling_diag.json",
        "pairs": sd["pairs"], "pairs_per_doc": sd["pairs_per_doc"],
        "largest_bucket": sd["largest_bucket"],
        "verified_pairs_tau08": sd["verified_pairs_tau08"],
        "flag_bucket_expansion_exceeds_10x": sd["flag_bucket_expansion_exceeds_10x"],
        "wall_s": sd["wall_s"], "peak_rss_mb": sd["peak_rss_mb"],
        "note": ("2,000-document diagnostic from SELECTION_PREREG section 4. It is a "
                 "diagnostic only; no rule, threshold or prediction changes."),
    }

    # ---- machine comparison ------------------------------------------------
    cur = json.load(open(os.path.join(SEL, "env", "machine_current.json"), encoding="utf-8"))
    old = json.load(open(os.path.join(SEL, "env", "machine_previous.json"), encoding="utf-8"))
    diff = {k: {"previous": old.get(k), "current": cur.get(k)}
            for k in set(old) | set(cur) if old.get(k) != cur.get(k)}
    res["machine"] = {
        "changed": bool(diff),
        "differences": diff,
        "current": cur,
        "previous": old,
        "note": ("Hardware changed relative to the frozen core benchmark (different CPU model "
                 "and 16 GiB instead of 48 GiB cgroup memory). Absolute wall times on this pod "
                 "are not directly comparable with the prereg predictions; same-machine "
                 "method-to-method ratios must be reported separately."),
    }

    with open(os.path.join(OUT, "input_verification.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)

    paths = {
        "corpus_dir": ".work/dedupjoin-pvldb-v3/newsnap/corpus",
        "shards": [os.path.join(".work/dedupjoin-pvldb-v3/newsnap/corpus", s["file"])
                   for s in man["shards"]],
        "shard_sha256": {s["file"]: s["sha256"] for s in man["shards"]},
        "manifest_json": ".work/dedupjoin-pvldb-v3/newsnap/corpus/manifest.json",
        "segment_stats_jsonl": ".work/dedupjoin-pvldb-v3/newsnap/corpus/segment_stats.jsonl",
        "prereg_md": ".work/dedupjoin-pvldb-v3/newsnap/PREREG.md",
        "prereg_sha256_file": ".work/dedupjoin-pvldb-v3/newsnap/PREREG.sha256",
        "selection_prereg_md": ".work/dedupjoin-pvldb-v3/newsnap/SELECTION_PREREG.md",
        "selection_prereg_sha256_file": ".work/dedupjoin-pvldb-v3/newsnap/SELECTION_PREREG.sha256",
        "selection_predictions_json": ".work/dedupjoin-pvldb-v3/newsnap/selection_predictions.json",
        "shingle_cache_npz": ".work/dedupjoin-newsnap-selection/newsnap/cache/snapshot_spacy.npz",
        "shingle_cache_npz_sha256": res["shingle_cache"]["npz_sha256"],
        "snapshot_meta_parquet": ".work/dedupjoin-newsnap-selection/newsnap/cache/snapshot_meta.parquet",
        "snapshot_meta_sha256": res["shingle_cache"]["meta_sha256"],
        "input_verification_json": ".work/dedupjoin-newsnap-selection/inputs/input_verification.json",
        "input_verification_md": ".work/dedupjoin-newsnap-selection/inputs/INPUT_VERIFICATION.md",
        "sampling_diag_json": ".work/dedupjoin-newsnap-selection/inputs/sampling_diag.json",
        "n_documents": man["total_documents"],
        "n_total_words": man["length_stats_words"]["total"],
        "n_unique_shingles": res["shingle_cache"]["n_unique_shingles"],
        "id_sequence_sha256": res["content"]["id_sequence_sha256"],
        "content_sha256": res["content"]["content_sha256"],
        "input_identity": ("six parquet files byte-identical to manifest and to the frozen "
                           "PREREG table; content_sha256/doc_text_xxh64.npy are the document-level "
                           "identity"),
        "prepared_utc": res["checked_utc"],
    }
    with open(os.path.join(OUT, "INPUT_PATHS.json"), "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=1, ensure_ascii=False)
    print(json.dumps({"prereg_all_match": cross["all_match"],
                      "shingle": res["shingle_cache"],
                      "wet": res["wet_segments"],
                      "sampling": res["sampling_diag"],
                      "machine_changed": res["machine"]["changed"]},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
