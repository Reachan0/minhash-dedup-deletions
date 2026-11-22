#!/usr/bin/env python3
"""Input verification for the CC-MAIN-2025-30 snapshot (step verify-rebuild-inputs).

Reads the six frozen parquet shards in
  .work/dedupjoin-pvldb-v3/newsnap/corpus/
and checks them against corpus/manifest.json:

  * file bytes and sha256 against manifest
  * row count, schema, nulls, empty text
  * language rule (lang == 'en' and lang_score >= 0.8)
  * per-segment English document counts against manifest['segments'] and
    corpus/segment_stats.jsonl
  * text-length statistics against manifest['length_stats_words']
  * document-id sequence: duplicates and an order-sensitive sha256
  * per-document text hash (xxh64 of the UTF-8 text as stored, the PREREG
    "raw_hash" definition) and an order-sensitive aggregate content sha256,
    saved as doc_text_xxh64.npy for downstream identity checks

Everything is streamed batch by batch; the whole text column is never held in
pandas.  Outputs go to this directory (inputs/).  No deduplication metric is
computed here.
"""
import hashlib
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
import xxhash

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BASE = os.path.join(ROOT, ".work", "dedupjoin-pvldb-v3", "newsnap")
CORPUS = os.path.join(BASE, "corpus")
OUT = HERE
BATCH = 5000


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def id_hash_update(h, doc_id):
    b = doc_id.encode("utf-8")
    h.update(len(b).to_bytes(4, "little"))
    h.update(b)


def main():
    t0 = time.time()
    man = json.load(open(os.path.join(CORPUS, "manifest.json")))
    res = {
        "snapshot": man["snapshot"],
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_dir": os.path.relpath(CORPUS, ROOT),
        "manifest_total_documents": man["total_documents"],
        "manifest_total_segments": man["total_segments"],
        "lang_rule": man["lang_rule"],
        "document_unit": man["document_unit"],
        "shards": [],
        "segment_counts": {},
        "checks": {},
        "warnings": [],
    }

    # ---- per-file bytes / sha256 / row count -------------------------------
    for s in man["shards"]:
        p = os.path.join(CORPUS, s["file"])
        rec = {"file": s["file"], "exists": os.path.exists(p)}
        if not rec["exists"]:
            rec.update({"bytes_ok": False, "sha256_ok": False, "rows_ok": False})
            res["shards"].append(rec)
            res["warnings"].append(f"missing shard {s['file']}")
            continue
        pf = pq.ParquetFile(p)
        rec.update({
            "bytes_measured": os.path.getsize(p),
            "bytes_manifest": s["bytes"],
            "bytes_ok": os.path.getsize(p) == s["bytes"],
            "sha256_measured": sha256_file(p),
            "sha256_manifest": s["sha256"],
            "rows_measured": pf.metadata.num_rows,
            "rows_manifest": s["rows"],
            "row_groups": pf.metadata.num_row_groups,
            "created_by": pf.metadata.created_by,
        })
        rec["sha256_ok"] = rec["sha256_measured"] == s["sha256"]
        rec["rows_ok"] = rec["rows_measured"] == s["rows"]
        res["shards"].append(rec)

    # ---- streaming content pass -------------------------------------------
    id_hasher = hashlib.sha256()
    content_hasher = hashlib.sha256()
    per_shard_content = []
    seen_ids = set()
    dup_ids = 0
    total_rows = 0
    total_nwords = 0
    nwords_all = []
    lang_counts = {}
    min_lang_score = float("inf")
    bad_lang = 0
    empty_text = 0
    null_text = 0
    null_id = 0
    nchar_mismatch = 0
    seg_counts = {}
    seg_first_order = []
    seg_seen = set()
    text_hashes = np.empty(man["total_documents"], dtype=np.uint64)
    pos = 0

    for s in man["shards"]:
        p = os.path.join(CORPUS, s["file"])
        if not os.path.exists(p):
            continue
        sh = hashlib.sha256()
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=BATCH,
                                 columns=["id", "segment", "text", "nchar",
                                          "nwords", "lang", "lang_score"]):
            d = b.to_pydict()
            ids, segs, texts = d["id"], d["segment"], d["text"]
            nch, nw, langs, scores = d["nchar"], d["nwords"], d["lang"], d["lang_score"]
            m = len(ids)
            for i in range(m):
                if ids[i] is None:
                    null_id += 1
                    continue
                if ids[i] in seen_ids:
                    dup_ids += 1
                else:
                    seen_ids.add(ids[i])
                id_hash_update(id_hasher, ids[i])
                if texts[i] is None:
                    null_text += 1
                    h = 0
                else:
                    tb = texts[i].encode("utf-8")
                    h = xxhash.xxh64_intdigest(tb)
                    if len(texts[i]) != nch[i]:
                        nchar_mismatch += 1
                    if len(texts[i]) == 0:
                        empty_text += 1
                text_hashes[pos] = h
                pos += 1
                sh.update(h.to_bytes(8, "little"))
                lg = langs[i] if langs[i] is not None else "<null>"
                lang_counts[lg] = lang_counts.get(lg, 0) + 1
                if langs[i] != "en" or (scores[i] is not None and scores[i] < 0.8):
                    bad_lang += 1
                if scores[i] is not None and scores[i] < min_lang_score:
                    min_lang_score = scores[i]
                total_nwords += int(nw[i])
                nwords_all.append(int(nw[i]))
                sg = segs[i]
                seg_counts[sg] = seg_counts.get(sg, 0) + 1
                if sg not in seg_seen:
                    seg_seen.add(sg)
                    seg_first_order.append(sg)
            total_rows += m
        digest = sh.digest()
        per_shard_content.append({"file": s["file"],
                                  "content_sha256": digest.hex(),
                                  "rows": pf.metadata.num_rows})
        content_hasher.update(digest)

    text_hashes = text_hashes[:pos]
    np.save(os.path.join(OUT, "doc_text_xxh64.npy"), text_hashes)

    nw_arr = np.asarray(nwords_all, dtype=np.int64)
    length_stats = {
        "n": int(nw_arr.size),
        "total": int(nw_arr.sum()),
        "mean": float(nw_arr.mean()),
        "median": float(np.quantile(nw_arr, 0.5)),
        "p10": float(np.quantile(nw_arr, 0.1)),
        "p90": float(np.quantile(nw_arr, 0.9)),
        "min": int(nw_arr.min()),
        "max": int(nw_arr.max()),
    }

    # ---- per-segment comparison -------------------------------------------
    man_seg = {s["segment"]: s["english_docs"] for s in man["segments"]}
    man_rank = {s["segment"]: s["rank"] for s in man["segments"]}
    stats = {}
    stats_path = os.path.join(CORPUS, "segment_stats.jsonl")
    if os.path.exists(stats_path):
        for line in open(stats_path, encoding="utf-8"):
            try:
                r = json.loads(line)
                stats[r["segment"]] = r
            except Exception:
                pass
    seg_rows = []
    mismatch = 0
    for seg, cnt in seg_counts.items():
        exp = man_seg.get(seg)
        st = stats.get(seg, {})
        ok = exp == cnt
        if not ok:
            mismatch += 1
        seg_rows.append({"segment": seg, "rank": man_rank.get(seg),
                         "measured_docs": cnt,
                         "manifest_english_docs": exp,
                         "segment_stats_english_docs": st.get("english_docs"),
                         "match": ok})
    seg_rows.sort(key=lambda r: (r["rank"] is None, r["rank"]))
    missing_segments = sorted(set(man_seg) - set(seg_counts))
    extra_segments = sorted(set(seg_counts) - set(man_seg))
    order_ok = seg_first_order == [s["segment"] for s in
                                   sorted(man["segments"], key=lambda x: x["rank"])]

    res["segment_counts"] = {
        "segments_in_manifest": len(man_seg),
        "segments_in_corpus": len(seg_counts),
        "mismatch_count": mismatch,
        "missing_segments": missing_segments,
        "extra_segments": extra_segments,
        "first_occurrence_order_matches_rank_order": order_ok,
        "rows": seg_rows,
    }
    import csv
    with open(os.path.join(OUT, "segment_counts_measured.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(seg_rows[0]))
        w.writeheader()
        w.writerows(seg_rows)

    # ---- summary and checks ------------------------------------------------
    res["content"] = {
        "rows_total": total_rows,
        "unique_ids": len(seen_ids),
        "duplicate_ids": dup_ids,
        "null_ids": null_id,
        "null_texts": null_text,
        "empty_texts": empty_text,
        "nchar_mismatch": nchar_mismatch,
        "lang_counts": dict(sorted(lang_counts.items(), key=lambda kv: -kv[1])),
        "documents_failing_lang_rule": bad_lang,
        "min_lang_score": None if min_lang_score == float("inf") else min_lang_score,
        "id_sequence_sha256": id_hasher.hexdigest(),
        "id_sequence_hash_definition":
            "sha256 over rows in shard order of LE32(len(utf8(id))) || utf8(id)",
        "content_sha256": content_hasher.hexdigest(),
        "content_hash_definition":
            "sha256 over per-shard sha256(concat LE64(xxh64_intdigest(utf8(text)))) "
            "in shard order; also doc_text_xxh64.npy",
        "per_shard_content": per_shard_content,
        "length_stats_words": length_stats,
        "length_stats_manifest": man["length_stats_words"],
    }

    all_bytes = all(x.get("bytes_ok") for x in res["shards"])
    all_sha = all(x.get("sha256_ok") for x in res["shards"])
    all_rows = all(x.get("rows_ok") for x in res["shards"])
    res["checks"] = {
        "all_shards_exist": all(x["exists"] for x in res["shards"]),
        "all_bytes_match_manifest": all_bytes,
        "all_sha256_match_manifest": all_sha,
        "all_row_counts_match_manifest": all_rows,
        "total_rows_equal_manifest": total_rows == man["total_documents"],
        "segment_counts_match_manifest": mismatch == 0 and not missing_segments
                                          and not extra_segments,
        "segment_order_matches_rank_order": order_ok,
        "language_rule_holds_for_every_document": bad_lang == 0,
        "no_duplicate_ids": dup_ids == 0,
        "no_null_or_empty_text": null_text == 0 and empty_text == 0,
        "nchar_equals_text_length_for_every_document": nchar_mismatch == 0,
        "total_words_match_manifest": total_nwords == man["length_stats_words"]["total"],
        "mean_words_matches_manifest_within_1e-6":
            abs(length_stats["mean"] - man["length_stats_words"]["mean"]) < 1e-6,
        "median_p10_p90_min_max_match_manifest":
            length_stats["median"] == man["length_stats_words"]["median"]
            and length_stats["p10"] == man["length_stats_words"]["p10"]
            and abs(length_stats["p90"] - man["length_stats_words"]["p90"]) < 1e-9
            and length_stats["min"] == man["length_stats_words"]["min"]
            and length_stats["max"] == man["length_stats_words"]["max"],
    }
    res["input_identity"] = {
        "file_level": "six parquet sha256 equal manifest (byte-identical file identity)"
        if all_sha else "file hashes differ from manifest",
        "content_level": "content_sha256 / doc_text_xxh64.npy define document-level identity",
        "note": "file sha256 is the primary input identity; the content hash is a second, "
                "order-sensitive identity that stays stable if parquet bytes are rewritten",
    }
    res["elapsed_s"] = round(time.time() - t0, 2)

    with open(os.path.join(OUT, "input_verification.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k not in ("segment_counts",)},
                     indent=1, ensure_ascii=False)[:6000])
    print("segment mismatches:", mismatch, "elapsed_s", res["elapsed_s"])


if __name__ == "__main__":
    main()
