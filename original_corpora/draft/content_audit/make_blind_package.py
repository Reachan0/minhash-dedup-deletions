#!/usr/bin/env python3
"""Write the blind-review package for the content audit.

The package lets a person label the sampled documents without seeing any machine label:
  artifacts/blind_review/README.md          category definitions and labelling rules
  artifacts/blind_review/samples_blind.jsonl  id, corpus, stratum, max J, text, counterpart text
  artifacts/blind_review/labels_template.csv  blank sheet to fill in
  artifacts/blind_review/labels_hidden.jsonl  machine + researcher labels (kept apart)
  artifacts/blind_review/hand_checked.jsonl   the 29 documents the researcher read, with labels
No label appears in README.md or samples_blind.jsonl.

usage: python make_blind_package.py
"""
import csv
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CA = os.path.join(ROOT, "output/dedupjoin-pvldb/results/content_audit")
BR = os.path.join(ROOT, "output/dedupjoin-pvldb/artifacts/blind_review")

README = """# Blind review package: what did the disputed deletions remove?

This package contains 260 documents sampled from the four-corpus deduplication pool.
260 rows: 160 C4 documents whose deletion is disputed across 100 MinHash seeds,
20 C4 documents every seed and method deleted (control), 20 C4 documents every seed and
method kept (control), 40 WET documents and 20 FineWeb documents disputed at seed 1.
The sample is stratified, not uniform over the pool; do not compute pool shares from raw
counts without the stratum weights in `results/content_audit/strata_counts.json`.

## What to do

Label every row of `samples_blind.jsonl` with exactly one category, using the definitions
below, and write the result to a copy of `labels_template.csv` (columns: `id,label,confidence,note`).
Where a document has a counterpart (exact 5-gram Jaccard >= 0.4), the counterpart text is
included in the `neighbor` field; judge whether the document adds anything the counterpart lacks.

## Categories

- `template`: boilerplate, navigation, listing/category page, auto-generated index, error page,
  cookie/legal notice, or a page whose text is mostly repeated site furniture. It carries no
  substantive information. **Slot rule**: a city, business name, product name, model number,
  phone number or event title substituted into otherwise identical boilerplate is a slot, not
  a fact; the page is still `template` unless it carries other substantive content.
- `repost`: a near-verbatim copy (or minor-format copy) of the counterpart or of a widely
  syndicated text; all substantive content is present in the counterpart.
- `partial_rewrite`: an edited, abridged, expanded or paraphrased version of the counterpart;
  its substantive facts are recoverable from the counterpart, only wording or ordering differs.
- `unique_facts`: a counterpart exists and is similar enough that a deduplicator treats the two
  as near-duplicates, but this document contains at least one specific substantive fact
  (a number, date, name, place, decision, event, measurement) absent from the counterpart.
- `unique_content`: no close counterpart, and the document carries substantive content that
  is absent from every counterpart. Deleting it loses information.

Precedence when several apply: `template` > `repost` > `unique_facts` > `partial_rewrite` > `unique_content`.
A number that appears only in a footer, date stamp, page counter, price widget or URL is not a
substantive fact. Judge only from the text shown.

`unique_facts` and `unique_content` are the *lost-information* classes; the audit's headline
quantity is their token share among disputed deletions.

## Files

| File | Contents |
|---|---|
| `samples_blind.jsonl` | one JSON row per document: `id`, `corpus`, `stratum`, `nwords`, `max_j`, `text`, `neighbor` |
| `labels_template.csv` | blank labelling sheet |
| `labels_hidden.jsonl` | labels produced by the machine labeller and by the researcher's hand check (do not open before labelling) |
| `hand_checked.jsonl` | the 29 documents read and labelled by the researcher, with evidence notes |
| `agreement.json` | agreement between hand labels and each automatic labeller, with Wilson intervals |

`samples_blind.jsonl` has no URL, domain or method decision, so the reviewer cannot tell
which method deleted the document.
"""


def main():
    os.makedirs(BR, exist_ok=True)
    samples = [json.loads(x) for x in open(os.path.join(CA, "samples.jsonl"))]
    hand = {}
    p = os.path.join(CA, "hand_labels.jsonl")
    if os.path.exists(p):
        hand = {json.loads(x)["id"]: json.loads(x) for x in open(p)}
    llm = {}
    p = os.path.join(CA, "llm_labels.jsonl")
    if os.path.exists(p):
        for x in open(p):
            r = json.loads(x)
            llm[r["id"]] = r

    with open(os.path.join(BR, "samples_blind.jsonl"), "w") as f:
        for r in samples:
            nb = r.get("neighbor")
            f.write(json.dumps(dict(
                id=r["id"], corpus=r["corpus"], stratum=r["stratum"],
                nwords=r.get("n_words_text") or r["nwords"], max_j=r.get("max_j"),
                text=r.get("text", ""),
                neighbor=None if not nb else dict(id=nb["id"], j=nb["j"], text=nb.get("text", "")),
            ), ensure_ascii=False) + "\n")

    with open(os.path.join(BR, "labels_template.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "confidence", "note"])
        for r in samples:
            w.writerow([r["id"], "", "", ""])

    with open(os.path.join(BR, "labels_hidden.jsonl"), "w") as f:
        for r in samples:
            h, m = hand.get(r["id"]), llm.get(r["id"])
            f.write(json.dumps(dict(
                id=r["id"], stratum=r["stratum"],
                hand_label=h["label"] if h else None,
                hand_confidence=h.get("confidence") if h else None,
                hand_note=h.get("note") if h else None,
                llm_label=m.get("label") if m else None,
                llm_confidence=m.get("confidence") if m else None,
                llm_lost_fact=m.get("lost_fact") if m else None,
                llm_qa=m.get("qa") if m else None,
            ), ensure_ascii=False) + "\n")

    with open(os.path.join(BR, "hand_checked.jsonl"), "w") as f:
        byid = {r["id"]: r for r in samples}
        for did, h in hand.items():
            r = byid.get(did, {})
            nb = r.get("neighbor")
            f.write(json.dumps(dict(
                id=did, stratum=r.get("stratum"), label=h["label"],
                confidence=h.get("confidence"), note=h.get("note"),
                nwords=r.get("n_words_text") or r.get("nwords"), max_j=r.get("max_j"),
                text=r.get("text", ""),
                neighbor=None if not nb else dict(id=nb["id"], j=nb["j"], text=nb.get("text", "")),
            ), ensure_ascii=False) + "\n")

    with open(os.path.join(BR, "README.md"), "w") as f:
        f.write(README)
    print(f"wrote {len(samples)} blind rows, {len(hand)} hand-checked, {len(llm)} machine labels")


if __name__ == "__main__":
    main()
