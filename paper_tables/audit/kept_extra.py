"""Facts about the kept-counterpart audit that summary.json does not hold, plus example candidates.
Run on AutoDL after analyze: python kept_extra.py [show DOC]"""
import json
import sys

import numpy as np

C = "/path/to/scratch/dedupjoin/dedupjoin-newsnap-selection/results_post_reveal/content_audit"
K = C + "/kept"
UNIQ = {"unique_facts", "unique_content"}


def load(base):
    rows = {json.loads(l)["doc"]: json.loads(l) for l in open(base + "/samples.jsonl")}
    lab = {}
    for m in ("gpt-6.2-sol", "qwen3.7-plus"):
        lab[m] = {r["doc"]: r for r in map(json.loads, open(f"{base}/labels_{m}.jsonl")) if r.get("label")}
    return rows, lab


def wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return float(x[o][np.searchsorted(c, 0.5)])


rows, lab = load(K)
orows, olab = load(C)
if len(sys.argv) > 2 and sys.argv[1] == "show":
    d = int(sys.argv[2])
    r = rows[d]
    print(json.dumps({k: r[k] for k in r if k not in ("text", "counterpart_text")}, indent=1))
    for m in lab:
        x = lab[m][d]
        print(m, x["label"], "|", x.get("lost_fact"), "|", x.get("reason"))
    print("--- DELETED\n" + r["text"][:2500])
    print("--- RETAINED\n" + (r["counterpart_text"] or "")[:2500])
    sys.exit()

out = {}
for s in ("unsupported", "supported"):
    docs = [d for d, r in rows.items() if r["stratum"] == s]
    w = np.array([rows[d]["weight"] for d in docs])
    jk = np.array([rows[d]["counterpart_jaccard"] for d in docs])
    jc = np.array([rows[d].get("closest_member_jaccard", rows[d]["counterpart_jaccard"]) for d in docs])
    same = np.array([rows[d].get("closest_member", rows[d]["counterpart"]) == rows[d]["counterpart"] for d in docs])
    both = np.array([all(lab[m].get(d, {}).get("label") in UNIQ for m in lab) for d in docs])
    oboth = np.array([all(olab[m].get(d, {}).get("label") in UNIQ for m in olab) for d in docs])
    # bootstrap within the stratum's word-count sub-strata is not stored; resample documents with weights
    rng = np.random.default_rng(5)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(docs), len(docs))
        bs.append((w[i] * both[i]).sum() / w[i].sum())
    out[s] = dict(n=len(docs), shown_share=float((w * (jk >= 0.4)).sum() / w.sum()),
                  median_j_kept=wmedian(jk, w), median_j_closest=wmedian(jc, w),
                  kept_is_closest=float((w * same).sum() / w.sum()),
                  both_kept=float((w * both).sum() / w.sum()), both_closest=float((w * oboth).sum() / w.sum()),
                  both_kept_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  moved_to_unique=float((w * (both & ~oboth)).sum() / w.sum()),
                  moved_from_unique=float((w * (~both & oboth)).sum() / w.sum()))
print(json.dumps(out, indent=1))
json.dump(out, open(K + "/extra.json", "w"), indent=1)

# example candidates: unsupported, both unique, counterpart shown, short, both name a lost fact
cand = []
for d, r in rows.items():
    if r["stratum"] != "unsupported" or r["counterpart_jaccard"] < 0.5 or not (80 <= r["words"] <= 400):
        continue
    if not all(lab[m].get(d, {}).get("label") in UNIQ and lab[m][d].get("lost_fact") for m in lab):
        continue
    cand.append((r["counterpart_jaccard"], d, r["words"], lab["gpt-6.2-sol"][d]["lost_fact"][:150],
                 lab["qwen3.7-plus"][d]["lost_fact"][:150]))
cand.sort(reverse=True)
print(len(cand), "candidates")
for c in cand[:25]:
    print(f"doc {c[1]} J={c[0]:.2f} words={c[2]}\n  sol: {c[3]}\n  qwen: {c[4]}")
