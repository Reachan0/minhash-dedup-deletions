"""Test-retest of the two labelling runs on rows whose prompt is identical in both
(retained documents, and deletions whose closest member is the retained document)."""
import json
import numpy as np
C = "/path/to/scratch/dedupjoin/dedupjoin-newsnap-selection/results_post_reveal/content_audit"
K = C + "/kept"
UNIQ = {"unique_facts", "unique_content"}
rows = {json.loads(l)["doc"]: json.loads(l) for l in open(K + "/samples.jsonl")}
same = {d for d, r in rows.items() if r["stratum"] == "retained" or r.get("closest_member", r["counterpart"]) == r["counterpart"]}
def lab(base, m):
    return {r["doc"]: r["label"] for r in map(json.loads, open(f"{base}/labels_{m}.jsonl")) if r.get("label")}
def kappa(a, b):
    cats = sorted(set(a) | set(b))
    a, b = np.array(a), np.array(b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe)
out = {"rows_same_prompt": len(same), "by_stratum": {s: sum(rows[d]["stratum"] == s for d in same) for s in ("unsupported", "supported", "retained")}}
O = {m: lab(C, m) for m in ("gpt-6.2-sol", "qwen3.7-plus")}
N = {m: lab(K, m) for m in ("gpt-6.2-sol", "qwen3.7-plus")}
for m in O:
    ds = [d for d in same if d in O[m] and d in N[m]]
    a = [O[m][d] for d in ds]; b = [N[m][d] for d in ds]
    out[m] = dict(n=len(ds), identical=float(np.mean([x == y for x, y in zip(a, b)])), kappa=float(kappa(a, b)),
                  uniq_flip=float(np.mean([(x in UNIQ) != (y in UNIQ) for x, y in zip(a, b)])))
    for s in ("unsupported", "retained"):
        dd = [d for d in ds if rows[d]["stratum"] == s]
        out[m]["identical_" + s] = float(np.mean([O[m][d] == N[m][d] for d in dd]))
# both-unique among unsupported deletions whose closest member is the retained one, both runs (weighted)
for name, L in (("old", O), ("new", N)):
    dd = [d for d in same if rows[d]["stratum"] == "unsupported" and all(d in L[m] for m in L)]
    w = np.array([rows[d]["weight"] for d in dd])
    b = np.array([all(L[m][d] in UNIQ for m in L) for d in dd])
    out["both_unique_unsup_samecp_" + name] = float((w * b).sum() / w.sum())
    out["n_unsup_samecp_" + name] = len(dd)
print(json.dumps(out, indent=1))
json.dump(out, open(K + "/retest.json", "w"), indent=1)
