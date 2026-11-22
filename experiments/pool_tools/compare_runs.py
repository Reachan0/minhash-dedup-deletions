"""Compare rebuilt runs.jsonl with reference by key. Usage: compare_runs.py REF NEW"""
import sys, json
SKIP = {"ts_utc", "hw", "threads", "evidence_tier", "hash_config"}
def load(p):
    return {json.loads(l)["key"]: json.loads(l) for l in open(p)}
ref, new = load(sys.argv[1]), load(sys.argv[2])
missing = [k for k in ref if k not in new]
print(f"ref rows={len(ref)} new rows={len(new)} missing_in_new={len(missing)}")
core = ("deleted", "unsupported")
nrow = nsame = ncore_same = 0; bad = []; cnt = {}
for k, r in ref.items():
    if k not in new: continue
    n = new[k]; nrow += 1
    diffs = [(f, r[f], n.get(f)) for f in r if f not in SKIP and not f.startswith("t_") and r[f] != n.get(f)
             and not (isinstance(r[f], float) and isinstance(n.get(f), float) and abs(r[f] - n[f]) < 1e-12)]
    if not any(d[0] in core for d in diffs): ncore_same += 1
    if not diffs: nsame += 1
    else:
        bad.append((k, diffs))
        for d in diffs: cnt[d[0]] = cnt.get(d[0], 0) + 1
print(f"compared={nrow} all_nontiming_fields_identical={nsame} deleted&unsupported_identical={ncore_same}")
print("field mismatch counts:", cnt)
for k, d in bad[:15]: print("MISMATCH", k, d[:4])
print("missing:", missing[:10])
