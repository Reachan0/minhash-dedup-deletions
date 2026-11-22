"""Compute workload id_sha256 via make_workloads.pool_ids and compare to workloads.json.
Usage: verify_ids.py CORE_DIR WORKLOADS_JSON"""
import sys, json, hashlib
sys.path.insert(0, sys.argv[1])
import core_lib as K, make_workloads as M
ref = json.load(open(sys.argv[2]))["workloads"]
ok = True
for w in K.WORKLOADS:
    ids = M.pool_ids(w)
    sha = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    r = ref[w]
    res = dict(workload=w, n=len(ids), n_ref=r["n"], first=ids[0], first_ref=r["first_id"], last=ids[-1],
               last_ref=r["last_id"], sha=sha, sha_ref=r["id_sha256"],
               match=(len(ids) == r["n"] and ids[0] == r["first_id"] and ids[-1] == r["last_id"] and sha == r["id_sha256"]))
    ok &= res["match"]; print(json.dumps(res), flush=True)
print("IDS_ALL_MATCH" if ok else "IDS_MISMATCH")
