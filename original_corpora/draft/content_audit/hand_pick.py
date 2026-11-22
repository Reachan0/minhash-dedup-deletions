#!/usr/bin/env python3
"""Pick a hand-checked subset spread over strata and similarity bins; write a blind view."""
import json, os, sys, re
import numpy as np
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..","..",".."))
CA=os.path.join(ROOT,"output/dedupjoin-pvldb/results/content_audit")
rows=[json.loads(x) for x in open(os.path.join(CA,"samples.jsonl"))]
rng=np.random.default_rng(4242)
def clip(t,h=1400,tl=600):
    t=re.sub(r"\s+"," ",t or "").strip()
    return t if len(t)<=h+tl else t[:h]+" [...] "+t[-tl:]
picked=[]
# c4 union: 4 bins x 3
bins=[(0,0.3),(0.3,0.5),(0.5,0.7),(0.7,0.8)]
rs=[r for r in rows if r["stratum"]=="c4_union_no_nb"]
for lo,hi in bins:
    cand=[r for r in rs if lo<=r["max_j"]<hi]
    for r in rng.choice(len(cand),size=min(3,len(cand)),replace=False):
        picked.append(cand[r])
for s,k in (("c4_agreed_deleted",5),("c4_agreed_kept",5),("wet_disputed",6),("fw_disputed",4)):
    rs=[r for r in rows if r["stratum"]==s]
    for r in rng.choice(len(rs),size=min(k,len(rs)),replace=False):
        picked.append(rs[r])
out=[]
for r in picked:
    nb=r.get("neighbor")
    out.append(dict(id=r["id"],stratum=r["stratum"],max_j=round(r["max_j"],3),
                    nwords=r["n_words_text"],text=clip(r["text"]),
                    neighbor_id=(nb or {}).get("id"),neighbor_j=(nb or {}).get("j"),
                    neighbor_text=clip((nb or {}).get("text",""),800,400)))
with open(os.path.join(ROOT,"output/dedupjoin-pvldb/artifacts/blind_review/hand_set.jsonl"),"w") as f:
    for r in out:
        f.write(json.dumps(r,ensure_ascii=False)+"\n")
print(len(out),"picked")
if len(sys.argv)>1:
    lo,hi=int(sys.argv[1]),int(sys.argv[2])
    for r in out[lo:hi]:
        print("="*100)
        print(f"ID {r['id']}  stratum={r['stratum']}  maxJ={r['max_j']}  nwords={r['nwords']}")
        print("TEXT:",r["text"])
        if r["neighbor_text"]:
            print(f"NEIGHBOR {r['neighbor_id']} J={r['neighbor_j']}:",r["neighbor_text"])
