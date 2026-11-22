import glob, os, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_lib as K
work = os.path.join(K.CORE, "crosscheck_datatrove")
n = 1400
S = np.zeros((n, 112), np.uint64)
for b in range(14):
    for r in range(14):
        p = os.path.join(work, "signatures", f"bucket_{b:03d}", f"{r:05d}.minhash.sig")
        dt = np.dtype([(f"f{i+1}", "<u8") for i in range(8)] + [("doc", "<u4")])
        a = np.fromfile(p, dtype=dt)
        assert a.size <= 100, (p, a.size)
        S[r*100 + a["doc"], b*8:(b+1)*8] = np.stack([a[f"f{i+1}"] for i in range(8)], 1)
# ours
off, flat, _ = K.load_shingles("wet")
import json
# reconstruct the same docs order as the crosscheck
ti, tj, _ = K.compute_truth("wet", 0.8, off)
root = K.uf_minid(len(off)-1, ti, tj); size = np.bincount(root, minlength=len(off)-1)
order = np.argsort(-size, kind="stable"); chosen=[]; seen=set()
for d in order:
    if size[d] < 2: break
    for c in np.flatnonzero(root == root[d]):
        c=int(c)
        if c not in seen: seen.add(c); chosen.append(c)
    if len(chosen) >= 1000: break
rng = np.random.RandomState(20261119)
while len(chosen) < 1400:
    c = int(rng.randint(len(off)-1))
    if c not in seen: seen.add(c); chosen.append(c)
docs = np.array(sorted(chosen), np.int64)
lens = off[docs+1]-off[docs]; so=np.zeros(n+1,np.int64); np.cumsum(lens,out=so[1:])
sf=np.concatenate([flat[off[d]:off[d+1]] for d in docs])
ours = K.L.signatures(so, sf, K.H, 1, "dt")
print("shape", ours.shape, S.shape, ours.dtype)
print("identical full:", bool((ours==S).all()))
print("per-column match count:", int((ours==S).all(0).sum()))
print("col0 ours", ours[:3,0].tolist(), "dt", S[:3,0].tolist())
print("col1 ours", ours[:3,1].tolist(), "dt", S[:3,1].tolist())
# maybe rows permuted
print("rowset equal:", bool(set(map(tuple, ours.tolist()))==set(map(tuple, S.tolist()))))
# compare band key equality instead
Ko = K.band_keys(ours); Kd = K.band_keys(S)
print("band keys equal frac:", float((Ko==Kd).mean()))
bad = np.flatnonzero((ours != S).any(1))
print("mismatching rows:", bad.tolist()[:20], "count", len(bad))
for r in bad[:5]:
    print("row", r, "doc", int(docs[r]), "lens", int(lens[r]),
          "ndiff", int((ours[r]!=S[r]).sum()), "ourband", Ko[:,r][:4].tolist(), "dtband", Kd[:,r][:4].tolist())
# pair sets
ei, ej, _ = K.chain_edges(Ko, np.ones(n,bool))
ei2, ej2, _ = K.chain_edges(Kd, np.ones(n,bool))
A={(int(a),int(b)) for a,b in zip(ei,ej)}; D={(min(int(a),int(b)),max(int(a),int(b))) for a,b in zip(ei2,ej2)}
print("our chain on ours", len(A), "on dt sig", len(D), "symdiff", len(A^D))
