import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_lib as K
from collections import Counter
work = os.path.join(K.CORE, "crosscheck_datatrove")
dt = np.dtype([(f"f{i+1}", "<u8") for i in range(8)] + [("doc", "<u4")])
exp_matches = {0:45,1:106,2:150,3:212,4:267,5:344,6:380,7:406,8:507,9:565,10:615,11:694,12:754,13:811}
for b in range(14):
    c = Counter()
    nrec = 0
    for r in range(14):
        p = os.path.join(work, "signatures", f"bucket_{b:03d}", f"{r:05d}.minhash.sig")
        a = np.fromfile(p, dtype=dt)
        nrec += a.size
        for x in a:
            c[tuple(int(x[f"f{i+1}"]) for i in range(8))] += 1
    exp = sum(v-1 for v in c.values() if v>=2)
    print(f"bucket {b}: recs {nrec} groups {len(c)} max {max(c.values())} exp_matches {exp} dt_matches {exp_matches[b]}")
