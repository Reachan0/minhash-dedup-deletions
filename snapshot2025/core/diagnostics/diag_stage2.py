import os, sys, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_lib as K
from datatrove.pipeline.dedup import minhash as MH
from datatrove.pipeline import dedup as DT
work = os.path.join(K.CORE, "crosscheck_datatrove")
cfg = MH.MinhashConfig(n_grams=5, num_buckets=14, hashes_per_bucket=8, seed=1)
out = os.path.join(work, "manual_bucket0")
shutil.rmtree(out, ignore_errors=True)
step = DT.MinhashDedupBuckets(input_folder=os.path.join(work,"signatures"), output_folder=out, config=cfg)
step.run(None, rank=0, world_size=14)
print("files:", os.listdir(out))
for f in os.listdir(out):
    print(f, os.path.getsize(os.path.join(out,f))//16, "pairs")
print("stats", step.stats)
