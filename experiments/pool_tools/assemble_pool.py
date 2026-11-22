"""Place the wet shards of a separately built wet pool at shard-00040.. so that
core_lib.SHARD_BASE (fw 0, c4 11, wet 40) resolves. Usage: assemble_pool.py POOL WETPOOL"""
import os, sys, shutil
pool, wp = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(pool, "wet"), exist_ok=True)
for fn in sorted(os.listdir(os.path.join(wp, "wet"))):
    k = int(fn.split("-")[1].split(".")[0])
    dst = os.path.join(pool, "wet", f"shard-{40 + k:05d}.parquet")
    shutil.copy2(os.path.join(wp, "wet", fn), dst); print(fn, "->", dst)
