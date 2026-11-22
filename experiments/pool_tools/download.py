"""Download rebuild sources with sha256 check. Usage: download.py MANIFEST RAWDIR"""
import json, sys, os, hashlib, subprocess
from concurrent.futures import ThreadPoolExecutor
man = json.load(open(sys.argv[1])); raw = sys.argv[2]
want = {"c4_en_train_00000","c4_en_train_00001","fineweb_sample10bt_000_00000","wet_00011","wet_00199","wet_00399","wet_00519"}
def dest(s):
    n = s["name"]; u = s["url"]
    sub = "c4" if n.startswith("c4") else "fw" if n.startswith("fine") else "wet"
    return os.path.join(raw, sub, os.path.basename(u))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""): h.update(c)
    return h.hexdigest()
def job(s):
    p = dest(s); u = s["url"]
    if os.environ.get("HF_MIRROR") and u.startswith("https://huggingface.co"): u = u.replace("https://huggingface.co", os.environ["HF_MIRROR"])
    for attempt in range(5):
        subprocess.run(["curl", "-fL", "-C", "-", "--retry", "3", "-sS", "-o", p, u], check=False)
        if os.path.exists(p) and os.path.getsize(p) == s["size_bytes"]: break
    h = sha(p) if os.path.exists(p) else None
    r = dict(name=s["name"], path=p, size=os.path.getsize(p) if h else None, want_size=s["size_bytes"], sha256=h, want_sha=s["sha256"], ok=(h == s["sha256"]))
    print(json.dumps(r), flush=True); return r
srcs = [s for s in man["sources"] if s["name"] in want]
with ThreadPoolExecutor(7) as ex: res = list(ex.map(job, srcs))
print("ALL_OK" if all(r["ok"] for r in res) else "MISMATCH")
