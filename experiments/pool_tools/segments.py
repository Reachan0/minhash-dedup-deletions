"""Write the WET segments JSON (order 00011,00199,00399,00519). Usage: segments.py RAWWETDIR OUT.json"""
import json, sys, os, glob
d, out = sys.argv[1], sys.argv[2]
segs = []
for name in ["00011", "00199", "00399", "00519"]:
    p = glob.glob(os.path.join(d, f"*-{name}.warc.wet.gz"))
    assert len(p) == 1, (name, p)
    segs.append({"name": name, "path": p[0]})
json.dump(segs, open(out, "w"), indent=1)
