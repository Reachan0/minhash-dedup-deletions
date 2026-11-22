#!/usr/bin/env python3
"""Build the exact-duplicate-collapsed corpus (one node per distinct shingle set)."""
import json
import os

import collapsed_lib as CL

info = CL.build_collapsed()
info["ts_utc"] = CL.S.now()
p = os.path.join(CL.S.SNAP, "collapsed_build.json")
json.dump(info, open(p, "w"), indent=1)
print(json.dumps(info))
