#!/usr/bin/env python3
"""Run the published run_snapshot_eval.py unchanged, with caches under this task."""
import os
import runpy
import sys

import env_paths

S = env_paths.prepared()
SCRIPT = os.path.join(env_paths.NEWSNAP_CODE, "run_snapshot_eval.py")
sys.argv = [SCRIPT] + sys.argv[1:]
runpy.run_path(SCRIPT, run_name="__main__")
