#!/usr/bin/env python3
"""Run the published run_timing.py unchanged, with caches under this task.

Usage: wrap_timing.py --method A --tau 0.8 --mode pathcold --seed 1 --repeat 0
"""
import os
import runpy
import sys

import env_paths

S = env_paths.prepared()
SCRIPT = os.path.join(env_paths.NEWSNAP_CODE, "run_timing.py")
sys.argv = [SCRIPT] + sys.argv[1:]
runpy.run_path(SCRIPT, run_name="__main__")
