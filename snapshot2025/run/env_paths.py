#!/usr/bin/env python3
"""Common paths for the reveal drivers (step run-prereg-selection).

The published method scripts live in `<slug>/code/newsnap` and are executed
verbatim; the only thing adjusted here is the cache root of *signatures* so
that they land under this task's process directory instead of the earlier
project core cache. No method definition is touched.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SLUG = os.path.dirname(HERE)                     # .work/dedupjoin-newsnap-selection
CODE = os.path.join(SLUG, "code")
NEWSNAP_CODE = os.path.join(CODE, "newsnap")
sys.path.insert(0, NEWSNAP_CODE)

import snap_common as S  # noqa: E402

K = S.K


def prepared():
    """Patch the signature cache root (core_lib.CACHE) to this task's cache dir."""
    K.CACHE = S.CACHE
    os.makedirs(os.path.join(S.CACHE, "sig"), exist_ok=True)
    return S
