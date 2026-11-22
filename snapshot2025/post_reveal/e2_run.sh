#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
exec "$here/../../../env/bin/python" -B "$here/e2_collapsed_timing.py" "$@"
