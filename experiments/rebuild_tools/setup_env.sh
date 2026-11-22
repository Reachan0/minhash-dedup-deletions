#!/usr/bin/env bash
# Build the pinned Python env for the post-reveal rebuild (versions from environment/pip_freeze_post_install.txt).
# conda's package servers are unreachable from this host, so use a venv on miniconda's Python 3.11 + PyPI.
set -ex
W=/path/to/home/dedupjoin-postreveal
rm -rf $W/env
~/miniconda3/bin/python -m venv $W/env
P=$W/env/bin/python
$P -m pip install --no-cache-dir -U pip
$P -m pip install --no-cache-dir numpy==2.4.6 pandas==3.1.1 pyarrow==26.0.1 scipy==1.17.1 xxhash==4.0.2 \
   spacy==3.8.18 regex==2026.11.19 numba==0.68.1 datatrove==0.6.0 fasttext-wheel psutil
$P -m pip freeze > $W/logs/pip_freeze.txt
$P -c "import numpy, pyarrow, datatrove, spacy, fasttext, xxhash, numba; print('env ok', numpy.__version__, pyarrow.__version__, spacy.__version__)"
[ -f $W/tools/lid.176.bin ] || curl -sSfL -o $W/tools/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
sha256sum $W/tools/lid.176.bin
echo SETUP_DONE
