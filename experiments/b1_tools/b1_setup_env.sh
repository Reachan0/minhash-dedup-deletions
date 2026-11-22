#!/usr/bin/env bash
# Python 3.11 env with the same pinned packages as the reveal (numpy 2.4 needs >= 3.11).
set -ex
W=/path/to/scratch/dedupjoin
uv venv --python 3.11 $W/env || /opt/conda/bin/conda create -y -p $W/env python=3.11
uv pip install --python $W/env/bin/python -i https://mirrors.aliyun.com/pypi/simple \
  numpy==2.4.6 pandas==3.1.1 pyarrow==26.0.1 scipy==1.17.1 xxhash==4.0.2 spacy==3.8.18 \
  regex==2026.11.19 numba==0.68.1 datatrove==0.6.0 fasttext-wheel psutil
$W/env/bin/python -c "import numpy, pyarrow, datatrove, spacy, fasttext, numba; print('env ok', numpy.__version__)"
[ -f $W/tools/lid.176.bin ] || curl -sSfL -o $W/tools/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
test "$(stat -c %s $W/tools/lid.176.bin)" = 131266198
echo SETUP_DONE
