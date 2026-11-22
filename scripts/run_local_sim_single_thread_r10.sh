#!/usr/bin/env bash
set -eo pipefail
OUR_MANI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/ubuntu/桌面/phsimani/physmani_reproduction/runtime.sh
export PYTHONPATH="$OUR_MANI_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUTF8=1
export XDG_CACHE_HOME="$OUR_MANI_ROOT/runtime/framework/cache"
export TMPDIR="$OUR_MANI_ROOT/runtime/framework/tmp"
mkdir -p "$TMPDIR"
cd "$OUR_MANI_ROOT"
exec xvfb-run -a python "$@"
