#!/usr/bin/env bash
set -eo pipefail
OUR_MANI_EXECUTOR_REPAIR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/ubuntu/桌面/phsimani/physmani_reproduction/runtime.sh
export PYTHONPATH="$OUR_MANI_EXECUTOR_REPAIR_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
OUR_MANI_EXECUTOR_REPAIR_SCRATCH="$(mktemp -d /tmp/our-mani-executor-repair.XXXXXXXX)"
export TMPDIR="$OUR_MANI_EXECUTOR_REPAIR_SCRATCH/tmp" XDG_CACHE_HOME="$OUR_MANI_EXECUTOR_REPAIR_SCRATCH/cache"
export XDG_CONFIG_HOME="$OUR_MANI_EXECUTOR_REPAIR_SCRATCH/config" XDG_RUNTIME_DIR="$OUR_MANI_EXECUTOR_REPAIR_SCRATCH/xdg-runtime"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
cd "$OUR_MANI_EXECUTOR_REPAIR_ROOT"
exec xvfb-run -a python -B -m our_mani.executor_release_settle_20261113_v18.worker "$@"
