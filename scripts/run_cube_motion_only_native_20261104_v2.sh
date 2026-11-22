#!/usr/bin/env bash
set -eo pipefail
OUR_MANI_CORRECTED_CUBE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/ubuntu/桌面/phsimani/physmani_reproduction/runtime.sh
export PYTHONPATH="$OUR_MANI_CORRECTED_CUBE_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$OUR_MANI_CORRECTED_CUBE_ROOT"
mkdir -p runtime/framework/cube_motion_only_native_20261104_v2/scratch
OUR_MANI_CORRECTED_CUBE_SCRATCH="$(mktemp -d "$OUR_MANI_CORRECTED_CUBE_ROOT/runtime/framework/cube_motion_only_native_20261104_v2/scratch/native.XXXXXXXX")"
export TMPDIR="$OUR_MANI_CORRECTED_CUBE_SCRATCH/tmp" XDG_CACHE_HOME="$OUR_MANI_CORRECTED_CUBE_SCRATCH/cache"
export XDG_CONFIG_HOME="$OUR_MANI_CORRECTED_CUBE_SCRATCH/config" XDG_RUNTIME_DIR="$OUR_MANI_CORRECTED_CUBE_SCRATCH/xdg-runtime"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
# Parent owns serialization, not these same file descriptors. Hold both exact
# global mutexes through the child's final Native shutdown; no old OS claim.
exec 8>"$OUR_MANI_CORRECTED_CUBE_ROOT/runtime/framework/fixed_TRAIN104_geometry_native.lock"
/usr/bin/flock -x 8
exec 9>"$OUR_MANI_CORRECTED_CUBE_ROOT/runtime/framework/TRAIN_postcomplete_diagnostics_v1106.lock"
/usr/bin/flock -x 9
exec xvfb-run -a python -B scripts/run_cube_motion_only_native_20261104_v2.py "$@"
