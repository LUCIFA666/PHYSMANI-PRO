#!/usr/bin/env bash
set -eo pipefail
OUR_MANI_EXECUTION_INTERFACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/ubuntu/桌面/phsimani/physmani_reproduction/runtime.sh
export PYTHONPATH="$OUR_MANI_EXECUTION_INTERFACE_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$OUR_MANI_EXECUTION_INTERFACE_ROOT"
OUR_MANI_EXECUTION_INTERFACE_SCRATCH="$(mktemp -d /tmp/our-mani-interface.XXXXXXXX)"
export TMPDIR="$OUR_MANI_EXECUTION_INTERFACE_SCRATCH/tmp" XDG_CACHE_HOME="$OUR_MANI_EXECUTION_INTERFACE_SCRATCH/cache"
export XDG_CONFIG_HOME="$OUR_MANI_EXECUTION_INTERFACE_SCRATCH/config" XDG_RUNTIME_DIR="$OUR_MANI_EXECUTION_INTERFACE_SCRATCH/xdg-runtime"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
exec 8>"$OUR_MANI_EXECUTION_INTERFACE_ROOT/runtime/framework/fixed_TRAIN104_geometry_native.lock"
/usr/bin/flock -n -x 8
exec 9>"$OUR_MANI_EXECUTION_INTERFACE_ROOT/runtime/framework/TRAIN_postcomplete_diagnostics_v1106.lock"
/usr/bin/flock -n -x 9
exec xvfb-run -a python -B scripts/cube_full_native_owner_20261111_v1.py --native-worker "$@"
