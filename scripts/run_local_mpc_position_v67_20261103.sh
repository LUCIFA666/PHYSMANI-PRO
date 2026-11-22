#!/usr/bin/env bash
set -eo pipefail
OUR_MANI_MPC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/ubuntu/桌面/phsimani/physmani_reproduction/runtime.sh
export PYTHONPATH="$OUR_MANI_MPC_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$OUR_MANI_MPC_ROOT"
mkdir -p runtime/framework/local_mpc_20261102/scratch
OUR_MANI_MPC_SCRATCH="$(mktemp -d "$OUR_MANI_MPC_ROOT/runtime/framework/local_mpc_20261102/scratch/native.XXXXXXXX")"
export TMPDIR="$OUR_MANI_MPC_SCRATCH/tmp" XDG_CACHE_HOME="$OUR_MANI_MPC_SCRATCH/cache"
export XDG_CONFIG_HOME="$OUR_MANI_MPC_SCRATCH/config" XDG_RUNTIME_DIR="$OUR_MANI_MPC_SCRATCH/xdg-runtime"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
exec 8>runtime/framework/fixed_TRAIN104_geometry_native.lock
/usr/bin/flock -x 8
exec 9>runtime/framework/TRAIN_postcomplete_diagnostics_v1106.lock
/usr/bin/flock -x 9
exec python scripts/run_frozen_system_environment_v820_r10.py \
  --manifest-sha256 1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30 \
  -- xvfb-run -a python -B scripts/run_local_mpc_position_v67_20261103.py "$@"
