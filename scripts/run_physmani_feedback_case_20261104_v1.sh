#!/usr/bin/env bash
set -eo pipefail
OUR_MANI_FEEDBACK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/ubuntu/桌面/phsimani/physmani_reproduction/runtime.sh
export PYTHONPATH="$OUR_MANI_FEEDBACK_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$OUR_MANI_FEEDBACK_ROOT"
mkdir -p runtime/framework/feedback_interface_20261104/scratch
OUR_MANI_FEEDBACK_SCRATCH="$(mktemp -d "$OUR_MANI_FEEDBACK_ROOT/runtime/framework/feedback_interface_20261104/scratch/native.XXXXXXXX")"
export TMPDIR="$OUR_MANI_FEEDBACK_SCRATCH/tmp" XDG_CACHE_HOME="$OUR_MANI_FEEDBACK_SCRATCH/cache"
export XDG_CONFIG_HOME="$OUR_MANI_FEEDBACK_SCRATCH/config" XDG_RUNTIME_DIR="$OUR_MANI_FEEDBACK_SCRATCH/xdg-runtime"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
# Backend owns the canonical Native/GPU mutex pair through shutdown.
exec python scripts/run_frozen_system_environment_v820_r10.py \
  --manifest-sha256 1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30 \
  -- xvfb-run -a python -B scripts/validate_physmani_feedback_case_20261104_v1.py --execute "$@"
