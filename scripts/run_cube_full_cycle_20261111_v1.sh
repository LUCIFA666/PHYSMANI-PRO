#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/桌面/our-mani
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
exec /home/ubuntu/physmani-conda/envs/physmani5090/bin/python -B -m our_mani.cube_full_cycle_20261111_v1.pipeline --execute
