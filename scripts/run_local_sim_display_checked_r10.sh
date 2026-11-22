#!/usr/bin/env bash
set -eo pipefail
OUR_MANI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/ubuntu/桌面/phsimani/physmani_reproduction/runtime.sh
export PYTHONPATH="$OUR_MANI_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUTF8=1
export XDG_CACHE_HOME="$OUR_MANI_ROOT/runtime/framework/cache"
export TMPDIR="$OUR_MANI_ROOT/runtime/framework/tmp"
mkdir -p "$TMPDIR"
cd "$OUR_MANI_ROOT"
OUR_MANI_XVFB_LOG="$(python scripts/native_display_guard_r10.py --error-path "$@")"
exec xvfb-run -a -e "$OUR_MANI_XVFB_LOG" python scripts/native_display_guard_r10.py "$@"
