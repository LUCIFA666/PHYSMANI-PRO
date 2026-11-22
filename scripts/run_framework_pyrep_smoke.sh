#!/usr/bin/env bash
set -eo pipefail
OUR_MANI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/ubuntu/桌面/phsimani/physmani_reproduction/runtime.sh
export PYTHONPATH="$OUR_MANI_ROOT:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$OUR_MANI_ROOT/runtime/framework/cache"
export TMPDIR="$OUR_MANI_ROOT/runtime/framework/tmp"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" "$OUR_MANI_ROOT/logs/framework"
cd "$OUR_MANI_ROOT"
exec xvfb-run -a python scripts/smoke_pyrep_framework.py
