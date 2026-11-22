#!/usr/bin/env bash
set -euo pipefail
OUR_MANI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="$OUR_MANI_ROOT/runtime/framework/uv-cache"
export UV_PYTHON_INSTALL_DIR="$OUR_MANI_ROOT/runtime/framework/python"
export UV_PROJECT_ENVIRONMENT="$OUR_MANI_ROOT/.faster-env"
export XDG_CACHE_HOME="$OUR_MANI_ROOT/runtime/framework/cache"
export HF_HOME="$OUR_MANI_ROOT/runtime/framework/huggingface"
export TMPDIR="$OUR_MANI_ROOT/runtime/framework/tmp"
export GIT_LFS_SKIP_SMUDGE=1
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME"
cd "$OUR_MANI_ROOT/third_party/FASTER"
exec "$OUR_MANI_ROOT/.conda-env/bin/uv" sync --python 3.11 --frozen --no-dev
