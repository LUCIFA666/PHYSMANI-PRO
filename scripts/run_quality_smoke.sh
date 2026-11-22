#!/usr/bin/env bash
set -euo pipefail

NGFF_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
NGFF_PYTHON="${NGFF_PYTHON:-$NGFF_PROJECT_ROOT/.conda-env/bin/python}"
export MPLCONFIGDIR="$NGFF_PROJECT_ROOT/.cache/matplotlib"
mkdir -p "$NGFF_PROJECT_ROOT/artifacts/quality_smoke"

"$NGFF_PYTHON" -u "$NGFF_PROJECT_ROOT/scripts/prepare_quality_smoke.py" \
  2>&1 | tee "$NGFF_PROJECT_ROOT/artifacts/quality_smoke/data.log"
"$NGFF_PYTHON" -u "$NGFF_PROJECT_ROOT/scripts/train_quality_smoke.py" \
  2>&1 | tee "$NGFF_PROJECT_ROOT/artifacts/quality_smoke/train.log"
"$NGFF_PYTHON" "$NGFF_PROJECT_ROOT/scripts/summarize_quality_smoke.py"
