#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
systemctl --user is-active our-mani-training.service || true
"$TASK_ROOT/.conda-env/bin/python" "$TASK_ROOT/scripts/status_continuous.py"
