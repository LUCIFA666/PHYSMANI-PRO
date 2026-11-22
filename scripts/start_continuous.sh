#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$TASK_ROOT/runtime/continuous/state" "$TASK_ROOT/logs/continuous"
rm -f -- "$TASK_ROOT/runtime/continuous/state/STOP"
systemctl --user link "$TASK_ROOT/runtime/continuous/our-mani-training.service"
systemctl --user daemon-reload
systemctl --user enable --now our-mani-training.service
bash "$TASK_ROOT/scripts/status_continuous.sh"
