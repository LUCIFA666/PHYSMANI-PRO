#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$TASK_ROOT/runtime/continuous/state"
touch "$TASK_ROOT/runtime/continuous/state/STOP"
systemctl --user stop our-mani-training.service
printf '%s\n' '已停止，最近权重和优化器状态已保存在 artifacts/continuous/latest.pt。'
