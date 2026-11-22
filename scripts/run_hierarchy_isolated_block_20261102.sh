#!/usr/bin/env bash
set -eo pipefail
OUR_MANI_BLOCK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUR_MANI_BLOCK_CASE="${1:?explicit new case runtime required}"
[[ "$OUR_MANI_BLOCK_CASE" == /* ]] || { echo "absolute case runtime required" >&2; exit 2; }
shift
[[ "${1:-}" == "--" ]] || { echo "expected -- before frozen wrapper argv" >&2; exit 2; }
shift
[[ "$#" -gt 0 ]] || { echo "explicit frozen wrapper argv required" >&2; exit 2; }
OUR_MANI_BLOCK_SLOT=""
OUR_MANI_BLOCK_ARGS=("$@")
for ((OUR_MANI_BLOCK_INDEX=0; OUR_MANI_BLOCK_INDEX<$#; OUR_MANI_BLOCK_INDEX++)); do
  if [[ "${OUR_MANI_BLOCK_ARGS[$OUR_MANI_BLOCK_INDEX]}" == "--resource-slot" ]]; then
    [[ -z "$OUR_MANI_BLOCK_SLOT" ]] || { echo "duplicate resource slot" >&2; exit 2; }
    OUR_MANI_BLOCK_SLOT="${OUR_MANI_BLOCK_ARGS[$((OUR_MANI_BLOCK_INDEX+1))]:-}"
  fi
done
[[ "$OUR_MANI_BLOCK_SLOT" =~ ^[0-3]$ ]] || { echo "one explicit resource slot 0..3 required" >&2; exit 2; }
source /home/ubuntu/桌面/phsimani/physmani_reproduction/runtime.sh
export PYTHONPATH="$OUR_MANI_BLOCK_ROOT:$PYTHONPATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
OUR_MANI_BLOCK_PARENT_PID="$BASHPID"
OUR_MANI_BLOCK_CPUS="$(python -B "$OUR_MANI_BLOCK_ROOT/our_mani/hierarchy_block_20261102/isolation.py" --prepare-shell "$OUR_MANI_BLOCK_ROOT" "$OUR_MANI_BLOCK_CASE" "$OUR_MANI_BLOCK_SLOT" "$OUR_MANI_BLOCK_PARENT_PID" -- "$@")"
export TMPDIR="$OUR_MANI_BLOCK_CASE/tmp" XDG_CACHE_HOME="$OUR_MANI_BLOCK_CASE/cache"
export XDG_RUNTIME_DIR="$OUR_MANI_BLOCK_CASE/xdg-runtime" XDG_CONFIG_HOME="$OUR_MANI_BLOCK_CASE/config"
cd "$OUR_MANI_BLOCK_ROOT"
# The explicit new wrapper performs one bwrap with frozen OS mounts+private
# network, then Xvfb. Serial mode explicitly selects its original-v820 branch.
exec /usr/bin/taskset --cpu-list "$OUR_MANI_BLOCK_CPUS" python "$@"
