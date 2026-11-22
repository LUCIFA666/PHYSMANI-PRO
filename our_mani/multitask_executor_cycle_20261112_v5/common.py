from pathlib import Path
from ..multitask_executor_cycle_20261112_v4.common import (
    ROOT, PREVIOUS_OUT, PYTHON, read, write, append, pointer, sha, verify, utc,
    stopped as previous_stopped, sources as previous_sources)
PLAN=ROOT/'config/framework/multitask_executor_cycle_20261112_v5/plan.json'
OUT=ROOT/'artifacts/framework/multitask_executor_cycle_20261112_v5'

def stopped():
    previous_stopped()
    if (OUT/'STOP').exists() or (ROOT/'runtime/framework/multitask_executor_cycle_20261112_v5/STOP').exists():
        raise InterruptedError('Scoped v5 bounded-path cycle STOP')

def sources():
    result=previous_sources()
    for path in [*Path(__file__).parent.glob('*.py'),OUT/'calibration_lock.json',
                 ROOT/'scripts/run_multitask_executor_cycle_native_20261112_v5.sh']:
        result[str(path.resolve())]=sha(path)
    return result
