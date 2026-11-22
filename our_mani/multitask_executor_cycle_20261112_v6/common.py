from pathlib import Path
from ..multitask_executor_cycle_20261112_v1.common import (
    ROOT,OUT as PREVIOUS_OUT,PYTHON,read,write,append,pointer,sha,verify,utc,HISTORICAL_STOP,identity)
from ..multitask_executor_cycle_20261112_v5.common import sources as previous_sources
PLAN=ROOT/'config/framework/multitask_executor_cycle_20261112_v6/plan.json'
OUT=ROOT/'artifacts/framework/multitask_executor_cycle_20261112_v6'

def stopped():
    retired=read(OUT/'retired_revision_stop.json')
    if (identity(ROOT/'runtime/framework/STOP')!=HISTORICAL_STOP
            or identity(retired['path'])!=retired['identity']
            or (OUT/'STOP').exists()
            or any((ROOT/f'runtime/framework/multitask_executor_cycle_20261112_v{v}/STOP').exists()for v in range(1,7))):
        raise InterruptedError('Scoped continuous-public-history cycle STOP')

def sources():
    result=previous_sources()
    for path in [*Path(__file__).parent.glob('*.py'),OUT/'calibration_lock.json',OUT/'retired_revision_stop.json',
                 ROOT/'scripts/run_multitask_executor_cycle_native_20261112_v6.sh']:
        result[str(path.resolve())]=sha(path)
    return result
