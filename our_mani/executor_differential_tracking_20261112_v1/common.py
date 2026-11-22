from pathlib import Path
from ..executor_cartesian_tracking_20261112_v1.common import ROOT,read,write,append,pointer,sha,verify
from ..executor_cartesian_tracking_20261112_v1.common import sources as previous_sources,stopped as previous_stopped
OUT=ROOT/'artifacts/framework/executor_differential_tracking_20261112_v1'

def stopped():
    previous_stopped()
    if (OUT/'STOP').exists()or(ROOT/'runtime/framework/executor_differential_tracking_20261112_v1/STOP').exists():
        raise InterruptedError('Executor timing STOP')

def sources():
    result=previous_sources()
    paths=[*Path(__file__).parent.glob('*.py'),OUT/'calibration_lock.json',
           ROOT/'scripts/run_executor_differential_native_20261112_v1.sh']
    result.update({str(p.resolve()):sha(p)for p in paths})
    return result
