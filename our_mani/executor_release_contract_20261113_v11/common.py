from pathlib import Path
from ..executor_robot_posture_20261113_v10.common import ROOT,read,write,append,pointer,sha,verify
from ..executor_robot_posture_20261113_v10.common import sources as previous_sources,stopped as previous_stopped
OUT=ROOT/'artifacts/framework/executor_release_contract_20261113_v11'

def stopped():
    previous_stopped()
    if (OUT/'STOP').exists()or(ROOT/'runtime/framework/executor_release_contract_20261113_v11/STOP').exists():
        raise InterruptedError('Release contract scoped STOP')

def sources():
    result=previous_sources()
    paths=[*Path(__file__).parent.glob('*.py'),OUT/'calibration_lock.json',
           ROOT/'scripts/run_executor_release_contract_native_20261113_v11.sh']
    result.update({str(p.resolve()):sha(p)for p in paths})
    return result
