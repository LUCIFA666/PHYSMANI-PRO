from pathlib import Path
from ..executor_reachable_approach_20261113_v21.common import ROOT,read,write,append,pointer,sha,verify
from ..executor_reachable_approach_20261113_v21.common import sources as previous_sources,stopped as previous_stopped
OUT=ROOT/'artifacts/framework/executor_continuous_posture_20261113_v22'

def stopped():
    previous_stopped()
    if (OUT/'STOP').exists()or(ROOT/'runtime/framework/executor_continuous_posture_20261113_v22/STOP').exists():
        raise InterruptedError('Continuous posture scoped STOP')

def sources():
    result=previous_sources()
    paths=[*Path(__file__).parent.glob('*.py'),OUT/'calibration_lock.json',
           OUT/'TRAIN_robot_posture_bank.npz',OUT/'posture_bank_manifest.json',
           ROOT/'scripts/run_executor_continuous_posture_native_20261113_v22.sh']
    result.update({str(p.resolve()):sha(p)for p in paths})
    return result
