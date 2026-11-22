from pathlib import Path
from ..executor_guided_approach_20261113_v20.common import ROOT,read,write,append,pointer,sha,verify
from ..executor_guided_approach_20261113_v20.common import sources as previous_sources,stopped as previous_stopped
OUT=ROOT/'artifacts/framework/executor_reachable_approach_20261113_v21'

def stopped():
    previous_stopped()
    if (OUT/'STOP').exists()or(ROOT/'runtime/framework/executor_reachable_approach_20261113_v21/STOP').exists():
        raise InterruptedError('Reachable approach scoped STOP')

def sources():
    result=previous_sources()
    paths=[*Path(__file__).parent.glob('*.py'),OUT/'calibration_lock.json',
           OUT/'TRAIN_robot_posture_bank.npz',OUT/'posture_bank_manifest.json',
           ROOT/'scripts/run_executor_reachable_approach_native_20261113_v21.sh']
    result.update({str(p.resolve()):sha(p)for p in paths})
    return result
