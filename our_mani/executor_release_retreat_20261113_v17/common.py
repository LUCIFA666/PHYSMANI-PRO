from pathlib import Path
from ..executor_drop_visual_20261113_v16.common import ROOT,read,write,append,pointer,sha,verify
from ..executor_drop_visual_20261113_v16.common import sources as previous_sources,stopped as previous_stopped
OUT=ROOT/'artifacts/framework/executor_release_retreat_20261113_v17'

def stopped():
    previous_stopped()
    if (OUT/'STOP').exists()or(ROOT/'runtime/framework/executor_release_retreat_20261113_v17/STOP').exists():
        raise InterruptedError('Release retreat scoped STOP')

def sources():
    result=previous_sources()
    paths=[*Path(__file__).parent.glob('*.py'),OUT/'calibration_lock.json',
           OUT/'TRAIN_robot_posture_bank.npz',OUT/'posture_bank_manifest.json',
           ROOT/'scripts/run_executor_release_retreat_native_20261113_v17.sh']
    result.update({str(p.resolve()):sha(p)for p in paths})
    return result
