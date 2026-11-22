from pathlib import Path
from ..executor_release_order_proposal_20261113_v19.common import ROOT,read,write,append,pointer,sha,verify
from ..executor_release_order_proposal_20261113_v19.common import sources as previous_sources,stopped as previous_stopped
OUT=ROOT/'artifacts/framework/executor_release_continuous_ik_20261113_v23'

def stopped():
    previous_stopped()
    if (OUT/'STOP').exists()or(ROOT/'runtime/framework/executor_release_continuous_ik_20261113_v23/STOP').exists():
        raise InterruptedError('Release continuous IK scoped STOP')

def sources():
    result=previous_sources()
    extra=ROOT/'our_mani/executor_continuous_posture_20261113_v22/posture.py'
    paths=[*Path(__file__).parent.glob('*.py'),extra,OUT/'calibration_lock.json',
           OUT/'TRAIN_robot_posture_bank.npz',OUT/'posture_bank_manifest.json',
           ROOT/'scripts/run_executor_release_continuous_ik_native_20261113_v23.sh']
    result.update({str(p.resolve()):sha(p)for p in paths})
    return result
