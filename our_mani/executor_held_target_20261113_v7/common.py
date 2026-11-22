from pathlib import Path
from ..executor_goal_continuity_20261113_v6.common import ROOT,read,write,append,pointer,sha,verify
from ..executor_goal_continuity_20261113_v6.common import sources as previous_sources,stopped as previous_stopped
OUT=ROOT/'artifacts/framework/executor_held_target_20261113_v7'

def stopped():
    previous_stopped()
    if (OUT/'STOP').exists()or(ROOT/'runtime/framework/executor_held_target_20261113_v7/STOP').exists():
        raise InterruptedError('Held target scoped STOP')

def sources():
    result=previous_sources()
    paths=[*Path(__file__).parent.glob('*.py'),OUT/'calibration_lock.json',
           ROOT/'scripts/run_executor_held_target_native_20261113_v7.sh']
    result.update({str(p.resolve()):sha(p)for p in paths})
    return result
