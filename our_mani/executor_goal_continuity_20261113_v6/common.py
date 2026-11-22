from pathlib import Path
from ..executor_placement_alignment_20261113_v5.common import ROOT,read,write,append,pointer,sha,verify
from ..executor_placement_alignment_20261113_v5.common import sources as previous_sources,stopped as previous_stopped
OUT=ROOT/'artifacts/framework/executor_goal_continuity_20261113_v6'

def stopped():
    previous_stopped()
    if (OUT/'STOP').exists()or(ROOT/'runtime/framework/executor_goal_continuity_20261113_v6/STOP').exists():
        raise InterruptedError('Goal continuity scoped STOP')

def sources():
    result=previous_sources()
    paths=[*Path(__file__).parent.glob('*.py'),OUT/'calibration_lock.json',
           ROOT/'scripts/run_executor_goal_continuity_native_20261113_v6.sh']
    result.update({str(p.resolve()):sha(p)for p in paths})
    return result
