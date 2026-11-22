from pathlib import Path
from ..executor_reference_unload_20261113_v4.common import ROOT,read,write,append,pointer,sha,verify
from ..executor_reference_unload_20261113_v4.common import sources as previous_sources,stopped as previous_stopped
OUT=ROOT/'artifacts/framework/executor_placement_alignment_20261113_v5'

def stopped():
    previous_stopped()
    if (OUT/'STOP').exists()or(ROOT/'runtime/framework/executor_placement_alignment_20261113_v5/STOP').exists():
        raise InterruptedError('Placement alignment scoped STOP')

def sources():
    result=previous_sources()
    paths=[*Path(__file__).parent.glob('*.py'),OUT/'calibration_lock.json',
           ROOT/'scripts/run_executor_placement_alignment_native_20261113_v5.sh']
    result.update({str(p.resolve()):sha(p)for p in paths})
    return result
