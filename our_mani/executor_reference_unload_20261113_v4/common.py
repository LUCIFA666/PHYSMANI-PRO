from pathlib import Path
from ..executor_contact_frame_20261113_v3.common import ROOT,read,write,append,pointer,sha,verify
from ..executor_contact_frame_20261113_v3.common import sources as previous_sources,stopped as previous_stopped
OUT=ROOT/'artifacts/framework/executor_reference_unload_20261113_v4'

def stopped():
    previous_stopped()
    if (OUT/'STOP').exists()or(ROOT/'runtime/framework/executor_reference_unload_20261113_v4/STOP').exists():
        raise InterruptedError('Reference unload scoped STOP')

def sources():
    result=previous_sources()
    paths=[*Path(__file__).parent.glob('*.py'),OUT/'calibration_lock.json',
           ROOT/'scripts/run_executor_reference_unload_native_20261113_v4.sh']
    result.update({str(p.resolve()):sha(p)for p in paths})
    return result
