from pathlib import Path
from ..executor_grasp_frame_20261113_v1.common import ROOT,read,write,append,pointer,sha,verify
from ..executor_grasp_frame_20261113_v1.common import sources as previous_sources,stopped as previous_stopped
OUT=ROOT/'artifacts/framework/executor_capture_precision_20261113_v2'

def stopped():
    previous_stopped()
    if (OUT/'STOP').exists()or(ROOT/'runtime/framework/executor_capture_precision_20261113_v2/STOP').exists():
        raise InterruptedError('Capture precision scoped STOP')

def sources():
    result=previous_sources()
    paths=[*Path(__file__).parent.glob('*.py'),OUT/'calibration_lock.json',
           ROOT/'scripts/run_executor_capture_precision_native_20261113_v2.sh']
    result.update({str(p.resolve()):sha(p)for p in paths})
    return result
