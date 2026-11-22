from pathlib import Path
from ..executor_protocol_bridge_20261113_v8.common import ROOT,read,write,append,pointer,sha,verify
from ..executor_protocol_bridge_20261113_v8.common import sources as previous_sources,stopped as previous_stopped
OUT=ROOT/'artifacts/framework/executor_control_lead_20261113_v9'

def stopped():
    previous_stopped()
    if (OUT/'STOP').exists()or(ROOT/'runtime/framework/executor_control_lead_20261113_v9/STOP').exists():
        raise InterruptedError('Control lead scoped STOP')

def sources():
    result=previous_sources()
    paths=[*Path(__file__).parent.glob('*.py'),OUT/'calibration_lock.json',
           ROOT/'scripts/run_executor_control_lead_native_20261113_v9.sh']
    result.update({str(p.resolve()):sha(p)for p in paths})
    return result
