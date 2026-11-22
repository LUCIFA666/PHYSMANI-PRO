from pathlib import Path
from ..executor_held_target_20261113_v7.common import ROOT,read,write,append,pointer,sha,verify
from ..executor_held_target_20261113_v7.common import sources as previous_sources,stopped as previous_stopped
OUT=ROOT/'artifacts/framework/executor_protocol_bridge_20261113_v8'

def stopped():
    previous_stopped()
    if (OUT/'STOP').exists()or(ROOT/'runtime/framework/executor_protocol_bridge_20261113_v8/STOP').exists():
        raise InterruptedError('Protocol bridge scoped STOP')

def sources():
    result=previous_sources()
    paths=[*Path(__file__).parent.glob('*.py'),
           *(ROOT/'our_mani/execution_protocol_20261113_v2').glob('*.py'),
           OUT/'calibration_lock.json',ROOT/'scripts/run_executor_protocol_bridge_native_20261113_v8.sh']
    result.update({str(p.resolve()):sha(p)for p in paths})
    return result
