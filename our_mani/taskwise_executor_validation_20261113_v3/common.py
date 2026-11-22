from pathlib import Path
from ..executor_continuous_posture_20261113_v22.common import ROOT,read,write,append,pointer,sha,verify
from ..executor_continuous_posture_20261113_v22.common import sources as previous_sources,stopped as previous_stopped
OUT=ROOT/'artifacts/framework/taskwise_executor_validation_20261113_v3/place_cups_on_rotating_frame'
PLAN=ROOT/'config/framework/taskwise_executor_validation_20261113_v3/place_cups_on_rotating_frame/stability_plan.json'

def stopped():
    previous_stopped()
    if (OUT/'STOP').exists()or(ROOT/'runtime/framework/taskwise_executor_validation_20261113_v3/STOP').exists():
        raise InterruptedError('Taskwise stability v3 scoped STOP')

def sources():
    result=previous_sources()
    paths=[*Path(__file__).parent.glob('*.py'),PLAN]
    result.update({str(p.resolve()):sha(p)for p in paths})
    return result
