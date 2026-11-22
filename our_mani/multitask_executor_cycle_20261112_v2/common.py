from pathlib import Path
from ..multitask_executor_cycle_20261112_v1.common import (
    ROOT, PLAN as PREVIOUS_PLAN, OUT as PREVIOUS_OUT, PYTHON, read, write, append, pointer, sha, verify, utc,
    stopped as original_stopped, sources as original_sources)

PLAN = ROOT / 'config/framework/multitask_executor_cycle_20261112_v2/plan.json'
OUT = ROOT / 'artifacts/framework/multitask_executor_cycle_20261112_v2'


def stopped():
    original_stopped()
    if (OUT/'STOP').exists() or (ROOT/'runtime/framework/multitask_executor_cycle_20261112_v2/STOP').exists():
        raise InterruptedError('Scoped v2 cold-reset cycle STOP')


def sources():
    result = original_sources()
    result.update({str(p.resolve()): sha(p) for p in Path(__file__).parent.glob('*.py')})
    return result
