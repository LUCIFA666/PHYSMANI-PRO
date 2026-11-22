from pathlib import Path
from ..multitask_executor_cycle_20261112_v1.common import (
    ROOT, PLAN as PREVIOUS_PLAN, OUT as PREVIOUS_OUT, PYTHON, read, write, append, pointer, sha, verify, utc,
    stopped as original_stopped, sources as original_sources)

PLAN = ROOT / 'config/framework/multitask_executor_cycle_20261112_v3/plan.json'
OUT = ROOT / 'artifacts/framework/multitask_executor_cycle_20261112_v3'


def stopped():
    original_stopped()
    if (OUT/'STOP').exists() or (ROOT/'runtime/framework/multitask_executor_cycle_20261112_v3/STOP').exists():
        raise InterruptedError('Scoped v2 cold-reset cycle STOP')


def sources():
    result = original_sources()
    result.update({str(p.resolve()): sha(p) for p in Path(__file__).parent.glob('*.py')})
    path = ROOT/'scripts/run_multitask_executor_cycle_native_20261112_v3.sh'
    result[str(path)] = sha(path)
    native = Path('/home/ubuntu/桌面/phsimani/PhysMani')
    for path in [native/'third_party/PyRep/pyrep/robots/configuration_paths/arm_configuration_path.py',
                 native/'third_party/PyRep/pyrep/robots/arms/arm.py',
                 native/'third_party/RLBench/rlbench/backend/scene.py',
                 native/'third_party/RLBench/rlbench/task_environment.py',
                 OUT/'calibration_lock.json']:
        result[str(path.resolve())] = sha(path)
    return result
