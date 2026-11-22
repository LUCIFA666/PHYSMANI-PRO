from pathlib import Path
from ..executor_contract_repair_20261112_v5.common import ROOT,read,write,append,pointer,sha,verify
from ..executor_contract_repair_20261112_v5.common import sources as previous_sources,stopped as previous_stopped
OUT=ROOT/'artifacts/framework/executor_contract_repair_20261112_v6'

def stopped():
    previous_stopped()
    if (OUT/'STOP').exists()or(ROOT/'runtime/framework/executor_contract_repair_20261112_v6/STOP').exists():
        raise InterruptedError('Executor timing STOP')

def sources():
    result=previous_sources()
    paths=[*Path(__file__).parent.glob('*.py'),OUT/'calibration_lock.json',
           ROOT/'scripts/run_executor_online_teacher_native_20261112_v6.sh',
           ROOT/'scripts/run_executor_policy_repair_native_20261112_v6.sh',
           ROOT/'scripts/run_repaired_button_executor_20261112_v6.py']
    result.update({str(p.resolve()):sha(p)for p in paths})
    return result
