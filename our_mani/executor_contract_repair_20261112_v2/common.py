from pathlib import Path
from ..executor_contract_repair_20261112_v1.common import ROOT, read, write, append, pointer, sha, verify, identity, HISTORICAL_STOP
from ..executor_contract_repair_20261112_v1.common import sources as previous_sources

OUT = ROOT/'artifacts/framework/executor_contract_repair_20261112_v2'

def stopped():
    retired=read(OUT/'retired_revision_stops.json')
    if (identity(ROOT/'runtime/framework/STOP')!=HISTORICAL_STOP or
            any(identity(r['path'])!=r['identity'] for r in retired) or
            (OUT/'STOP').exists() or (ROOT/'runtime/framework/executor_contract_repair_20261112_v2/STOP').exists()):
        raise InterruptedError('Continuous-reference executor STOP')

def sources():
    result=previous_sources()
    paths=[*Path(__file__).parent.glob('*.py'),OUT/'calibration_lock.json',OUT/'retired_revision_stops.json',
           ROOT/'scripts/run_executor_contract_repair_native_20261112_v2.sh',
           ROOT/'scripts/run_executor_policy_repair_native_20261112_v2.sh',
           ROOT/'scripts/run_executor_reference_replay_native_20261112_v2.sh']
    result.update({str(p.resolve()):sha(p)for p in paths})
    return result
