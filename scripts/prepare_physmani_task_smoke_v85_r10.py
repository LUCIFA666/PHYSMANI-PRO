"""Freeze indexed verification around the unchanged original PhysMani actor.

Only unstarted requests can be migrated by the separate queue coordinator.
This preparation does not start simulations, touch the old lock, or refit models.
"""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs

B = ROOT/'artifacts/framework/multitask_r10'
PREVIOUS = B/'physmani_task_smoke_v40_setup'
SETUP = B/'physmani_task_smoke_v85_setup'
OUT = B/'physmani_task_smoke_v85'
OWNER = ROOT/'runtime/framework/native_resource_owner_v19'


def main():
    if SETUP.exists():
        raise FileExistsError('preserve previous frozen or failed preparation')
    old_lock = json.loads((PREVIOUS/'frozen_inputs.json').read_text())
    verify_inputs(ROOT, old_lock['files'])
    previous = json.loads((PREVIOUS/'plan.json').read_text())
    files = dict(old_lock['files'])
    worker = ROOT/'scripts/run_task_executor_smoke_physmani_v85_r10.py'
    for p in (Path(__file__), worker, ROOT/'our_mani/multitask/input_stat_cache_v79.py',
              ROOT/'our_mani/multitask/frozen_input_index_v83.py',
              ROOT/'tests/test_input_stat_cache_v79.py', ROOT/'tests/test_frozen_input_index_v83.py'):
        p = p.resolve()
        if str(p) in files:
            assert files[str(p)] == file_sha(p)
        files[str(p)] = file_sha(p)
    requests = []
    for row in previous['requests']:
        row = deepcopy(row)
        argv = row['argv']
        argv[2] = str(worker.relative_to(ROOT))
        argv[argv.index('--frozen-inputs')+1] = str((SETUP/'frozen_inputs.json').relative_to(ROOT))
        argv[argv.index('--out')+1] = str((OUT/'development_demo_only'/row['task']/row['raw_episode_id']).relative_to(ROOT))
        row['original_job_id'] = 'physmani_v40_'+row['task']
        row['job_id'] = 'physmani_v85_'+row['task']
        row['unstarted_at_preparation'] = not (OWNER/'jobs'/row['original_job_id']).exists()
        requests.append(row)
    SETUP.mkdir()
    atomic_json(SETUP/'frozen_inputs.json', dict(
        schema='r10_PhysMani_same_actor_indexed_input_lock_v85', files=files,
        approved_for_execution=True, formal_VAL_TEST_ready=False, native_started=False,
        GPU_started=False, original_frozen_inputs_preserved=len(old_lock['files'])))
    atomic_json(SETUP/'plan.json', dict(
        schema='r10_PhysMani_unstarted_fixed_TRAIN_migration_plan_v85', requests=requests,
        original_plan_sha256=file_sha(PREVIOUS/'plan.json'), tasks=14,
        unchanged_policy=True, unchanged_native_backend=True,
        unchanged_checkpoint=True, unchanged_configuration=True,
        selection=previous['selection'], native_started=False, GPU_started=False,
        automatic_retry=False, formal_metric=False, TEST_read=False))
    try:
        checks = []
        for row in requests:
            if not row['unstarted_at_preparation']:
                continue
            result = subprocess.run(row['argv']+['--check'], cwd=ROOT,
                text=True, capture_output=True, check=True)
            receipt = json.loads(result.stdout)
            assert receipt['complete'] and receipt['check_only']
            assert not receipt['native_started'] and not receipt['GPU_started']
            atomic_json(SETUP/(row['task']+'_CPU_preflight.json'), receipt)
            checks.append(dict(task=row['task'], raw_episode_id=row['raw_episode_id'], argv=row['argv']))
        verify_inputs(ROOT, files)
        atomic_json(SETUP/'CPU_acceptance.json', dict(
            complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
            actual_pending_argv_checks=checks, frozen_inputs=len(files),
            cache_verifier_tests_reused_each_environment=6,
            canonical_index_tests_reused_each_environment=7,
            native_started=False, GPU_started=False, queue_migrated=False,
            policy_and_actuator_unchanged=True, formal_release=False))
    except BaseException as error:
        atomic_json(SETUP/'preparation_failed.json', dict(
            type=type(error).__name__, message=str(error), native_started=False,
            queue_migrated=False))
        raise
    print(json.dumps(dict(prepared=True, pending_checked=len(checks),
        frozen_inputs=len(files), native_started=False, GPU_started=False)), flush=True)


if __name__ == '__main__':
    main()
