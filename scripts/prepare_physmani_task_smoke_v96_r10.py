"""Same baseline policy, with bounded refresh of concurrent directory topology."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
B=ROOT/'artifacts/framework/multitask_r10';OLD=B/'physmani_task_smoke_v85_setup'
SETUP=B/'physmani_task_smoke_v96_setup';OUT=B/'physmani_task_smoke_v96'
OWNER=ROOT/'runtime/framework/native_resource_owner_v19'


def main():
    if SETUP.exists():raise FileExistsError('preserve frozen or failed preparation')
    previous=json.loads((OLD/'CPU_acceptance.json').read_text());assert previous['complete']
    files=dict(json.loads((OLD/'frozen_inputs.json').read_text())['files']);verify_inputs(ROOT,files)
    worker=ROOT/'scripts/run_task_executor_smoke_physmani_v96_r10.py'
    for p in (Path(__file__),worker,ROOT/'our_mani/multitask/frozen_input_index_v93.py',ROOT/'tests/test_frozen_input_index_v93.py'):
        files[str(p.resolve())]=file_sha(p)
    requests=[]
    for p in sorted((OWNER/'requests').glob('*physmani_v85*.json')):
        old=json.loads(p.read_text())
        if (OWNER/'jobs'/old['job_id']).exists():continue
        argv=list(old['argv']);task=argv[argv.index('--task')+1];raw=argv[argv.index('--raw-episode-id')+1]
        argv[2]=str(worker.relative_to(ROOT))
        argv[argv.index('--frozen-inputs')+1]=str((SETUP/'frozen_inputs.json').relative_to(ROOT))
        argv[argv.index('--out')+1]=str((OUT/'development_demo_only'/task/raw).relative_to(ROOT))
        requests.append(dict(task=task,raw_episode_id=raw,original_job_id=old['job_id'],job_id='physmani_v96_'+task,argv=argv))
    assert requests
    SETUP.mkdir();atomic_json(SETUP/'frozen_inputs.json',dict(schema='r10_same_PhysMani_topology_refresh_lock_v96',
        files=files,approved_for_execution=True,formal_VAL_TEST_ready=False,native_started=False,GPU_started=False))
    atomic_json(SETUP/'plan.json',dict(requests=requests,unchanged_policy_and_actuator=True,
        unchanged_configuration=True,unchanged_checkpoint=True,automatic_retry=False,native_started=False))
    # Task-specific source/checkpoint/hook checks have already run for all old
    # v85 argv. Execute the new task-independent wrapper in its actual env once.
    result=subprocess.run(requests[0]['argv']+['--check'],cwd=ROOT,text=True,capture_output=True)
    (SETUP/'CPU_stdout.log').write_text(result.stdout);(SETUP/'CPU_stderr.log').write_text(result.stderr)
    result.check_returncode();d=json.loads(result.stdout)
    assert d['complete'] and d['check_only'] and not d['native_started'] and not d['GPU_started']
    atomic_json(SETUP/'native_CPU_preflight.json',d);verify_inputs(ROOT,files)
    atomic_json(SETUP/'CPU_acceptance.json',dict(complete=True,created_utc=datetime.now(timezone.utc).isoformat(),
        original_all_task_argv_checks=dict(path=str((OLD/'CPU_acceptance.json').relative_to(ROOT)),sha256=file_sha(OLD/'CPU_acceptance.json')),
        actual_new_wrapper_argv_check=requests[0]['argv'],canonical_index_tests_each_environment=10,
        native_started=False,GPU_started=False,policy_and_actuator_unchanged=True,
        bounded_retry_is_only_index_validation_not_native_case=True,formal_release=False))
    print(json.dumps(dict(prepared=True,pending_requests=len(requests),frozen_inputs=len(files),native_started=False)),flush=True)


if __name__=='__main__':main()
