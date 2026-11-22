"""Preserve400 policy bytes, add validated owner-compatible entrypoint402."""
from pathlib import Path
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_native_resource_owner_v401_r10 import checked_request
from scripts.run_native_resource_owner_v19_r10 import validate_request
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs


def main():
    base = ROOT/'artifacts/framework/multitask_r10'
    setup = base/'continuous_bridge_v402_setup'
    if setup.exists():
        raise FileExistsError('preserve prior preparation')
    files, checks = {}, []
    for label in ('continuous_buttons_v400', 'continuous_bin_v400'):
        old = base/(label+'_setup')
        for name, sha in json.loads((old/'frozen_inputs.json').read_text())['files'].items():
            if name in files and files[name] != sha:
                raise ValueError('inconsistent inherited frozen bytes')
            files[name] = sha
        checks.extend(json.loads((old/'CPU_acceptance.json').read_text())['actual_argv_preflights'])
    for name in ('scripts/run_task_executor_smoke_continuous_r10.py',
                 'scripts/run_native_resource_owner_v401_r10.py',
                 'scripts/prepare_continuous_bridge_r10.py'):
        files[str(ROOT/name)] = file_sha(ROOT/name)
    verify_inputs(ROOT, files)
    setup.mkdir()
    atomic_json(setup/'frozen_inputs.json', dict(files=files, approved_for_execution=True,
        schema='r10_same400_policy_owner_bridge402_lock', formal_VAL_TEST=False))
    accepted = []
    for index, row in enumerate(checks):
        argv = list(row['argv'])
        argv[2] = 'scripts/run_task_executor_smoke_continuous_r10.py'
        argv[argv.index('--frozen-inputs')+1] = str((setup/'frozen_inputs.json').relative_to(ROOT))
        argv[argv.index('--out')+1] = str((base/'continuous_bridge_v402'/'development_demo_only'/
            row['task']/row['raw']/row['method']).relative_to(ROOT))
        request = dict(schema='r10_owned_native_TRAIN_request_v19',
            job_id='continuous_bridge_v402_'+row['task']+'_'+row['raw']+'_'+row['method'].split('_')[0],
            kind='executor_smoke', argv=argv, use_gpu=False, timeout_seconds=1800,
            source_sha256=file_sha(ROOT/argv[2]))
        validate_request(request)
        result = subprocess.run(argv+['--check'], cwd=ROOT, text=True, capture_output=True)
        if result.returncode:
            atomic_json(setup/'preparation_failed.json', dict(index=index, stderr=result.stderr,
                stdout=result.stdout, native_started=False))
        result.check_returncode()
        proof = json.loads(result.stdout)
        assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
        atomic_json(setup/(request['job_id']+'_CPU_preflight.json'), proof)
        accepted.append(request)
        print(json.dumps(dict(CPU_pass=index+1, total=len(checks), task=row['task'], method=row['method'])), flush=True)
    verify_inputs(ROOT, files)
    atomic_json(setup/'CPU_acceptance.json', dict(complete=True, requests=accepted,
        original400_policy_unchanged=True, rejected400_submission_never_started_native=True,
        formal_VAL_TEST=False))
    owner = ROOT/'runtime/framework/native_resource_owner_v401'
    for index, request in enumerate(accepted):
        atomic_json(owner/'requests'/f'0010_{index}_{request["job_id"]}.json', request)
    atomic_json(setup/'queued_once.json', dict(queued_once=True, jobs=[r['job_id'] for r in accepted],
        automatic_retry=False, native_case_retry=False, formal_VAL_TEST=False))
    print(json.dumps(dict(queued=len(accepted), frozen_inputs=len(files))), flush=True)


if __name__ == '__main__':
    main()
