"""Fixed remaining V64 development cases; check only unless --execute.

The host launcher waits and closes Native for every case. A task result of false
does not stop or replace a case; infrastructure/closure/source errors do stop.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'config/framework/P4_position_v64_development_canaries_v1/plan.json'
PLAN_SHA256='8da608d8a968c4cf8406c430c23f70a7966a424306fd13f4be3a6ca375dcfbca'
FREEZE=ROOT/'artifacts/framework/local_mpc_20261102/V64_independent_worker_preparation_v1/execution_source_freeze.json'
FREEZE_SHA256='39dedd0a5d8a80489a2e46c48aff41ef7496a26bbdcc60965597afa22f705050'
HOST=ROOT/'scripts/launch_local_mpc_position_v64_20261103.py'
EXPECTED=('00_val_static','03_val_uniform','uniform_002','static_no_proxy')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def append(path,row):
    with path.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()


def verify_sources():
    if sha(PLAN)!=PLAN_SHA256 or sha(FREEZE)!=FREEZE_SHA256:
        raise ValueError('Original fixed V64 plan/freeze changed')
    sources=load(FREEZE)['files_sha256']
    if any(sha(ROOT/name)!=value for name,value in sources.items()):
        raise ValueError('Frozen V64 source/model/config changed')
    return sources


def audit_closed(job,sources):
    output=ROOT/job['path'];prefix=output.parent/(output.name+'.owner')
    owner=load(Path(str(prefix)+'.actual_exit.json'))
    worker=load(output/'worker_exit.json');result=load(output/'result.json');inputs=load(output/'run_inputs.json')
    if (owner.get('actual_process_returncode')!=0 or owner.get('actual_wait_completed')is not True
        or owner.get('source_unchanged')is not True or owner.get('stop_reason')is not None
        or worker.get('exit_code')!=0 or worker.get('native_closed')is not True
        or worker.get('result_recorded')is not True or result.get('complete')is not True
        or result.get('source_unchanged')is not True
        or inputs.get('source_sha256')!=sources or inputs.get('config')!=load(ROOT/job['config'])):
        raise ValueError('Actual clean exit/native closure/exact frozen case identity required')
    if job['id']=='static_no_proxy':
        config=inputs['config']
        if config.get('diagnostic_no_proxy_attachment')is not True or config.get('counterfactual_not_benchmark')is not True:
            raise ValueError('Original NoProxy counterfactual negative cannot change')
    return dict(case_id=job['id'],actual_process_returncode=0,native_closed=True,
        source_unchanged=True,public_hold_final=result.get('public_lift_hold_observed'),
        public_stop_reason=result.get('canary_stop_reason'),
        task_success_used_to_select_or_retry=False)


def check_queue():
    sources=verify_sources();plan=load(PLAN);jobs=plan['jobs']
    if tuple(j['id']for j in jobs)!=EXPECTED:raise ValueError('Fixed four original development parents required')
    first=audit_closed(jobs[0],sources)
    for job in jobs[1:]:
        output=ROOT/job['path'];prefix=output.parent/(output.name+'.owner')
        if output.exists() or any(Path(str(prefix)+suffix).exists()for suffix in ('.json','.log','.actual_exit.json')):
            raise FileExistsError('Remaining cases must be fresh; no resumption or retry')
        if any(p.exists()for p in (output/'STOP',output.parent/(output.name+'.STOP'),output.parent/'STOP',ROOT/'runtime/framework/local_mpc_20261102/STOP')):
            raise ValueError('Administrative STOP is preserved')
        if job['id']=='static_no_proxy':
            config=load(ROOT/job['config'])
            if config.get('diagnostic_no_proxy_attachment')is not True or config.get('counterfactual_not_benchmark')is not True:
                raise ValueError('Original counterfactual NoProxy negative required')
    return sources,jobs[1:],first


def command(job):
    return [sys.executable,'-B',str(HOST),'--config',str(ROOT/job['config']),'--scene',job['scene'],
        '--output',str(ROOT/job['path']),'--source-freeze',str(FREEZE),
        '--source-freeze-sha256',FREEZE_SHA256,'--execute']


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args(argv)
    sources,jobs,first=check_queue()
    if not args.execute:
        print(json.dumps(dict(complete=True,Native_started=False,first_actual_closed=first,
            source_freeze_sha256=FREEZE_SHA256,jobs=jobs,commands=[command(j)for j in jobs]),allow_nan=False));return 0
    root=ROOT/'runtime/framework/local_mpc_20261102/public_candidate_position_v64_development_v1'
    ledger=root/'remaining3_queue_v1.jsonl'
    with ledger.open('x')as stream:stream.write(json.dumps(dict(event='start',utc=now(),queue_source_sha256=sha(__file__),fixed_order=[j['id']for j in jobs],first_actual_closed=first))+'\n')
    try:
        for job in jobs:
            verify_sources()
            completed=subprocess.run(command(job),cwd=ROOT,check=False)
            if completed.returncode!=0:raise RuntimeError('Host infrastructure/administrative error; no next case: '+job['id'])
            outcome=audit_closed(job,sources);verify_sources()
            append(ledger,dict(event='actual_closed',utc=now(),**outcome))
        append(ledger,dict(event='complete',utc=now(),complete=True,all3_actual_closed=True,no_case_replacement_or_retry=True))
        return 0
    except BaseException as error:
        append(ledger,dict(event='infrastructure_stop',utc=now(),complete=False,error_type=type(error).__name__,error=str(error),no_retry=True))
        raise


if __name__=='__main__':raise SystemExit(main())
