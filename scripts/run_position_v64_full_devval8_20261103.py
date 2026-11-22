"""Complete reused-development V64 evaluation, despite retained public gate FAIL.

Default check only. Exactly eight original V56 assignments run once in order;
task false never causes retries/replacement, while infrastructure errors stop.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PLAN=ROOT/'config/framework/P4_student_closedloop_devVAL_v64_v1/plan.json'
HOST=ROOT/'scripts/launch_local_mpc_position_v64_devval8_20261103.py'
EXPECTED=('00_val_static','01_val_static','02_val_uniform','03_val_uniform',
          '04_val_uniform','05_val_uniform','06_val_speed_change','07_val_speed_change')


def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def append(path,row):
    with path.open('a')as stream:stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()


def check_plan(source_freeze=None,expected_sha256=None):
    from our_mani.local_mpc_20261103.position_v64_devval8_worker import check_config
    plan=load(PLAN)
    if (tuple(j['id']for j in plan['jobs'])!=EXPECTED
        or plan.get('protocol_revision')!='development_system_evaluation_while_public_hold_canary_gate_failed_v1'
        or plan.get('public_hold_canary_gate',{}).get('passed')is not False
        or plan.get('all8_regardless_of_task_outcomes')is not True
        or plan.get('task_failure_retry')is not False or plan.get('case_replacement')is not False
        or plan.get('reuse_development')is not True or plan.get('formal_TEST_used')is not False):
        raise ValueError('Exact fixed reused-development protocol with public gate FAIL preserved required')
    reports=[]
    for job in plan['jobs']:
        output=ROOT/job['path'];prefix=output.parent/(output.name+'.owner')
        if output.exists()or any(Path(str(prefix)+suffix).exists()for suffix in ('.json','.log','.actual_exit.json')):
            raise FileExistsError('All8 cases must be fresh, no retry/resumption/replacement')
        if any(p.exists()for p in (output/'STOP',output.parent/'STOP',output.parent/(output.name+'.STOP'),ROOT/'runtime/framework/local_mpc_20261102/STOP')):
            raise ValueError('Administrative STOP is preserved')
        report=check_config(ROOT/job['config'],job['scene'],source_freeze=source_freeze,
                            expected_source_freeze_sha256=expected_sha256)
        reports.append({k:v for k,v in report.items()if k!='source_sha256'})
    return plan,reports


def verify_sources(path,expected):
    if sha(path)!=expected:raise ValueError('Source freeze SHA changed')
    sources=load(path)['files_sha256']
    if any(sha(ROOT/p)!=digest for p,digest in sources.items()):raise ValueError('Frozen code/model/config changed')
    return sources


def audit_closed(job,sources):
    output=ROOT/job['path'];prefix=output.parent/(output.name+'.owner')
    owner=load(Path(str(prefix)+'.actual_exit.json'));worker=load(output/'worker_exit.json')
    result=load(output/'result.json');run=load(output/'run_inputs.json')
    if (owner.get('actual_process_returncode')!=0 or owner.get('actual_wait_completed')is not True
        or owner.get('source_unchanged')is not True or owner.get('stop_reason')is not None
        or worker.get('exit_code')!=0 or worker.get('native_closed')is not True
        or worker.get('result_recorded')is not True or result.get('complete')is not True
        or result.get('source_unchanged')is not True or run.get('source_sha256')!=sources
        or run.get('config')!=load(ROOT/job['config'])):
        raise ValueError('Clean actual exit/Native closure/source/assignment identity required')
    return dict(case_id=job['id'],actual_process_returncode=0,native_closed=True,
                public_hold_final=result.get('public_lift_hold_observed'),
                public_stop_reason=result.get('canary_stop_reason'),
                task_outcome_used_for_retry_or_case_selection=False)


def command(job,args):
    return [sys.executable,'-B',str(HOST),'--config',str(ROOT/job['config']),'--scene',job['scene'],
            '--output',str(ROOT/job['path']),'--source-freeze',str(args.source_freeze.resolve()),
            '--source-freeze-sha256',args.source_freeze_sha256,'--execute']


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-freeze',type=Path);parser.add_argument('--source-freeze-sha256')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--acknowledge-public-hold-canary-failed',action='store_true')
    args=parser.parse_args(argv)
    plan,reports=check_plan(args.source_freeze,args.source_freeze_sha256)
    if not args.execute:
        print(json.dumps(dict(complete=True,ready=all(r['ready']for r in reports),
            Native_started=False,public_hold_canary_gate_passed=False,reused_development=True,
            source_freeze_sha256=args.source_freeze_sha256,cases=reports),allow_nan=False));return 0
    if not args.acknowledge_public_hold_canary_failed:
        parser.error('system evaluation requires explicit acknowledgement that public-hold acceptance remains FAIL')
    if args.source_freeze is None or not args.source_freeze_sha256 or not all(r['ready']for r in reports):
        parser.error('real complete prospective sourcefreeze/admission required')
    sources=verify_sources(args.source_freeze,args.source_freeze_sha256)
    root=ROOT/'runtime/framework/local_mpc_20261102/P4_student_closedloop_devVAL_v64_v1'
    root.mkdir(parents=True,exist_ok=False);ledger=root/'serial_queue_v1.jsonl'
    with ledger.open('x')as stream:stream.write(json.dumps(dict(event='start',utc=now(),fixed_order=list(EXPECTED),public_hold_canary_gate_passed=False,acknowledged_failed_public_hold_gate=True,reused_development=True,queue_source_sha256=sha(__file__)))+'\n')
    try:
        for job in plan['jobs']:
            verify_sources(args.source_freeze,args.source_freeze_sha256)
            completed=subprocess.run(command(job,args),cwd=ROOT,check=False)
            if completed.returncode!=0:raise RuntimeError('Infrastructure/administrative failure; stop without retry: '+job['id'])
            row=audit_closed(job,sources);verify_sources(args.source_freeze,args.source_freeze_sha256)
            append(ledger,dict(event='actual_closed',utc=now(),**row))
        append(ledger,dict(event='complete',utc=now(),all8_actual_closed=True,no_task_retries_or_case_replacement=True,public_hold_canary_gate_passed=False))
        return 0
    except BaseException as error:
        append(ledger,dict(event='infrastructure_stop',utc=now(),error_type=type(error).__name__,error=str(error),no_retry=True));raise


if __name__=='__main__':raise SystemExit(main())
