"""Fixed reserved16 evaluation queue; no generation/Native in default check.

Execution needs later Root-authorized materialization and a prospectively
frozen exact V64 source/model/config map. Original public acceptance is FAIL.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.local_mpc_20261103 import position_v64_test16_protocol as protocol
HOST=ROOT/'scripts/launch_local_mpc_position_v64_test16_20261103.py'


def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def append(path,row):
    with path.open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()


def verify_sources(ref):
    source=protocol.checked(ref);sources=load(source)['files_sha256']
    if any(sha(ROOT/p)!=v for p,v in sources.items()):raise ValueError('Frozen TEST16 source/model/config changed')
    return sources


def check_execution_plan(path):
    path=Path(path).resolve(strict=True)
    if path!=protocol.EXECUTION_PLAN.resolve(strict=True):raise ValueError('Only the independently authorized fixed16 execution plan allowed')
    plan=load(path);profile,preparation=protocol.metadata();receipt=load(protocol.checked(plan['materialization_receipt']))
    if (plan.get('schema')!='V64_reserved_TEST16_evaluation_only_execution_plan_v1'
        or plan.get('evaluation_only')is not True or plan.get('public_hold_canary_gate',{}).get('passed')is not False
        or plan.get('partition')!=preparation['partition']or plan.get('protocol')!=preparation['protocol']
        or plan.get('all16_regardless_of_task_outcomes')is not True
        or plan.get('raw_model_qualified')is not False or plan.get('formal_pose_real_time_force_qualified')is not False
        or receipt.get('schema')!='V64_reserved_TEST16_evaluation_materialization_receipt_v1'
        or receipt.get('complete')is not True or receipt.get('evaluation_only')is not True
        or receipt.get('public_hold_gate_passed')is not False or receipt.get('source_freeze')!=plan['source_freeze']
        or receipt.get('partition')!=plan['partition']or receipt.get('protocol')!=plan['protocol']):
        raise ValueError('Separate evaluation authorization cannot impersonate old quality gate pass')
    expected=[{k:v for k,v in j.items()if k!='assignment'}for j in preparation['jobs']]
    if plan['jobs']!=expected or len(receipt['configs'])!=16:raise ValueError('All16 original parents/order/motion must be retained')
    if protocol.checked(receipt['pending_materialization'])!=protocol.PENDING_RECEIPT.resolve(strict=True):
        raise ValueError('Original configuration materialization record required')
    sources=verify_sources(plan['source_freeze'])
    from our_mani.local_mpc_20261103.position_v64_test16_worker import check_config
    for row,job in zip(receipt['configs'],plan['jobs']):
        if protocol.checked(row['config'])!=(ROOT/job['config']).resolve(strict=True):raise ValueError('Fixed config pointer differs')
        output=ROOT/job['path'];prefix=output.parent/(output.name+'.owner')
        if output.exists()or any(Path(str(prefix)+x).exists()for x in ('.json','.log','.actual_exit.json')):
            raise FileExistsError('No TEST resumption/task retry or replacement')
        if any(p.exists()for p in (output/'STOP',output.parent/'STOP',output.parent/(output.name+'.STOP'),ROOT/'runtime/framework/local_mpc_20261102/STOP')):
            raise ValueError('Administrative STOP is preserved')
        report=check_config(ROOT/job['config'],job['scene'],source_freeze=ROOT/plan['source_freeze']['path'],
            expected_source_freeze_sha256=plan['source_freeze']['sha256'],require_ready=True)
        if report.get('ready')is not True:raise ValueError('Genuine position-only admission required')
    return plan,sources


def audit_closed(job,sources):
    output=ROOT/job['path'];prefix=output.parent/(output.name+'.owner')
    owner=load(Path(str(prefix)+'.actual_exit.json'));worker=load(output/'worker_exit.json')
    result=load(output/'result.json');run=load(output/'run_inputs.json')
    if (owner.get('actual_process_returncode')!=0 or owner.get('actual_wait_completed')is not True
        or owner.get('source_unchanged')is not True or owner.get('stop_reason')is not None
        or worker.get('native_closed')is not True or worker.get('exit_code')!=0
        or result.get('complete')is not True or result.get('source_unchanged')is not True
        or run.get('source_sha256')!=sources or run.get('config')!=load(ROOT/job['config'])):
        raise ValueError('Actual clean closure/source identity required before next case')
    return dict(case_id=job['id'],actual_process_returncode=0,native_closed=True,
        public_hold_final=result.get('public_lift_hold_observed'),task_outcome_used_for_retry=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--acknowledge-public-hold-gate-failed',action='store_true')
    args=p.parse_args(argv)
    if args.plan is None:
        if args.execute:p.error('later authorized materialized/frozen execution plan required')
        print(json.dumps(protocol.check_preparation(),allow_nan=False));return 0
    plan,sources=check_execution_plan(args.plan)
    if not args.execute:
        print(json.dumps(dict(complete=True,ready=True,Native_started=False,cases=16,public_hold_gate_passed=False,source_freeze=plan['source_freeze']),allow_nan=False));return 0
    if not args.acknowledge_public_hold_gate_failed:p.error('evaluation-only public gate FAIL acknowledgement required')
    root=ROOT/'runtime/framework/local_mpc_20261102/P4_frozen_V64_newTEST16_eval_v1';root.mkdir(parents=True,exist_ok=False)
    ledger=root/'serial_queue_v1.jsonl';plan_sha=sha(args.plan)
    with ledger.open('x')as f:f.write(json.dumps(dict(event='start',utc=now(),fixed_order=[j['id']for j in plan['jobs']],public_hold_gate_passed=False,evaluation_only=True,execution_plan_sha256=plan_sha))+'\n')
    try:
        for job in plan['jobs']:
            verify_sources(plan['source_freeze'])
            if sha(args.plan)!=plan_sha:raise ValueError('Fixed execution plan changed during TEST')
            command=[sys.executable,'-B',str(HOST),'--config',str(ROOT/job['config']),'--scene',job['scene'],
                '--output',str(ROOT/job['path']),'--source-freeze',str(ROOT/plan['source_freeze']['path']),
                '--source-freeze-sha256',plan['source_freeze']['sha256'],'--execute']
            completed=subprocess.run(command,cwd=ROOT,check=False)
            if completed.returncode!=0:raise RuntimeError('Infrastructure/administrative stop, no retry: '+job['id'])
            outcome=audit_closed(job,sources);verify_sources(plan['source_freeze'])
            if sha(args.plan)!=plan_sha:raise ValueError('Fixed execution plan changed during TEST')
            append(ledger,dict(event='actual_closed',utc=now(),**outcome))
        append(ledger,dict(event='complete',utc=now(),all16_actual_closed=True,no_task_retry_or_case_replacement=True,public_hold_gate_passed=False));return 0
    except BaseException as error:
        append(ledger,dict(event='infrastructure_stop',utc=now(),error_type=type(error).__name__,error=str(error),no_retry=True));raise


if __name__=='__main__':raise SystemExit(main())
