"""Run all eight fixed V64 development assignments with V65, never TEST.

Default checks only. A complete prospective source freeze is mandatory for
execution. Task failures do not cause retries, replacement or early stopping.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = ROOT/'config/framework/P4_student_closedloop_devVAL_v65_v1/plan.json'
HOST = ROOT/'scripts/launch_local_mpc_position_v65_devval8_20261103.py'
EXPECTED = ('00_val_static', '01_val_static', '02_val_uniform', '03_val_uniform', '04_val_uniform', '05_val_uniform', '06_val_speed_change', '07_val_speed_change')


def load(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n'); stream.flush()


def verify_sources(path, digest):
    if sha(path) != digest: raise ValueError('Execution freeze changed')
    sources = load(path)['files_sha256']
    if any(sha(ROOT/name) != value for name, value in sources.items()):
        raise ValueError('Prospectively frozen code/model/config changed')
    return sources


def check_plan(args):
    from our_mani.local_mpc_20261103.position_v65_devval8_worker import check_config
    plan = load(PLAN)
    if (tuple(j['id'] for j in plan['jobs']) != EXPECTED
            or plan.get('all8_regardless_of_task_outcomes') is not True
            or plan.get('retry_or_case_replacement_based_on_outcome') is not False
            or plan.get('formal_TEST_read') is not False
            or plan.get('natural_witness_only') is not True
            or plan.get('active_new_probe_execution') is not False
            or plan.get('evaluation_only') is not True or plan.get('reuse_development') is not True
            or plan.get('public_hold_canary_gate',{}).get('passed')is not False):
        raise ValueError('Fixed eight reused-development cases and natural-only witness required')
    reports = []
    for job in plan['jobs']:
        output = ROOT/job['path']; prefix = output.with_name(output.name+'.owner')
        if output.exists() or any(Path(str(prefix)+s).exists() for s in ('.json','.log','.actual_exit.json')):
            raise FileExistsError('Fresh cases required; never resume or retry')
        if any(p.exists() for p in (output/'STOP',output.parent/'STOP',output.with_name(output.name+'.STOP'),ROOT/'runtime/framework/local_mpc_20261102/STOP')):
            raise ValueError('Administrative STOP is preserved')
        report = check_config(ROOT/job['config'], job['scene'],
            source_freeze=args.source_freeze,
            expected_source_freeze_sha256=args.source_freeze_sha256)
        if report.get('case_id') != job['id']:
            raise ValueError('Plan id and authenticated referenced case must match')
        reports.append({k:v for k,v in report.items() if k!='source_sha256'})
    return plan, reports


def audit_closed(job, sources):
    output = ROOT/job['path']
    owner = load(output.with_name(output.name+'.owner.actual_exit.json'))
    worker = load(output/'worker_exit.json'); result = load(output/'result.json')
    inputs = load(output/'run_inputs.json')
    if (type(owner.get('actual_process_returncode')) is not int or owner['actual_process_returncode'] != 0
            or owner.get('case_id') != job['id'] or Path(owner.get('output','')).resolve() != output.resolve()
            or owner.get('actual_wait_completed') is not True
            or owner.get('source_unchanged') is not True or owner.get('stop_reason') is not None
            or type(worker.get('exit_code')) is not int or worker['exit_code'] != 0
            or worker.get('native_closed') is not True
            or worker.get('result_recorded') is not True or result.get('complete') is not True
            or result.get('source_unchanged') is not True
            or result.get('canary_stop_reason') not in ('budget_exhausted',
                'visible_public_grasp_and_goal_complete', 'visible_public_lift_hold_observed')
            or inputs.get('source_sha256') != sources or inputs.get('config') != load(ROOT/job['config'])):
        raise ValueError('Actual clean closure and fixed source/config required')
    return dict(case_id=job['id'],actual_process_returncode=0,native_closed=True,
        source_unchanged=True,task_outcome_used_for_retry_or_selection=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-freeze', type=Path)
    parser.add_argument('--source-freeze-sha256')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--acknowledge-public-hold-canary-failed',action='store_true')
    args = parser.parse_args(argv)
    plan, reports = check_plan(args)
    if not args.execute:
        print(json.dumps(dict(complete=True,ready=all(r['ready'] for r in reports),
            Native_started=False,formal_TEST_read=False,cases=reports),allow_nan=False)); return 0
    if (not args.acknowledge_public_hold_canary_failed or plan.get('execution_authorized')is not True
            or plan.get('execution_ready')is not True):
        parser.error('Root authorization and acknowledgement of retained failed public gate required')
    if not args.source_freeze or not args.source_freeze_sha256 or not all(r['ready'] for r in reports):
        parser.error('Real complete prospective freeze/admission required')
    sources = verify_sources(args.source_freeze,args.source_freeze_sha256)
    queue = ROOT/'runtime/framework/local_mpc_20261102/P4_student_closedloop_devVAL_v65_v1'
    queue.mkdir(parents=True,exist_ok=False); ledger = queue/'serial_queue_v1.jsonl'
    append(ledger,dict(event='start',utc=now(),fixed_order=list(EXPECTED),
        queue_source_sha256=sha(__file__),formal_TEST_read=False))
    try:
        for job in plan['jobs']:
            output = ROOT/job['path']
            if any(p.exists() for p in (output/'STOP',output.parent/'STOP',
                    output.with_name(output.name+'.STOP'),ROOT/'runtime/framework/local_mpc_20261102/STOP')):
                raise ValueError('Administrative STOP prevents every subsequent case')
            verify_sources(args.source_freeze,args.source_freeze_sha256)
            command = [sys.executable,'-B',str(HOST),'--config',str(ROOT/job['config']),
                '--scene',job['scene'],'--output',str(ROOT/job['path']),
                '--source-freeze',str(args.source_freeze.resolve()),
                '--source-freeze-sha256',args.source_freeze_sha256,'--execute']
            outcome = subprocess.run(command,cwd=ROOT,check=False)
            if outcome.returncode != 0: raise RuntimeError('Infrastructure/admin error; no retry: '+job['id'])
            row = audit_closed(job,sources); verify_sources(args.source_freeze,args.source_freeze_sha256)
            append(ledger,dict(event='actual_closed',utc=now(),**row))
        append(ledger,dict(event='complete',utc=now(),all8_actual_closed=True,
            task_failures_not_retried_or_replaced=True)); return 0
    except BaseException as error:
        append(ledger,dict(event='infrastructure_stop',utc=now(),error_type=type(error).__name__,
            error=str(error),no_retry=True)); raise


if __name__ == '__main__': raise SystemExit(main())
