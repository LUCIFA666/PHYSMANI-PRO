"""Wait for the observed v16 process, then execute the fixed v17 infra roster.

This never retries a case, changes a roster or scores a policy. A missing PID
alone cannot launch v17: all 54 original v16 raw completions must verify first.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/framework/multitask_r10'
sys.path.insert(0, str(ROOT))
from scripts import collect_multitask_recovery_r10 as runner


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_identity(pid):
    try:
        value = (Path('/proc')/str(pid)/'stat').read_text()
    except FileNotFoundError:
        return None
    fields = value[value.rfind(')')+2:].split()
    return dict(starttime=fields[19], state=fields[0])


def run(pid, starttime):
    setup16 = BASE/'TRAIN_recovery_v16_known_failures_setup'
    setup17 = BASE/'TRAIN_recovery_v17_additional_known_failures_setup'
    out16 = BASE/'TRAIN_recovery_v16_known_failures'
    out17 = BASE/'TRAIN_recovery_v17_additional_known_failures'
    status_path = setup17/'continuation_status.json'
    if status_path.exists(): raise ValueError('preserve an existing continuation; no automatic restart')
    bindings = {str(path.relative_to(ROOT)): sha(path) for path in (
        setup16/'plan.json', setup16/'frozen/execution_registration.json',
        setup16/'preflight.json', setup17/'plan.json',
        setup17/'frozen/execution_registration.json', setup17/'preflight.json', Path(__file__))}
    def status(stage, **fields):
        value = dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(),
            predecessor_observed_process_id=pid, predecessor_observed_starttime=starttime,
            immutable_inputs_sha256=bindings, automatic_retries=False,
            policy_success_is_not_a_gate=True, **fields)
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
        temporary.replace(status_path)
    def frozen():
        for name, checksum in bindings.items():
            if sha(ROOT/name) != checksum: raise ValueError('registered recovery input changed: '+name)
    try:
        frozen()
        status('waiting_for_verified_v16_process_terminal')
        while True:
            identity = process_identity(pid)
            if identity is None or identity['starttime'] != starttime or identity['state'] == 'Z': break
            time.sleep(10)
        frozen()
        preflight16 = read(setup16/'preflight.json')
        ids16 = preflight16['check_stdout']['fixed_failed_cases']
        report = read(out16/'queue_report.json')
        if (len(ids16) != 54 or len(set(ids16)) != 54
                or len(report.get('selected_case_ids', [])) != 54
                or set(report.get('selected_case_ids', [])) != set(ids16)
                or report.get('selected_complete') is not True
                or len(report.get('receipts', [])) != 54
                or {row.get('case_id') for row in report['receipts']} != set(ids16)
                or not all(row.get('status') in ('completed', 'resumed_verified') for row in report['receipts'])):
            raise ValueError('v16 has not verified all54 original completions; no v17 launch')
        context = runner.verify_execution(setup16/'plan.json', setup16/'frozen/execution_registration.json',
            output=out16, case_ids=ids16)
        cases = {case['case_id']: case for case in context['plan']['cases']}
        for cid in ids16:
            runner.validate_recovery_result(runner.collection.case_directory(out16, cases[cid])/'case.json',
                context, cases[cid])
        context17 = runner.verify_execution(setup17/'plan.json', setup17/'frozen/execution_registration.json',
            output=out17)
        ids17 = context17['allowed']
        if len(ids17) != 5 or len(set(ids17)) != 5:
            raise ValueError('v17 must retain exactly five registered cases')
        expected = [str(ROOT/'.conda-env/bin/python'), '-B',
            str(ROOT/'scripts/collect_multitask_recovery_r10.py'), '--check',
            '--plan', str(setup17/'plan.json'), '--execution-registration',
            str(setup17/'frozen/execution_registration.json'), '--output', str(out17)]
        expected += [value for cid in ids17 for value in ('--queue-case-id', cid)]
        if read(setup17/'preflight.json')['check_command'] != expected:
            raise ValueError('v17 preflight argv differs from exact registered interpreter/runner/plan/output/case list')
        command = list(expected)
        if any((out17/'infrastructure_attempts'/cid).exists() for cid in ids17):
            raise ValueError('v17 already attempted; preserve original evidence, no automatic retry')
        command[command.index('--check')] = '--run-queue'
        command += ['--workers', '2']
        frozen()
        status('v16_all54_verified_v17_fixed5_running_parallel2', command=command,
            predecessor_completed=54, next_planned_cases=5)
        with (setup17/'registered_queue.log').open('xb') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            status('v16_all54_verified_v17_fixed5_running_parallel2', command=command,
                predecessor_completed=54, next_planned_cases=5, next_queue_process_id=process.pid)
            code = process.wait()
        frozen()
        report17 = read(out17/'queue_report.json')
        complete = (code == 0 and report17.get('selected_complete') is True
            and len(report17.get('selected_case_ids', [])) == 5
            and set(report17.get('selected_case_ids', [])) == set(ids17)
            and len(report17.get('receipts', [])) == 5
            and {row.get('case_id') for row in report17['receipts']} == set(ids17)
            and all(row.get('status') in ('completed', 'resumed_verified') for row in report17['receipts']))
        if complete:
            fresh17 = runner.verify_execution(setup17/'plan.json', setup17/'frozen/execution_registration.json',
                output=out17, case_ids=ids17)
            cases17 = {case['case_id']: case for case in fresh17['plan']['cases']}
            for cid in ids17:
                runner.validate_recovery_result(runner.collection.case_directory(out17, cases17[cid])/'case.json',
                    fresh17, cases17[cid])
        status('v17_fixed5_completed' if complete else 'v17_pending_infrastructure_review',
            returncode=code, next_planned_cases=5, next_selected_complete=complete)
        return 0 if complete else 2
    except Exception as error:
        status('pending_infrastructure_review_no_automatic_retry', error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predecessor-pid', type=int, required=True)
    parser.add_argument('--predecessor-starttime', required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.predecessor_pid, args.predecessor_starttime))


if __name__ == '__main__': main()
