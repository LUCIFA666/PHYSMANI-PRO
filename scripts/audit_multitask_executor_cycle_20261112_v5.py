"""Independent read-only completed-cycle timing, Native-outcome and wait audit."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/framework/multitask_executor_cycle_20261112_v5'
PLAN = ROOT/'config/framework/multitask_executor_cycle_20261112_v5/plan.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    report, plan = read(OUT/'report.json'), read(PLAN)
    assert report['complete'] and report['cases_completed'] == 84
    for path, digest in report['source_sha256'].items():
        assert sha(path) == digest, path
    assert sha(report['selected_model']['path']) == report['selected_model']['sha256']
    ledger = rows(OUT/'case_ledger.jsonl')
    assert len(ledger) == 84
    expected = {c['id']:c for c in plan['cases']}
    assert {r['case_id'] for r in ledger} == set(expected)
    checks, tick_sum, rml_retries = [], 0, []
    for row in ledger:
        case = expected[row['case_id']]
        assert row['role'] == case['role'] and row['task'] == case['task']
        assert row['seed'] == case['cold_reset_seeds'][row['cold_reset_attempt']]
        exit_path = Path(row['actual_exit']['path'])
        assert sha(exit_path) == row['actual_exit']['sha256']
        exit_receipt = read(exit_path)
        assert exit_receipt['complete'] and exit_receipt['actual_wait_completed']
        assert exit_receipt['actual_process_returncode'] == 0 and not exit_receipt['unreaped_descendants']
        folder = exit_path.parent/'worker'
        actual = read(folder/'result.json')
        assert actual['complete'] and actual['native_closed'] and actual['source_unchanged']
        ticks = actual['actual_ticks']
        assert type(ticks) is int and 0 <= ticks <= 1200
        assert abs(actual['actual_seconds']-ticks*.05) < 2e-5
        judge = actual['native_task_judge']
        assert actual['episode_success'] == judge['native_task_success_latched']
        assert judge['original_native_metric']['native_ever_success'] == actual['episode_success']
        if actual['episode_success']:
            assert abs(judge['first_success_time']-actual['actual_seconds']) < 2e-5
        receipts = rows(folder/'execution.jsonl') if (folder/'execution.jsonl').exists() else []
        assert sum(r['executed_ticks'] for r in receipts) == ticks
        ended = 0.
        for receipt in receipts:
            assert receipt['unexpected_physics_ticks'] == 0
            assert receipt['tick_calls_completed'] == receipt['executed_ticks']
            assert abs(receipt['start_clock']-ended) < 2e-5
            for tick in receipt['ticks']:
                assert tick['tick_completed']
                assert abs(tick['clock_after']-tick['clock_before']-.05) < 2e-5
            ended = receipt['end_clock']
        assert abs(ended-actual['actual_seconds']) < 2e-5
        assert actual['servo_reference_readback_max_error_rad'] <= 1e-6
        assert actual['exact_servo_writes'] == ticks
        frames = rows(folder/'public_frame_provenance.jsonl') if (folder/'public_frame_provenance.jsonl').exists() else []
        assert len(frames) == actual['model_forward_calls']
        for frame in frames:
            if frame['saved_RGBD']:
                import numpy as np
                with np.load(frame['saved_RGBD'], allow_pickle=False) as arrays:
                    assert abs(float(arrays['timestamp'])-frame['timestamp']) < 1e-8
                    for key, digest in frame['array_sha256'].items():
                        assert hashlib.sha256(np.ascontiguousarray(arrays[key]).tobytes()).hexdigest() == digest
        decisions = rows(folder/'decisions.jsonl') if (folder/'decisions.jsonl').exists() else []
        for decision in decisions:
            audit = decision.get('native_RML_sampling_audit') or {}
            if len(audit.get('velocity_trials', [])) > 1:
                rml_retries.append(dict(case_id=row['case_id'], query=decision['index'], audit=audit))
            if decision.get('gripper_intent_issued'):
                assert decision['arrival']['measured_arm_arrived']
        tick_sum += ticks
        checks.append(dict(case_id=row['case_id'], passed=True, ticks=ticks,
                           policy_forward_calls=actual['model_forward_calls'], success=actual['episode_success']))
    assert sha(ROOT/'artifacts/framework/cube_full_reuse_20261112_v1/report.json') == 'e1b240fe224615e6a47fb83b0b79f7bd2f2b7c6df86fd8761924b6b49d894c09'
    global_stop = ROOT/'runtime/framework/STOP'
    stop_stat = global_stop.stat()
    assert (sha(global_stop), stop_stat.st_mtime_ns, stop_stat.st_ino) == (
        'e2bb2d8aff8697cf3f0c18a7a0707b2bd7e8e817d18b563513c365bf0e5a1850',1793780243677293196,220594231)
    locks = []
    for name in ('fixed_TRAIN104_geometry_native.lock', 'TRAIN_postcomplete_diagnostics_v1106.lock',
                 'current_cube_centre_large_training_20261104_v1.lock'):
        with (ROOT/'runtime/framework'/name).open('a+') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX|fcntl.LOCK_NB)
            locks.append(dict(name=name, free=True))
            fcntl.flock(stream, fcntl.LOCK_UN)
    service = subprocess.check_output(['systemctl','--user','show','our-mani-multitask-small-20261112-v5.service',
                                       '-p','MainPID','-p','Result','-p','ExecMainStatus'],text=True)
    assert 'MainPID=0' in service and 'Result=success' in service and 'ExecMainStatus=0' in service
    value = dict(complete=True, independently_audited_cases=84, physical_ticks=tick_sum,
                 all_actual_wait_and_native_closed=True, source_model_and_Cube_baseline_unchanged=True,
                 first_native_success_no_extra_steps=True, reference_readback_within_1e_minus6=True,
                 locks=locks, service=service, RML_multiple_velocity_previews=rml_retries,
                 official_archived_VAL_TEST=False, checks=checks)
    output = OUT/'final_audit.json'
    with output.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:value[k] for k in ('complete','independently_audited_cases','physical_ticks')},indent=2))


if __name__ == '__main__':
    main()
