"""Report the complete fixed original TRAIN two-block PhysMani smoke grid."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha


def main():
    base = ROOT/'artifacts/framework/multitask_r10'
    setup = base/'block_physmani_v348_setup'
    out = base/'block_physmani_v348_complete_audit_v357'
    if out.exists():
        raise FileExistsError('preserve complete audit')
    acceptance = json.loads((setup/'CPU_acceptance.json').read_text())
    owner = ROOT/'runtime/framework/native_resource_owner_v19'
    requests = [json.loads(f.read_text()) for f in (owner/'requests').glob('*.json')]
    status = json.loads((owner/'status.json').read_text())
    checks = acceptance['actual_argv_preflights']
    assert acceptance['complete'] and len(checks) == 2
    rows = []
    for check in checks:
        argv = check['argv']
        matches = [r for r in requests if r['argv'] == argv]
        assert len(matches) == 1
        job = matches[0]['job_id']
        terminal = status['completed'][job]
        assert terminal['complete'] and terminal['returncode'] == 0
        path = ROOT/argv[argv.index('--out')+1]/'case.json'
        case = json.loads(path.read_text())
        assert case['complete'] and case['split'] == 'train' and case['method'] == 'physmani'
        assert not case['formal_test_used'] and case['development_demo_only']
        assert case['source_raw_episode_id'] == check['raw']
        metrics = case['all_stage_counted_joint_motion']
        events = metrics['events']
        assert len(events) == case['counted_clock']['ticks']
        assert metrics['original_planning_model_acceleration_unchanged']
        assert metrics['actual_window_acceleration_criterion_rad_s2'] == 6.
        assert not metrics['policy_or_physics_modified']
        motion = metrics['all_stage_actual_window_checks_passed']
        native = case['official_task_success']
        rows.append(dict(task=case['task'], raw=case['source_raw_episode_id'], method='physmani',
            case=dict(path=str(path.relative_to(ROOT)), sha256=file_sha(path)),
            owner_terminal=terminal, native_success=native, actual_motion_passed=motion,
            constrained_native_success=native and motion, clock=case['counted_clock'],
            infrastructure_error=case['infrastructure_error'], cleanup_completed=case['cleanup_completed'],
            peak_actual_window_acceleration_rad_s2=metrics['peak_actual_window_acceleration'],
            speed_violating_windows=sum(e['speed_passed'] is False for e in events),
            acceleration_violating_windows=sum(e['acceleration_passed'] is False for e in events),
            original_planning_model_acceleration_limits=metrics['original_planning_model_acceleration_limits'],
            baseline_policy_and_native_physics_unchanged=True))
    out.mkdir()
    report = dict(schema='r10_fixed_first_original_two_block_PhysMani_complete_audit_v357',
        created_utc=datetime.now(timezone.utc).isoformat(), denominator=len(rows), cases=rows,
        native_successes=sum(r['native_success'] for r in rows),
        actual_motion_passed=sum(r['actual_motion_passed'] for r in rows),
        constrained_native_successes=sum(r['constrained_native_success'] for r in rows),
        infrastructure_errors=sum(r['infrastructure_error'] is not None for r in rows),
        original_author_weights_world_model_and_RML=True,
        original_R09_V6_block_reset_and_terminal_API_difference_disclosed=True,
        original_unused_planning12_unchanged_actual6_reported_independently=True,
        each_task_first_source_only=True, full_four_source_qualification=False,
        formal_VAL_TEST=False, TEST_read=False, source_outcome_filtering=False,
        v337_tick0_infrastructure_failures_preserved_not_task_scores=True,
        source=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=file_sha(__file__)))
    atomic_json(out/'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}), flush=True)


if __name__ == '__main__':
    main()
