"""Read-only fixed TRAIN audit; every checked argv remains in the denominator."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha

B = ROOT / 'artifacts/framework/multitask_r10'
BATCHES = {'high_peg_alignment_v355_setup': 2, 'button_author_intent_v361_setup': 4, 'high_peg_retarget_v374_setup': 2}


def main():
    out = B / 'executor_fixed_TRAIN_complete_audit_v391'
    if out.exists():
        raise FileExistsError('preserve previous audit')
    owner = ROOT / 'runtime/framework/native_resource_owner_v19'
    status = json.loads((owner / 'status.json').read_text())
    requests = [json.loads(p.read_text()) for p in (owner / 'requests').glob('*.json')]
    reports = {}
    for name, count in BATCHES.items():
        setup = B / name
        acceptance = json.loads((setup / 'CPU_acceptance.json').read_text())
        checks = acceptance['actual_argv_preflights']
        assert acceptance['complete'] is True and len(checks) == count
        rows = []
        for row in checks:
            argv = row['argv']
            matches = [r for r in requests if r['argv'] == argv]
            assert len(matches) == 1, (name, row['raw'], row['method'])
            job = matches[0]['job_id']
            terminal = status['completed'][job]
            assert terminal['complete'] and terminal['returncode'] == 0
            path = ROOT / argv[argv.index('--out') + 1] / 'case.json'
            case = json.loads(path.read_text())
            assert case['complete'] and case['split'] == 'train'
            assert case['method'] == row['method'] and case['task'] == row['task']
            assert case['source_raw_episode_id'] == row['raw']
            assert not case['formal_test_used'] and case['demo_only']
            metrics = case['actor_metrics']
            reference = metrics['continuous_reference_feedforward']
            failed = reference['observed_motion_qualification_failed']
            assert isinstance(failed, bool)
            events = reference['observed_events']
            bad = [e for e in events if e['violations']]
            assert bool(bad) == failed
            peak = max((abs(v) for e in events
                        for v in (e['backward_window_acceleration'] or [])), default=0.)
            stop = metrics.get('reference_stop_guard_v274', {})
            native = case['official_task_success']
            assert isinstance(native, bool)
            rows.append(dict(
                task=case['task'], raw=row['raw'], method=row['method'],
                case=dict(path=str(path.relative_to(ROOT)), sha256=file_sha(path)),
                job_id=job, owner_terminal=terminal,
                native_success=native, actual_motion_passed=not failed,
                constrained_native_success=native and not failed,
                infrastructure_error=case['infrastructure_error'],
                cleanup_completed=case['cleanup_completed'], clock=case['counted_clock'],
                peak_actual_backward_window_acceleration_rad_s2=peak,
                violating_windows=len(bad),
                first_motion_violation=({k: bad[0][k] for k in
                    ('time', 'violations', 'backward_window_acceleration')} if bad else None),
                additional_stop_proof_queries=stop.get('queries'),
                additional_stop_proof_rejections=stop.get('rejections')))
        reports[name] = dict(
            complete=True, fixed_checked_argv_count=count, cases=rows,
            native_successes=sum(r['native_success'] for r in rows),
            actual_motion_passed=sum(r['actual_motion_passed'] for r in rows),
            constrained_native_successes=sum(r['constrained_native_success'] for r in rows),
            infrastructure_errors=sum(r['infrastructure_error'] is not None for r in rows),
            all16_executor_qualification=False, formal_VAL_TEST=False,
            source_outcome_filtering=False, original_failures_retained=True)
    out.mkdir()
    for name, report in reports.items():
        atomic_json(out / (name + '_report.json'), report)
    summary = dict(
        schema='r10_completed_fixed_TRAIN_batch_audit_v391',
        created_utc=datetime.now(timezone.utc).isoformat(),
        batches={k: {field: v[field] for field in
            ('fixed_checked_argv_count', 'native_successes', 'actual_motion_passed',
             'constrained_native_successes', 'infrastructure_errors')}
                 for k, v in reports.items()},
        ordinary_bin_v255_qualification_unchanged=True,
        TEST_read=False, formal_VAL_TEST=False, all16_executor_qualification=False,
        source=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=file_sha(__file__)))
    atomic_json(out / 'report.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
