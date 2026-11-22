"""Separate runtime, measured motion, native success and control interventions."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha
BASE = ROOT/'artifacts/framework/multitask_r10'


def summarize(path):
    case = json.loads(path.read_text())
    if case['split'] != 'train' or case.get('formal_test_used') or case.get('TEST_used_for_fitting'):
        raise ValueError('pilot audit only reads original TRAIN development cases')
    actor = case.get('actor_metrics') or {}
    manager = actor.get('continuous_reference_feedforward') or {}
    dispatcher = actor.get('task_dispatch_v43') or {}
    motion = dispatcher.get('actual_observed_window_limits_passed')
    if type(motion) is not bool:
        failed = manager.get('observed_motion_qualification_failed')
        motion = None if type(failed) is not bool or not manager.get('observed_events') else not failed
    observed = manager.get('observed_events') or []
    accelerations = [abs(v) for event in observed for v in (event.get('backward_window_acceleration') or [])]
    transit = manager.get('bounded_approach_transit_v18') or {}
    common = transit.get('continuous_transit') or {}
    return dict(task=case['task'], raw=case['raw_episode_id'], method=case['method'],
        runtime_clean=case.get('complete') is True and case.get('infrastructure_error') is None
            and case.get('cleanup_completed') is True,
        native_success=case['official_task_success'], actual_motion_pass=motion,
        simulation_seconds=case.get('simulation_seconds'),
        peak_observed_acceleration=max(accelerations, default=None),
        braking_ticks=sum(e.get('reason') == 'continuous_reference_braking' for e in actor.get('public_events', [])),
        first_violation=next((e for e in observed if e.get('violations')), None),
        successful_transit_renewals=sum(e.get('renewed') is True for e in common.get('renewals', [])),
        moving_reference_renewals=sum(e.get('renewed') is True
            for e in actor.get('moving_reference', {}).get('events', [])),
        source=dict(path=str(path.relative_to(ROOT)), sha256=file_sha(path)),
        formal_VAL_TEST=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = (ROOT/args.out).resolve()
    if not out.is_relative_to(BASE) or out.exists():
        raise ValueError('fresh pilot audit artifact required')
    groups = {}
    for name in ('button_reference_v389', 'bin_physical_smoke_v240',
                 'continuous_bridge_v402', 'moving_button_v404'):
        rows = [summarize(path) for path in sorted((BASE/name).rglob('case.json'))]
        groups[name] = dict(cases=rows, completed=len(rows),
            runtime_clean=sum(r['runtime_clean'] for r in rows),
            motion_pass=sum(r['actual_motion_pass'] is True for r in rows),
            native_success=sum(r['native_success'] for r in rows))
    report = dict(schema='r10_continuous_executor_pilot_comparison_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), cohorts=groups,
        method_and_source_denominators_kept_separate=True,
        earlier_success_does_not_select_a_source=True, formal_VAL_TEST=False, TEST_read=False)
    atomic_json(out/'report.json', report)
    print(json.dumps({k:{a:b for a,b in v.items() if a != 'cases'} for k,v in groups.items()}))


if __name__ == '__main__':
    main()
