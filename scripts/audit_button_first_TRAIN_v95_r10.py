"""Audit all four fixed button cases, retaining every failure and limit breach."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,validate_completed_case
from our_mani.multitask.input_stat_cache_v79 import verify_inputs

B=ROOT/'artifacts/framework/multitask_r10'
REPORT=B/'button_first_sources_v95/report.json'


def main():
    if REPORT.exists():raise FileExistsError('preserve previous acceptance report')
    lock=B/'button_author_intent_v95_setup/frozen_inputs.json'
    verify_inputs(ROOT,json.loads(lock.read_text())['files'])
    rows=[]
    for task,raw in (('push_moving_button','episode100'),('push_moving_button_high_speed','episode110')):
        for method in ('cv_geometry','context_geometry'):
            p=B/'button_author_intent_v95/development_demo_only'/task/raw/method/'case.json'
            d=json.loads(p.read_text());validate_completed_case(d,d['identity'])
            assert d['split']=='train' and d['demo_only'] and not d['formal_test_used']
            assert d['complete'] and d['cleanup_completed'] and d['infrastructure_error'] is None
            assert d['counted_clock']['dt']==.05 and d['counted_clock']['ticks']<=200
            a=d['actor_metrics'];m=a['continuous_reference_feedforward'];dispatch=a['task_dispatch_v43']
            events=dispatch['events']
            assert len(events)==d['counted_clock']['ticks']
            violations=[e for e in events if e['actual_window_acceleration_within_limits'] is False
                or not e['actual_window_velocity_within_limits']]
            maximum=max(abs(x) for e in events for x in (e['backward_window_acceleration'] or []))
            rows.append(dict(task=task,raw_episode_id=raw,method=method,
                case=dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p)),
                native_success=d['official_task_success'],counted_clock=d['counted_clock'],
                complete=True,cleanup_completed=True,infrastructure_error=None,
                motion_qualified=not violations and not m['observed_motion_qualification_failed'],
                maximum_actual_acceleration_rad_s2=maximum,
                motion_violations=[dict(tick=e['tick_after'],time=e['time_after']) for e in violations],
                source_guard_events=a['button_author_intent_v74']['source_state_events'],
                supervision='original_author_commands',completed_skill_support_claimed=False,
                public_press_completion_claimed=False))
    atomic_json(REPORT,dict(schema='r10_button_first_fixed_TRAIN_acceptance_v95',
        created_utc=datetime.now(timezone.utc).isoformat(),rows=rows,
        native_successes=sum(r['native_success'] for r in rows),fixed_cases=4,
        motion_passes=sum(r['motion_qualified'] for r in rows),infrastructure_errors=0,
        normal_CV_first_source_success=next(r["native_success"] for r in rows if r["task"]=="push_moving_button" and r["method"]=="cv_geometry"),all_button_executors_qualified=False,
        formal_VAL_TEST=False,all16_executors_qualified=False,failed_cases_preserved=True,
        lock=dict(path=str(lock.relative_to(ROOT)),sha256=file_sha(lock)),
        source=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=file_sha(__file__))))
    print(json.dumps(dict(native_successes=sum(r['native_success'] for r in rows),
        fixed_cases=4,motion_passes=sum(r['motion_qualified'] for r in rows),formal_VAL_TEST=False)))


if __name__=='__main__':main()
