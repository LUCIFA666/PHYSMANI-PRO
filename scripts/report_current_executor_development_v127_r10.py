"""Read completed native diagnostics without changing labels or rollout data."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import gc,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json
B=ROOT/'artifacts/framework/multitask_r10'


def pointer(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return dict(path=str(path.relative_to(ROOT)),sha256=h.hexdigest())


def terminal(record):
    assert record['complete'] and record['cleanup_completed']
    assert record['counted_clock']==dict(ticks=200,dt=.05,seconds=10.)
    assert record['split']=='train' and record['demo_only'] and not record['formal_test_used']
    return dict(complete=record['complete'],cleanup_completed=record['cleanup_completed'],
        infrastructure_error=record['infrastructure_error'],official_task_success=record['official_task_success'],
        counted_clock=record['counted_clock'],
        actual_motion_gate_passed=record['actor_metrics']['task_dispatch_v43']['actual_observed_window_limits_passed'])


def main():
    out=B/'executor_development_readonly_v127'
    if out.exists():raise FileExistsError('preserve existing diagnostic report')
    out.mkdir();created=datetime.now(timezone.utc).isoformat()
    path=B/'button_author_intent_v102/development_demo_only/push_moving_button/episode100/context_geometry/case.json'
    with path.open() as f:record=json.load(f)
    components=[e['button_success_components_v100'] for e in record['actor_metrics']['native_metric_diagnostics']['events']
        if e['stage']=='after_dispatch' and 'button_success_components_v100' in e]
    assert len(components)==200 and all(e['clock_unchanged'] for e in components)
    joint=[e for e in components if e['joint_component_met']]
    descent=[e for e in components if e['descent_component_met']]
    both=[e for e in components if e['simultaneous_components_met']]
    button=dict(schema='r10_button_success_component_readonly_diagnosis_v127',created_utc=created,
        case=pointer(path),**terminal(record),after_dispatch_samples=len(components),
        joint_condition_ticks=len(joint),descent_condition_ticks=len(descent),simultaneous_condition_ticks=len(both),
        joint_condition_samples=[{k:e[k] for k in ('time','absolute_displacement','descent_over_native_window')} for e in joint],
        peak_joint_displacement_m=max(e['absolute_displacement'] for e in components),
        original_joint_threshold_m=.003,original_descent_threshold_m=.03,original_descent_window_steps=10,
        diagnosis='native joint condition was met, but never simultaneously with required native TCP descent',
        physical_cause_not_inferred_from_metric_only=True,policy_commands_modified=False,
        native_success_callback_not_reinvoked=True,formal_VAL_TEST=False)
    assert (len(joint),len(descent),len(both))==(13,58,0)
    atomic_json(out/'button_v102_report.json',button)
    del record;gc.collect()
    rows=[]
    for version in (113,120):
        for method in ('cv_geometry','context_geometry'):
            path=B/f'normal_cup_smoke_v{version}/development_demo_only/place_cups_on_rotating_frame/episode102'/method/'case.json'
            with path.open() as f:record=json.load(f)
            metrics=record['actor_metrics'];trace=metrics['executor_observed_state_trace']
            prefix=metrics['continuous_reference_feedforward']['bounded_approach_transit_v18']
            rows.append(dict(version=version,method=method,case=pointer(path),**terminal(record),
                phase_index_counts=dict(Counter(str(e['phase_index']) for e in trace)),
                selected_candidates=len(metrics['continuation_attempts']),
                observable_skill_events=metrics['observable_skill_events'],
                TCP_initial=trace[0]['tcp_xyzw'],TCP_last=trace[-1]['tcp_xyzw'],
                minimum_TCP_z=min(e['tcp_xyzw'][2] for e in trace),
                prefix_event_counts=dict(Counter(e['event'] for e in prefix['events'])),
                prefix_trial_failure_counts=dict(Counter(trial.get('reason','unknown') for e in prefix['events']
                    for trial in e.get('trials',[]))),
                current_scene_ready=metrics['last_diagnostics']['collision_scene_readiness']))
            print(json.dumps(dict(version=version,method=method,task_success=record['official_task_success'],
                motion_pass=rows[-1]['actual_motion_gate_passed'],minimum_TCP_z=rows[-1]['minimum_TCP_z'])),flush=True)
            del record,metrics,trace,prefix;gc.collect()
    atomic_json(out/'normal_cup_v113_v120_report.json',dict(schema='r10_normal_cup_lag_reservation_native_acceptance_v127',
        created_utc=created,cases=rows,original_case_outputs_preserved=True,
        v113_diagnosis='positive fitted lag queried beyond original horizon; no prefix staged',
        v120_diagnosis='lag reservation enabled prefixes; horizontal-before-vertical heuristic prevented useful descent',
        no_completion_or_motion_failure_relabelled=True,formal_VAL_TEST=False))
    print(json.dumps(dict(complete=True,button_joint_ticks=13,button_descent_ticks=58,button_overlap=0,cup_cases=len(rows))),flush=True)


if __name__=='__main__':main()
