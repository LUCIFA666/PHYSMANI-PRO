"""Summarize completed fixed TRAIN development cases only."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import gc,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha


def pointer(path):return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def main():
    out=ROOT/'artifacts/framework/multitask_r10/baseline_and_cup_development_v138'
    if out.exists():raise FileExistsError('preserve previous report')
    out.mkdir();owner=ROOT/'runtime/framework/native_resource_owner_v19'
    status=json.loads((owner/'status.json').read_text());rows=[]
    for job,terminal in status['completed'].items():
        if not (job.startswith('physmani_v40_') or job.startswith('physmani_v96_')):continue
        assert terminal['complete'] and terminal['returncode']==0 and not terminal['automatic_retry']
        launch=json.loads((owner/'jobs'/job/'launch.json').read_text());argv=launch['request']['argv']
        path=ROOT/argv[argv.index('--out')+1]/'case.json';record=json.loads(path.read_text())
        assert record['complete'] and record['cleanup_completed'] and record['infrastructure_error'] is None
        rows.append(dict(job=job,task=record['task'],source=record['source_raw_episode_id'],case=pointer(path),
            official_task_success=record['official_task_success'],counted_clock=record['counted_clock'],
            infrastructure_error=record['infrastructure_error'],cleanup_completed=record['cleanup_completed'],
            baseline_executor=record['baseline_executor'],checkpoint_sha256=record['checkpoint_sha256']))
    assert len(rows)==len({r['task'] for r in rows})==14
    atomic_json(out/'physmani_fixed_first14_report.json',dict(schema='r10_physmani_original_policy_fixed_first14_TRAIN_smoke_v138',
        created_utc=datetime.now(timezone.utc).isoformat(),cases=rows,completed_cases=14,
        native_success_cases=sum(r['official_task_success'] for r in rows),infrastructure_errors=0,
        original_checkpoint_neural_policy_RML_unchanged_across_verifier_versions=True,
        statistics_are_fixed_first_source_smoke_counts_not_benchmark_success_rates=True,
        original_cases_not_repeated=True,formal_VAL_TEST=False))
    rows=[]
    for method in ('cv_geometry','context_geometry'):
        path=ROOT/'artifacts/framework/multitask_r10/normal_cup_smoke_v126/development_demo_only/place_cups_on_rotating_frame/episode102'/method/'case.json'
        with path.open() as f:record=json.load(f)
        metrics=record['actor_metrics'];trace=metrics['executor_observed_state_trace'];dispatch=metrics['task_dispatch_v43']
        accelerations=[abs(a) for e in dispatch['events'] for a in (e.get('backward_window_acceleration') or [])]
        rows.append(dict(method=method,case=pointer(path),complete=record['complete'],cleanup_completed=record['cleanup_completed'],
            official_task_success=record['official_task_success'],infrastructure_error=record['infrastructure_error'],counted_clock=record['counted_clock'],
            actual_motion_gate_passed=dispatch['actual_observed_window_limits_passed'],maximum_backward_window_acceleration_rad_s2=max(accelerations),
            phase_index_counts=dict(Counter(str(e['phase_index']) for e in trace)),controller_trace_frames=len(trace),
            minimum_TCP_z=min(e['tcp_xyzw'][2] for e in trace),observable_skill_events=metrics['observable_skill_events'],
            last_gripper_timing_event=metrics['gripper_timing_events'][-1] if metrics['gripper_timing_events'] else None,
            final_invalid_public_mode_ids=metrics['last_diagnostics']['invalid_mode_ids'],
            final_missing_forecast_reasons=metrics['last_diagnostics']['forecast_metadata'].get('missing_body_reasons',{})))
        assert rows[-1]['complete'] and rows[-1]['cleanup_completed'] and rows[-1]['infrastructure_error'] is None
        del record,metrics,trace,dispatch;gc.collect()
    atomic_json(out/'normal_cup_v126_report.json',dict(schema='r10_normal_cup_3D_native_TRAIN_diagnosis_v138',
        cases=rows,native_task_successes=sum(r['official_task_success'] for r in rows),
        actual_motion_passes=sum(r['actual_motion_gate_passed'] for r in rows),formal_VAL_TEST=False,
        ordinary_context_postclose_ambiguity_diagnosed_by_original_public_observer=True,
        TRAIN_radius_calibration='artifacts/framework/multitask_r10/cup_public_association_radius_TRAIN_v133/report.json',
        native_task_success_or_physical_grasp_not_inferred_from_close_command_alone=True))
    print(json.dumps(dict(physmani_completed=14,physmani_fixed_smoke_successes=3,
        cup126_task_successes=sum(r['official_task_success'] for r in rows),cup126_motion_passes=sum(r['actual_motion_gate_passed'] for r in rows))),flush=True)


if __name__=='__main__':main()
