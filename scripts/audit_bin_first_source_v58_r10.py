"""Record the exact fixed TRAIN pair without converting it to a formal score."""
from pathlib import Path
import sys,json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,verify_inputs
BASE=ROOT/'artifacts/framework/multitask_r10'
OUT=BASE/'bin_first_source_acceptance_v58/report.json'

def main():
    if OUT.exists():raise RuntimeError('preserve immutable report')
    lock=BASE/'task_executor_smoke_v58_setup/frozen_inputs.json';files=json.loads(lock.read_text())['files'];verify_inputs(ROOT,files)
    rows=[]
    for method in ('cv_geometry','context_geometry'):
        path=BASE/'task_executor_smoke_v58/development_demo_only/put_rubbish_in_moving_bin/episode105'/method/'case.json'
        d=json.loads(path.read_text());a=d['actor_metrics'];c=a['continuous_reference_feedforward']
        assert d['complete'] and d['cleanup_completed'] and d['infrastructure_error'] is None
        assert d['official_task_success'] and not c['observed_motion_qualification_failed']
        assert d['split']=='train' and not d['formal_test_used'] and d['demo_only']
        max_acc=max(abs(x) for e in c['observed_events'] for x in (e['backward_window_acceleration'] or []))
        max_vel=max(abs(x) for e in c['observed_events'] for x in (e['backward_window_velocity'] or []))
        rows.append(dict(method=method,case=dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path)),
            raw_episode_id=d['raw_episode_id'],official_task_success=d['official_task_success'],counted_clock=d['counted_clock'],
            observed_motion_qualification_failed=False,maximum_backward_window_acceleration_rad_s2=max_acc,
            maximum_backward_window_joint_speed_rad_s=max_vel,cleanup_completed=True,infrastructure_error=None,
            phase_timing=a['observed_feedback_controller']['phase_timing_events'],
            clearance_goal_binding=a['placement_clearance_v50']['binding'],
            first_selected_clearance_goal=a['placement_clearance_v50']['events'][0]['goal']))
    report=dict(schema='r10_fixed_TRAIN_first_bin_pair_acceptance_v58',created_utc=datetime.now(timezone.utc).isoformat(),
        task='put_rubbish_in_moving_bin',fixed_TRAIN_sources_complete=1,fixed_TRAIN_sources_required=4,
        rows=rows,frozen_inputs_verified=len(files),frozen_lock=dict(path=str(lock.relative_to(ROOT)),sha256=file_sha(lock)),
        source=str(Path(__file__).relative_to(ROOT)),source_sha256=file_sha(__file__),
        formal_VAL_or_TEST=False,all16_executors_qualified=False,formal_success_rate_claimed=False,
        new_joint16_training_started=False,old_models_retained_for_development=True,
        other_fixed_sources_queued_once=True,prohibited_methods=['Value','NGFF','FASTER'])
    atomic_json(OUT,report);print(json.dumps(dict(report=str(OUT.relative_to(ROOT)),rows=[{k:r[k] for k in ('method','official_task_success','counted_clock','maximum_backward_window_acceleration_rad_s2')} for r in rows])))

if __name__=='__main__':main()
