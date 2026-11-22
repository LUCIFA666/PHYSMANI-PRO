"""Offline hold/twist and completed-skill support diagnostics; no label edits."""
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import analyze_multitask_smoke_r10 as common
from scripts import collect_multitask_training_r10 as c
from our_mani.multitask.specs import get_task_spec
BASE=c.BASE


def comparison(reported,derived,mask):
    if not mask.any():return dict(pairs=0,vector_rms_error=None,mean_bias=None)
    error=reported[mask]-derived[mask]
    return dict(pairs=int(mask.sum()),vector_rms_error=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),
        mean_bias=error.mean(0).tolist(),maximum_error=float(np.linalg.norm(error,axis=1).max()))


def summarize():
    manifest_path=BASE/'TRAIN_smoke_collection_v5/manifest.json';manifest=common.read(manifest_path)
    common.require(manifest['complete'] is True and len(manifest['entries'])==28,'complete28 teacher/hold acquisition required')
    audit_path=BASE/'v5_interaction_adapter_audit.json';adapter=common.read(audit_path)
    common.require(adapter['collector_manifest']['sha256']==common.sha(manifest_path) and not adapter['conversion_errors'],
        'adapter audit mismatch/errors')
    rows=[]
    for entry in manifest['entries']:
        if not entry['context_primary_view']:continue
        path=ROOT/entry['cache'];common.require(common.sha(path)==entry['cache_sha256'],'hold cache changed')
        with np.load(path,allow_pickle=False) as stored:a={key:stored[key].copy() for key in stored.files}
        t=a['times'];dt=np.diff(t);body_rows={}
        for j,body in enumerate(map(str,a['body_ids'])):
            p,r=a['target_positions'][:,j],a['target_rotations'][:,j]
            dv=np.diff(p,axis=0)/dt[:,None]
            dw=Rotation.from_matrix(r[1:]@np.swapaxes(r[:-1],-1,-2)).as_rotvec()/dt[:,None]
            valid=(a['native_pose_valid'][:-1,j]&a['native_pose_valid'][1:,j]&a['mode_valid'][:-1,j]
                &a['mode_valid'][1:,j]&a['dynamics_segment_valid'][:-1,j]&a['dynamics_segment_valid'][1:,j]
                &(a['modes'][:-1,j]==a['modes'][1:,j]))
            body_rows[body]=dict(history_valid_rows=int(a['valid'][:,j].sum()),
                history_modes=dict(Counter(('external','attached','released')[int(m)] for m in a['modes'][:,j])),
                orientation_observable_rows=int(a['rotation_observable'][:,j].sum()),
                context_windows=entry['context_window_eligibility']['by_body'][body],velocity={})
            for name,reported,derived,threshold in (
                ('linear_m_per_s',a['target_linear_velocities'][:,j],dv,.001),
                ('angular_rad_per_s',a['target_angular_velocities'][:,j],dw,.01)):
                middle=(reported[:-1]+reported[1:])*.5
                body_rows[body]['velocity'][name]=dict(
                    pose_moving_pairs=int((valid&(np.linalg.norm(derived,axis=1)>threshold)).sum()),
                    midpoint_all=comparison(middle,derived,valid),
                    midpoint_excluding_first_two_intervals=comparison(middle,derived,valid&(t[:-1]>=.1-1e-8)),
                    reported_end_of_interval=comparison(reported[1:],derived,valid),
                    reported_one_extra_frame_later=comparison(reported[2:],derived[:-1],valid[:-1]&valid[1:]),
                    startup_exclusion_is_diagnostic_only=True,
                    later_reported_value_never_used_as_current_policy_feature=True)
        rows.append(dict(task=entry['task'],parent_source_group=entry['parent_source_group'],cache=common.pointer(path),
            frames=len(t),seconds=float(t[-1]),history_hold_only=True,bodies=body_rows,
            source_group_not_replica_count=True))
    skills=[]
    for row in adapter['rows']:
        if not row.get('interaction_eligible'):continue
        events=row['events'];spec=get_task_spec(row['task'])
        phases={}
        for phase in spec.phase_skills:
            subset=[event for event in events if event['phase']==phase.name]
            phases[phase.name]=dict(completion_events=len(subset),parent_support=int(bool(subset)),
                basic_curve_eligible_events=sum(event['eligible_zero_lag_curve_rows']>=2 and event['terminal_endpoint_rows']>=1 for event in subset),
                zero_lag_curve_rows=sum(event['eligible_zero_lag_curve_rows'] for event in subset),
                terminal_endpoint_rows=sum(event['terminal_endpoint_rows'] for event in subset),
                durations_seconds=[event['duration_seconds'] for event in subset],
                excluded_curve_rows=dict(sum((Counter(event['excluded_rows']) for event in subset),Counter())),
                required=not phase.optional_after_native_completion,
                formal_minimum_eight_parents_met=False,
                BC_fit_performed=False)
        skills.append(dict(task=row['task'],parent_source_group=row['parent_source_group'],phases=phases,
            event_exclusions=row['audit']['excluded_phase_completion_events'],
            raw_commands=row['audit']['total_raw_commands'],accepted_phase_events=len(events)))
    return dict(version='r10_v5_hold_motion_and_completed_skill_support_v1',created_utc=c.utc(),
        manifest=common.pointer(manifest_path),interaction_adapter_audit=common.pointer(audit_path),
        script=common.pointer(__file__),hold_sources=len(rows),hold_rows=rows,skill_rows=skills,
        accepted_phase_events=sum(row['accepted_phase_events'] for row in skills),
        BC_interpretation='completion event and basic rows do not establish a fitted skill; one parent/task cannot meet eight-parent formal support; failed/incomplete teacher attempts retained',
        velocity_interpretation='native midpoint-vs-pose difference plus startup/end/one-frame-later comparators are descriptive TRAIN-only time-alignment diagnostics; no raw labels or model parameters changed',
        root_cause_boundary='a large raw mismatch may reflect scripted object pose update order, physics/contact impulses or timing; comparators alone do not authorize label rewriting',
        nonvisual_role_notice='remove spoke2 is collision-only in the independent component audit; zero direct history windows cannot be fixed by asserting fresh visibility',
        model_fitting_performed=False,labels_modified=False,VAL_read=False,TEST_read=False,native_or_GPU_launched=False)


def main():
    result=summarize();path=BASE/'TRAIN_hold_teacher_diagnostics_v5.json';c.write_json(path,result,overwrite=True)
    text=[f"v5纯离线：14hold父源，{result['accepted_phase_events']}真实phase completion事件。91 command_done不是91成功技能。",
        '每task只有1父源；所有skill尚未满足正式8父源支持，不宣称BC已训练。',
        '速度对比包含启动前两interval剔除及1帧时序诊断；仅解释数据，不改标签、模型或阈值。']
    for row in result['skill_rows']:
        text.append(row['task']+' | '+', '.join(f"{phase}:{item['completion_events']}event/{item['zero_lag_curve_rows']}rows" for phase,item in row['phases'].items()))
    path.with_suffix('.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(hold_sources=result['hold_sources'],accepted_phase_events=result['accepted_phase_events'],labels_modified=False)))
if __name__=='__main__':main()
