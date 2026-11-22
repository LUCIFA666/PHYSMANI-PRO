"""Read-only metrics on fixed initial12 TRAIN native acquisitions.

Current public RGBD diagnostics/poses are compared with the separate offline
native GT namespace after acquisition. No errors filter sources, labels,
training windows or the continuing full collection; this is not policy SR.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.multitask.evaluation import file_sha,atomic_json
from our_mani.multitask.specs import get_task_spec

BASE=ROOT/'artifacts/framework/multitask_r10'
MODES={0:'external',1:'attached',2:'released'}


def require(value,message):
    if not value:raise ValueError(message)


def quantiles(values):
    values=np.asarray(values,float)
    if not len(values):return dict(n=0,p50=None,p95=None,max=None)
    require(np.isfinite(values).all(),'finite recorded pose metrics required')
    return dict(n=len(values),p50=float(np.quantile(values,.5)),p95=float(np.quantile(values,.95)),max=float(values.max()))


def summarize(rows):
    scorable=[r for r in rows if r['current_registration'] and r['GT_pose_valid']]
    full=[r for r in scorable if r['public_full_orientation']]
    cache_full=[r for r in scorable if r['cache_rotation_observable']]
    def metrics(selected):
        return dict(position_error_m=quantiles([r['position_error_m'] for r in selected]),
            full_SO3_error_degrees=quantiles([r['SO3_error_degrees'] for r in selected]),
            SO3_above_10_degrees=sum(r['SO3_error_degrees']>10 for r in selected),
            SO3_above_30_degrees=sum(r['SO3_error_degrees']>30 for r in selected),
            SO3_above_90_degrees=sum(r['SO3_error_degrees']>90 for r in selected),
            SO3_above_150_degrees=sum(r['SO3_error_degrees']>150 for r in selected))
    return dict(body_frames=len(rows),native_GT_pose_valid=sum(r['GT_pose_valid'] for r in rows),
        current_registration=sum(r['current_registration'] for r in rows),
        current_directly_visible=sum(r['directly_visible'] for r in rows),
        current_public_full_R_evidence_frames=sum(r['public_full_orientation'] for r in rows),
        current_public_full_R_evidence_fraction=sum(r['public_full_orientation'] for r in rows)/len(rows) if rows else None,
        cache_valid_frames=sum(r['cache_valid'] for r in rows),
        cache_rotation_observable_frames=sum(r['cache_rotation_observable'] for r in rows),
        target_valid_frames=sum(r['target_valid'] for r in rows),mode_identity_valid_frames=sum(r['mode_valid'] for r in rows),
        missing_current_registration=sum(not r['current_registration'] for r in rows),
        missing_GT_pose=sum(not r['GT_pose_valid'] for r in rows),
        all_current_registration_GT_valid=metrics(scorable),
        declared_public_full_R_GT_valid=metrics(full),cache_rotation_observable_GT_valid=metrics(cache_full),
        feature_status_counts=dict(Counter(r['feature_status'] for r in rows)),
        feature_ambiguous_frames=sum(r['feature_ambiguous'] is True for r in rows),
        large_R_diagnostic_thresholds_are_not_training_or_collection_gates=True)


def main():
    initial_path=BASE/'TRAIN_full_collection_initial12_v11.json';initial=json.loads(initial_path.read_text())
    plan_path=ROOT/initial['plan']['path'];require(file_sha(plan_path)==initial['plan']['sha256'],'fixed initial12 plan SHA changed')
    plan=json.loads(plan_path.read_text());require(plan['plan_sha256']==initial['plan']['internal_sha256'],'fixed initial12 internal plan differs')
    prototype_path=BASE/'TRAIN_cup_feature_replay_plan_v1.json';prototype=json.loads(prototype_path.read_text())
    prototype_parents=set(prototype['independent_TRAIN_parents'])
    verified={str(initial_path.relative_to(ROOT)):file_sha(initial_path),str(plan_path.relative_to(ROOT)):file_sha(plan_path),
        str(prototype_path.relative_to(ROOT)):file_sha(prototype_path),str(Path(__file__).relative_to(ROOT)):file_sha(__file__)}
    rows=[];cases=[];pending=[];case_status=[];phase_parents=defaultdict(set)
    for selected in initial['cases']:
        path=BASE/'TRAIN_full_collection_v11/cases/train'/selected['task']/selected['case_id']/'case.json'
        if not path.exists():
            pending.append(selected['case_id']);case_status.append(dict(**selected,status='pending_canonical_receipt'));continue
        record=json.loads(path.read_text());entry=record['entry']
        require(record.get('status')=='completed' and record['case_id']==selected['case_id']
            and record['plan_sha256']==plan['plan_sha256'] and record['split']=='train'
            and entry['parent_source_group']==selected['parent_source_group']
            and entry['behavior']==selected['behavior'] and record['policy_history_contains_native_GT'] is False,
            'canonical initial12 receipt differs from the fixed acquisition')
        parent=record['parent_process_receipt'];raw=ROOT/parent['child_case_path']
        require(file_sha(raw)==parent['child_case_sha256'] and json.loads(raw.read_text())==
            {k:v for k,v in record.items() if k!='parent_process_receipt'},'raw child/canonical acquisition differs')
        verified[str(path.relative_to(ROOT))]=file_sha(path);verified[str(raw.relative_to(ROOT))]=file_sha(raw)
        for name,sha in record['files_sha256'].items():
            require(file_sha(ROOT/name)==sha,'completed acquisition output SHA changed: '+name);verified[name]=sha
        case_status.append(dict(**selected,status='completed_verified'))
        if 'cups' not in selected['task']:continue
        with np.load(ROOT/entry['cache'],allow_pickle=False) as cache:a={k:cache[k].copy() for k in cache.files}
        diagnostics_path=(ROOT/entry['cache']).parent/'tracker_diagnostics.json'
        diagnostics=json.loads(diagnostics_path.read_text())['frames'];ids=list(map(str,a['body_ids']))
        require(len(diagnostics)==len(a['times']) and np.allclose([d['timestamp'] for d in diagnostics],a['times'],rtol=0,atol=1e-9),
            'public diagnostic timestamps do not match the acquired cache')
        behavior='teacher' if entry['teacher_used'] else 'hold';case_rows=[]
        for i,time in enumerate(a['times']):
            for role in ('cup0','cup1','cup2'):
                j=ids.index(role);meta=diagnostics[i]['bodies'][role];feature=meta.get('cup_feature_registration') or {}
                registered=meta.get('registration_valid') is True;gt=bool(a['native_pose_valid'][i,j])
                row=dict(case_id=selected['case_id'],task=selected['task'],parent_source_group=selected['parent_source_group'],
                    behavior=behavior,role=role,frame_index=i,time=float(time),mode=MODES[int(a['modes'][i,j])],
                    teacher_phase=str(a['teacher_phase'][i]) if entry['teacher_used'] else 'not_applicable_hold',
                    current_registration=registered,directly_visible=meta.get('directly_visible') is True,
                    public_full_orientation=meta.get('orientation_observable') is True,
                    cache_rotation_observable=bool(a['rotation_observable'][i,j]),cache_valid=bool(a['valid'][i,j]),
                    GT_pose_valid=gt,target_valid=bool(a['target_valid'][i,j]),mode_valid=bool(a['mode_valid'][i,j]),
                    feature_status=feature.get('status','no_current_feature_pose'),feature_ambiguous=feature.get('ambiguous'),
                    observed_surface_rms_m=feature.get('observed_surface_rms_m'),
                    feature_candidate_cost_gap_m2=feature.get('candidate_cost_gap_m2'),
                    position_error_m=None,SO3_error_degrees=None)
                if registered and gt:
                    row['position_error_m']=float(np.linalg.norm(a['positions'][i,j]-a['target_positions'][i,j]))
                    row['SO3_error_degrees']=float(np.degrees(Rotation.from_matrix(
                        a['rotations'][i,j]@a['target_rotations'][i,j].T).magnitude()))
                rows.append(row);case_rows.append(row)
        phase_coverage=None
        if entry['teacher_used']:
            command=json.loads((ROOT/entry['teacher_command_receipt']).read_text())
            events=command['phase_completion_events'];required=[p.name for p in get_task_spec(selected['task']).phase_skills
                if not p.optional_after_native_completion]
            phases=sorted(set(e['phase'] for e in events))
            for phase in phases:phase_parents[(selected['task'],phase)].add(selected['parent_source_group'])
            phase_coverage=dict(command_count=len(command['commands']),command_completed_count=sum('command_completed_at' in c for c in command['commands']),
                required_skill_order=required,completed_phase_event_count=len(events),
                completed_phase_counts=dict(Counter(e['phase'] for e in events)),
                missing_required_completed_skills=[p for p in required if p not in phases],
                ordered_completed_events=[{k:e[k] for k in ('phase','to_phase','evidence','origin_command_id','origin_reference_body') if k in e} for e in events],
                command_arrival_is_not_phase_completion=True,independent_parent_count_per_completed_skill=1,
                minimum_eight_parent_library_support_established=False)
        cases.append(dict(**selected,behavior_partition=behavior,frames=len(a['times']),cup_body_frames=len(case_rows),
            prototype_parent_overlap=selected['parent_source_group'] in prototype_parents,
            summary=summarize(case_rows),by_mode={mode:summarize([r for r in case_rows if r['mode']==mode]) for mode in MODES.values()},
            by_role={role:summarize([r for r in case_rows if r['role']==role]) for role in ('cup0','cup1','cup2')},
            phase_coverage=phase_coverage))
    groups=defaultdict(list)
    for row in rows:groups[(row['behavior'],row['mode'])].append(row)
    cup_parents={r['parent_source_group'] for r in cases};large=[r for r in rows if r['public_full_orientation'] and
        r['current_registration'] and r['GT_pose_valid'] and r['SO3_error_degrees']>30]
    full_by_mode={behavior:{mode:summarize(groups[(behavior,mode)]) for mode in MODES.values()} for behavior in ('hold','teacher')}
    report=dict(version='r10_fixed_initial12_cup_native_GT_diagnostic_v11',created_utc=datetime.now(timezone.utc).isoformat(),
        status='complete' if not pending else 'pending_initial12_completion',complete=not pending,
        fixed_initial12_completed=12-len(pending),fixed_initial12_expected=12,pending_case_ids=pending,
        all_fixed_cases=case_status,cup_cases_completed=len(cases),cup_cases_expected=8,
        TRAIN_only=True,VAL_read=False,TEST_read=False,native_or_GPU_launched_by_audit=False,
        namespaces=dict(public_observation=['positions','rotations','rotation_observable','valid','tracker_diagnostics.bodies'],
            offline_metric_only=['target_positions','target_rotations','native_pose_valid'],
            mode_partition='public inferred modes plus mode_valid; not native hidden contact schedule',
            dynamics_target_valid_is_not_the_metric_pose_availability_mask=True),
        source_generalization=dict(prototype_independent_TRAIN_parents=sorted(prototype_parents),initial12_cup_parents=sorted(cup_parents),
            overlapping_parents=sorted(cup_parents&prototype_parents),disjoint_parents=sorted(cup_parents-prototype_parents),
            all_initial_cup_parents_disjoint_from_prototype=not bool(cup_parents&prototype_parents),
            same_parent_reacquisition_is_not_independent_generalization=True,
            new_evidence='longer fresh native hold and teacher/contact sequences on the same fixed first TRAIN parents'),
        overall=summarize(rows),by_behavior={b:summarize([r for r in rows if r['behavior']==b]) for b in ('hold','teacher')},
        by_behavior_mode=full_by_mode,cases=cases,
        independently_completed_phase_parent_counts={task:{phase:len(parents) for (t,phase),parents in phase_parents.items() if t==task}
            for task in sorted({t for t,p in phase_parents})},
        full_R_evidence_but_SO3_above_30deg_count=len(large),
        largest_full_R_error_examples=sorted(large,key=lambda r:r['SO3_error_degrees'],reverse=True)[:20],
        all_current_registered_position_and_rotation_samples=rows,
        files_read_SHA256=verified,all_completed_case_inputs_SHA_rechecked_at_end=all(file_sha(ROOT/p)==sha for p,sha in verified.items()),
        original_data_or_core_plan_labels_modified=False,samples_excluded_from_training_or_collection_by_audit=0,
        learned_policy_test_success_rate_evaluated=False,arbitrary_collection_gate_added=False,
        next_actions=['Continue the frozen full1400 TRAIN-parent teacher+hold collection without GT-error filtering.',
            'Audit independent remaining TRAIN parents separately from these four overlapping parents.',
            'Use all acquired parents for the planned grouped Context/library fits; preserve failed and incomplete teacher skills.',
            'Inspect any confident >30deg cases by current mask/occlusion/handle coverage and mode; do not tune on VAL/TEST.'])
    require(report['all_completed_case_inputs_SHA_rechecked_at_end'],'a completed input changed while auditing')
    destination=BASE/'initial12_cup_feature_native_audit_v11.json'
    if destination.exists():
        old=json.loads(destination.read_text());require(old.get('complete') is not True,'completed initial12 audit is immutable')
        destination.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    else:atomic_json(destination,report)
    print(json.dumps(dict(status=report['status'],report=str(destination.relative_to(ROOT)),
        cup_cases=len(cases),body_frames=len(rows),full_R_big_errors=len(large),
        by_behavior_mode=full_by_mode,source_overlap=report['source_generalization']),ensure_ascii=False))


if __name__=='__main__':main()
