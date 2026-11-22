"""Freeze zero-bias static priors and audit all original buzz teacher parents.

New, derived TRAIN-only artifacts. No collector/tracker/protocol modification,
no simulator, GPU, teacher execution, or additional acquisition is performed.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a
from our_mani.multitask.training_view import validate_arrays
from our_mani.multitask import two_center_pose as pose
from scripts import diagnose_buzz_public_position_pose_r10 as diagnostic

BASE=ROOT/'artifacts/framework/multitask_r10'
CV=BASE/'buzz_public_position_pose_TRAIN_diagnostic_20261018T0740Z/report.json'
CV_SHA='6a3354e778cdb807b4bb0ad636a376e596356d864a7376fa36480dc4d83553a2'
HOLD='TRAIN_hold_motion_identification';TEACHER='TRAIN_teacher_committed_waypoint'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def math_parameters(prior):
    return dict(local_axis=prior['local_axis'],world_axis=prior['world_axis'],
        local_frame_to_wand_offset=prior['native_local_frame_to_wand_offset'],
        local_basis=prior['local_basis'],frame_to_wand_rotation=prior['frame_to_wand_rotation'])


def public_row(arrays,index):
    modes=('external','attached','released')
    return {role:pose.PublicCenter(arrays['positions'][index,j],bool(arrays['valid'][index,j]),
        bool(arrays['mode_valid'][index,j]),modes[int(arrays['modes'][index,j])] if int(arrays['modes'][index,j]) in (0,1,2) else 'unknown',
        float(arrays['times'][index])) for j,role in enumerate(('frame','wand'))}


def load_arrays(entry):
    with np.load(ROOT/entry['cache'],allow_pickle=False) as stored:arrays={k:stored[k].copy() for k in stored.files}
    ids=list(map(str,arrays['body_ids']));order=[ids.index('frame'),ids.index('wand')]
    for key in ('positions','rotations','valid','modes','mode_valid','rotation_observable','target_positions','target_rotations',
        'native_pose_valid','target_pose_valid_before_twist_mask','target_rotation_observable','target_valid'):
        arrays[key]=arrays[key][:,order]
    return arrays


def fit_row(case,arrays):
    return dict(parent_source_group=case['parent_source_group'],
        public={k:arrays[k] for k in ('positions','valid','mode_valid','modes')},
        labels=dict(positions=arrays['target_positions'],rotations=arrays['target_rotations'],
            native_pose_valid=arrays['native_pose_valid'],pose_valid_before_twist=arrays['target_pose_valid_before_twist_mask'],
            rotation_label_valid=arrays['target_rotation_observable']))


def main(output):
    output=Path(output).resolve()
    if not output.is_relative_to(BASE) or output.exists():raise ValueError('fresh ROOT artifact directory required')
    output.mkdir(parents=True);files={}
    cv=a.read(a.checked_pointer(ROOT,dict(path=str(CV.relative_to(ROOT)),sha256=CV_SHA),files))
    original_plan=a.read(a.checked_pointer(ROOT,cv['fixed_plan'],files))
    snapshot=a.read(a.checked_pointer(ROOT,original_plan['snapshot'],files))
    registrations=a.read(a.checked_pointer(ROOT,original_plan['registrations'],files));contexts={}
    for registration in registrations:
        context=a.load_archived_plan(ROOT,registration,files)
        if 'predecessor_plan_sha256' in registration:
            a.verify_recovery_transition(ROOT,contexts[registration['predecessor_plan_sha256']],context,files)
        contexts[context['plan']['plan_sha256']]=context
    base=contexts[next(iter(contexts))]
    expected={c['case_id']:c for c in base['plan']['cases'] if c['task'] in pose.TASKS and c['behavior'] in (HOLD,TEACHER)}
    inventory=[r for r in snapshot['snapshot_committed_receipts'] if r['case_id'] in expected]
    assert len(expected)==len(inventory)==400 and len({r['case_id'] for r in inventory})==400
    write(output/'fixed_plan.json',dict(schema='r10_buzz_two_center_static_prior_and_teacher_scope_plan_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),parent_crossvalidation=a.pointer(ROOT,CV),
        fixed_case_ids=sorted(expected),hold_parents_per_task=100,teacher_parents_per_task=100,
        teacher_primary_metric='fixed parent-crossfit zero-bias model (other 50 hold TRAIN parents), every teacher frame retained',
        final_model='one static zero-bias prior fit to all100 hold TRAIN parents per task; no parent-specific model at runtime',
        no_outlier_removal=True,no_parameter_search=True,mean_bias_variant_not_selected=True,
        freshness='current same-frame observed_at only within 1e-9s numeric tolerance; archived rows lack independent per-body timestamps',
        native_or_GPU_started=False,live_sources_modified=False,orientation_permission_granted=False))
    loaded={};provenance=[]
    for item in inventory:
        case=expected[item['case_id']];context=contexts[item['actual_plan_sha256']]
        proof=a.verify_completed(ROOT,item,context,case,files)
        if context is not base:
            preceding=[r for r in snapshot['infrastructure_receipts'] if r['case_id']==case['case_id']]
            preceding=sorted(preceding,key=lambda r:(a.instant(r['finished_utc']),r['parent_receipt']['path']))
            a.verify_recovery_execution(ROOT,proof,context,contexts[context['registration']['predecessor_plan_sha256']],preceding,files)
        record=a.read(ROOT/item['canonical_receipt']['path']);entry=record['entry']
        a.checked_pointer(ROOT,dict(path=entry['cache'],sha256=entry['cache_sha256']),files)
        with np.load(ROOT/entry['cache'],allow_pickle=False) as stored:raw={k:stored[k].copy() for k in stored.files}
        validate_arrays(raw,entry,record)
        loaded[case['case_id']]=load_arrays(entry)
        provenance.append(dict(case_id=case['case_id'],task=case['task'],behavior=case['behavior'],parent_source_group=case['parent_source_group'],
            actual_plan_sha256=item['actual_plan_sha256'],canonical_receipt=item['canonical_receipt'],
            raw_child_receipt=proof['child_receipt'],actual_parent_receipt=proof['parent_receipt'],
            cache=dict(path=entry['cache'],sha256=entry['cache_sha256'])))
    canonical={};tails={}
    for task in pose.TASKS:
        cases=[c for c in expected.values() if c['task']==task]
        paths={c['template_asset'] for c in cases};assert len(paths)==1
        asset_path=Path(next(iter(paths)));asset=a.read(asset_path);asset_pointer=a.pointer(ROOT,asset_path)
        a.checked_pointer(ROOT,asset_pointer,files)
        assert asset['task']==task and asset['world_pose_or_motion_schedule_in_asset'] is False
        canonical[task]=dict(TRAIN_CAD=asset_pointer,asset_sha256=asset['asset_sha256'],
            role_native_roots={'frame':'Cuboid','wand':'wand'},
            frame_definition='Native role-root local axes used both by frozen TRAIN CAD export and TRAIN target_R labels')
        tails[task]={}
        for variant,metrics in cv['tasks'][task]['variants'].items():
            pointer=next(p for p in cv['prediction_files'] if p['path'].endswith(task+'__'+variant+'.npz'))
            path=a.checked_pointer(ROOT,pointer,files)
            with np.load(path,allow_pickle=False) as stored:errors=stored['SO3_error_rad'].copy()
            tails[task][variant]=dict(frame_errors_above_point15_rad=int((errors[:,0]>.15).sum()),
                wand_errors_above_point15_rad=int((errors[:,1]>.15).sum()),
                frame_parent_q95_above_point15_rad=sum(p['frame_SO3_error_rad']['q95']>.15 for p in metrics['per_parent']),
                wand_parent_q95_above_point15_rad=sum(p['wand_SO3_error_rad']['q95']>.15 for p in metrics['per_parent']),
                all_errors_retained=True,prediction_archive=pointer)
    suffixes={
        'third_party/RLBench/rlbench/tasks/beat_the_rotating_buzz.py':[(165,185),(201,211),(231,247)],
        'third_party/RLBench/rlbench/tasks/beat_the_rotating_buzz_high_speed.py':[(165,185),(201,211),(231,247)],
        'third_party/PyRep/pyrep/objects/object.py':[(497,510)],
        'our_mani/multitask/specs.py':[(125,141)],
        'our_mani/multitask/training_geometry.py':[(132,151)],
        'scripts/collect_multitask_training_r10.py':[(842,858)]}
    sources=[]
    for suffix,spans in suffixes.items():
        source_context,original,pointer=next((context,name,p) for context in contexts.values()
            for name,p in context['sources'].items() if name.endswith(suffix))
        path=a.checked_pointer(ROOT,pointer,files);lines=path.read_text().splitlines()
        sources.append(dict(original_source_path=original,executed_source_archive=pointer,
            actual_archived_plan_sha256=source_context['plan']['plan_sha256'],
            source_archive_is_original_v11=source_context is base,
            excerpts=[dict(line=i,text=lines[i-1]) for low,high in spans for i in range(low,min(high,len(lines))+1)]))
    sidecar=dict(schema='r10_TRAIN_buzz_two_center_pose_source_basis_and_tail_audit_v1',
        parent_crossvalidation=a.pointer(ROOT,CV),canonical_bases=canonical,unfiltered_tails=tails,source_evidence=sources,
        outliers_removed=False,exact_rigid_pair_claimed=False,
        rigidity='Observed hold frame-to-wand offsets/relative rotations have nonzero temporal and between-parent spread. A static learned approximate relation is empirically useful; it is not an exact physical constraint.',
        rotation_axis='Actual Object.rotate forms current world y_axis from the object matrix. TRAIN independently learned local y maps near world Z; world Z was not an estimator assumption.',
        orientation_permission_granted=False)
    write(output/'source_basis_and_tails.json',sidecar);tail_pointer=a.pointer(ROOT,output/'source_basis_and_tails.json')
    helper_pointer=a.pointer(ROOT,Path(pose.__file__));model_pointers={};full_priors={};fold_priors={};parity=[]
    for task in pose.TASKS:
        holds=[c for c in expected.values() if c['task']==task and c['behavior']==HOLD];assert len(holds)==100
        prior=diagnostic.fit_static_prior([fit_row(c,loaded[c['case_id']]) for c in holds]);parameters=math_parameters(prior)
        receipt=dict(schema='r10_buzz_two_center_TRAIN_static_prior_receipt_v1',task=task,split='train',behavior=HOLD,
            fixed_original_hold_parents=100,training_parent_source_groups=sorted(c['parent_source_group'] for c in holds),
            actual_case_provenance=[p for p in provenance if p['task']==task and p['behavior']==HOLD],
            static_prior=parameters,canonical_basis=canonical[task],helper_source=helper_pointer,
            parent_crossvalidation=a.pointer(ROOT,CV),tails_and_canonical_basis=tail_pointer,
            static_fitter_source=a.pointer(ROOT,Path(diagnostic.__file__)),derivation_source=a.pointer(ROOT,Path(__file__)),
            actual_registration_list=original_plan['registrations'],public_centroid_bias_correction=False,
            full_fit_diagnostics={k:v for k,v in prior.items() if k not in ('training_parents','training_row_counts')},
            orientation_permission_granted=False)
        receipt_path=output/'models'/task/'TRAIN_static_fit_receipt.json';write(receipt_path,receipt)
        model=dict(schema=pose.MODEL_VERSION,method=pose.METHOD,task=task,roles={'frame':'frame','wand':'wand'},static_prior=parameters,
            canonical_basis=canonical[task],TRAIN_evidence=dict(parent_crossvalidation=a.pointer(ROOT,CV),
                tails_and_canonical_basis=tail_pointer,training_model_receipt=a.pointer(ROOT,receipt_path)),helper_source=helper_pointer,
            configuration=dict(public_centroid_bias_correction=False,numeric_radial_epsilon_m=pose.NUMERIC_RADIAL_EPS_M,
                timestamp_numeric_tolerance_seconds=pose.TIMESTAMP_NUMERIC_TOLERANCE_SECONDS),
            runtime_scope='conditional_fresh_external_pair_only',orientation_permission_granted=False)
        path=output/'models'/task/'model.json';write(path,model);model_pointers[task]=a.pointer(ROOT,path)
        full_priors[task]=pose.load_two_center_pose_prior(ROOT,model_pointers[task],canonical_asset_pointer=canonical[task]['TRAIN_CAD'])
        fold_priors[task]={}
        for fold in ('0','1'):
            source=cv['tasks'][task]['folds'][fold];parameters=math_parameters(source['prior'])
            projection=dict(schema='r10_CPU_parent_crossfit_static_prior_projection_v1',task=task,fold=int(fold),
                static_prior=parameters,training_parent_source_groups=source['prior']['training_parents'],
                held_parent_source_groups=source['held_parents'],parent_crossvalidation=a.pointer(ROOT,CV),
                canonical_basis=canonical[task],helper_source=helper_pointer,method=pose.METHOD,orientation_permission_granted=False)
            path=output/'models'/task/('CPU_parent_fold'+fold+'.json');write(path,projection);pointer=a.pointer(ROOT,path)
            prior=pose.TwoCenterPosePrior(task,pointer['sha256'],canonical[task]['asset_sha256'],**parameters)
            fold_priors[task][fold]=(prior,pointer)
        # Check that the independent pure helper reproduces every previous
        # fixed hold-CV prediction, not merely one synthetic identity example.
        archive=next(p for p in cv['prediction_files'] if p['path'].endswith(task+'__no_public_bias_correction.npz'))
        with np.load(ROOT/archive['path'],allow_pickle=False) as stored:
            ids=stored['case_id'].copy();indices=stored['frame_index'].copy();previous=stored['predicted_rotations'].copy()
        max_difference=0.
        for cid,index,old in zip(ids,indices,previous):
            case=expected[str(cid)];fold=str(original_plan['parent_folds'][case['parent_source_group']]);arrays=loaded[str(cid)]
            pred=pose.estimate_two_center_pose(fold_priors[task][fold][0],public_row(arrays,int(index)),current_timestamp=float(arrays['times'][index]))
            assert pred.valid
            max_difference=max(max_difference,float(np.max(np.abs(np.stack((pred.frame_rotation,pred.wand_rotation))-old))))
        assert max_difference<1e-12
        parity.append(dict(task=task,frames=len(ids),maximum_rotation_matrix_difference=max_difference))
    teacher_reports={};teacher_prediction_files=[]
    for task in pose.TASKS:
        cases=[c for c in expected.values() if c['task']==task and c['behavior']==TEACHER];assert len(cases)==100
        per_parent=[];all_errors=[];all_valid=[];all_reasons=[];all_ids=[];all_indices=[];phases=defaultdict(list)
        for case in cases:
            arrays=loaded[case['case_id']];fold=str(original_plan['parent_folds'][case['parent_source_group']]);prior,pointer=fold_priors[task][fold]
            predictions=[pose.estimate_two_center_pose(prior,public_row(arrays,i),current_timestamp=float(t)) for i,t in enumerate(arrays['times'])]
            valid=np.asarray([p.valid for p in predictions]);R=np.full((len(valid),2,3,3),np.nan)
            for i,pred in enumerate(predictions):
                if pred.valid:R[i]=np.stack((pred.frame_rotation,pred.wand_rotation))
            # All current-only predictions finish before teacher GT scoring.
            label_valid=(arrays['native_pose_valid']&arrays['target_rotation_observable']).all(axis=1)
            label_valid&=np.isfinite(arrays['target_rotations']).all(axis=(1,2,3))
            errors=np.full((len(valid),2),np.nan);mask=valid&label_valid
            errors[mask]=diagnostic.angles(R[mask],arrays['target_rotations'][mask])
            reasons=[p.reason for p in predictions]
            phase=list(map(str,arrays['teacher_phase']))
            for i,name in enumerate(phase):phases[name].append((bool(valid[i]),errors[i].copy(),reasons[i]))
            per_parent.append(dict(case_id=case['case_id'],parent_source_group=case['parent_source_group'],fold=int(fold),model=pointer,
                frames=len(valid),accepted=int(valid.sum()),failures=dict(Counter(reason for ok,reason in zip(valid,reasons) if not ok)),
                frame_SO3_error_rad=diagnostic.describe(errors[:,0]),wand_SO3_error_rad=diagnostic.describe(errors[:,1]),
                frame_errors_above_point15_rad=int((errors[:,0]>.15).sum()),wand_errors_above_point15_rad=int((errors[:,1]>.15).sum()),
                public_wand_mode_counts=dict(Counter(map(int,arrays['modes'][:,1])))))
            all_errors.append(errors);all_valid.append(valid);all_reasons.extend(reasons)
            all_ids.extend([case['case_id']]*len(valid));all_indices.extend(range(len(valid)))
        errors=np.concatenate(all_errors);valid=np.concatenate(all_valid)
        teacher_reports[task]=dict(parents=100,total_frames=len(valid),accepted_frames=int(valid.sum()),coverage_fraction=float(valid.mean()),
            failures=dict(Counter(reason for ok,reason in zip(valid,all_reasons) if not ok)),
            frame_SO3_error_rad=diagnostic.describe(errors[:,0]),wand_SO3_error_rad=diagnostic.describe(errors[:,1]),
            frame_errors_above_point15_rad=int((errors[:,0]>.15).sum()),wand_errors_above_point15_rad=int((errors[:,1]>.15).sum()),
            per_phase={name:dict(frames=len(values),accepted=sum(v[0] for v in values),
                failures=dict(Counter(v[2] for v in values if not v[0])),
                frame_SO3_error_rad=diagnostic.describe([v[1][0] for v in values]),
                wand_SO3_error_rad=diagnostic.describe([v[1][1] for v in values])) for name,values in sorted(phases.items())},
            per_parent=per_parent)
        path=output/(task+'__all100_teacher_parent_crossfit_predictions.npz')
        np.savez_compressed(path,prediction_valid=valid,SO3_error_rad=errors,reason=np.asarray(all_reasons),
            case_id=np.asarray(all_ids),frame_index=np.asarray(all_indices))
        teacher_prediction_files.append(a.pointer(ROOT,path))
    for path,checksum in files.items():assert a.file_sha(path)==checksum
    report=dict(schema='r10_buzz_two_center_prior_and_full_teacher_TRAIN_scope_diagnostic_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),fixed_plan=a.pointer(ROOT,output/'fixed_plan.json'),
        helper_source=helper_pointer,builder_source=a.pointer(ROOT,Path(__file__)),models=model_pointers,
        canonical_bases=canonical,source_basis_and_tails=tail_pointer,parent_crossvalidation=a.pointer(ROOT,CV),
        actual_case_provenance=provenance,all400_original_hold_and_teacher_cases_retained=True,
        hold_parent_crossfit_parity=parity,teacher_parent_crossfit=teacher_reports,prediction_archives=teacher_prediction_files,
        public_none_orientation_permission_changed=False,live_tracker_connected=False,all_input_SHA_stable_before_after=True,
        verified_files_sha256=files,native_or_GPU_started=False,
        caveats=[
            'External public modes are a necessary guard, not proof that the native frame/wand relation remains rigid. Teacher-domain errors test this conditional assumption.',
            'Archived rows have one frame time and valid flags, not independently saved per-body observed_at. Replay uses row time; future live integration must supply actual current observation timestamps.',
            'Attached/released/unknown/stale/degenerate observations return no rotations. No prior estimate is carried forward.',
            'Zero-bias is fixed. The mean-bias diagnostic is not selected. Raw arrays, failures, terminal and censored teacher frames are retained.',
            'Static priors include privileged TRAIN supervision as an explicit shared prior. They do not constitute full RGBD pose observability or authorize a new runtime protocol.'])
    path=output/'report.json';write(path,report)
    lines=['Buzz two-center pure helper; static TRAIN priors frozen, no runtime wiring.']
    for task,metrics in teacher_reports.items():lines.append(task+': teacher100 parents, accepted '+str(metrics['accepted_frames'])+'/'+str(metrics['total_frames'])+', frame q95='+str(metrics['frame_SO3_error_rad']['q95'])+', wand q95='+str(metrics['wand_SO3_error_rad']['q95']))
    (output/'report.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=a.pointer(ROOT,path),summary=lines),ensure_ascii=False),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    main(parser.parse_args().out)
