"""Fixed TRAIN-only buzz pose inference diagnostic; no native/GPU execution.

Fitting sees TRAIN pose labels. The runtime-shaped estimator receives only two
current public centroids and their public validity/mode fields, plus the static
prior learned on other parents. It never reads native yaw, time or future poses.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as lineage
from our_mani.multitask.training_view import validate_arrays

BASE=ROOT/'artifacts/framework/multitask_r10'
SNAPSHOT=BASE/'collection_snapshot_20261018T071841976208Z/report.json'
SNAPSHOT_SHA='9598fd71df519352cf19b0502130295066844267e87994650efb1b02084182a7'
REGISTRATIONS=BASE/'TRAIN_composite_registration_preflight_v4/registrations.json'
REGISTRATIONS_SHA='1daf0b65ce6cdf590bc4c955b49ebb00ced06234a127568bd8bb183b487d0983'
TASKS=('beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed')
BEHAVIOR='TRAIN_hold_motion_identification'
EPS=1e-8


def utc():return datetime.now(timezone.utc).isoformat()


def rotation_mean(matrices):
    u,_,vt=np.linalg.svd(np.mean(matrices,axis=0))
    return u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt


def angles(prediction,truth):
    relative=np.einsum('...ji,...jk->...ik',prediction,truth)
    return np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1.,1.))


def describe(values):
    values=np.asarray(values,dtype=float).reshape(-1);finite=values[np.isfinite(values)]
    return dict(total=int(values.size),finite=int(finite.size),nonfinite=int(values.size-finite.size),
        mean=None if not finite.size else float(np.mean(finite)),
        q50=None if not finite.size else float(np.quantile(finite,.5)),
        q95=None if not finite.size else float(np.quantile(finite,.95)),
        maximum=None if not finite.size else float(np.max(finite)))


def native_mask(row):
    # target_valid additionally masks backward-difference boundary samples;
    # pose validity and orientation-label validity are separate saved fields.
    truth=row['labels']
    return (truth['native_pose_valid'].all(axis=1)&truth['pose_valid_before_twist'].all(axis=1)
        &truth['rotation_label_valid'].all(axis=1)
        &np.isfinite(truth['positions']).all(axis=(1,2))
        &np.isfinite(truth['rotations']).all(axis=(1,2,3)))


def public_mask(public):
    return (public['valid'].all(axis=-1)&public['mode_valid'].all(axis=-1)
        &(public['modes']==0).all(axis=-1)&np.isfinite(public['positions']).all(axis=(-1,-2)))


def fit_static_prior(training_rows):
    """TRAIN labels fit static geometry/axis only; no per-held-parent labels."""
    matrices=[];offsets=[];relatives=[];motion=[];biases=[];counts={}
    for row in training_rows:
        mask=native_mask(row);truth=row['labels'];R=truth['rotations'][mask,0]
        if not len(R):raise ValueError('TRAIN parent has no finite native pose labels')
        matrices.append(R)
        delta=truth['positions'][mask,1]-truth['positions'][mask,0]
        offsets.append(np.mean(np.einsum('tji,tj->ti',R,delta),axis=0))
        relatives.append(rotation_mean(np.einsum('tji,tjk->tik',R,truth['rotations'][mask,1])))
        difference=R-R[0]
        motion.append(np.mean(np.einsum('tji,tjk->tik',difference,difference),axis=0))
        observed=mask&public_mask(row['public'])
        public_delta=row['public']['positions'][observed,1]-row['public']['positions'][observed,0]
        native_delta=truth['positions'][observed,1]-truth['positions'][observed,0]
        if not len(public_delta):raise ValueError('TRAIN parent has no public/native bias pairs')
        biases.append(np.mean(public_delta-native_delta,axis=0))
        counts[row['parent_source_group']]=dict(native_pose_rows=int(mask.sum()),public_bias_rows=int(observed.sum()))
    eigenvalues,eigenvectors=np.linalg.eigh(np.mean(motion,axis=0))
    axis=eigenvectors[:,0];axis*=1 if axis[np.argmax(np.abs(axis))]>=0 else -1
    all_R=np.concatenate(matrices);world_vectors=all_R@axis
    world_axis=np.mean(world_vectors,axis=0);world_axis/=np.linalg.norm(world_axis)
    local_offset=np.mean(offsets,axis=0)
    radial=local_offset-axis*np.dot(local_offset,axis)
    if eigenvalues[-1]<=1e-10 or np.linalg.norm(radial)<=EPS:
        raise ValueError('static axis or radial offset is unidentifiable under the fixed numeric checks')
    radial/=np.linalg.norm(radial)
    local_basis=np.column_stack((radial,np.cross(axis,radial),axis))
    return dict(local_axis=axis.tolist(),world_axis=world_axis.tolist(),
        native_local_frame_to_wand_offset=local_offset.tolist(),
        frame_to_wand_rotation=rotation_mean(np.asarray(relatives)).tolist(),
        local_basis=local_basis.tolist(),mean_public_minus_native_delta=np.mean(biases,axis=0).tolist(),
        axis_motion_eigenvalues=eigenvalues.tolist(),
        closest_local_cardinal_axis='xyz'[int(np.argmax(np.abs(axis)))],
        world_axis_angular_spread_rad=describe(np.arccos(np.clip(world_vectors@world_axis,-1.,1.))),
        training_parents=[row['parent_source_group'] for row in training_rows],training_row_counts=counts,
        parent_weights_equal=True,native_yaw_parameter_used=False)


def estimate_current_pose(public_current,prior,*,subtract_training_mean_bias=False):
    """The estimator has no access to labels, future samples or native driver."""
    if set(public_current)!={'positions','valid','mode_valid','modes'}:
        raise ValueError('estimator input must contain only public current centroids and masks')
    if not bool(public_mask(public_current)):
        return dict(valid=False,failure='public_invalid_nonexternal_or_nonfinite')
    positions=public_current['positions'];delta=positions[1]-positions[0]
    if subtract_training_mean_bias:delta=delta-np.asarray(prior['mean_public_minus_native_delta'])
    axis=np.asarray(prior['world_axis']);radial=delta-axis*np.dot(delta,axis)
    norm=np.linalg.norm(radial)
    if not np.isfinite(norm) or norm<=EPS:
        return dict(valid=False,failure='public_centroid_radial_vector_degenerate')
    radial/=norm
    frame=np.column_stack((radial,np.cross(axis,radial),axis))@np.asarray(prior['local_basis']).T
    wand=frame@np.asarray(prior['frame_to_wand_rotation'])
    return dict(valid=True,frame_rotation=frame,wand_rotation=wand,radial_length_m=float(norm))


def rigid_diagnostics(rows):
    local=[];relative=[];parents=[];cardinals=[[],[],[]]
    for row in rows:
        mask=native_mask(row);truth=row['labels'];R=truth['rotations'][mask,0]
        delta=truth['positions'][mask,1]-truth['positions'][mask,0]
        offset=np.einsum('tji,tj->ti',R,delta)
        Q=np.einsum('tji,tjk->tik',R,truth['rotations'][mask,1])
        center=np.mean(offset,axis=0);rotation=rotation_mean(Q)
        local.append(offset);relative.append(Q)
        for j in range(3):cardinals[j].append(R[:,:,j])
        parents.append(dict(parent_source_group=row['parent_source_group'],frames=len(mask),native_rows=int(mask.sum()),
            local_offset_mean=center.tolist(),offset_temporal_spread_m=describe(np.linalg.norm(offset-center,axis=1)),
            relative_rotation_temporal_spread_rad=describe(angles(Q,rotation)),
            relative_rotation_mean=rotation.tolist()))
    local=np.concatenate(local);relative=np.concatenate(relative)
    center=np.mean(local,axis=0);rotation=rotation_mean(relative)
    cardinal_report={}
    for j,parts in enumerate(cardinals):
        vectors=np.concatenate(parts);mean=np.mean(vectors,axis=0);norm=np.linalg.norm(mean)
        cardinal_report['xyz'[j]]=dict(mean_world_vector=mean.tolist(),mean_resultant_norm=float(norm),
            angular_spread_rad=describe(np.arccos(np.clip(vectors@(mean/norm),-1.,1.))) if norm>EPS else None)
    return dict(all_native_local_offset_mean=center.tolist(),
        all_parent_and_time_offset_spread_m=describe(np.linalg.norm(local-center,axis=1)),
        parent_mean_offset_spread_m=describe(np.linalg.norm(np.asarray([p['local_offset_mean'] for p in parents])-center,axis=1)),
        all_native_relative_rotation_mean=rotation.tolist(),
        all_parent_and_time_relative_rotation_spread_rad=describe(angles(relative,rotation)),
        parent_mean_relative_rotation_spread_rad=describe(angles(np.asarray([p['relative_rotation_mean'] for p in parents]),rotation)),
        native_local_cardinal_axes_in_world=cardinal_report,parents=parents)


def synthetic_self_check():
    # Deliberately not world Z; two vector correspondences recover full SO3.
    local_axis=np.array([0.,1.,0.]);world_axis=np.array([1.,0.,0.])
    local_basis=np.column_stack(([1.,0.,0.],[0.,0.,-1.],local_axis))
    world_basis=np.column_stack(([0.,1.,0.],[0.,0.,1.],world_axis))
    R=world_basis@local_basis.T;offset=np.array([.2,.03,0.]);delta=R@offset;bias=np.array([.01,-.02,.03])
    prior=dict(world_axis=world_axis,local_basis=local_basis,frame_to_wand_rotation=np.eye(3),mean_public_minus_native_delta=bias)
    public=dict(positions=np.stack((np.zeros(3),delta+bias)),valid=np.ones(2,bool),mode_valid=np.ones(2,bool),modes=np.zeros(2,int))
    result=estimate_current_pose(public,prior,subtract_training_mean_bias=True)
    assert result['valid'] and np.max(np.abs(result['frame_rotation']-R))<1e-12
    public['valid'][1]=False;assert not estimate_current_pose(public,prior)['valid']
    public['valid'][1]=True;public['positions'][1]=world_axis*.1
    assert not estimate_current_pose(public,prior)['valid']
    try:estimate_current_pose(dict(public,native_rotation=R),prior)
    except ValueError:pass
    else:raise AssertionError('runtime input accepted native rotation')


def main(output):
    started=utc();output=Path(output).resolve()
    if not output.is_relative_to(BASE) or output.exists():raise ValueError('fresh diagnostic output under ROOT required')
    output.mkdir(parents=True);synthetic_self_check()
    files={};snapshot_path=lineage.checked_pointer(ROOT,dict(path=str(SNAPSHOT.relative_to(ROOT)),sha256=SNAPSHOT_SHA),files)
    snapshot=lineage.read(snapshot_path);registrations=lineage.read(lineage.checked_pointer(ROOT,
        dict(path=str(REGISTRATIONS.relative_to(ROOT)),sha256=REGISTRATIONS_SHA),files))
    contexts={}
    for registration in registrations:
        context=lineage.load_archived_plan(ROOT,registration,files)
        if 'predecessor_plan_sha256' in registration:
            lineage.verify_recovery_transition(ROOT,contexts[registration['predecessor_plan_sha256']],context,files)
        contexts[context['plan']['plan_sha256']]=context
    base=contexts[next(iter(contexts))];expected={c['case_id']:c for c in base['plan']['cases'] if c['task'] in TASKS and c['behavior']==BEHAVIOR}
    inventory=[row for row in snapshot['snapshot_committed_receipts'] if row['case_id'] in expected]
    assert len(expected)==len(inventory)==200 and len({r['case_id'] for r in inventory})==200
    folds={}
    for task in TASKS:
        parents=sorted([r['parent_source_group'] for r in inventory if r['task']==task],key=lambda p:hashlib.sha256(p.encode()).hexdigest())
        assert len(parents)==len(set(parents))==100
        folds.update({parent:index%2 for index,parent in enumerate(parents)})
    plan=dict(schema='r10_fixed_TRAIN_buzz_public_centroid_pose_diagnostic_plan_v1',created_utc=utc(),
        snapshot=lineage.pointer(ROOT,SNAPSHOT),registrations=lineage.pointer(ROOT,REGISTRATIONS),tasks=list(TASKS),
        fixed_case_ids=sorted(expected),parent_folds=folds,fold_rule='Per task sort SHA256(parent_source_group), alternate fold0/fold1; 50 training/50 held parents.',
        variants=['no_public_bias_correction','subtract_training_mean_public_minus_native_delta'],
        numerical_radial_degeneracy_threshold_m=EPS,numerical_motion_eigenvalue_threshold=1e-10,
        axis_inferred_from_TRAIN_R_differences=True,world_Z_assumed=False,
        runtime_inputs=['current public positions','valid','mode_valid','external modes'],
        GT_roles=['other-parent TRAIN static rigid/axis prior','held-parent scoring only'],
        native_yaw_or_future_pose_runtime_inputs=False,outlier_filtering=False,parameters_searched=False,
        frozen_production_sources_modified=False,orientation_none_protocol_changed=False,
        native_or_GPU_started=False,original_angular_error_gate_rad=.15)
    (output/'fixed_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    rows=[]
    for item in inventory:
        context=contexts[item['actual_plan_sha256']];case=expected[item['case_id']]
        proof=lineage.verify_completed(ROOT,item,context,case,files)
        if context is not base:
            preceding=[r for r in snapshot['infrastructure_receipts'] if r['case_id']==item['case_id']]
            preceding=sorted(preceding,key=lambda r:(lineage.instant(r['finished_utc']),r['parent_receipt']['path']))
            lineage.verify_recovery_execution(ROOT,proof,context,contexts[context['registration']['predecessor_plan_sha256']],preceding,files)
        record=lineage.read(ROOT/item['canonical_receipt']['path']);entry=record['entry']
        cache=lineage.checked_pointer(ROOT,dict(path=entry['cache'],sha256=entry['cache_sha256']),files)
        with np.load(cache,allow_pickle=False) as stored:arrays={name:stored[name].copy() for name in stored.files}
        validate_arrays(arrays,entry,record)
        assert arrays['teacher_action_valid'].sum()==0 and entry['teacher_used'] is False
        ids=list(map(str,arrays['body_ids']));indices=[ids.index('frame'),ids.index('wand')]
        rows.append(dict(case_id=item['case_id'],task=item['task'],parent_source_group=item['parent_source_group'],
            actual_plan_sha256=item['actual_plan_sha256'],canonical_receipt=item['canonical_receipt'],
            child_receipt=proof['child_receipt'],parent_receipt=proof['parent_receipt'],cache=lineage.pointer(ROOT,cache),times=arrays['times'],
            public={key:arrays[key][:,indices].copy() for key in ('positions','valid','mode_valid','modes')},
            public_rotation_observable=arrays['rotation_observable'][:,indices],
            labels=dict(positions=arrays['target_positions'][:,indices],rotations=arrays['target_rotations'][:,indices],
                native_pose_valid=arrays['native_pose_valid'][:,indices],pose_valid_before_twist=arrays['target_pose_valid_before_twist_mask'][:,indices],
                rotation_label_valid=arrays['target_rotation_observable'][:,indices],target_valid=arrays['target_valid'][:,indices])))
    reports={};prediction_files=[]
    for task in TASKS:
        selected=[r for r in rows if r['task']==task]
        report=dict(parents=len(selected),frames=sum(len(r['times']) for r in selected),rigid_geometry=rigid_diagnostics(selected),folds={},variants={})
        all_results={variant:[] for variant in plan['variants']}
        for fold in (0,1):
            training=[r for r in selected if folds[r['parent_source_group']]!=fold]
            held=[r for r in selected if folds[r['parent_source_group']]==fold]
            prior=fit_static_prior(training);report['folds'][str(fold)]=dict(prior=prior,held_parents=[r['parent_source_group'] for r in held])
            for variant in plan['variants']:
                for row in held:
                    before=time.perf_counter();predictions=[estimate_current_pose({k:v[i].copy() for k,v in row['public'].items()},prior,
                        subtract_training_mean_bias=variant==plan['variants'][1]) for i in range(len(row['times']))]
                    elapsed=time.perf_counter()-before
                    valid=np.asarray([p['valid'] for p in predictions]);rotations=np.full((len(valid),2,3,3),np.nan)
                    for i,pred in enumerate(predictions):
                        if pred['valid']:rotations[i]=np.stack((pred['frame_rotation'],pred['wand_rotation']))
                    # Predictions for this held parent are complete before its
                    # native pose labels are consulted for scoring.
                    labels=native_mask(row);scorable=valid&labels;errors=np.full((len(valid),2),np.nan)
                    errors[scorable]=angles(rotations[scorable],row['labels']['rotations'][scorable])
                    all_results[variant].append(dict(row=row,prediction_valid=valid,predictions=rotations,errors=errors,label_valid=labels,
                        seconds=elapsed,failures=Counter(p.get('failure','') for p in predictions if not p['valid'])))
        for variant,results in all_results.items():
            valid=np.concatenate([r['prediction_valid'] for r in results]);errors=np.concatenate([r['errors'] for r in results])
            label_valid=np.concatenate([r['label_valid'] for r in results]);failures=sum((r['failures'] for r in results),Counter())
            report['variants'][variant]=dict(total_parents=len(results),total_frames=len(valid),predicted_frames=int(valid.sum()),
                coverage_fraction=float(valid.mean()),GT_scorable_frames=int((valid&label_valid).sum()),failures=dict(failures),
                frame_SO3_error_rad=describe(errors[:,0]),wand_SO3_error_rad=describe(errors[:,1]),
                frame_q95_below_original_point15_rad=bool(np.nanquantile(errors[:,0],.95)<.15),
                wand_q95_below_original_point15_rad=bool(np.nanquantile(errors[:,1],.95)<.15),
                estimation_seconds=sum(r['seconds'] for r in results),per_parent=[dict(case_id=r['row']['case_id'],
                    parent_source_group=r['row']['parent_source_group'],fold=folds[r['row']['parent_source_group']],
                    frames=len(r['prediction_valid']),predicted=int(r['prediction_valid'].sum()),failures=dict(r['failures']),
                    frame_SO3_error_rad=describe(r['errors'][:,0]),wand_SO3_error_rad=describe(r['errors'][:,1])) for r in results])
            path=output/(task+'__'+variant+'.npz')
            np.savez_compressed(path,predicted_rotations=np.concatenate([r['predictions'] for r in results]),prediction_valid=valid,
                native_label_valid=label_valid,SO3_error_rad=errors,
                parent_source_group=np.concatenate([np.repeat(r['row']['parent_source_group'],len(r['prediction_valid'])) for r in results]),
                case_id=np.concatenate([np.repeat(r['row']['case_id'],len(r['prediction_valid'])) for r in results]),
                frame_index=np.concatenate([np.arange(len(r['prediction_valid'])) for r in results]))
            prediction_files.append(lineage.pointer(ROOT,path))
        report['mask_counts']=dict(public_valid=sum(int(r['public']['valid'].sum()) for r in selected),
            public_mode_valid=sum(int(r['public']['mode_valid'].sum()) for r in selected),
            public_external=sum(int((r['public']['modes']==0).sum()) for r in selected),
            public_rotation_observable=sum(int(r['public_rotation_observable'].sum()) for r in selected),
            native_rotation_label_valid=sum(int(r['labels']['rotation_label_valid'].sum()) for r in selected),
            target_valid_includes_twist_boundary_mask=sum(int(r['labels']['target_valid'].sum()) for r in selected))
        reports[task]=report
    for path,expected_sha in files.items():assert lineage.file_sha(path)==expected_sha
    report=dict(schema='r10_TRAIN_buzz_public_centroid_pose_fixed_parent_crossvalidation_v1',started_utc=started,finished_utc=utc(),
        fixed_plan=lineage.pointer(ROOT,output/'fixed_plan.json'),script=lineage.pointer(ROOT,Path(__file__)),
        prior_runtime_separation='Native labels fit static priors only on other 50 TRAIN parents. Current estimator sees only public centroids/masks/external modes. Held labels score after all predictions per parent.',
        all_200_original_hold_parents_retained=True,original_8_actual_registrations_verified=True,
        runtime_future_or_native_driver_inputs=False,orientation_none_protocol_changed=False,
        no_outlier_filter_or_threshold_search=True,native_or_GPU_started=False,full_RGBD_pose_observability_claimed=False,
        actual_case_provenance=[{k:r[k] for k in ('case_id','task','parent_source_group','actual_plan_sha256','canonical_receipt','child_receipt','parent_receipt','cache')} for r in rows],
        tasks=reports,prediction_files=prediction_files,all_verified_input_bytes_stable_before_after=True,verified_files_sha256=files,
        limitation='Hold-only rigid/external diagnostic cannot establish post-grasp/released behavior, unseen tasks, full RGBD orientation observability, or runtime authorization. No production source or pose protocol was changed.')
    path=output/'report.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['TRAIN-only buzz public-centroid pose feasibility; fixed two-fold parent CV, no outlier removal.']
    for task,result in reports.items():
        lines.append(task+': '+str(result['parents'])+' hold parents / '+str(result['frames'])+' frames')
        for variant,metrics in result['variants'].items():lines.append('  '+variant+': coverage='+str(metrics['coverage_fraction'])+', frame q95='+str(metrics['frame_SO3_error_rad']['q95'])+', wand q95='+str(metrics['wand_SO3_error_rad']['q95']))
    lines.append('No production permissions changed; no full RGBD pose observability or GPU/native execution claim.')
    (output/'report.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=lineage.pointer(ROOT,path),summary=lines),ensure_ascii=False),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    main(parser.parse_args().out)
