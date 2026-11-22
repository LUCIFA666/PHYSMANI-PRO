"""First fixed fit parent13 structural/label QA; no model or Native is run.

Default --check reads only metadata/source definitions. After actualclosed0,
--audit-closed-first-fit reads only this fixed fit parent. It is not a complete
40-parent manifest, visibility qualification, source-selection or fit run.
"""
from pathlib import Path
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.current_cube_rgbd_centre_20261103_v1.data import (
    PLAN_SCHEMA,SCOPE,SHOT_IDS,PARTITION_IDS,REGISTRY_SHA,pilot_metadata,sha256,read_json,checked_pointer,
    load_npz,validated_target_metadata,derive_targets)
from our_mani.current_cube_rgbd_centre_20261103_v1.training import check,pointer,write_new


def checked_first_closed(plan_path,capture_root):
    plan_path=Path(plan_path).resolve(strict=True);plan=read_json(plan_path);old=pilot_metadata()[0]
    if plan.get('schema')!=PLAN_SCHEMA or plan.get('scope')!=SCOPE or plan.get('split')!='train' or plan.get('preparation_only')is not False or tuple(p['parent_id']for p in plan.get('parents',[]))!=PARTITION_IDS:
        raise ValueError('Exact fixed40 actual execution plan required')
    parent=plan['parents'][0];pid=PARTITION_IDS[0]
    if parent['role']!='fit'or parent['seed']!=old['seed']:raise ValueError('Only fixed first fit0, never reserved')
    registry=checked_pointer(ROOT,plan['TRAIN_q_registry'])
    if sha256(registry)!=REGISTRY_SHA or read_json(registry)['rows'][0]!=parent['TRAIN_q_source']:raise ValueError('Original first q row required')
    root=Path(capture_root).resolve(strict=True);case=root/pid
    if any('test'in p.lower()for p in root.parts)or case.is_symlink():raise ValueError('Only original TRAIN capture case')
    owner_path=root/(pid+'.owner.actual_exit.json');owner=read_json(owner_path);result=read_json(case/'result.json');run=read_json(case/'run_inputs.json')
    if type(owner.get('actual_process_returncode'))is not int or owner['actual_process_returncode']!=0 or owner.get('actual_wait_completed')is not True or owner.get('source_unchanged')is not True or owner.get('stop_reason')is not None or owner.get('parent_id')!=pid or Path(owner.get('output','')).resolve()!=case:
        raise ValueError('Actual samecase rc0/wait/source terminal owner required before labels')
    if any(result.get(k)is not True for k in ('complete','native_closed','source_unchanged'))or result.get('parent_id')!=pid or type(result.get('controlled_physics_ticks'))is not int or result['controlled_physics_ticks']!=0:
        raise ValueError('Complete13 zero-tick closed source case required')
    if run.get('parent_id')!=pid or run.get('fold')!='fit'or run.get('TRAIN_q_source')!=parent['TRAIN_q_source']or run.get('plan_sha256')!=sha256(plan_path)or run.get('source_sha256')!=plan.get('source_sha256')or run.get('external_source_sha256')!=plan.get('external_source_sha256')or not run.get('Native_sources_sha256')or result.get('Native_sources_sha256_after')!=run['Native_sources_sha256']:
        raise ValueError('Exact first-fit planned source/run/Native fingerprint required')
    sources=read_json(case/'source_snapshot/manifest.json');archived={}
    for row in sources['files']:
        path=(case/'source_snapshot'/row['archive']).resolve(strict=True)
        if case/'source_snapshot'not in path.parents or row['original']in archived or sha256(path)!=row['sha256']:raise ValueError('Exact byte archive required')
        archived[row['original']]=row['sha256']
    expected={str((ROOT/k).resolve()):v for k,v in run['source_sha256'].items()};expected.update(run['external_source_sha256'])
    if not expected or any(archived.get(k)!=v or sha256(k)!=v for k,v in expected.items())or sha256(case/'source_snapshot/plan.json')!=sha256(plan_path):raise ValueError('Complete unchanged live/archive/plan graph required')
    if tuple(f.get('shot_id')for f in result['shots'])!=SHOT_IDS or tuple(s.get('shot_id')for s in parent['shots'])!=SHOT_IDS:raise ValueError('All13 original ordered shots required')
    return plan,parent,case,result,dict(owner=pointer(owner_path),result=pointer(case/'result.json'),run=pointer(case/'run_inputs.json'),source_manifest=pointer(case/'source_snapshot/manifest.json'),plan=pointer(plan_path),live_sources=len(expected),archive_sources_verified=len(archived),all_actual_source_gate_before_label_arrays=True)


def audit(plan_path,capture_root,output):
    started=time.perf_counter();plan,parent,case,result,gate=checked_first_closed(plan_path,capture_root)
    out=Path(output).resolve()
    if out.exists()or case==out or case in out.parents:raise ValueError('Fresh report output outside case required')
    import numpy as np
    from our_mani.near_goal_segmentation_20261103.data import validate_public
    from our_mani.current_cube_rgbd_centre_20261103_v1.features import public_model_arrays,make_dual_goal_crops
    from our_mani.near_goal_depth_refiner_20261103.data import binary_depth_labels
    initial=load_npz(case/'initialization/public_rgbd.npz');validate_public(initial);sphere=read_json(case/'public_initial_sphere.json')
    if sphere.get('Native_pose_used')is not False or sphere.get('metadata',{}).get('simulator_pose_used')is not False or not np.array_equal(initial['public_sphere_center'],sphere['sphere']['center']):raise ValueError('Only original public Sphere bootstrap')
    frames=[];source_pointers=[];images=[]
    for spec,frame in zip(parent['shots'],result['shots']):
        sid=spec['shot_id']
        if frame.get('scene_id')!=parent['parent_id']+'/'+sid or any(frame.get(k)!=spec.get(k)for k in ('shot_id','condition','condition_detail'))or frame.get('physics_ticks')!=0 or frame.get('timestamp')!=0 or frame.get('augmentation',{}).get('restored')is not True:
            raise ValueError('Original independent13shot identity/restore required')
        camera=frame.get('camera',{})
        if camera.get('warmup_rounds_fixed')!=12 or any(camera.get(k)is not True for k in ('original_clone_equal','all_counterfactual_flags_restored','all_restore_RGBD_equal','clock_unchanged','handling_restored','global_vision_sensors_enabled','created_nonpassive')):raise ValueError('Normal renderer restore evidence required')
        paths={k:checked_pointer(case,frame[k])for k in ('input','labels','annotation_metadata')}
        if paths['input'].parent!=case/'inputs'or any(paths[k].parent!=case/'labels'for k in ('labels','annotation_metadata')):raise ValueError('Original input/labels siblings required')
        source_pointers.extend(pointer(p)for p in paths.values())
        inputs=load_npz(paths['input']);validate_public(inputs)
        if not np.array_equal(inputs['public_sphere_center'],initial['public_sphere_center']):raise ValueError('No Sphere refit allowed')
        model_inputs=public_model_arrays(inputs) # Public inputs assembled before any target arrays.
        if set(model_inputs)!={'features','wide_features','K','camera_rotations','initial_sphere'}:raise ValueError('Exact public-only feature bundle')
        labels=load_npz(paths['labels']);metadata=read_json(paths['annotation_metadata'])
        if metadata.get('role')!='fit'or metadata.get('source_parent_group')!=parent['TRAIN_q_source']['source_parent_group']:raise ValueError('Actual first fit source group required')
        metadata,_=validated_target_metadata(case,metadata,sid)
        target=derive_targets(inputs,labels,metadata,sid)
        serial={k:(v.tolist()if isinstance(v,np.ndarray)else v)for k,v in target.items()}
        frames.append(dict(shot_id=sid,scene_id=frame['scene_id'],condition=frame['condition'],condition_detail=frame['condition_detail'],
            target=serial,model_input_shapes={k:list(v.shape)for k,v in model_inputs.items()},
            actual_cube_centre_labels_only=np.asarray(labels['cube_pose_world'])[:3].tolist(),distance_from_public_sphere_labels_only=float(np.linalg.norm(labels['cube_pose_world'][:3]-inputs['public_sphere_center']))))
        images.append((inputs,make_dual_goal_crops(inputs),binary_depth_labels(inputs,labels)==1))
    out.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(13,4,figsize=(12,35),squeeze=False)
    for i,((inputs,crops,owner),row)in enumerate(zip(images,frames)):
        for camera in range(4):
            ax=axes[i,camera];ax.imshow(inputs['rgb'][camera]);ax.contour(owner[camera],levels=[.5],colors=['red'],linewidths=.5)if owner[camera].any()else None
            for scale,size,color in (('near',64,'orange'),('wide',128,'cyan')):
                if crops[scale]['projection_available'][camera]:
                    x,y=crops[scale]['origins_xy'][camera];ax.add_patch(Rectangle((x-.5,y-.5),size,size,fill=False,edgecolor=color,linewidth=.7))
            ax.set_xlim(-.5,255.5);ax.set_ylim(255.5,-.5);ax.axis('off')
            ax.set_title(f"{row['shot_id']} {inputs['camera_names'][camera]} target={row['target']['presence_target']} sumNear/Wide={row['target']['near_owner_pixels']}/{row['target']['wide_owner_pixels']}",fontsize=7)
    fig.suptitle('First fixed FIT parent: labels-only owner red; public near orange / wide stride2 window cyan. No model or qualification.',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.99));fig.savefig(out/'RGBD_owner_dual_crop_QA.png',dpi=100);plt.close(fig)
    if any(sha256(p['path'])!=p['sha256']for p in source_pointers):raise ValueError('Source changed during QA')
    report=dict(schema='CF_pilot40_first_fixed_fit13_structural_label_QA_v1',complete=True,parent_id=parent['parent_id'],role='fit',
        full40_dataset_ready=False,first_parent_only_not_complete_manifest=True,frames=frames,
        target_counts={str(k):sum(f['target']['presence_target']==k for f in frames)for k in (-1,0,1)},source_gate=gate,
        original_input_label_pointers=source_pointers,figure=pointer(out/'RGBD_owner_dual_crop_QA.png'),QA_script=pointer(__file__),
        seconds=time.perf_counter()-started,actual_new_model_or_visual_NN_forwards=0,optimizer_steps=0,
        reserved_parents_read=0,Native_started=False,training_started=False,TEST_read=False,
        GT_arrays_used_only_for_labels_QA=True,GT_never_input_to_features_or_policy=True,model_qualified=False,source_or_sample_replacement=False)
    write_new(out/'report.json',report)
    return dict(report=pointer(out/'report.json'),target_counts=report['target_counts'],figure=report['figure'])


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture-plan');p.add_argument('--capture-root');p.add_argument('--output')
 mode=p.add_mutually_exclusive_group();mode.add_argument('--check',action='store_true');mode.add_argument('--audit-closed-first-fit',action='store_true');a=p.parse_args()
 if a.audit_closed_first_fit:
  if not all((a.capture_plan,a.capture_root,a.output)):p.error('closed-first-fit requires plan/root/fresh output')
  value=audit(a.capture_plan,a.capture_root,a.output)
 else:
  value=check();value.update(first_parent_id=PARTITION_IDS[0],first_role='fit',first_shots=list(SHOT_IDS),case_read=False)
 print(json.dumps(value,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
