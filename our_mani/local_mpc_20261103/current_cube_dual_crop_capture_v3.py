"""Independent fixed40/520 current-RGBD counterfactual data capture.

All source parents/roles/recipes precede rendering. Original V10 camera,
kinematic transaction and restore code is reused, with no model or policy.
Counterfactual images and actual Cube pose remain labels only. Each snapshot
is static independent augmentation, never a physical execution transition.
"""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
import json,os
import h5py,numpy as np
from . import depth_owner_position_v10_calibration8 as frozen
from . import current_cube_dual_crop_layout_cap12_v3 as layout

old=frozen.old
b=old.b
SCHEMA='CURRENT_CUBE_dual_crop_CF_pilot40_plan_20261103_v1'
SCOPE='current_cube_dual_crop_CF_pilot40'
SHARED_REGISTRY_SHA='76e836b6a872dce703598b18a694953d4ae48be69458bfde7bbde0c214c3cb9c'
PREPARATION_SHA='befefa20fd2c0d26635729f38138ba4ba8b459c6c31ab3083f753f01cbf5e32c'
Q_REGISTRY_SHA='c482d57a9145924471ddf4bb0b7f4887805d09e89a42d2574a85c8d22684c272'
ASSIGNMENT_SHA='c2178763ad95cfb98fbb00349702f6721c23136dafeb790a6d59fd534f210d50'
SHOT_IDS=tuple(f'shot{i:02d}'for i in range(13))
ROLE_COUNTS={'fit':16,'selection':16,'reserved_fresh_centre_VAL':8}
SOURCE_TASKS=(b.TASK,b.TASK+'_high_speed')
DLS=old.DLS
SEED_DELTAS=old.SEED_DELTAS
PoEKinematics=old.PoEKinematics
FAMILY=b.FAMILY
CAMERAS=b.CAMERAS
LABEL_METHOD=b.LABEL_METHOD
CaptureInvariantError=b.CaptureInvariantError
sha=b.sha
pointer=b.pointer
clock=b.clock
archive_capture_sources=b.archive_capture_sources
write_json=b.write_json
write_npz=b.write_npz
bootstrap_robot_diagnostics=b.bootstrap_robot_diagnostics
normalize_baseline_fingers=b.normalize_baseline_fingers
capture_checked=b.capture_checked
fit_public_sphere=b.fit_public_sphere
public_payload=b.public_payload
snapshot_pose=b.snapshot_pose
vector=b.vector
assert_close=b.assert_close
assert_pose_close=b.assert_pose_close
# Exact frozen body, including all finally/readback/clock and velocity disclosures.
zero_clock_shot=frozen.zero_clock_shot
materialize_dual_crop_shots=layout.materialize_dual_crop_shots


def stop_paths(root,output):return frozen.stop_paths(root,output)
def stop_requested(root,output):return any(p.exists()for p in stop_paths(root,output))


def checked_capture_plan(root,plan_path,parent_id=None):
    root=Path(root).resolve();path=Path(plan_path).resolve(strict=True)
    if root/'config/framework'not in path.parents:raise ValueError('Versioned config/framework CF plan required')
    plan=json.loads(path.read_text())
    tools={'collector_script':'scripts/collect_current_cube_dual_crop_20261104_v3.py',
        'owner_launcher':'scripts/launch_current_cube_dual_crop_20261104_v3.py',
        'owner_shell':'scripts/run_current_cube_dual_crop_20261104_v3.sh'}
    if any(plan.get(k)!=v for k,v in tools.items()):raise ValueError('Exact independent collector/owner/shell identities required')
    if (plan.get('schema')!=SCHEMA or plan.get('scope')!=SCOPE or plan.get('split')!='train'
        or type(plan.get('preparation_only'))is not bool or plan.get('perception_only')is not True
        or plan.get('label_method')!=LABEL_METHOD
        or any(plan.get(k)is not False for k in('Native_started','fit_started','TEST_read','benchmark_episode','execution_transition','native_instance_id_masks','material_or_alpha_edits'))):
        raise ValueError('Explicit new static TRAIN-origin dual-crop data protocol required')
    if (plan.get('DLS')!=DLS or not np.array_equal(plan.get('fixed_seed_deltas_rad'),SEED_DELTAS)
        or type(plan.get('render_warmup_rounds'))is not int or plan['render_warmup_rounds']!=12
        or plan.get('baseline_finger_normalization')!='source_reset_open_kinematic_before_snapshot'):
        raise ValueError('Original fixed9start/64iteration,12render/legal finger mechanics required')
    profile=json.loads(pointer(root,plan['layout_profile']).read_text())
    if (plan.get('hand_pad_radius_cap_m')!=.12 or profile.get('hand_pad_radius_cap_m')!=.12
        or profile.get('schema')!='current_cube_dual_crop_fixed13_handcap120_profile_20261104_v3'):
        raise ValueError('Exact predeclared all-parent120mm hand pad cap/profile required')
    cfg=json.loads(pointer(root,plan['native_config']).read_text())
    if cfg.get('dataset_use')!='train'or cfg.get('task')!=b.TASK or cfg.get('preserve_native_height')is not True:
        raise ValueError('Original normal/static native-height TRAIN asset config required')
    shared=json.loads(pointer(root,plan['source_parent_registry']).read_text())
    prep=json.loads(pointer(root,plan['metadata_preparation_manifest']).read_text())
    registry=json.loads(pointer(root,plan['TRAIN_q_registry']).read_text())
    if (plan['source_parent_registry']['sha256']!=SHARED_REGISTRY_SHA or plan['metadata_preparation_manifest']['sha256']!=PREPARATION_SHA
        or plan['TRAIN_q_registry']['sha256']!=Q_REGISTRY_SHA or shared.get('complete')is not True or registry.get('complete')is not True):
        raise ValueError('Frozen shared140/16/8/36 and preassigned pilot40/q0 identities required')
    if registry['assignment']['sha256']!=ASSIGNMENT_SHA:raise ValueError('Metadata-first40 assignment required')
    pointer(root,registry['assignment'])
    partition=json.loads(pointer(root,plan['partition']).read_text())
    parents=plan['parents'];declared=partition['parents'];qrows=registry['rows']
    expected_ids=prep['minimal_pilot_capture']['parent_ids'];jobs={j['parent_id']:j for j in prep['jobs']}
    shared_rows={j['parent_source_group']:j for j in shared['rows']}
    if (len(parents)!=40 or len(declared)!=40 or len(qrows)!=40
        or [p['parent_id']for p in parents]!=expected_ids
        or [p['parent_id']for p in declared]!=expected_ids
        or [p['parent_id']for p in qrows]!=expected_ids
        or dict(Counter(p['role']for p in parents))!=ROLE_COUNTS
        or partition.get('scope')!=SCOPE or partition.get('pilot')is not True):
        raise ValueError('Exact metadata-first40 order and16/16/8 source-role split required')
    limits=json.loads(pointer(root,plan['joint_limits']).read_text())['joint_limits']
    selected=None;groups=set()
    for p,qrow,d in zip(parents,qrows,declared):
        job=jobs[p['parent_id']];group=job['source_parent_group'];origin=shared_rows[group]
        if (p['seed']!=job['seed']or p['role']!=job['role']or p['fold']!=p['role']
            or p.get('pilot')is not True or d.get('pilot')is not True or d['seed']!=p['seed']or d['role']!=p['role']
            or tuple(d.get('shot_ids',()))!=SHOT_IDS or p['source_task']!=job['source_task']
            or p['TRAIN_q_source']!=qrow or qrow['parent_seed']!=p['seed']or qrow['parent_role']!=p['role']
            or qrow['source_parent_id']!=group or qrow['source_parent_group']!=group
            or origin['role']!=p['role']or origin['source_HDF5']!=qrow['file']or qrow['file']!=job['source_HDF5']
            or qrow['task']!=job['source_task']or qrow['task']not in SOURCE_TASKS
            or p['near_direction_ordinal']!=job['direction_ordinal']or not np.array_equal(p['near_direction_unit_world'],job['direction_unit_world'])
            or p['cube_rotation_degrees']!=job['cube_rotation_degrees']or p['cube_rotation_axis']!=job['cube_rotation_axis']):
            raise ValueError('Parent/source/role/seed/direction/rotation assignment changed')
        specs=layout.shot_specs_for_parent(p)
        if p['shots']!=specs or tuple(s['shot_id']for s in specs)!=SHOT_IDS:
            raise ValueError('Original fixed13 recipes/order required; no observed-output replacement')
        if any(s['condition']!=j['condition']for s,j in zip(specs,job['recipes'])):
            raise ValueError('Four exact predeclared dual-crop condition literals required')
        qpath=Path(qrow['file']['path'])
        if (group in groups or 'train_package_compressed'not in qpath.parts
            or qpath.parent.name!=qrow['raw_episode_id']or qrow['dataset']!='low_dim_obs/joint_positions'or qrow['index']!=0
            or qrow.get('finite')is not True or qrow.get('within_native_bounds')is not True
            or qrow.get('joint_targets_clipped')is not False):raise ValueError('Unique legal fixed TRAINq0 source group required')
        groups.add(group);q=vector(qrow['joint_positions'],7,'source TRAINq0')
        if np.any(q<limits['lower'])or np.any(q>limits['upper']):raise ValueError('TRAINq0 outside frozen Native bounds')
        if parent_id is None or p['parent_id']==parent_id:
            pointer(root,qrow['file'])
            with h5py.File(qpath,'r')as h:
                dataset=h[qrow['dataset']]
                if list(dataset.shape)!=qrow['original_dataset_shape']or str(dataset.dtype)!=qrow['original_dataset_dtype']:
                    raise ValueError('Fixed q-source dataset identity changed')
                actual=np.asarray(dataset[0,:7],float)
            if not np.array_equal(actual,q):raise ValueError('TRAINq0 source values changed')
        if p['parent_id']==parent_id:selected=p
    if parent_id and selected is None:raise ValueError('Parent is outside predeclared40')
    if sha(pointer(root,plan['kinematics']))!='9a5b2dd908957878f2b5793a8199cc6ef1912e27a5e9f20c56388d2176c4ae28':
        raise ValueError('Original frozen world Panda_tip chain required')
    cal=json.loads(pointer(root,plan['goal_calibration']).read_text())
    if plan['marker_offset_train_m']!=cal['offset_train_calibrated_m']:raise ValueError('Original TRAIN-only xyz marker offset required')
    if plan['world_outside_position_m']!=[100.,100.,-100.]:raise ValueError('Original fixed outsideCube bootstrap required')
    required={
        'our_mani/local_mpc_20261103/current_cube_dual_crop_capture_v3.py','our_mani/local_mpc_20261103/current_cube_dual_crop_layout_cap12_v3.py',
        'our_mani/local_mpc_20261103/current_cube_dual_crop_capture_v1.py','our_mani/local_mpc_20261103/current_cube_dual_crop_layout_v1.py',
        'our_mani/local_mpc_20261103/depth_owner_position_v10_calibration8.py','our_mani/local_mpc_20261103/depth_owner_capture_sensors.py',
        'scripts/collect_current_cube_dual_crop_20261104_v3.py','scripts/launch_current_cube_dual_crop_20261104_v3.py','scripts/run_current_cube_dual_crop_20261104_v3.sh'}
    if not required<=set(plan['source_sha256']):raise ValueError('Complete new collector/owner/layout/frozen mechanics source pins required')
    for name,digest in plan['source_sha256'].items():
        if sha(root/name)!=digest:raise CaptureInvariantError('Source changed '+name)
    for name,digest in plan['external_source_sha256'].items():
        if sha(name)!=digest:raise CaptureInvariantError('External source changed '+name)
    return plan,cfg,selected


checked_pilot_plan=checked_capture_plan


def check_capture_plan(root,plan_path,parent_id=None):
    plan,_,_=checked_capture_plan(root,plan_path,parent_id)
    return dict(complete=True,Native_started=False,fit_started=False,TEST_read=False,schema=SCHEMA,scope=SCOPE,
        preparation_only=plan['preparation_only'],capture_ready=plan['preparation_only']is False,
        parents=40,shots=520,shots_per_parent=13,role_counts=ROLE_COUNTS,selected_parent=parent_id,
        plan_sha256=sha(plan_path),current_public_RGBD_inputs_only=True,
        actual_Native_capture_task=b.TASK,source_task_is_TRAIN_q0_provenance_not_task_replay=True,
        numerical_world_arrays='all13 materialized after one real publicSphere/outsideCube/TRAINq0 bootstrap and before anyshot; no adaptive retries')


def annotation_metadata(root,output,parent,shot,identity,public_sphere,actual_labels):
    distance=float(np.linalg.norm(np.asarray(actual_labels['cube_pose_world'])[:3]-public_sphere['center']))
    absence=shot['absence_from_all_camera_frusta_by_construction']
    scope_pointer=None
    if shot['shot_id']=='shot01':
        if not distance>.20:raise CaptureInvariantError('Outside-scope construction did not remain outside .20m')
        name='labels/shot01.scope_negative_proof.json'
        write_json(output/name,dict(verified=True,construction='outside_scope',intended_scope_m=.20,
            distance_from_initial_public_sphere_m=distance,actual_cube_center_labels_only=np.asarray(actual_labels['cube_pose_world'])[:3],
            initial_public_sphere_center=public_sphere['center'],scene_object_exists=True,semantic_absence_claimed=False,
            source='predeclared0p25offset_readback_sameframe',physics_ticks=0,model_input_allowed=False))
        scope_pointer=dict(path=name,sha256=sha(output/name))
    return dict(scene_id=identity,shot_id=shot['shot_id'],condition=shot['condition'],condition_detail=shot['condition_detail'],
        role=parent['role'],source_parent_group=parent['TRAIN_q_source']['source_parent_group'],source_task=parent['source_task'],
        intended_scope_m=.20,construction=('outside_all_frusta'if absence else 'unverified_outside_position_requested')if shot['shot_id']in('shot00','shot02')else 'outside_scope'if scope_pointer else 'current_cube_static_pose',
        position_band=shot['position_band'],distance_from_initial_public_sphere_m=distance,
        near_direction_ordinal=parent['near_direction_ordinal'],near_direction_unit_world=parent['near_direction_unit_world'],
        cube_scene_present=True,absence_from_all_camera_frusta_by_construction=absence,
        absence_proof=dict(path='labels/outside_frusta_proof.json',sha256=sha(output/'labels/outside_frusta_proof.json'))if shot['shot_id']in('shot00','shot02')else None,
        scope_negative_proof=scope_pointer,visibility_or_centre_loss_not_assumed_from_construction=True)


def capture_parent(root,plan_path,parent_id,output):
 root=Path(root).resolve();output=Path(output).resolve();plan,cfg,parent=checked_capture_plan(root,plan_path,parent_id)
 if plan['preparation_only'] is not False:raise ValueError('Preparation-only plan cannot create Native/output')
 plan_digest=sha(plan_path)
 if root/FAMILY not in output.parents or output.exists():raise ValueError('fresh perception runtime output required')
 if stop_requested(root,output):raise InterruptedError('perception capture STOP before Native startup')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=''or not os.environ.get('DISPLAY'):raise CaptureInvariantError('CPU-only explicit DISPLAY launcher required')
 # No lock reacquisition: outer host wrapper holds both original Native locks
 # through its actual wait. Root launcher's source/plan receipt is independent.
 from scripts.run_frozen_system_environment_v820_r10 import namespace_context
 environment=namespace_context(plan['frozen_environment_sha256'])
 from ..local_mpc_20261102.worker import native_source_snapshot,make_backend
 native_before=native_source_snapshot(cfg);cfg=deepcopy(cfg);cfg['seed']=parent['seed']
 output.mkdir(parents=True,exist_ok=False)
 for name in('inputs','labels'): (output/name).mkdir()
 archive_capture_sources(root,output,plan_path,plan)
 write_json(output/'run_inputs.json',dict(plan_sha256=sha(plan_path),parent_id=parent_id,seed=parent['seed'],fold=parent['fold'],near_direction_ordinal=parent['near_direction_ordinal'],near_direction_unit_world=parent['near_direction_unit_world'],
  TRAIN_q_source=parent['TRAIN_q_source'],native_config=plan['native_config'],
  source_sha256=plan['source_sha256'],Native_sources_sha256=native_before,environment=environment,
  external_source_sha256=plan['external_source_sha256'],
  perception_only=True,calibration_only=False,position_measurement_calibration_only=False,data_cohort=SCOPE,scope=SCOPE,source_task=parent['source_task'],source_parent_group=parent['TRAIN_q_source']['source_parent_group'],declared_role=parent['role'],intended_scope_m=.20,actual_Native_capture_task=b.TASK,initialization_method='outsideCube_TRAINq0_then_real_publicSphere_no_refit',Native_task_success_read_for_policy=False,execution_transition=False,TEST_read=False,
  label_method=LABEL_METHOD,native_instance_id_masks=False,render_warmup_rounds=12,render_warmup_is_physics=False))
 backend=bundle=None;frames=[];failure=None;result={};public_sphere=None
 try:
  if stop_requested(root,output):raise InterruptedError('perception capture STOP before constructor')
  backend=make_backend(cfg,'static',budget_ticks=1)
  write_json(output/'labels/staging_receipt.json',dict(staging=backend.staging_receipt,
   excluded_from_control_budget=True,controlled_physics_ticks=0,not_a_data_transition=True))
  write_json(output/'labels/bootstrap_robot_diagnostics_raw.json',bootstrap_robot_diagnostics(backend,parent['TRAIN_q_source']['joint_positions']))
  normalization={}
  try:normalize_baseline_fingers(backend,parent['TRAIN_q_source']['joint_positions'],normalization)
  finally:write_json(output/'labels/baseline_finger_normalization.json',normalization)
  from .depth_owner_capture_sensors import create_depth_owner_cameras
  from .perception_capture_sensors import resolve_annotation_roles
  bundle=create_depth_owner_cameras(backend,warmup_rounds=12);roles,bindings,role_receipt=resolve_annotation_roles(backend)
  write_json(output/'labels/role_bindings.json',dict(bindings=bindings,receipt=role_receipt,
   current_handles_for_bookkeeping_only=True,pixel_instance_ids_produced=False,model_input_allowed=False))
  initial_clock=clock(backend)
  native_baseline=snapshot_pose(backend)
  write_json(output/'labels/native_normalized_baseline_snapshot.json',native_baseline)
  bootstrap_shot=dict(cube_mode='fixed_world_pose',fixed_cube_pose_world=plan['world_outside_position_m']+[0.,0.,0.,1.],
   joint_positions=parent['TRAIN_q_source']['joint_positions'],
   finger_positions_m=frozen.layout.finger_layout(frozen.layout.robot_asset_geometry(),'full_open_asset_max')['finger_positions_m'])
  bootstrap_receipt={}
  try:
   with zero_clock_shot(backend,bootstrap_shot,None)as bootstrap_receipt:
    write_json(output/'labels/bootstrap_robot_diagnostics.json',bootstrap_robot_diagnostics(backend,parent['TRAIN_q_source']['joint_positions']))
    # Initialization is separate evidence, never a training shot or a synthetic
    # temporal history. Public sphere is fixed before any kinematic augmentation.
    arrays,annotations,camera_receipt,initial_labels=capture_checked(bundle,roles,output,'initialization')
    if clock(backend)!=initial_clock:raise CaptureInvariantError('initialization capture advanced clocks')
    if stop_requested(root,output):raise InterruptedError('perception capture STOP after initialization')
    public_sphere,fit=fit_public_sphere(arrays,root/plan['goal_calibration']['path'])
    write_json(output/'public_initial_sphere.json',dict(sphere=public_sphere,metadata=fit,input_scene_id=parent_id+'/initialization',Native_pose_used=False))
    (output/'initialization').mkdir()
    write_npz(output/'initialization/public_rgbd.npz',public_payload(arrays,public_sphere,float(backend.clock())))
    camera_geometry={name:dict(K=arrays['camera_intrinsics'][i],T=arrays['camera_to_world'][i],near=bundle.rgb_cameras[name].get_near_clipping_plane(),far=bundle.rgb_cameras[name].get_far_clipping_plane(),width=256,height=256)for i,name in enumerate(CAMERAS)}
    hardware_limits=json.loads(pointer(root,plan['joint_limits']).read_text())['joint_limits'];actual_limits=backend.joint_limits()
    if any(not np.array_equal(actual_limits[k],hardware_limits[k])for k in('lower','upper')):raise CaptureInvariantError('actual Native robot bounds differ from frozenTRAIN hardware')
    kin=PoEKinematics(json.loads(pointer(root,plan['kinematics']).read_text()))
    materialized=materialize_dual_crop_shots(plan,parent,public_sphere,native_baseline['cube_pose'],initial_labels['cube_local_bbox'],camera_geometry,kin,actual_limits)
    write_json(output/'labels/materialized_shot_plan.json',materialized)
    materialized_sha=sha(output/'labels/materialized_shot_plan.json')
    write_json(output/'initialization/receipt.json',dict(clock=initial_clock,camera=camera_receipt,
     not_a_training_shot=True,materialized_shot_plan=dict(path='labels/materialized_shot_plan.json',sha256=materialized_sha)))
    if not materialized['complete']:raise CaptureInvariantError('materialization infeasible before anyshot: '+str(materialized['failures']))
    if tuple(s['shot_id']for s in materialized['shots'])!=SHOT_IDS:raise CaptureInvariantError('Complete13ordered materialization required before anyshot')
    write_json(output/'labels/outside_frusta_proof.json',materialized['outside_frusta_proof'])
    for shot in materialized['shots']:
     if stop_requested(root,output):raise InterruptedError('perception capture STOP before shot')
     with zero_clock_shot(backend,shot,public_sphere)as intervention:
      arrays,annotations,camera_receipt,after=capture_checked(bundle,roles,output,shot['shot_id'])
      if clock(backend)!=initial_clock:raise CaptureInvariantError('capture advanced clocks')
      if stop_requested(root,output):raise InterruptedError('perception capture STOP after sensor capture')
      semantic=annotations['semantic_role_ids']
      identity=parent_id+'/'+shot['shot_id'];timestamp=float(backend.clock())
      inputs=public_payload(arrays,public_sphere,timestamp)
      labels=dict(**annotations,scene_id=np.asarray(identity),timestamp=np.asarray(timestamp),
       counted_tick=np.asarray(backend.raw.ledger.ticks),**{k:np.asarray(v)for k,v in after.items()})
      filename=shot['shot_id']+'.npz';write_npz(output/'inputs'/filename,inputs);write_npz(output/'labels'/filename,labels)
      metadata=annotation_metadata(root,output,parent,shot,identity,public_sphere,after)
      metadata_path='labels/'+shot['shot_id']+'.annotation_metadata.json';write_json(output/metadata_path,metadata)
      frame=dict(scene_id=identity,shot_id=shot['shot_id'],timestamp=timestamp,clock=clock(backend),camera=camera_receipt,
       condition=shot['condition'],condition_detail=shot['condition_detail'],declared_role=parent['role'],source_parent_group=parent['TRAIN_q_source']['source_parent_group'],source_task=parent['source_task'],intended_scope_m=.20,position_band=shot['position_band'],distance_from_initial_public_sphere_m=metadata['distance_from_initial_public_sphere_m'],near_direction_ordinal=shot['near_direction_ordinal'],near_direction_unit_world=shot['near_direction_unit_world'],independent_scene=True,physics_ticks=0,
       annotation_metadata=dict(path=metadata_path,sha256=sha(output/metadata_path)),
       input=dict(path='inputs/'+filename,sha256=sha(output/'inputs'/filename)),labels=dict(path='labels/'+filename,sha256=sha(output/'labels'/filename)),
       cube_pixels_per_view=[int((mask==1).sum())for mask in semantic],goal_pixels_per_view=[int((mask==2).sum())for mask in semantic],
       augmentation=intervention,public_sphere_radius_m=public_sphere['radius_m'],
       label_method=LABEL_METHOD,native_instance_id_masks=False,
       accepted_for_training=False,qualification_scope='current_cube_dual_crop_CF_pilot40 independent13shots; render/snapshot contract only; no automatic strong supervision or model qualification')
     frames.append(frame);write_json(output/(shot['shot_id']+'.receipt.json'),frame)
     print(json.dumps(dict(event='perception_snapshot',scene_id=identity,controlled_ticks=backend.raw.ledger.ticks)),flush=True)
    if clock(backend)!=initial_clock:raise CaptureInvariantError('parent changed clock')
    result=dict(complete=True,shots=frames,controlled_physics_ticks=0,physical_clock=initial_clock,
     initialization_separate_not_a_shot=True,materialized_shot_plan_sha256=materialized_sha,
     same_clock_RGBD_and_labels=True,not_a_continuous_trajectory=True,Native_success_claimed=False,
     label_method=LABEL_METHOD,native_instance_id_masks=False,counterfactual_arrays_saved_only_in_labels=True,
     fixed_render_warmup_rounds=12,render_warmup_is_physics=False,
     baseline_fingers_normalized_without_physics=True,baseline_normalization_separate_from_Native_staging=True,
     semantic_rules='Cube and Goal each qualified own-depth positive interiors; raw positive conflict/negative/1pixel boundary and nonownerGoalRGB response remain unknown',
     oracle_center_evaluation_required_before_fit=False,
     current_CF_supervision_and_visibility_validation_required_before_fit=True,
     old_axis_face_oracle_not_a_rotated_centre_head_admission_gate=True,
     render_sync_only_not_5mm_or_task_qualification=True)
  finally:
   write_json(output/'labels/public_initialization_bootstrap_transaction.json',bootstrap_receipt)
  result.update(initialization_cube_was_outside_all_frusta_by_plan=True,
   outside_frusta_proof_verified=materialized['outside_frusta_proof'].get('verified')is True,
   unverified_outside_rows_remain_unknown_not_hard_negative=True,
   initialization_TRAIN_q0_set_before_true_RGBD_Sphere_fit=True,
   Sphere_refit_after_shots=False,bootstrap_legal_Native_baseline_restored=bootstrap_receipt.get('restored')is True,
   original_dynamic_finger_overshoot_restoration_claimed=False)

 except BaseException as e:
  failure=e;result=dict(complete=False,error_type=type(e).__name__,error=str(e),shots=frames)
 finally:
  cleanup=[];native_closed=False
  for obj in(bundle,backend):
   if obj is not None:
    try:
     obj.close()
     if obj is backend:native_closed=True
    except BaseException as e:cleanup.append(type(e).__name__+': '+str(e))
  source_errors=[]
  try:
   current_sources=all(sha(root/k)==v for k,v in plan['source_sha256'].items())and all(sha(k)==v for k,v in plan['external_source_sha256'].items())
   current_sources=current_sources and sha(plan_path)==plan_digest
   pointer(root,parent['TRAIN_q_source']['file'])
   native_after=native_source_snapshot(cfg)
   current_sources=current_sources and native_after==native_before
  except BaseException as e:
   current_sources=False;source_errors.append(type(e).__name__+': '+str(e));native_after=None
  result.update(parent_id=parent_id,role=parent['role'],scope=SCOPE,finished_utc=datetime.now(timezone.utc).isoformat(),source_unchanged=current_sources,
   Native_sources_sha256_after=native_after,source_errors=source_errors,
   native_closed=native_closed,cleanup_errors=cleanup,perception_only=True,execution_transition=False,TEST_read=False,fit_started=False,
   data_admitted_for_training=False,no_task_success_evaluation=True,calibration_only=False,position_measurement_calibration_only=False,excluded_from_fit_and_selection=parent['role']=='reserved_fresh_centre_VAL',declared_role=parent['role'],intended_scope_m=.20,source_task=parent['source_task'],actual_Native_capture_task=b.TASK,supervision_requires_current_CF_visibility_or_explicit_negative_proof=True)
  if cleanup or not current_sources:result['complete']=False
  write_json(output/'result.json',result)
 if failure:raise failure
 if not result['complete']:raise CaptureInvariantError('capture cleanup/source qualification failed')
 return result
