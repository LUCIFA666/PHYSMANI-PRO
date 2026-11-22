"""Independent V10 eight-parent/ten-shot static calibration capture, no model evaluation.

The original full32 producer stays frozen. All materialization/render/restore
mechanics are reused unchanged; this version owns only the new eight identities
and their explicitly separate calibration-only source/plan contract.
"""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import json,os
import h5py,numpy as np
from . import depth_owner_full32 as old
from . import position_v9_robot_layout as layout
from contextlib import contextmanager

b=old.b
SCHEMA='P3_near_goal_depth_owner_position_calibration8_v10_plan_20261103_v1'
SCOPE='position_calibration8_v10'
SEEDS=tuple(range(261103656,261103664))
RAW_IDS=('episode399','episode405','episode416','episode427','episode430','episode431','episode439','episode442')
PRIOR_REGISTRY_SHA='57106527829fa9c23d796a0e6e5ceb265ada804fcc3dd48cdc6f460af669c317'
POSITION_REGISTRY_SHA='f93a9e0878bb34df1dbf300c74d288763359c1253c162460da980afa2c6c6f86'
PRIOR_V9_REGISTRY_SHA='d446f42570ad367c29ea2ff2330475a87c2968802c2b59a48b0b61ebea380555'
PRIOR_ADAPTIVE_REGISTRY_SHA='874321a01945b3ab29ff954184546782c0c36acb766e29033c67e5bb4df4e02f'
PRIOR_POSITION_REGISTRY_SHA='4ad28cef4363f988a5be2ffbdbe6fd4efa41091e15eeb232b64ce581228122f4'
PRIOR_RAW_CALIBRATION_REGISTRY_SHA='aa794f4a12a429cb9be364b87d6a557fe1d75499d4a89788b72d4ba28593dce8'
DLS=old.DLS
SEED_DELTAS=old.SEED_DELTAS
DIRECTION_RAW=old.DIRECTION_RAW
DIRECTIONS=old.DIRECTIONS
DIRECTION_ORDINALS=(0,1,2,3,4,5,6,7)
PoEKinematics=old.PoEKinematics
materialize_v10_shots=layout.materialize_v9_shots
outside_frusta_proof=old.outside_frusta_proof
solve_fixed_starts=old.solve_fixed_starts


def shot_specs_for_parent(ordinal):
    if type(ordinal) is not int or not 0<=ordinal<8:raise ValueError('One of the eight fixed calibration ordinals required')
    return layout.shot_specs_for_parent(DIRECTION_ORDINALS[ordinal])


def checked_pilot_plan(root,plan_path,parent_id=None):
    root=Path(root).resolve();path=Path(plan_path).resolve(strict=True)
    if root/'config/framework' not in path.parents:raise ValueError('Versioned config/framework position_calibration8 plan required')
    plan=json.loads(path.read_text())
    tool_metadata={'collector_script':'scripts/collect_near_goal_position_v10_calibration8_20261103.py',
        'owner_launcher':'scripts/launch_near_goal_position_v10_calibration8_20261103.py',
        'owner_shell':'scripts/run_near_goal_position_v10_calibration8_20261103.sh'}
    if any(plan.get(k)!=v for k,v in tool_metadata.items()):raise ValueError('Exact V10 collector/owner/shell metadata identity required')
    if type(plan.get('preparation_only')) is not bool:raise ValueError('Explicit preparation_only bool required')
    if (plan.get('schema')!=SCHEMA or plan.get('scope')!=SCOPE or plan.get('split')!='train'
        or plan.get('calibration_only') is not True or plan.get('position_measurement_calibration_only') is not True
        or plan.get('excluded_from_fit_and_selection') is not True
        or plan.get('perception_only') is not True or plan.get('label_method')!=b.LABEL_METHOD
        or any(plan.get(k) is not False for k in ('Native_started','fit_started','TEST_read','benchmark_episode',
            'execution_transition','native_instance_id_masks','material_or_alpha_edits'))):
        raise ValueError('Explicit position_calibration8 calibration-only TRAIN-origin capture required')
    if (plan.get('DLS')!=DLS or not np.array_equal(plan.get('fixed_seed_deltas_rad'),SEED_DELTAS)
        or type(plan.get('render_warmup_rounds')) is not int or plan['render_warmup_rounds']!=12
        or plan.get('baseline_finger_normalization')!='source_reset_open_kinematic_before_snapshot'):
        raise ValueError('Unchanged fixed9start/64iter,12render/legal finger baseline required')
    cfg=json.loads(b.pointer(root,plan['native_config']).read_text())
    if cfg.get('dataset_use')!='train' or cfg.get('task')!=b.TASK or cfg.get('preserve_native_height') is not True:
        raise ValueError('Original native-height TRAIN asset configuration required')
    prior=json.loads(b.pointer(root,plan['prior_full32_q_registry']).read_text())['rows']
    if plan['prior_full32_q_registry']['sha256']!=PRIOR_REGISTRY_SHA:raise ValueError('Original32 qsource registry identity required')
    prior_raw=json.loads(b.pointer(root,plan['prior_raw_calibration4_q_registry']).read_text())['rows']
    if plan['prior_raw_calibration4_q_registry']['sha256']!=PRIOR_RAW_CALIBRATION_REGISTRY_SHA:raise ValueError('Consumed raw calibration4 source identity required')
    prior_position=json.loads(b.pointer(root,plan['prior_position_calibration4_q_registry']).read_text())['rows']
    if plan['prior_position_calibration4_q_registry']['sha256']!=PRIOR_POSITION_REGISTRY_SHA:raise ValueError('Consumed position4 qsource identity required')
    prior_adaptive=json.loads(b.pointer(root,plan['prior_adaptive_position_calibration8_q_registry']).read_text())['rows']
    if plan['prior_adaptive_position_calibration8_q_registry']['sha256']!=PRIOR_ADAPTIVE_REGISTRY_SHA:raise ValueError('Consumed original adaptive8 q registry identity required')
    prior_v9=json.loads(b.pointer(root,plan['prior_position_v9_calibration8_q_registry']).read_text())['rows']
    if plan['prior_position_v9_calibration8_q_registry']['sha256']!=PRIOR_V9_REGISTRY_SHA:raise ValueError('Consumed V9 source registry identity required')
    registry=json.loads(b.pointer(root,plan['TRAIN_q_registry']).read_text())['rows']
    if plan['TRAIN_q_registry']['sha256']!=POSITION_REGISTRY_SHA:raise ValueError('Preassigned first-unused4 qsource registry required')
    partition=json.loads(b.pointer(root,plan['partition']).read_text())
    if (partition.get('schema')!='P3_near_goal_segmentation_position_calibration8_v10_partition_20261103_v1'
        or partition.get('scope')!=SCOPE or partition.get('pilot') is not False):
        raise ValueError('Separate position_calibration8 nonpilot calibration partition required')
    parents=plan['parents'];declared=partition['parents']
    if (tuple(p['seed'] for p in parents)!=SEEDS or len(declared)!=8
        or tuple(r['raw_episode_id'] for r in registry)!=RAW_IDS
        or tuple(r['parent_seed'] for r in registry)!=SEEDS
        or plan.get('direction_ordinals')!=list(DIRECTION_ORDINALS)):
        raise ValueError('Exact656..663/numeric-first-unused399/405/416/427/430/431/439/442/direction0..7 required')
    prior_groups={r['source_parent_id'] for r in prior+prior_raw+prior_position+prior_adaptive+prior_v9};groups=set();selected=None
    for i,(p,r,d) in enumerate(zip(parents,registry,declared)):
        if (p['parent_id']!=r['parent_id'] or p['parent_id']!=d['parent_id'] or p['seed']!=d['seed']
            or p['role']!='calibration' or p['fold']!='calibration' or d['role']!='calibration'
            or p.get('pilot') is not False or d.get('pilot') is not False
            or r['parent_role']!='calibration' or p['TRAIN_q_source']!=r
            or tuple(d.get('shot_ids',()))!=layout.SHOT_IDS or p['shots']!=shot_specs_for_parent(i) or p['near_direction_ordinal']!=DIRECTION_ORDINALS[i]
            or not np.array_equal(p['near_direction_unit_world'],DIRECTIONS[DIRECTION_ORDINALS[i]])):
            raise ValueError('Newparent/source/calibration role/10recipe assignment changed')
        qpath=Path(r['file']['path'])
        if (r['source_parent_id'] in prior_groups or r['source_parent_id'] in groups
            or r['dataset']!='low_dim_obs/joint_positions' or r['index']!=0
            or 'train_package_compressed' not in qpath.parts or qpath.parent.name!=r['raw_episode_id'] or r['task']!=b.TASK):
            raise ValueError('Position unique normal TRAIN q0 group required; old groups may not be reused')
        groups.add(r['source_parent_id']);b.vector(r['joint_positions'],7,'TRAIN q')
        if parent_id is None or p['parent_id']==parent_id:
            b.pointer(root,r['file'])
            with h5py.File(qpath,'r') as h:q=np.asarray(h[r['dataset']][0],float)
            if not np.array_equal(q,r['joint_positions']):raise ValueError('Position TRAINq0 array changed')
        if p['parent_id']==parent_id:selected=p
    if parent_id and selected is None:raise ValueError('Parent is not in fresh calibration4')
    if b.sha(b.pointer(root,plan['kinematics']))!='9a5b2dd908957878f2b5793a8199cc6ef1912e27a5e9f20c56388d2176c4ae28':
        raise ValueError('Original frozen world Panda_tip chain required')
    b.pointer(root,plan['joint_limits']);cal=json.loads(b.pointer(root,plan['goal_calibration']).read_text())
    if plan['marker_offset_train_m']!=cal['offset_train_calibrated_m']:raise ValueError('Exact originalTRAIN xyz marker offset required')
    if plan['world_outside_position_m']!=[100.,100.,-100.] or plan['tool_quaternion_xyzw']!=[1.,0.,0.,0.]:
        raise ValueError('Unchanged fixed outside/tool targets required')
    required={'our_mani/local_mpc_20261103/depth_owner_position_v10_calibration8.py',
        'our_mani/local_mpc_20261103/position_v9_robot_layout.py',
        'artifacts/framework/local_mpc_20261102/P3_finger_contact_geometry_v1/robot_geometry_calibration.bytecopy.json',
        'our_mani/local_mpc_20261103/depth_owner_full32.py',
        'scripts/collect_near_goal_position_v10_calibration8_20261103.py',
        'scripts/launch_near_goal_position_v10_calibration8_20261103.py',
        'scripts/run_near_goal_position_v10_calibration8_20261103.sh'}
    if not required<=set(plan['source_sha256']):raise ValueError('All new versioned collector/owner/source pins required')
    for name,digest in plan['source_sha256'].items():
        if b.sha(root/name)!=digest:raise b.CaptureInvariantError('Source changed '+name)
    for name,digest in plan['external_source_sha256'].items():
        if b.sha(name)!=digest:raise b.CaptureInvariantError('External source changed '+name)
    return plan,cfg,selected


def check_capture_plan(root,plan_path,parent_id=None):
    plan,_,_=checked_pilot_plan(root,plan_path,parent_id)
    return dict(complete=True,Native_started=False,fit_started=False,TEST_read=False,
        schema=SCHEMA,scope=SCOPE,preparation_only=plan['preparation_only'],capture_ready=plan['preparation_only'] is False,parents=8,shots=80,selected_parent=parent_id,plan_sha256=b.sha(plan_path),
        role='calibration_only_not_fit_or_selection',direction_ordinals=list(DIRECTION_ORDINALS),
        numerical_world_arrays='derive all10 from each actual firstpublicSphere after outsideCube/TRAINq0 bootstrap, before anyshot; no adaptive retry')


# Exact completed full32 mechanics, including false-setter/readback restoration,
# legal baseline normalization, clock guard, owned cameras and source archive.
FAMILY=b.FAMILY
CAMERAS=b.CAMERAS
LABEL_METHOD=b.LABEL_METHOD
CaptureInvariantError=b.CaptureInvariantError
sha=b.sha
pointer=b.pointer
clock=b.clock
def stop_paths(root,output):
 return b.stop_paths(root,output)+(Path(output).parent/'STOP',)
def stop_requested(root,output):return any(p.exists()for p in stop_paths(root,output))
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


@contextmanager
def zero_clock_shot(backend,shot,public_sphere):
 """Kinematic data augmentation; no actuate/grasp/step and no fake clock reset."""
 saved=snapshot_pose(backend);receipt={'clock_before':saved['clock'],'restored':False}
 try:
  q=vector(shot['joint_positions'],7,'TRAIN q');f=vector(shot['finger_positions_m'],2,'finger positions')
  lim=backend.joint_limits()
  if np.any(q<lim['lower'])or np.any(q>lim['upper']):raise CaptureInvariantError('requested q outside native limits')
  backend.arm.set_joint_positions(q.tolist(),disable_dynamics=False);backend.arm.set_joint_target_positions(q.tolist())
  backend.gripper.set_joint_positions(f.tolist(),disable_dynamics=False);backend.gripper.set_joint_target_positions(f.tolist())
  backend.gripper.set_joint_target_velocities([0.,0.])
  if shot['cube_mode']=='public_sphere_offset':
   if public_sphere is None:raise CaptureInvariantError('initial public sphere must be fixed before near pose')
   pose=np.r_[np.asarray(public_sphere['center'])+shot['offset_world_m'],shot['quaternion_xyzw']]
   if 'materialized_cube_pose_world'in shot:assert_pose_close(pose,shot['materialized_cube_pose_world'],'frozen near pose',1e-12)
   backend.task.target_block.set_pose(pose.tolist(),reset_dynamics=False)
  elif shot['cube_mode']=='fixed_world_pose':
   pose=vector(shot['fixed_cube_pose_world'],7,'explicit bootstrap world pose')
   backend.task.target_block.set_pose(pose.tolist(),reset_dynamics=False)
   assert_pose_close(backend.task.target_block.get_pose(),pose,'fixed bootstrap Cube pose')
  elif 'materialized_cube_pose_world'in shot:assert_pose_close(backend.task.target_block.get_pose(),shot['materialized_cube_pose_world'],'original native pose')
  if clock(backend)!=saved['clock']:raise CaptureInvariantError('kinematic setter advanced any clock')
  assert_close(backend.arm.get_joint_positions(),q,'q')
  assert_close(backend.gripper.get_joint_positions(),f,'finger q')
  receipt.update(kinematic_augmented=True,simulation_steps=0,not_execution_transition=True)
  yield receipt
 finally:
  errors=[]
  restores=((backend.arm.set_joint_positions,(saved['arm_q'].tolist(),),{'disable_dynamics':False}),
   (backend.arm.set_joint_target_positions,(saved['arm_target'].tolist(),),{}),
   (backend.gripper.set_joint_positions,(saved['finger_q'].tolist(),),{'disable_dynamics':False}),
   (backend.gripper.set_joint_target_positions,(saved['finger_target'].tolist(),),{}),
   (backend.gripper.set_joint_target_velocities,(saved['finger_target_velocity'].tolist(),),{}),
   (backend.task.target_block.set_pose,(saved['cube_pose'].tolist(),),{'reset_dynamics':False}),
   (backend.arm._ik_target.set_pose,(saved['ik_pose'].tolist(),),{'reset_dynamics':False}))
  for call,args,kwargs in restores:
   try:call(*args,**kwargs)
   except BaseException as e:errors.append(type(e).__name__+': '+str(e))
  try:
   after=snapshot_pose(backend)
   if after['clock']!=saved['clock']:raise CaptureInvariantError('shot/restoration advanced clocks')
   for key in('arm_q','arm_target','finger_q','finger_target','finger_target_velocity'):assert_close(after[key],saved[key],key)
   for key in('cube_pose','ik_pose'):assert_pose_close(after[key],saved[key],key)
   # Native exposes no general setter for measured qdot. Kinematic shots are
   # independent and the process closes; never claim dynamic-state restoration.
   receipt.update(arm_velocity_before=saved['arm_velocity'],arm_velocity_after=after['arm_velocity'],
    dynamics_restoration_claimed=False,fresh_parent_shutdown_required=True)
  except BaseException as e:errors.append(type(e).__name__+': '+str(e))
  receipt.update(clock_after=clock(backend),restored=not errors,restoration_errors=errors)
  if errors:raise CaptureInvariantError('shot restoration failed: '+'; '.join(errors))



def capture_parent(root,plan_path,parent_id,output):
 root=Path(root).resolve();output=Path(output).resolve();plan,cfg,parent=checked_pilot_plan(root,plan_path,parent_id)
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
  perception_only=True,calibration_only=True,position_measurement_calibration_only=True,calibration_cohort='position_calibration8_v10_20261103_v1',scope=SCOPE,initialization_method='outsideCube_TRAINq0_then_real_publicSphere_no_refit',Native_task_success_read_for_policy=False,execution_transition=False,TEST_read=False,
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
   finger_positions_m=layout.finger_layout(layout.robot_asset_geometry(),'full_open_asset_max')['finger_positions_m'])
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
    materialized=materialize_v10_shots(plan,parent,public_sphere,native_baseline['cube_pose'],initial_labels['cube_local_bbox'],camera_geometry,kin,actual_limits)
    write_json(output/'labels/materialized_shot_plan.json',materialized)
    materialized_sha=sha(output/'labels/materialized_shot_plan.json')
    write_json(output/'initialization/receipt.json',dict(clock=initial_clock,camera=camera_receipt,
     not_a_training_shot=True,materialized_shot_plan=dict(path='labels/materialized_shot_plan.json',sha256=materialized_sha)))
    if not materialized['complete']:raise CaptureInvariantError('materialization infeasible before anyshot: '+str(materialized['failures']))
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
      metadata=dict(scene_id=identity,shot_id=shot['shot_id'],condition=shot['condition'],condition_detail=shot['condition_detail'],position_band=shot['position_band'],distance_from_initial_public_sphere_m=shot['distance_from_initial_public_sphere_m'],near_direction_ordinal=shot['near_direction_ordinal'],near_direction_unit_world=shot['near_direction_unit_world'],cube_scene_present=True,
       absence_from_all_camera_frusta_by_construction=shot['absence_from_all_camera_frusta_by_construction'],
       absence_proof=dict(path='labels/outside_frusta_proof.json',sha256=sha(output/'labels/outside_frusta_proof.json'))if shot['absence_from_all_camera_frusta_by_construction']else None)
      metadata_path='labels/'+shot['shot_id']+'.annotation_metadata.json';write_json(output/metadata_path,metadata)
      frame=dict(scene_id=identity,shot_id=shot['shot_id'],timestamp=timestamp,clock=clock(backend),camera=camera_receipt,
       condition=shot['condition'],condition_detail=shot['condition_detail'],position_band=shot['position_band'],distance_from_initial_public_sphere_m=shot['distance_from_initial_public_sphere_m'],near_direction_ordinal=shot['near_direction_ordinal'],near_direction_unit_world=shot['near_direction_unit_world'],independent_scene=True,physics_ticks=0,
       annotation_metadata=dict(path=metadata_path,sha256=sha(output/metadata_path)),
       input=dict(path='inputs/'+filename,sha256=sha(output/'inputs'/filename)),labels=dict(path='labels/'+filename,sha256=sha(output/'labels'/filename)),
       cube_pixels_per_view=[int((mask==1).sum())for mask in semantic],goal_pixels_per_view=[int((mask==2).sum())for mask in semantic],
       augmentation=intervention,public_sphere_radius_m=public_sphere['radius_m'],
       label_method=LABEL_METHOD,native_instance_id_masks=False,
       accepted_for_training=False,qualification_scope='position_calibration8_v10 calibration-only per-parent10newshots, render/snapshot contract only; no model/criterion qualification')
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
     oracle_center_evaluation_required_before_fit=True,render_sync_only_not_5mm_or_task_qualification=True)
  finally:
   write_json(output/'labels/public_initialization_bootstrap_transaction.json',bootstrap_receipt)
  result.update(initialization_cube_was_outside_all_frusta_by_plan=True,
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
   data_admitted_for_training=False,no_task_success_evaluation=True,calibration_only=True,position_measurement_calibration_only=True,excluded_from_fit_and_selection=True)
  if cleanup or not current_sources:result['complete']=False
  write_json(output/'result.json',result)
 if failure:raise failure
 if not result['complete']:raise CaptureInvariantError('capture cleanup/source qualification failed')
 return result
