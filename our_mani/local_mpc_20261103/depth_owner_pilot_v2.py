"""Predeclared TRAIN-only pilot64, with all poses frozen before shot00.

No Native import until capture_parent. Materialization is pure NumPy with the
frozen world Panda_tip PoE chain; algebraic IK is not a collision certificate.
"""
from copy import deepcopy
from datetime import datetime,timezone
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import json,os
import h5py,numpy as np
from . import depth_owner_capture as b
from ..local_mpc_p2_20261103_v2.kinematics import PoEKinematics
from ..local_mpc_p2_20261103_v2.action_adapter import BatchPoseJointAdapter,DLSConfig

SCHEMA='P3_near_goal_depth_owner_pilot64_plan_20261103_v2'
SEEDS=(261103600,261103601,261103602,261103603,261103604,261103605,261103624,261103628)
RAW_IDS=('episode0','episode1','episode11','episode16','episode25','episode26','episode32','episode34')
DLS=dict(iterations=64,damping=.01,position_tolerance=.002,angle_tolerance=.02,rotation_weight=.25,max_joint_step=.25)
SEED_DELTAS=np.array([[0,0,0,0,0,0,0],[0,0,0,0,0,0,np.pi],[0,0,0,0,0,0,-np.pi],
 [1.2,0,0,0,0,0,0],[-1.2,0,0,0,0,0,0],[0,0,1.2,0,0,0,0],[0,0,-1.2,0,0,0,0],
 [1.2,0,0,0,0,0,-np.pi],[-1.2,0,0,0,0,0,np.pi]],float)
SHOT_SPECS=[
 dict(shot_id='shot00',condition='sphere_only',condition_detail='sphere_only_cube_out_all_frusta',cube_recipe='outside',q_recipe='TRAIN_seed'),
 dict(shot_id='shot01',condition='cube_far',condition_detail='table_far',cube_recipe='native_table',q_recipe='TRAIN_seed'),
 dict(shot_id='shot02',condition='near_goal',condition_detail='near_offset_006m',cube_recipe='near006',q_recipe='TRAIN_seed'),
 dict(shot_id='shot03',condition='near_goal',condition_detail='partial_offset_003m',cube_recipe='near003',q_recipe='TRAIN_seed'),
 dict(shot_id='shot04',condition='near_goal',condition_detail='strong_offset_000m',cube_recipe='near000',q_recipe='TRAIN_seed'),
 dict(shot_id='shot05',condition='near_goal',condition_detail='gripper_adjacent',cube_recipe='near006',q_recipe='DLS_adjacent'),
 dict(shot_id='shot06',condition='near_goal',condition_detail='gripper_occlusion',cube_recipe='near006',q_recipe='DLS_occlusion'),
 dict(shot_id='shot07',condition='sphere_only',condition_detail='sphere_only_gripper_near',cube_recipe='outside',q_recipe='DLS_occlusion'),
]


def checked_pilot_plan(root,plan_path,parent_id=None):
 root=Path(root).resolve();path=Path(plan_path).resolve(strict=True)
 if root/'config/framework'not in path.parents:raise ValueError('fixed config/framework pilot plan required')
 plan=json.loads(path.read_text())
 if (plan.get('schema')!=SCHEMA or plan.get('scope')!='pilot64' or plan.get('split')!='train' or plan.get('label_method')!=b.LABEL_METHOD
  or plan.get('perception_only')is not True or any(plan.get(k)is not False for k in ('Native_started','fit_started','TEST_read','benchmark_episode','execution_transition','native_instance_id_masks','material_or_alpha_edits'))):raise ValueError('explicit TRAIN-only depth-owner pilot64 required')
 if plan.get('DLS')!=DLS or not np.array_equal(plan.get('fixed_seed_deltas_rad'),SEED_DELTAS)or plan.get('render_warmup_rounds')!=12 or type(plan['render_warmup_rounds'])is not int:raise ValueError('fixed deterministic9start solver/render budgets required')
 if plan.get('baseline_finger_normalization')!='source_reset_open_kinematic_before_snapshot':raise ValueError('legal source-open baseline required')
 cfg=json.loads(b.pointer(root,plan['native_config']).read_text())
 if cfg.get('dataset_use')!='train' or cfg.get('task')!=b.TASK or cfg.get('preserve_native_height')is not True:raise ValueError('unchanged native-height TRAIN pick calibration required')
 partition=json.loads(b.pointer(root,plan['partition']).read_text());declared={p['seed']:p for p in partition['parents']if p['pilot']}
 registry=json.loads(b.pointer(root,plan['TRAIN_q_registry']).read_text())['rows']
 if tuple(p['seed']for p in plan['parents'])!=SEEDS or tuple(r['raw_episode_id']for r in registry)!=RAW_IDS:raise ValueError('exact preassigned8pilot parents and numericTRAIN sources required')
 selected=None
 for p,r in zip(plan['parents'],registry):
  original=declared[p['seed']]
  if (p['parent_id']!=original['parent_id']or p['role']!=original['role']or p['fold']!=p['role']or p['TRAIN_q_source']!=r
   or p['shots']!=SHOT_SPECS or p['seed']!=r['parent_seed']or p['role']!=r['parent_role']):raise ValueError('parent/source/fold/8shot assignment changed')
  qref=p['TRAIN_q_source'];qpath=Path(qref['file']['path'])
  if qref['index']!=0 or qref['dataset']!='low_dim_obs/joint_positions'or 'train_package_compressed'not in qpath.parts or qpath.parent.name!=qref['raw_episode_id']or qref['task']!=b.TASK:raise ValueError('real TRAIN rawID q0 only')
  b.vector(qref['joint_positions'],7,'TRAIN q')
  if parent_id is None or parent_id==p['parent_id']:
   b.pointer(root,qref['file'])
   with h5py.File(qpath,'r')as h:q=np.asarray(h[qref['dataset']][0],float)
   if not np.array_equal(q,qref['joint_positions']):raise ValueError('TRAIN q0 changed')
  if p['parent_id']==parent_id:selected=p
 if parent_id and selected is None:raise ValueError('parent notin pilot64')
 if len({p['TRAIN_q_source']['source_parent_id']for p in plan['parents']})!=8:raise ValueError('cross-fold source parent reuse forbidden')
 if b.sha(b.pointer(root,plan['kinematics']))!='9a5b2dd908957878f2b5793a8199cc6ef1912e27a5e9f20c56388d2176c4ae28':raise ValueError('frozen world Panda_tip chain required')
 b.pointer(root,plan['joint_limits']);cal=json.loads(b.pointer(root,plan['goal_calibration']).read_text())
 if plan['marker_offset_train_m']!=cal['offset_train_calibrated_m']:raise ValueError('exact originalTRAIN xyz marker offset required')
 if plan['world_outside_position_m']!=[100.,100.,-100.]or plan['tool_quaternion_xyzw']!=[1.,0.,0.,0.]:raise ValueError('fixed no-retry outside/tool targets required')
 required={'our_mani/local_mpc_20261103/depth_owner_pilot_v2.py','our_mani/local_mpc_20261103/depth_owner_capture.py','our_mani/local_mpc_20261103/depth_owner_capture_sensors.py','scripts/collect_near_goal_depth_owner_20261103.py'}
 if not required<=set(plan['source_sha256']):raise ValueError('complete producer source pins required')
 for name,digest in plan['source_sha256'].items():
  if b.sha(root/name)!=digest:raise b.CaptureInvariantError('source changed '+name)
 for name,digest in plan['external_source_sha256'].items():
  if b.sha(name)!=digest:raise b.CaptureInvariantError('external source changed '+name)
 return plan,cfg,selected


def check_capture_plan(root,plan_path,parent_id=None):
 p,_,_=checked_pilot_plan(root,plan_path,parent_id)
 return dict(complete=True,Native_started=False,fit_started=False,TEST_read=False,schema=SCHEMA,
  parents=8,shots=64,selected_parent=parent_id,plan_sha256=b.sha(plan_path),
  numerical_world_arrays='materialize from thatparent firstpublicSphere before anyshot; not yet Native-materialized',DLS=DLS)


def outside_frusta_proof(position,quaternion,bbox,cameras):
 from scipy.spatial.transform import Rotation
 corners=np.array(list(product(*zip(bbox[::2],bbox[1::2]))))@Rotation.from_quat(quaternion).as_matrix().T+position
 views={}
 for name in b.CAMERAS:
  c=cameras[name];K=np.asarray(c['K'],float);T=np.asarray(c['T'],float);near,far=float(c['near']),float(c['far'])
  if K.shape!=(3,3)or T.shape!=(4,4)or not np.isfinite(K).all()or not np.isfinite(T).all()or not 0<near<far:raise ValueError('actualfinite camera K/T+near/far required')
  pc=(corners-T[:3,3])@T[:3,:3];h=pc@K.T;z=pc[:,2]
  planes={'near':z-near,'far':far-z,'left':h[:,0]+.5*z,'right':(c['width']-.5)*z-h[:,0],
   'top':h[:,1]+.5*z,'bottom':(c['height']-.5)*z-h[:,1]}
  excluded=[k for k,v in planes.items()if np.all(v<0)]
  views[name]=dict(near_m=near,far_m=far,K=K.tolist(),camera_to_world=T.tolist(),camera_corners=pc.tolist(),
   halfspace_values={k:v.tolist()for k,v in planes.items()},all8corners_outside_planes=excluded,verified=bool(excluded))
 return dict(verified=all(v['verified']for v in views.values()),all_four_camera_frusta_outside=all(v['verified']for v in views.values()),
  source='predeclared_analytic_four_camera_frusta',world_corners=corners.tolist(),views=views)


def solve_fixed_starts(adapter,seed,targets,quaternions):
 starts=np.clip(seed[None]+SEED_DELTAS,adapter.lower,adapter.upper)
 result=adapter.solve(np.tile(starts,(len(targets),1)),np.repeat(targets,9,axis=0),np.repeat(quaternions,9,axis=0))
 n=len(targets);distance=np.asarray(result['ik_position_error']).reshape(n,9);angle=np.asarray(result['ik_angle_error']).reshape(n,9)
 goals=np.asarray(result['joint_goal']).reshape(n,9,7)
 finite=np.isfinite(distance)&np.isfinite(angle)&np.isfinite(goals).all(axis=-1)
 feasible=np.asarray(result['joint_goal_feasible']).reshape(n,9)&finite
 raw_scores=distance**2+(.25*angle)**2;scores=np.where(feasible,raw_scores,np.inf)
 chosen=np.full(n,-1,int);selected=np.tile(seed,(n,1));p=np.full(n,np.nan);a=np.full(n,np.nan);valid=np.any(feasible,axis=1)
 for i in range(n):
  if valid[i]:
   chosen[i]=int(np.argmin(scores[i]));selected[i]=goals[i,chosen[i]];p[i]=distance[i,chosen[i]];a[i]=angle[i,chosen[i]]
 def safe(value):
  if isinstance(value,np.ndarray):return safe(value.tolist())
  if isinstance(value,list):return [safe(x)for x in value]
  if isinstance(value,float)and not np.isfinite(value):return None
  return value
 return dict(joint_goal=selected,joint_goal_feasible=valid,ik_position_error=safe(p),ik_angle_error=safe(a),
  all_starts=dict(order='target-major, then fixed9start order',seed_deltas_rad=SEED_DELTAS.tolist(),clipped_start_q=starts.tolist(),
   raw_results={k:safe(v)for k,v in result.items()},feasible=feasible.tolist(),finite=finite.tolist(),
   scores=safe(raw_scores),selected_index=chosen.tolist(),placeholder_for_infeasible_rows=(~valid).tolist(),
   selection='evaluateall9, stableargmin feasible position_error²+(.25angle_error)², noearlystop'))


def materialize_pilot_shots(plan,parent,sphere,native_cube_pose,bbox,cameras,kinematics,joint_limits):
 S=b.vector(sphere['center'],3,'public Sphere');G=S+np.asarray(plan['marker_offset_train_m']);C=G+[.06,0,0]
 seed=b.vector(parent['TRAIN_q_source']['joint_positions'],7,'TRAIN q seed')
 lower=np.asarray(joint_limits['lower']);upper=np.asarray(joint_limits['upper'])
 adapter=BatchPoseJointAdapter(kinematics,SimpleNamespace(lower=lower,upper=upper),DLSConfig(**plan['DLS']))
 targets=np.stack((C+[0,-.075,.025],C));quats=np.tile(plan['tool_quaternion_xyzw'],(2,1))
 solved=solve_fixed_starts(adapter,seed,targets,quats);failures=[]
 if not np.all(solved['joint_goal_feasible']):failures.append('predeclared_fixed9start_DLS_infeasible_no_source_or_target_retry')
 proof=outside_frusta_proof(np.array(plan['world_outside_position_m']),[0,0,0,1],bbox,cameras)
 if not proof['verified']:failures.append('fixed_outside_pose_not_outside_all_frusta_no_retry')
 shots=[]
 for spec in parent['shots']:
  s=deepcopy(spec);s.update(finger_positions_m=[.04,.04],finger_source='source_defined_Native_reset_open',quaternion_xyzw=[0.,0.,0.,1.])
  mode=s['cube_recipe']
  if mode=='native_table':pose=np.asarray(native_cube_pose).copy();s['cube_mode']='native_initial'
  else:
   p=np.array(plan['world_outside_position_m'])if mode=='outside'else G+{'near006':[.06,0,0],'near003':[.03,0,0],'near000':[0,0,0]}[mode]
   pose=np.r_[p,[0,0,0,1]];s.update(cube_mode='public_sphere_offset',offset_world_m=(p-S).tolist())
  q=seed if s['q_recipe']=='TRAIN_seed'else solved['joint_goal'][0 if s['q_recipe']=='DLS_adjacent'else 1]
  if np.any(q<lower)or np.any(q>upper):failures.append(s['shot_id']+'_joint_bounds_invalid')
  s.update(joint_positions=q.tolist(),materialized_cube_pose_world=pose.tolist(),cube_scene_present=True,
   absence_from_all_camera_frusta_by_construction=mode=='outside')
  shots.append(s)
 return dict(complete=not failures,failures=failures,parent_id=parent['parent_id'],seed=parent['seed'],public_Sphere_center=S.tolist(),
  public_goal_reference=G.tolist(),TRAIN_marker_offset=plan['marker_offset_train_m'],shots=shots,outside_frusta_proof=proof,
  DLS=dict(config=plan['DLS'],target_positions=targets.tolist(),target_quaternions=quats.tolist(),
   results={k:v.tolist()if isinstance(v,np.ndarray)else v for k,v in solved.items()},Native_IK_called=False,physical_steps=0,collision_or_reachability_qualified=False),
  derived_before_any_shot=True,no_adaptive_retry=True,not_a_continuous_trajectory=True)

# Immutable completed-canary mechanics; no copies with changed tolerances.
FAMILY=b.FAMILY
CAMERAS=b.CAMERAS
LABEL_METHOD=b.LABEL_METHOD
CaptureInvariantError=b.CaptureInvariantError
sha=b.sha
pointer=b.pointer
clock=b.clock
stop_requested=b.stop_requested
archive_capture_sources=b.archive_capture_sources
write_json=b.write_json
write_npz=b.write_npz
bootstrap_robot_diagnostics=b.bootstrap_robot_diagnostics
normalize_baseline_fingers=b.normalize_baseline_fingers
capture_checked=b.capture_checked
fit_public_sphere=b.fit_public_sphere
public_payload=b.public_payload
zero_clock_shot=b.zero_clock_shot

def capture_parent(root,plan_path,parent_id,output):
 root=Path(root).resolve();output=Path(output).resolve();plan,cfg,parent=checked_pilot_plan(root,plan_path,parent_id)
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
 write_json(output/'run_inputs.json',dict(plan_sha256=sha(plan_path),parent_id=parent_id,seed=parent['seed'],fold=parent['fold'],
  TRAIN_q_source=parent['TRAIN_q_source'],native_config=plan['native_config'],
  source_sha256=plan['source_sha256'],Native_sources_sha256=native_before,environment=environment,
  external_source_sha256=plan['external_source_sha256'],
  perception_only=True,Native_task_success_read_for_policy=False,execution_transition=False,TEST_read=False,
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
  materialized=materialize_pilot_shots(plan,parent,public_sphere,initial_labels['cube_pose_world'],initial_labels['cube_local_bbox'],camera_geometry,kin,actual_limits)
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
    metadata=dict(scene_id=identity,shot_id=shot['shot_id'],condition=shot['condition'],condition_detail=shot['condition_detail'],cube_scene_present=True,
     absence_from_all_camera_frusta_by_construction=shot['absence_from_all_camera_frusta_by_construction'],
     absence_proof=dict(path='labels/outside_frusta_proof.json',sha256=sha(output/'labels/outside_frusta_proof.json'))if shot['absence_from_all_camera_frusta_by_construction']else None)
    metadata_path='labels/'+shot['shot_id']+'.annotation_metadata.json';write_json(output/metadata_path,metadata)
    frame=dict(scene_id=identity,shot_id=shot['shot_id'],timestamp=timestamp,clock=clock(backend),camera=camera_receipt,
     condition=shot['condition'],condition_detail=shot['condition_detail'],independent_scene=True,physics_ticks=0,
     annotation_metadata=dict(path=metadata_path,sha256=sha(output/metadata_path)),
     input=dict(path='inputs/'+filename,sha256=sha(output/'inputs'/filename)),labels=dict(path='labels/'+filename,sha256=sha(output/'labels'/filename)),
     cube_pixels_per_view=[int((mask==1).sum())for mask in semantic],goal_pixels_per_view=[int((mask==2).sum())for mask in semantic],
     augmentation=intervention,public_sphere_radius_m=public_sphere['radius_m'],
     label_method=LABEL_METHOD,native_instance_id_masks=False,
     accepted_for_training=False,qualification_scope='pilot per-parent8shots render/snapshot only; aggregate manifest and oracle required before fit')
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
  result.update(parent_id=parent_id,role=parent['role'],finished_utc=datetime.now(timezone.utc).isoformat(),source_unchanged=current_sources,
   Native_sources_sha256_after=native_after,source_errors=source_errors,
   native_closed=native_closed,cleanup_errors=cleanup,perception_only=True,execution_transition=False,TEST_read=False,fit_started=False,
   data_admitted_for_training=False,no_task_success_evaluation=True)
  if cleanup or not current_sources:result['complete']=False
  write_json(output/'result.json',result)
 if failure:raise failure
 if not result['complete']:raise CaptureInvariantError('capture cleanup/source qualification failed')
 return result
