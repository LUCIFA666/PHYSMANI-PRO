"""Current RGBD annotation snapshots, not a task/transition data collector.

Native is imported only in capture_parent after strict source/CPU/STOP checks.
All poses/instance IDs stay in labels. Inputs are raw RGBD/K/T and a sphere fit
from the first unchanged public image. Shots never imply elapsed50ms motion.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,re,shutil
import h5py
import numpy as np

SCHEMA='P3_near_goal_perception_capture_plan_20261103_v1'
FAMILY=Path('runtime/framework/near_goal_perception_20261103')
CAMERAS=('front','8','16','36')
TASK='pick_moving_target_on_the_table'
PUBLIC_ARRAY_KEYS={'rgb','depth_m','camera_intrinsics','camera_to_world','camera_names'}
INPUT_KEYS={'rgb','depth_m','K','T','public_sphere_center','camera_names','timestamp'}
REQUIRED_SOURCES={
 'our_mani/local_mpc_20261103/__init__.py',
 'our_mani/local_mpc_20261103/perception_capture.py',
 'our_mani/local_mpc_20261103/perception_capture_sensors.py',
 'scripts/collect_near_goal_perception_20261103.py',
 'scripts/run_near_goal_perception_capture_20261103.sh',
 'scripts/launch_near_goal_perception_capture_20261103.py',
 'scripts/run_frozen_system_environment_v820_r10.py',
 'our_mani/local_mpc_20261102/worker.py','our_mani/local_mpc_20261102/native.py',
 'our_mani/pyrep_env.py','our_mani/goal_perception.py','our_mani/perception.py',
 'our_mani/local_mpc_control_20261103/goal_aware_perception.py',
 'artifacts/framework/local_policy/goal_marker_calibration.json',
}
REQUIRED_EXTERNAL=(
 'third_party/PyRep/pyrep/objects/vision_sensor.py',
 'third_party/PyRep/pyrep/objects/object.py','third_party/PyRep/pyrep/objects/shape.py',
 'third_party/PyRep/pyrep/robots/robot_component.py','third_party/PyRep/pyrep/robots/end_effectors/gripper.py',
 'third_party/PyRep/pyrep/backend/sim.py','third_party/PyRep/pyrep/backend/simConst.py',
)

class CaptureInvariantError(RuntimeError):pass

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:plain(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v)for v in x]
 return x

def write_json(path,value):
 with Path(path).open('x')as f:json.dump(plain(value),f,indent=2,allow_nan=False)
def write_npz(path,value):
 with Path(path).open('xb')as f:np.savez_compressed(f,**value)
 with np.load(path,allow_pickle=False)as back:
  if set(back.files)!=set(value)or any(not np.array_equal(back[k],v)for k,v in value.items()):raise CaptureInvariantError('NPZ roundtrip changed capture')

def archive_capture_sources(root,output,plan_path,plan):
 """Byte-copy evidence; the launched Native source/assets are never rewritten."""
 base=output/'source_snapshot';base.mkdir();rows=[]
 refs=[(Path(root)/name,'repo/'+name,digest)for name,digest in plan['source_sha256'].items()]
 refs.extend((Path(name),'external/'+name.lstrip('/'),digest)for name,digest in plan['external_source_sha256'].items())
 refs.append((Path(plan_path),'plan.json',sha(plan_path)))
 for original,relative,digest in refs:
  if sha(original)!=digest:raise CaptureInvariantError('source changed before archive: '+str(original))
  target=base/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
  if sha(target)!=digest:raise CaptureInvariantError('source byte-copy mismatch: '+str(original))
  rows.append(dict(original=str(original),archive=relative,sha256=digest))
 write_json(base/'manifest.json',dict(files=rows,Native_sources_modified=False))
def vector(value,size,name):
 a=np.asarray(value,float)
 if a.shape!=(size,)or not np.isfinite(a).all():raise ValueError('finite '+name+f'[{size}] required')
 return a

def pointer(root,ref):
 if not isinstance(ref,dict)or set(ref)!={'path','sha256'}:raise ValueError('exact path/SHA pointer required')
 p=Path(ref['path']);p=p if p.is_absolute()else Path(root)/p;p=p.resolve(strict=True)
 if sha(p)!=ref['sha256']:raise CaptureInvariantError('bound source changed: '+str(p))
 return p

def stop_paths(root,output):
 output=Path(output)
 return(output/'STOP',output.parent/(output.name+'.STOP'),Path(root)/FAMILY/'STOP')
def stop_requested(root,output):return any(p.exists()for p in stop_paths(root,output))

def checked_plan(root,plan_path,parent_id=None):
 root=Path(root).resolve();plan_path=Path(plan_path).resolve(strict=True)
 if root/'config/framework'not in plan_path.parents:raise ValueError('new config/framework capture plan required')
 plan=json.loads(plan_path.read_text())
 if (plan.get('schema')!=SCHEMA or plan.get('split')!='train' or plan.get('perception_only')is not True
     or any(plan.get(k)is not False for k in('benchmark_episode','execution_transition','task_success_supervision','Native_started','fit_started','TEST_read'))):
  raise ValueError('explicit independent TRAIN perception-only plan required')
 cfgpath=pointer(root,plan['native_config']);cfg=json.loads(cfgpath.read_text())
 if (root/'config/framework'not in cfgpath.parents or cfg.get('dataset_use')!='train' or cfg.get('task')!=TASK
     or cfg.get('preserve_native_height')is not True):raise ValueError('original native-height TRAIN pick calibration config only')
 qpath=pointer(root,plan['TRAIN_q_source']['file'])
 if ('train_package_compressed'not in qpath.parts or qpath.name!='data.h5'
     or qpath!=Path(cfg['calibration_train_episode']).resolve() or plan['TRAIN_q_source'].get('dataset')!='low_dim_obs/joint_positions'
     or type(plan['TRAIN_q_source'].get('index'))is not int or plan['TRAIN_q_source']['index']!=0):
  raise ValueError('smoke uses exact original TRAIN calibration q0, no expert/GT source')
 with h5py.File(qpath,'r')as h:
  train_q=np.asarray(h['low_dim_obs/joint_positions'][0],float)
 if not np.array_equal(vector(plan['TRAIN_q_source']['joint_positions'],7,'TRAIN q'),train_q):raise ValueError('q differs from real TRAIN q0')
 goalpath=pointer(root,plan['goal_calibration'])
 if goalpath!=root/'artifacts/framework/local_policy/goal_marker_calibration.json':raise ValueError('fixed original public-goal TRAIN calibration required')
 parents=plan.get('parents',[])
 if not parents or len({p['parent_id']for p in parents})!=len(parents):raise ValueError('distinct preassigned parents required')
 for p in parents:
  if not isinstance(p.get('parent_id'),str)or not re.fullmatch(r'[A-Za-z0-9_-]+',p['parent_id']):raise ValueError('safe nonempty parent ID required')
  if (type(p.get('seed'))is not int or not 261103600<=p['seed']<=261103631 or p.get('fold')not in('fit','selection','calibration')):raise ValueError('preassigned TRAIN parent seed/fold required')
  shots=p.get('shots',[])
  if len(shots)!=2 or [s.get('cube_mode')for s in shots]!=['native_initial','public_sphere_offset']:raise ValueError('minimal fixed two-shot canary only')
  if len({s['shot_id']for s in shots})!=2:raise ValueError('independent unique shot IDs required')
  for shot in shots:
   if not isinstance(shot.get('shot_id'),str)or not re.fullmatch(r'[A-Za-z0-9_-]+',shot['shot_id']):raise ValueError('flat nonempty independent shot ID required')
   if not np.array_equal(vector(shot['joint_positions'],7,'shot q'),train_q):raise ValueError('smoke q must be exact TRAIN q0')
   if shot.get('finger_source')!='source_defined_Native_reset_open' or not np.array_equal(vector(shot['finger_positions_m'],2,'finger positions'),[.04,.04]):raise ValueError('explicit native reset-open, not fictional TRAIN finger pair')
   if shot['cube_mode']=='public_sphere_offset':
    if not np.array_equal(vector(shot['offset_world_m'],3,'public sphere offset'),[.06,0.,0.])or not np.array_equal(vector(shot['quaternion_xyzw'],4,'cube quaternion'),[0.,0.,0.,1.]):raise ValueError('fixed versioned near-sphere smoke pose required')
 sources=plan.get('source_sha256')
 if not isinstance(sources,dict)or not REQUIRED_SOURCES<=set(sources):raise ValueError('complete capture/owner/native/public source pins required')
 for name,digest in sources.items():
  path=(root/name).resolve()
  if root not in path.parents or sha(path)!=digest:raise CaptureInvariantError('capture source changed: '+name)
 if str(cfgpath.relative_to(root))not in sources:raise ValueError('native calibration configuration source pin required')
 for name,digest in plan.get('external_source_sha256',{}).items():
  if not Path(name).is_absolute()or sha(name)!=digest:raise CaptureInvariantError('external capture source changed: '+name)
 if not {str((Path(cfg['native_repo'])/p).resolve())for p in REQUIRED_EXTERNAL}<=set(plan.get('external_source_sha256',{})):raise ValueError('installed sensor/setter Native source pins required')
 found=next((p for p in parents if p['parent_id']==parent_id),None)if parent_id else None
 if parent_id and found is None:raise ValueError('parent not in fixed TRAIN plan')
 return plan,cfg,found

def check_capture_plan(root,plan_path,parent_id=None):
 plan,cfg,parent=checked_plan(root,plan_path,parent_id)
 return dict(complete=True,Native_started=False,fit_started=False,TEST_read=False,
  perception_only=True,benchmark_episode=False,execution_transition=False,
  parents=len(plan['parents']),selected_parent=parent_id,shots_per_parent=2,
  plan_sha256=sha(plan_path),q_source_scope='TRAIN q0 only; fingers are source-defined Native reset-open, not HDF independent finger labels')

def clock(backend):
 value=tuple(backend._clock_tuple());raw_control=backend.clock();control=float(raw_control)
 if len(value)!=4 or any(isinstance(v,(bool,np.bool_))for v in(*value,raw_control))or not np.isfinite(np.asarray(value,float)).all()or not np.isfinite(control):raise CaptureInvariantError('finite physical/task/step/ledger clocks required')
 return value,control

def snapshot_pose(backend):
 return dict(clock=clock(backend),arm_q=np.asarray(backend.arm.get_joint_positions(),float),
  arm_target=np.asarray(backend.arm.get_joint_target_positions(),float),arm_velocity=np.asarray(backend.arm.get_joint_velocities(),float),
  finger_q=np.asarray(backend.gripper.get_joint_positions(),float),finger_target=np.asarray(backend.gripper.get_joint_target_positions(),float),
  finger_target_velocity=np.asarray(backend.gripper.get_joint_target_velocities(),float),
  cube_pose=np.asarray(backend.task.target_block.get_pose(),float),ik_pose=np.asarray(backend.arm._ik_target.get_pose(),float))

def assert_close(a,b,name,tolerance=1e-6):
 if np.asarray(a).shape!=np.asarray(b).shape or not np.isfinite(a).all()or not np.isfinite(b).all()or np.max(np.abs(np.asarray(a)-np.asarray(b)))>tolerance:raise CaptureInvariantError(name+' readback/restoration mismatch')

def assert_pose_close(a,b,name,tolerance=1e-6):
 a=vector(a,7,name);b=vector(b,7,name)
 assert_close(a[:3],b[:3],name+' position',tolerance)
 if min(np.max(np.abs(a[3:]-b[3:])),np.max(np.abs(a[3:]+b[3:])))>tolerance:raise CaptureInvariantError(name+' quaternion restoration mismatch')

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

def fit_public_sphere(arrays,calibration_file):
 from ..goal_perception import RedGoalMarker
 from ..perception import CameraCalibration
 from ..contracts import Pose
 from ..local_mpc_control_20261103.goal_aware_perception import sphere_from_public_goal_fit
 meta=json.loads(Path(calibration_file).read_text());marker=RedGoalMarker(meta['offset_train_calibrated_m'])
 cal={name:CameraCalibration(arrays['camera_intrinsics'][i],Pose(arrays['camera_to_world'][i,:3,3],arrays['camera_to_world'][i,:3,:3]))for i,name in enumerate(CAMERAS)}
 observation={'timestamp':0.,'observation/images':dict(zip(CAMERAS,arrays['rgb'])),'observation/depths':dict(zip(CAMERAS,arrays['depth_m'])),'observation/calibrations':cal}
 pose,metadata=marker.observe(observation);sphere=sphere_from_public_goal_fit(pose,metadata,0.)
 return asdict(sphere),metadata

def public_payload(arrays,public_sphere,timestamp):
 if set(arrays)!=PUBLIC_ARRAY_KEYS:raise CaptureInvariantError('undeclared/privileged public array channel')
 result={key:np.asarray(arrays[key]).copy()for key in('rgb','depth_m','camera_names')}
 result.update(K=np.asarray(arrays['camera_intrinsics']).copy(),T=np.asarray(arrays['camera_to_world']).copy(),
  public_sphere_center=vector(public_sphere['center'],3,'public sphere centre'),timestamp=np.asarray(timestamp))
 if set(result)!=INPUT_KEYS:raise CaptureInvariantError('public NPZ contract changed')
 return result

def preserve_capture_diagnostic(output,shot_id,arrays,instances,poses,camera_receipt):
 """Keep rejected renderer evidence too. Privileged masks/poses stay labels-only."""
 write_json(output/(shot_id+'.camera_diagnostic.json'),dict(camera=camera_receipt,
  accepted_for_training=False,requires_complete_parent_and_freshness_qualification=True))
 if arrays is not None:
  (output/'diagnostics_inputs').mkdir(exist_ok=True)
  # No annotation-derived crop seed; pre-fit diagnostics contain only raw sensors.
  if set(arrays)!=PUBLIC_ARRAY_KEYS:raise CaptureInvariantError('unexpected diagnostics public channels')
  write_npz(output/'diagnostics_inputs'/(shot_id+'.npz'),arrays)
 if instances is not None:
  write_npz(output/'labels'/('diagnostics_'+shot_id+'.npz'),dict(instances=instances,**poses))

def materialize_shots(parent,public_sphere,native_pose):
 """Freeze all deterministic interventions before seeing any augmented image."""
 shots=deepcopy(parent['shots']);initial=vector(native_pose,7,'initial Native Cube label pose')
 for shot in shots:
  pose=initial if shot['cube_mode']=='native_initial'else np.r_[vector(public_sphere['center'],3,'public sphere')+shot['offset_world_m'],shot['quaternion_xyzw']]
  shot['materialized_cube_pose_world']=pose.tolist()
 return dict(parent_id=parent['parent_id'],seed=parent['seed'],shots=shots,
  derived_before_any_kinematic_shot=True,derive_inputs='initial public RGBD sphere plus exact preassigned offsets; native_initial pose is annotation only',
  adaptive_pose_or_retry=False,perception_only=True,not_continuous_trajectory=True)

def capture_checked(bundle,roles,output,shot_id):
 before=annotation_labels(roles)
 try:arrays,instances,camera_receipt=bundle.capture()
 except BaseException:
  write_json(output/(shot_id+'.camera_diagnostic.json'),dict(camera=getattr(bundle,'last_receipt',None),
   accepted_for_training=False,capture_returned=False))
  raise
 after=annotation_labels(roles)
 for key in before:
  (assert_pose_close if key.endswith('_pose_world')else assert_close)(before[key],after[key],key)
 annotations=getattr(bundle,'last_annotation_arrays',None)
 if not isinstance(annotations,dict)or set(annotations)!={'mask_depth_m'}:raise CaptureInvariantError('same-render label-only mask depth evidence required')
 mask_depth=annotations['mask_depth_m']
 if not isinstance(mask_depth,np.ndarray)or mask_depth.shape!=(4,256,256)or mask_depth.dtype!=np.float32 or not np.isfinite(mask_depth).all()or np.any(mask_depth<0):raise CaptureInvariantError('finite metric label-only mask depth required')
 after['mask_depth_m']=mask_depth.copy()
 preserve_capture_diagnostic(output,shot_id,arrays,instances,after,camera_receipt)
 if not all(camera_receipt.get(k)is True for k in('original_clone_equal','global_vision_sensors_enabled','created_nonpassive','handling_restored','mask_rgb_projection_equal')):raise CaptureInvariantError('RGBD clone/current-render qualification failed')
 if camera_receipt.get('mask_rgb_depth_equal')is not True:raise CaptureInvariantError('instance-label/RGBD occlusion-depth qualification failed')
 if not isinstance(instances,np.ndarray)or instances.shape!=(4,256,256)or instances.dtype!=np.uint32:raise CaptureInvariantError('aligned integer instance planes required')
 return arrays,instances,camera_receipt,after

def annotation_labels(roles):
 # The module is safe to import on CPU; its Native getter resolution stays lazy.
 from .perception_capture_sensors import annotation_pose_labels
 return annotation_pose_labels(roles)

def bootstrap_robot_diagnostics(backend,source_q):
 """Settled initialization is measured, not assumed to equal an unapplied shot.

 Native home staging has real settling steps. Record that result separately;
 strict requested-q readback is applied only after each zero-clock setter.
 """
 before=clock(backend)
 q=vector(backend.arm.get_joint_positions(),7,'bootstrap measured q')
 dq=vector(backend.arm.get_joint_velocities(),7,'bootstrap measured dq')
 target=vector(backend.arm.get_joint_target_positions(),7,'bootstrap servo target')
 fingers=vector(backend.gripper.get_joint_positions(),2,'bootstrap measured fingers')
 finger_target=vector(backend.gripper.get_joint_target_positions(),2,'bootstrap finger targets')
 finger_velocity=vector(backend.gripper.get_joint_target_velocities(),2,'bootstrap finger target velocities')
 source=vector(source_q,7,'bound TRAIN q0')
 if clock(backend)!=before:raise CaptureInvariantError('bootstrap robot getters advanced clocks')
 return dict(clock=before,measured_joint_positions=q,measured_joint_velocities=dq,joint_target_positions=target,
  measured_finger_positions_m=fingers,finger_target_positions_m=finger_target,finger_target_velocities_m_s=finger_velocity,
  source_TRAIN_q0=source,source_defined_reset_open_m=[.04,.04],
  measured_q_minus_TRAIN_q0_rad=q-source,measured_fingers_minus_reset_open_m=fingers-[.04,.04],
  max_abs_q_minus_TRAIN_q0_rad=float(np.max(np.abs(q-source))),
  max_abs_finger_minus_reset_open_m=float(np.max(np.abs(fingers-[.04,.04]))),
  equality_to_unapplied_shot_required=False,source_q_or_pose_changed=False,labels_only=True,
  bootstrap_is_not_a_training_shot=True,not_a_transition=True)

def capture_parent(root,plan_path,parent_id,output):
 root=Path(root).resolve();output=Path(output).resolve();plan,cfg,parent=checked_plan(root,plan_path,parent_id)
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
  TRAIN_q_source=plan['TRAIN_q_source'],native_config=plan['native_config'],
  source_sha256=plan['source_sha256'],Native_sources_sha256=native_before,environment=environment,
  external_source_sha256=plan['external_source_sha256'],
  perception_only=True,Native_task_success_read_for_policy=False,execution_transition=False,TEST_read=False))
 backend=bundle=None;frames=[];failure=None;result={};public_sphere=None
 try:
  if stop_requested(root,output):raise InterruptedError('perception capture STOP before constructor')
  backend=make_backend(cfg,'static',budget_ticks=1)
  write_json(output/'labels/staging_receipt.json',dict(staging=backend.staging_receipt,
   excluded_from_control_budget=True,controlled_physics_ticks=0,not_a_data_transition=True))
  from .perception_capture_sensors import create_capture_cameras,resolve_annotation_roles
  bundle=create_capture_cameras(backend);roles,bindings,role_receipt=resolve_annotation_roles(backend)
  write_json(output/'labels/instance_bindings.json',dict(bindings=bindings,receipt=role_receipt,model_input_allowed=False))
  initial_clock=clock(backend)
  write_json(output/'labels/bootstrap_robot_diagnostics.json',bootstrap_robot_diagnostics(backend,plan['TRAIN_q_source']['joint_positions']))
  # Initialization is separate evidence, never a training shot or a synthetic
  # temporal history. Public sphere is fixed before any kinematic augmentation.
  arrays,instances,camera_receipt,initial_labels=capture_checked(bundle,roles,output,'initialization')
  if clock(backend)!=initial_clock:raise CaptureInvariantError('initialization capture advanced clocks')
  if stop_requested(root,output):raise InterruptedError('perception capture STOP after initialization')
  public_sphere,fit=fit_public_sphere(arrays,root/plan['goal_calibration']['path'])
  write_json(output/'public_initial_sphere.json',dict(sphere=public_sphere,metadata=fit,input_scene_id=parent_id+'/initialization',Native_pose_used=False))
  (output/'initialization').mkdir()
  write_npz(output/'initialization/public_rgbd.npz',public_payload(arrays,public_sphere,float(backend.clock())))
  materialized=materialize_shots(parent,public_sphere,initial_labels['cube_pose_world'])
  write_json(output/'labels/materialized_shot_plan.json',materialized)
  materialized_sha=sha(output/'labels/materialized_shot_plan.json')
  write_json(output/'initialization/receipt.json',dict(clock=initial_clock,camera=camera_receipt,
   not_a_training_shot=True,materialized_shot_plan=dict(path='labels/materialized_shot_plan.json',sha256=materialized_sha)))
  for shot in materialized['shots']:
   if stop_requested(root,output):raise InterruptedError('perception capture STOP before shot')
   with zero_clock_shot(backend,shot,public_sphere)as intervention:
    arrays,instances,camera_receipt,after=capture_checked(bundle,roles,output,shot['shot_id'])
    if clock(backend)!=initial_clock:raise CaptureInvariantError('capture advanced clocks')
    if stop_requested(root,output):raise InterruptedError('perception capture STOP after sensor capture')
    semantic=np.zeros_like(instances,dtype=np.uint8)
    for key,label in(('cube',1),('goal',2)):
     semantic[np.isin(instances,bindings[key])]=label
    identity=parent_id+'/'+shot['shot_id'];timestamp=float(backend.clock())
    inputs=public_payload(arrays,public_sphere,timestamp)
    labels=dict(instances=instances,semantic_role_ids=semantic,scene_id=np.asarray(identity),timestamp=np.asarray(timestamp),
     counted_tick=np.asarray(backend.raw.ledger.ticks),**{k:np.asarray(v)for k,v in after.items()})
    filename=shot['shot_id']+'.npz';write_npz(output/'inputs'/filename,inputs);write_npz(output/'labels'/filename,labels)
    frame=dict(scene_id=identity,shot_id=shot['shot_id'],timestamp=timestamp,clock=clock(backend),camera=camera_receipt,
     input=dict(path='inputs/'+filename,sha256=sha(output/'inputs'/filename)),labels=dict(path='labels/'+filename,sha256=sha(output/'labels'/filename)),
     cube_pixels_per_view=[int((mask==1).sum())for mask in semantic],goal_pixels_per_view=[int((mask==2).sum())for mask in semantic],
     augmentation=intervention,public_sphere_radius_m=public_sphere['radius_m'],
     accepted_for_training=False,qualification_scope='two-shot renderer/labels smoke only; no training manifest')
   frames.append(frame);write_json(output/(shot['shot_id']+'.receipt.json'),frame)
   print(json.dumps(dict(event='perception_snapshot',scene_id=identity,controlled_ticks=backend.raw.ledger.ticks)),flush=True)
  if clock(backend)!=initial_clock:raise CaptureInvariantError('parent changed clock')
  result=dict(complete=True,shots=frames,controlled_physics_ticks=0,physical_clock=initial_clock,
   initialization_separate_not_a_shot=True,materialized_shot_plan_sha256=materialized_sha,
   same_clock_RGBD_and_labels=True,not_a_continuous_trajectory=True,Native_success_claimed=False,
   RGBD_instance_occlusion_depth_exact=True,mask_depth_evidence_saved_only_in_labels=True,
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
   pointer(root,plan['TRAIN_q_source']['file'])
   native_after=native_source_snapshot(cfg)
   current_sources=current_sources and native_after==native_before
  except BaseException as e:
   current_sources=False;source_errors.append(type(e).__name__+': '+str(e));native_after=None
  result.update(parent_id=parent_id,finished_utc=datetime.now(timezone.utc).isoformat(),source_unchanged=current_sources,
   Native_sources_sha256_after=native_after,source_errors=source_errors,
   native_closed=native_closed,cleanup_errors=cleanup,perception_only=True,execution_transition=False,TEST_read=False,fit_started=False,
   data_admitted_for_training=False,no_task_success_evaluation=True)
  if cleanup or not current_sources:result['complete']=False
  write_json(output/'result.json',result)
 if failure:raise failure
 if not result['complete']:raise CaptureInvariantError('capture cleanup/source qualification failed')
 return result
