"""Standalone, explicit-execute replay of frozen PUBLIC commands, then query C.

No policy/model is loaded. No GT object/held/state restore is a replay input.
The original backend/gripper/executor and original endpoint+7path helper remain
unchanged. All replay matching gates precede the candidate collision query.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import hashlib,json,os,re
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
REQUEST_SHA='36910b8c7033bd5938f5071bc5b8c2032c8936e7f394ead2f3251fd74c59918f'
ANCHORS=('00_val_static_t7.9','01_val_static_t4.1','04_val_uniform_t2.6')
FAMILY=ROOT/'runtime/framework/local_mpc_20261102'
SCHEMA='public_prefix_C_static_collision_replay_source_freeze_v1'
OWN=('our_mani/local_mpc_20261103/public_prefix_static_collision_replay_20261103.py',
 'scripts/replay_public_prefix_C_static_collision_20261103.py','scripts/launch_public_prefix_C_static_collision_20261103.py','scripts/run_public_prefix_C_static_collision_20261103.sh')
ROBOT_KEYS=('timestamp','tcp_position','tcp_quaternion','q','dq','finger_open_amounts')
# Tight, predeclared replay checks; never retry/relax them based on a result.
TOL=dict(clock=2e-5,q=1e-6,dq=1e-6,tcp_position=1e-6,tcp_angle_rad=5e-6,finger_open_amounts=1e-6,reference=1e-12)


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def lines(path):return[json.loads(s)for s in Path(path).read_text().splitlines()]
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return{k:plain(v)for k,v in x.items()}
 if isinstance(x,(tuple,list)):return[plain(v)for v in x]
 return x
def write(path,value):
 with Path(path).open('x')as f:json.dump(plain(value),f,indent=2,allow_nan=False);f.write('\n')
def pointer(ref):
 p=Path(ref['path']).resolve(strict=True)
 if sha(p)!=ref['sha256']:raise ValueError('Bound source changed: '+str(p))
 return p
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def near(actual,expected,tolerance,label):
 a,b=np.asarray(actual,float),np.asarray(expected,float)
 if a.shape!=b.shape or not np.isfinite(a).all()or not np.isfinite(b).all():raise ValueError('Invalid '+label)
 error=float(np.max(np.abs(a-b)))if a.size else 0.
 if error>tolerance:raise ValueError('Replay mismatch '+label+': '+repr(error)+' > '+repr(tolerance))
 return error

def robot(backend):
 before=backend.clock();pose=backend.arm.get_tip().get_pose()
 result=dict(timestamp=float(before),tcp_position=list(pose[:3]),tcp_quaternion=list(pose[3:]),
  q=list(backend.get_joint_positions()),dq=list(backend.get_joint_velocities()),finger_open_amounts=list(backend.gripper.get_open_amount()))
 if backend.clock()!=before:raise ValueError('Robot-only sensor read stepped physics')
 return result

def match_robot(actual,expected):
 if set(actual)!=set(ROBOT_KEYS)or set(expected)!=set(ROBOT_KEYS):raise ValueError('Only original public robot fields')
 errors={k:near(actual[k],expected[k],TOL['clock']if k=='timestamp'else TOL[k],k)for k in ROBOT_KEYS if k!='tcp_quaternion'}
 a=np.asarray(actual['tcp_quaternion'],float);b=np.asarray(expected['tcp_quaternion'],float)
 angle=float(2*np.arccos(np.clip(abs(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)),0,1)))
 if angle>TOL['tcp_angle_rad']:raise ValueError('Replay mismatch TCP rotation: '+repr(angle))
 errors['tcp_angle_rad']=angle;return errors

def match_h(actual,expected):
 if set(actual)!=set(expected):raise ValueError('Actuator state keys changed')
 errors={}
 for key in actual:
  a,b=actual[key],expected[key]
  if b is None or type(b)is bool:
   if a is not b:raise ValueError('Replay actuator state differs: '+key)
  else:errors[key]=near(a,b,TOL['clock']if key=='reference_clock'else TOL['reference'],'h.'+key)
 return errors

def stop_paths(output):return(output/'STOP',output.with_name(output.name+'.STOP'),output.parent/'STOP',FAMILY/'STOP')
def stop_requested(output):return any(p.exists()for p in stop_paths(output))


def load_request(request_path):
 path=Path(request_path).resolve(strict=True)
 if sha(path)!=REQUEST_SHA:raise ValueError('Only the fixed sealed3anchor request')
 request=read(path)
 if tuple(request.get('ordered_anchor_ids',()))!=ANCHORS or tuple(a['id']for a in request['anchors'])!=ANCHORS:raise ValueError('Fixed complete3anchors')
 if request['requested_static_query']['alpha_order']!=[1.,.125,.25,.375,.5,.625,.75,.875]or request['requested_static_query']['execute_candidate_command']is not False:raise ValueError('Original endpoint/path query only')
 sources={str(path):REQUEST_SHA}
 def bind(p,d):
  name=str(Path(p).resolve(strict=True))
  if name in sources and sources[name]!=d:raise ValueError('Conflicting source identity')
  if sha(name)!=d:raise ValueError('Changed source: '+name)
  sources[name]=d
 for name,digest in request['source_sha256'].items():bind(name,digest)
 pointer(request['known_kinematics'])
 if request['known_kinematics']['sha256']!='9a5b2dd908957878f2b5793a8199cc6ef1912e27a5e9f20c56388d2176c4ae28':raise ValueError('Original qualified world Panda_tip FK')
 prepared=[]
 for anchor in request['anchors']:
  cfgpath=pointer(anchor['config']);cfg=read(cfgpath);bind(cfgpath,anchor['config']['sha256'])
  if cfg['seed']!=anchor['seed']or cfg.get('gripper_completion_latch')is not True or cfg.get('diagnostic_no_proxy_attachment',False)is not False:raise ValueError('Unchanged original positive scene/gripper physics required')
  runpath=pointer(anchor['original_run_inputs']);run=read(runpath);bind(runpath,anchor['original_run_inputs']['sha256'])
  if run['config']!=cfg:raise ValueError('Actual recorded config differs')
  for ref in anchor['closed_case'].values():pointer(ref);bind(ref['path'],ref['sha256'])
  worker=read(pointer(anchor['closed_case']['worker_exit']));result=read(pointer(anchor['closed_case']['result']))
  if worker['exit_code']!=0 or worker['native_closed']is not True or result['complete']is not True or result['source_unchanged']is not True:raise ValueError('Closed unchanged case required')
  for name,digest in run['source_sha256'].items():bind(ROOT/name,digest)
  from ..local_mpc_20261102.worker import native_source_snapshot
  native=native_source_snapshot(cfg)
  if run['native_sources_sha256']!=native:raise ValueError('Original Native asset chain changed')
  for name,digest in native.items():bind(Path(cfg['native_repo'])/name,digest)
  log=pointer(anchor['public_decision']);bind(log,anchor['public_decision']['sha256']);rows=lines(log);at=anchor['timestamp'];index=anchor['public_decision']['line_index0'];row=rows[index]
  if canonical(row)!=anchor['public_decision']['row_canonical_sha256']or abs(row['time']-at)>1e-8:raise ValueError('Exact saved anchor row')
  if row['public_robot']!=anchor['public_robot']or row['actuator_state_before']!=anchor['current_actuator_h']:raise ValueError('Saved public anchor/state differs')
  prefix=[r for r in rows if r['time']<at-1e-9]
  if len(prefix)!=round(at/.1)or any(abs(r['time']-i*.1)>1e-8 for i,r in enumerate(prefix)):raise ValueError('Complete ordered100ms public dispatch prefix')
  ticks=[]
  for r in prefix:
   receipt=r['executor'];action=r['executed_action']
   if receipt['executed_ticks']!=2 or receipt['unexpected_physics_ticks']!=0 or len(receipt['ticks'])!=2 or receipt['target_joints']!=action['joint_goal']or receipt['gripper_target']!=action['opening']:raise ValueError('Original applied two-tick reference trace required')
   for t in receipt['ticks']:
    if t['tick_completed']is not True:raise ValueError('Incomplete old tick')
    ticks.append(t)
  if not ticks or abs(ticks[-1]['clock_after']-at)>1e-8:raise ValueError('Stop BEFORE anchor command')
  public_path=Path(anchor['case_directory'])/'public_shadow/public_outputs.jsonl';sources[str(public_path.resolve())]=sha(public_path)
  packets=lines(public_path);robots={round(p['public_robot']['timestamp'],9):{k:p['public_robot'][k]for k in ROBOT_KEYS}for p in packets if p['public_robot']['timestamp']<=at+1e-8}
  if len(robots)!=round(at/.05)+1:raise ValueError('Every real50ms PUBLIC robot checkpoint required')
  for t in ticks:
   for k in('clock_before','clock_after'):
    if round(t[k],9)not in robots:raise ValueError('Missing public tick')
  C=anchor['C_first_command']
  if C['command_executed']is not False or C['collision_qualified']is not False or C['opening']!=0.:raise ValueError('C remains unexecuted and unqualified')
  pointer(C['exact_frozen_step']);sources[str(Path(C['exact_frozen_step']['path']).resolve())]=C['exact_frozen_step']['sha256']
  prepared.append(dict(anchor=anchor,config=cfg,prefix=prefix,ticks=ticks,robots=robots))
 for name in OWN:sources[str((ROOT/name).resolve())]=sha(ROOT/name)
 return request,prepared,sources


def check(request_path,anchor_id=None,source_freeze=None,expected_source_freeze_sha256=None,collect_sources=False):
 request,prepared,sources=load_request(request_path)
 if anchor_id is not None and anchor_id not in ANCHORS:raise ValueError('Fixed anchor only')
 if collect_sources:return dict(schema=SCHEMA,request=dict(path=str(Path(request_path).resolve()),sha256=REQUEST_SHA),files_sha256=sources,tolerances=TOL,C_executed=False,Native_started=False)
 ready=False
 if source_freeze is not None:
  if not isinstance(expected_source_freeze_sha256,str)or re.fullmatch('[0-9a-f]{64}',expected_source_freeze_sha256)is None or sha(source_freeze)!=expected_source_freeze_sha256:raise ValueError('Exact external freeze SHA required')
  frozen=read(source_freeze)
  if frozen.get('schema')!=SCHEMA or frozen.get('files_sha256')!=sources or frozen.get('tolerances')!=TOL or frozen.get('request',{}).get('sha256')!=REQUEST_SHA:raise ValueError('Complete same current replay source map required')
  ready=True
 return dict(complete=True,ready=ready,missing=[]if ready else['execution_source_freeze'],anchors=[dict(id=p['anchor']['id'],prefix_macros=len(p['prefix']),prefix_ticks=len(p['ticks']))for p in prepared],anchor_id=anchor_id,Native_started=False,NN_forward_calls=0,C_command_executed=False,tolerances=TOL)


class ReplayBackend:
 def __init__(self,backend,prepared,output):self.backend=backend;self.prepared=prepared;self.output=output;self.index=0;self.robot_checks=[];self.query_receipts=[]
 def __getattr__(self,name):return getattr(self.backend,name)
 def read_state(self):return{'opening':float(np.mean(self.backend.gripper.get_open_amount()))}
 def set_joint_target_positions(self,q):
  near(q,self.prepared['ticks'][self.index]['reference_positions'],TOL['reference'],'applied original qref');self.backend.set_joint_target_positions(q)
 def actuate_gripper(self,opening):
  expected=self.prepared['ticks'][self.index];near(opening,expected['gripper_target'],0,'opening');done=self.backend.actuate_gripper(opening)
  if done is not expected['gripper_done']:raise ValueError('Original gripper completion diverged')
  return done
 def tick(self,phase):
  if stop_requested(self.output):raise InterruptedError('Administrative STOP before replay tick')
  expected=self.prepared['ticks'][self.index];match_robot(robot(self.backend),self.prepared['robots'][round(expected['clock_before'],9)])
  self.backend.tick(phase) # Returned privileged state is deliberately ignored.
  errors=match_robot(robot(self.backend),self.prepared['robots'][round(expected['clock_after'],9)])
  self.robot_checks.append(dict(clock=float(self.backend.clock()),errors=errors));self.index+=1
 def probe_joint_collision(self,q):
  if self.index!=len(self.prepared['ticks']):raise ValueError('No candidate queries before complete matching prefix')
  before=float(self.clock());snapshot=robot(self.backend);collision=self.backend.probe_joint_collision(q)
  if self.clock()!=before:raise ValueError('Collision query stepped physics')
  match_robot(robot(self.backend),snapshot);record=deepcopy(self.backend.last_collision_probe)
  if record.get('valid')is not True or record.get('actual_q_qdot_servo_ik_and_target_native_state_preserved')is not True:raise ValueError('Original static query restoration failed')
  self.query_receipts.append(record);return collision


def execute(request_path,anchor_id,output,source_freeze,expected_source_freeze_sha256):
 admission=check(request_path,anchor_id,source_freeze,expected_source_freeze_sha256)
 if not admission['ready']or anchor_id not in ANCHORS:raise ValueError('Explicit complete fixed-anchor admission required')
 output=Path(output).resolve()
 if FAMILY not in output.parents or output.exists()or stop_requested(output):raise ValueError('Fresh output and preserved STOP required')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=''or not os.environ.get('DISPLAY'):raise ValueError('Frozen CPU-only Native wrapper required')
 from ..local_mpc_20261102 import worker as old
 from ..local_mpc_control_20261103.completed_gripper import CompletedGripperBackend
 from ..local_mpc_control_20261103.joint_goals import PoEJointGoalHelper,JointGoalConfig
 from ..local_mpc_p2_20261103.kinematics import PoEKinematics
 from ..local_mpc_p2_20261103.action_adapter import matrix_quaternion
 request,prepared,sources=load_request(request_path);p=next(x for x in prepared if x['anchor']['id']==anchor_id);a=p['anchor'];cfg=p['config']
 old.environment_proof();native_before=old.native_source_snapshot(cfg);output.mkdir(parents=True,exist_ok=False);(output/'labels').mkdir()
 write(output/'run_inputs.json',dict(admission=admission,request_sha256=REQUEST_SHA,source_sha256=sources,config=cfg,anchor_id=anchor_id,GT_restore_used=False,C_executed=False,NN_forward_calls=0))
 backend=raw=None;success=False
 try:
  raw=old.make_backend(cfg,a['scene']);backend=ReplayBackend(CompletedGripperBackend(raw),p,output);executor=old.make_executor(backend,cfg)
  match_robot(robot(backend),p['robots'][0.])
  for macro in p['prefix']:
   if stop_requested(output):raise InterruptedError('Administrative STOP before macro')
   match_h(executor.state(),macro['actuator_state_before']);action=macro['executed_action'];actual=executor.execute_joint_target(action['joint_goal'],opening=action['opening'],phase=macro['executor']['phase']).to_dict()
   for got,expected in zip(actual['ticks'],macro['executor']['ticks']):
    for key in('reference_positions','reference_velocities','reference_accelerations'):near(got[key],expected[key],TOL['reference'],key)
   match_h(executor.state(),macro['actuator_state_after'])
  match_robot(robot(backend),a['public_robot']);match_h(executor.state(),a['current_actuator_h'])
  if abs(backend.clock()-a['timestamp'])>TOL['clock']or backend.index!=len(p['ticks']):raise ValueError('Anchor clock/prefix gate failed')
  kin=PoEKinematics(read(pointer(request['known_kinematics'])));helper=PoEJointGoalHelper(kin,executor.limits,JointGoalConfig())
  C=a['C_first_command'];q=np.asarray(C['joint_goal'],float);T=kin(q);expectedFK=C['static_FK'];near(T,expectedFK['tcp_matrix'],1e-12,'frozen C static FK')
  command=dict(joint_goal=q,tcp_position=T[:3,3],tcp_quaternion=matrix_quaternion(T[:3,:3]),opening=C['opening'])
  before=backend.clock();goal,reason,qualification=helper.qualify_action(backend,dict(q=np.asarray(backend.get_joint_positions()),time=before),command,require_predicted_joint_goal=True)
  if backend.clock()!=before or backend.index!=len(p['ticks']):raise ValueError('Candidate query changed physics')
  result=dict(complete=True,anchor_id=anchor_id,anchor_matched=True,prefix_macros=len(p['prefix']),prefix_ticks=len(p['ticks']),current_public_robot=robot(backend),current_actuator_h=executor.state(),
   C_first_q=q,C_command_executed=False,C_response_model_qualified=False,collision_scope_only=True,collision_helper_qualified=goal is not None,collision_helper_rejection_reason=reason,original_helper_qualification=qualification,query_receipts=backend.query_receipts,
   static_arm_map_only_dynamic_target_and_held_geometry_excluded=True,whole_path_or_physical_execution_certified=False,Native_IK_calls=0,query_physics_steps=0,NN_forward_calls=0,
   replay_robot_checks=backend.robot_checks,GT_or_q_pose_held_restoration_used=False,Native_sources_unchanged=old.native_source_snapshot(cfg)==native_before,source_unchanged=all(sha(name)==digest for name,digest in sources.items()))
  if not result['Native_sources_unchanged']or not result['source_unchanged']:raise ValueError('Sources changed during replay')
  write(output/'result.json',result);success=True
 except BaseException as error:
  write(output/'failure.json',dict(type=type(error).__name__,message=str(error),anchor_id=anchor_id,C_command_executed=False,query_count=0 if backend is None else len(backend.query_receipts),anchor_qualified=False,NN_forward_calls=0));raise
 finally:
  closed=False;cleanup_errors=[]
  if raw is not None:
   # Quarantine original Native labels AFTER all replay/query decisions; none
   # is decoded or used by this driver for state recovery/admission.
   try:write(output/'labels/native_replay_ticks.json',raw.pop_tick_records())
   except BaseException as error:cleanup_errors.append('posthoc label flush: '+str(error))
   try:raw.close();closed=True
   except BaseException as error:cleanup_errors.append('Native close: '+str(error))
  write(output/'worker_exit.json',dict(exit_code=0 if success and not cleanup_errors else 1,native_closed=closed,C_command_executed=False,result_recorded=success,cleanup_errors=cleanup_errors))
  if cleanup_errors:raise RuntimeError('Replay cleanup failed: '+repr(cleanup_errors))
 return result
