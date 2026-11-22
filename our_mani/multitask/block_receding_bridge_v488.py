"""Shared measured-position bridges for legacy block global/local references."""
from contextlib import contextmanager
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ..intercept_planner import JointTrajectory
from ..reachability import TimedReachability
from .block_fresh_tail_v448 import SmoothBlockEnvironment as OriginalBoundary
from .block_clip_brake_v315 import public_decision_scope as original_scope
from .block_reference_v197 import DualReferenceGate,ReferenceModel,clipped_terminal
from .receding_position_bridge_v484 import repair_first,position_step
from .reference_stop_guard_v274 import offset_gate
from .evaluation import file_sha
MARGIN=.005  # Same planning reserve already used by the core14 geometry actors.
SOURCES=('our_mani/multitask/block_receding_bridge_v488.py',
 'our_mani/multitask/receding_position_bridge_v484.py','our_mani/multitask/next_position_actual_v473.py')

def binding(root):
 return dict(schema='r10_block_global_local_measured_position_bridge_v488',
  sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},planning_joint_interior_margin_rad=MARGIN,
  margin_origin='existing core14 joint_limit_recovery_margin_rad, an extra planning reserve',
  previous_extra_rejection_only455_and473_policies_not_installed=True,
  native_joint_intervals_and_motors_unchanged=True,reference4_actual6_and_complete_collisions_retained=True,
  current_counted_tail_observations_retained=True,actual_motor_tracking_guaranteed=False,formal_VAL_TEST=False)

class InteriorModel:
 def __init__(self,original):self.original=original
 def __getattr__(self,name):return getattr(self.original,name)
 @property
 def lower(self):return np.asarray(self.original.lower)+MARGIN
 @property
 def upper(self):return np.asarray(self.original.upper)-MARGIN

class SmoothBlockEnvironment(OriginalBoundary):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.bridge_events=[]
 def _gates(self):
  actual,reference=super()._gates()
  return actual,TimedReachability(InteriorModel(reference.model))
 def qualify(self,robot,plan,kind):
  actual,reference=self._gates();q,v=plan.at(min(robot.timestamp+self.env.ledger.dt,plan.end))
  first=position_step(actual,robot,q,float(self.env.ledger.dt))
  if first.feasible:return plan,dict(kind=kind,exact_next_actual_position=first.reason)
  bridge,proof=repair_first(actual,reference,robot,self.reference_q,self.reference_v,plan,float(self.env.ledger.dt))
  self.bridge_events.append(dict(time=float(robot.timestamp),kind=kind,admitted=bridge is not None,proof=proof))
  return bridge,dict(kind=kind,measured_position_bridge=proof)
 def stopped_backup(self,robot,q,v):
  actual,reference=self._gates();dt=float(self.env.ledger.dt);now=float(robot.timestamp)
  first=position_step(actual,robot,q,dt)
  if not first.feasible:return None
  lower=max(.2,1.5*float(np.max(abs(v)/4.)))
  for duration in sorted({lower,.2,.3,.4,.5,.6,.8,1.}):
   if duration<lower-1e-8 or now+dt+duration>10.+1e-8 or dt+duration>1.1+1e-8:continue
   end=q+.5*v*duration;zero=np.zeros_like(v)
   ref=offset_gate(reference,dt).check_joints(replace(robot,timestamp=now+dt,joints=q,joint_velocities=v),end,zero,duration)
   measured=offset_gate(actual,dt).check_joints(replace(robot,timestamp=now+dt,joints=q,joint_velocities=first.joint_velocities),end,zero,duration)
   current=actual.check_joints(robot,end,zero,dt+duration)
   if ref.feasible and measured.feasible and current.feasible:
    return JointTrajectory(now+dt,now+dt+duration,q.copy(),v.copy(),end,zero)
  return None
 def advance(self,pose,*,opening=1.):
  decision,robot=self.current_decision,self.public_robot
  if (decision is None or decision.get('command')!='local_capture' or robot is None
   or abs(robot.timestamp-self.env.time)>1e-8 or not np.allclose(pose.xyzw(),decision['pose'].xyzw(),rtol=0,atol=1e-9)):
   raise ValueError('unchanged original current local decision required')
  self.local_active=True;self._initialize_reference();now=float(self.env.time);dt=float(self.env.ledger.dt)
  actual_gate,reference_gate=self._gates();before=int(self.env.ledger.ticks);failures=[]
  try:
   with self._collision_scope():
    goal=self.env.robot_model.inverse(pose,robot.joints)
    if goal is not None:
     solved=self.env.robot_model.forward(goal)
     if (np.linalg.norm(solved.position-pose.position)>actual_gate.position_tolerance
      or Rotation.from_matrix(np.array(solved.rotation@pose.rotation.T,copy=True)).magnitude()>actual_gate.orientation_tolerance):goal=None
    if goal is not None:
     desired=np.clip((goal-robot.joints)/dt,-self.env.robot_model.velocity_limits,self.env.robot_model.velocity_limits)
     reference_robot=replace(robot,joints=self.reference_q,joint_velocities=self.reference_v)
     for duration in (.15,.25,.4,.5,.7,.9):
      if now+duration>10.+1e-8:continue
      for scale in (1.,.5,.25,.125):
       terminal=clipped_terminal(self.reference_v,desired,duration,scale)
       end=self.reference_q+.5*(self.reference_v+terminal)*duration
       actual=actual_gate.check_joints(robot,end,terminal,duration);reference=reference_gate.check_joints(reference_robot,end,terminal,duration)
       failures.append(dict(duration=duration,scale=scale,actual=actual.reason,reference=reference.reason))
       if not(actual.feasible and reference.feasible):continue
       plan=JointTrajectory(now,now+duration,self.reference_q,self.reference_v,end,terminal)
       plan,proof=self.qualify(robot,plan,'local_reference')
       if plan is None:continue
       q,v=plan.at(now+dt);backup=self.stopped_backup(robot,q,v)
       if backup is None:continue
       self.backup=backup
       return self._issue(q,v,opening,dict(kind='same_deadline_measured_local_reference_bridge',
        actual_gate=actual.reason,reference_gate=reference.reason,full_gate_trials=failures,
        original_duration=duration,original_scale=scale,position_bridge=proof,
        complete_actual_and_reference_offset_stop=True,stop_at=backup.end))
    return self._brake(self.env._requested_opening,'local_goal_or_full_trajectory_rejected')
  except (ValueError,IndexError,KeyError) as error:
   if int(self.env.ledger.ticks)!=before:raise
   self.reference_events.append(dict(event='public_local_reference_unavailable',time=now,error=str(error)))
   return self._stationary_existing_target('current_public_collision_forecast_unavailable')
 def _brake(self,opening,reason):
  now,dt=float(self.env.time),float(self.env.ledger.dt);robot=self.public_robot
  if robot is None or abs(float(robot.timestamp)-now)>1e-8:return self._stationary_existing_target('current_actual_frame_required_for_braking')
  actual,reference=self._gates();ref=replace(robot,joints=self.reference_q,joint_velocities=self.reference_v);candidates=[]
  if self.backup is not None and self.backup.start-1e-8<=now<self.backup.end-1e-8:
   q,v=self.backup.at(now)
   if np.allclose(q,self.reference_q,rtol=0,atol=1e-6) and np.allclose(v,self.reference_v,rtol=0,atol=1e-6):candidates.append(self.backup)
  lower=max(float(np.max(abs(self.reference_v)/4.)),float(np.max(abs(robot.joint_velocities)/6.)))
  for duration in sorted({.05,.1,.15,.2,.25,.3,.4,.5,.6,.8,1.1,*(lower*f for f in (1.02,1.25,1.5))}):
   if duration<dt-1e-8 or duration>1.1+1e-8 or now+duration>10.+1e-8:continue
   candidates.append(JointTrajectory(now,now+duration,self.reference_q.copy(),self.reference_v.copy(),
    self.reference_q+.5*self.reference_v*duration,np.zeros_like(self.reference_v)))
  for plan in candidates:
   duration=float(plan.end-now)
   if not(actual.check_joints(robot,plan.q1,plan.v1,duration).feasible and reference.check_joints(ref,plan.q1,plan.v1,duration).feasible):continue
   plan,proof=self.qualify(robot,plan,'braking_reference')
   if plan is None:continue
   self.backup=plan;q,v=plan.at(min(now+dt,plan.end))
   return self._issue(q,v,opening,dict(kind='current_measured_same_deadline_block_brake',reason=reason,position_bridge=proof))
  return self._stationary_existing_target('no_complete_current_measured_reference_brake')
 def receipt(self):return dict(super().receipt(),block_receding_bridge_v488=dict(binding=binding(Path(__file__).resolve().parents[2]),events=deepcopy(self.bridge_events)))

@contextmanager
def public_decision_scope(evaluator,boundary):
 with original_scope(evaluator,boundary):
  previous=evaluator.CandidateInterception
  def construct(*args,**kwargs):
   workflow=previous(*args,**kwargs);actual_model=args[1]
   reference_model=InteriorModel(ReferenceModel(actual_model));gate=DualReferenceGate(actual_model,reference_model)
   workflow.reachability=workflow.selector.reachability=gate;old_command=workflow.joint_command
   def command(robot):
    result=old_command(robot)
    if result.get('command')!='joint_trajectory':return result
    plan=workflow.plan;now=float(robot.timestamp);query=actual_model.collision_query;origin=workflow.collision_scene.origin
    if query is not None and origin is not None:actual_model.collision_query=lambda offset:query(now-float(origin)+float(offset))
    try:
     reference_robot=replace(robot,joints=boundary.reference_q,joint_velocities=boundary.reference_v)
     check=gate.reference_gate.check_joints(reference_robot,plan.q1,plan.v1,plan.end-now)
     if not check.feasible:return dict(command='hold',reason='global_interior_reference_reserve_rejected',gate_reason=check.reason)
     plan,proof=boundary.qualify(robot,plan,'global_reference')
     if plan is None:return dict(command='hold',reason='global_measured_reference_bridge_unavailable')
     workflow.plan=plan;q,v=plan.at(now+float(boundary.env.ledger.dt))
     return dict(result,joints=q,reference_joints=q,reference_velocities=v,
      global_reference_proof_v197=dict(result['global_reference_proof_v197'],
       planning_joint_interior_margin_rad=MARGIN,measured_position_bridge=proof,original_absolute_arrival_deadline=plan.end))
    finally:actual_model.collision_query=query
   workflow.joint_command=command;return workflow
  evaluator.CandidateInterception=construct
  try:yield
  finally:evaluator.CandidateInterception=previous
