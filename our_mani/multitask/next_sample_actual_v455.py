"""Check the exact next issued reference from the current public measured state.

A long-horizon actual-state Hermite and a reference-state Hermite can have
entirely different first samples. Retain both proofs and additionally check
that same next command over one counted tick. Measured backward-window speed
is still a proxy, so this is a command admission test, not a motor guarantee.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MethodType
import numpy as np
from ..contracts import Reachability
from ..intercept_planner import JointTrajectory
from .evaluation import file_sha,digest


def binding(root):
 return dict(schema='r10_exact_next_command_public_actual_state_proof_v455',
  source_sha256=file_sha(Path(root)/'our_mani/multitask/next_sample_actual_v455.py'),
  complete_existing_actual_and_reference_path_and_collision_proofs_retained=True,
  extra_check='current public actual q/backward-window v to exact next issued q/nominal v over original dt',
  velocity_semantics='measured backward-window proxy, not instantaneous endpoint velocity',
  native_motor_parameters_changed=False,actual_motor_tracking_guaranteed=False,
  additional_physics_ticks=0,additional_camera_queries=0,formal_VAL_TEST=False)


def check_sample(native_gate,robot,q,v,dt):
 if not np.allclose(native_gate.model.acceleration_limits,6.) or abs(dt-.05)>1e-9:
  raise ValueError('retain original measured six-unit native/.05s contract')
 return native_gate.check_joints(robot,np.asarray(q),np.asarray(v),dt)


def check_first(native_gate,robot,q0,v0,q1,v1,duration,dt):
 if duration<dt-1e-9:return Reachability(False,'next_command_outside_counted_horizon')
 plan=JointTrajectory(float(robot.timestamp),float(robot.timestamp)+duration,
  np.asarray(q0),np.asarray(v0),np.asarray(q1),np.asarray(v1))
 q,v=plan.at(float(robot.timestamp)+dt)
 return check_sample(native_gate,robot,q,v,dt)


class CandidateGate:
 def __init__(self,original,manager,events):self.original,self.manager,self.events=original,manager,events
 def __getattr__(self,name):return getattr(self.original,name)
 def check(self,*args,**kwargs):
  from ..reachability import TimedReachability
  return TimedReachability.check(self,*args,**kwargs)
 def check_joints(self,robot,q,v,duration):
  checked=self.original.check_joints(robot,q,v,duration)
  if not checked.feasible:return checked
  m=self.manager;actual=m.current_state.robot if m.current_state is not None else robot
  boundary=m.continuity_state(actual)
  if boundary is None:return Reachability(False,'next_sample_requires_confirmed_reference')
  if abs(actual.timestamp-robot.timestamp)>1e-8:raise ValueError('current actual frame mismatch')
  q0,v0,_=boundary
  proof=check_first(m.native_gate,actual,q0,v0,checked.joints,checked.joint_velocities,duration,m.config.dt)
  if len(self.events)<512:self.events.append(dict(time=float(actual.timestamp),event='candidate_exact_next_actual_proof',
   passed=proof.feasible,reason=proof.reason,evidence=deepcopy(proof.evidence)))
  return checked if proof.feasible else Reachability(False,'candidate_exact_next_actual_'+proof.reason,evidence=proof.evidence)


def install(actor):
 m,c=actor.continuous_reference_manager,actor.controller
 if m.confirmations or c.feedback_step_calls:raise ValueError('install before execution')
 events=[]
 m.dual_gate=CandidateGate(m.dual_gate,m,events);c.original_reachability=c.generator.reachability=m.dual_gate
 prefix=m.approach_transit;prefix.reference_gate=CandidateGate(prefix.reference_gate,m,events)
 previous=m.check_stream
 def stream(instance,robot,duration):
  result=previous(robot,duration)
  if result is None or not result.feasible or instance.pending is None:return result
  pending=instance.pending
  proof=check_sample(instance.native_gate,robot,pending['reference_to_joints'],pending['reference_to_velocity'],instance.config.dt)
  event=dict(time=float(robot.timestamp),event='stream_exact_next_actual_proof',passed=proof.feasible,
   reason=proof.reason,evidence=deepcopy(proof.evidence),proposal_id=pending['proposal_id'])
  events.append(event)
  if not proof.feasible:
   instance.pending=None
   return Reachability(False,'stream_exact_next_actual_'+proof.reason,evidence=proof.evidence)
  pending['exact_next_actual_state_proof']=deepcopy(event)
  return result
 m.check_stream=MethodType(stream,m)
 # Reuse the installed actor-local stopping authority, without replacing its
 # existing offset-stop or dispatch-confirmation services.
 from .reference_stop_guard_v274 import NextReferenceStopGuard
 function=getattr(m.make_brake,'__func__',m.make_brake)
 guards=[cell.cell_contents for cell in (getattr(function,'__closure__',None)or())
  if isinstance(cell.cell_contents,NextReferenceStopGuard)]
 if len(guards)!=1:raise ValueError('exactly one existing consistent stop authority required')
 guard=guards[0]
 def brake(service,robot,remaining_seconds,reason):
  service.constraint_contract.verify(m);boundary=m.continuity_state(robot)
  if boundary is None:return False
  q0,v0,basis=boundary;now=float(robot.timestamp);trials=[]
  for duration in service.durations((v0,robot.joint_velocities),service.available(now,remaining_seconds)):
   phase=c.library.phase_order[c.phase_index] if c.phase_index<len(c.library.phase_order) else 'verify'
   intent=c.goal if c.goal is not None else m.retired_brake_goal;same=intent is not None and intent.phase==phase
   c.generator.bind_candidate_scene(m.current_state,m.current_forecast,phase,
    intent.reference_id if same else '',intent.manipulated_ids if same else (),now+duration)
   end,zero=q0+.5*v0*duration,np.zeros_like(v0)
   reference=service.reference_gate().check_joints(replace(robot,joints=q0,joint_velocities=v0),end,zero,duration)
   actual=m.native_gate.check_joints(robot,end,zero,duration)
   first=check_first(m.native_gate,robot,q0,v0,end,zero,duration,m.config.dt) if reference.feasible and actual.feasible else None
   trials.append(dict(duration=duration,reference_reason=reference.reason,actual_reason=actual.reason,
    exact_next_actual_reason=None if first is None else first.reason))
   if first is None or not first.feasible:continue
   m.plan=JointTrajectory(now,now+duration,q0.copy(),v0.copy(),end,zero)
   m.plan_kind='braking';m.plan_candidate_id='next_sample_brake:'+digest(dict(time=now,q=q0.tolist(),v=v0.tolist(),duration=duration))[:16]
   m.invalidated=False
   event=dict(time=now,event='same_next_sample_and_full_path_brake_staged',reason=reason,boundary_basis=basis,
    trials=trials,phase_clock_reset=False,total_budget_extended=False,actual_tracking_guaranteed=False)
   events.append(event);service.record(**deepcopy(event));return True
  event=dict(time=now,event='no_same_next_sample_full_path_brake',reason=reason,trials=trials,
   raw_hold_not_relabelled_safe=True);events.append(event);service.record(**deepcopy(event));return False
 guard.make_brake=MethodType(brake,guard)
 old=actor.metrics_report
 def metrics(instance):return dict(old(),next_sample_actual_v455=dict(binding=binding(Path(__file__).resolve().parents[2]),events=deepcopy(events)))
 actor.metrics_report=MethodType(metrics,actor)
 return events
