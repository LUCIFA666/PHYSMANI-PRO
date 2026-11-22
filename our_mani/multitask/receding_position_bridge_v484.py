"""Project an infeasible next setpoint into jointly reachable public-state bounds.

Both the measured-state and confirmed-reference paths pass complete first and
suffix checks. The original endpoint, endpoint derivative and deadline stay
fixed. No empirical motor model, extra observation or physics tick is used.
"""
from dataclasses import dataclass,replace
from copy import deepcopy
from pathlib import Path
from types import MethodType
import numpy as np
from ..contracts import Reachability
from ..intercept_planner import JointTrajectory
from ..reachability import TimedReachability
from .next_position_actual_v473 import check_sample
from .reference_stop_guard_v274 import offset_gate,NextReferenceStopGuard
from .evaluation import file_sha,digest


def binding(root):
 return dict(schema='r10_public_actual_reference_piecewise_position_bridge_v484',
  sources_sha256={p:file_sha(Path(root)/p) for p in (
   'our_mani/multitask/receding_position_bridge_v484.py','our_mani/multitask/next_position_actual_v473.py')},
  exact_next_position_and_complete_two_branch_first_suffix_collision_proofs=True,
  original_endpoint_derivative_and_absolute_deadline_retained=True,
  original_reference4_actual6_native_actuator_and_grasp_retained=True,
  initial_actual_velocity_basis='public backward-window proxy, not instantaneous endpoint velocity',
  existing_phase_speed_reserve_retained_for_both_branches=True,
  additional_observations=0,additional_physics_ticks=0,empirical_motor_model=False,
  actual_motor_tracking_guaranteed=False,formal_VAL_TEST=False)


@dataclass(frozen=True)
class PiecewiseReference:
 first:object
 tail:object
 @property
 def start(self):return self.first.start
 @property
 def end(self):return self.tail.end
 @property
 def q0(self):return self.first.q0
 @property
 def v0(self):return self.first.v0
 @property
 def q1(self):return self.tail.q1
 @property
 def v1(self):return self.tail.v1
 def at(self,time):return self.first.at(time) if time<=self.first.end else self.tail.at(time)


class VelocityCapModel:
 def __init__(self,original,limits):self.original,self.velocity_limits=original,np.asarray(limits)
 def __getattr__(self,name):return getattr(self.original,name)


def capped_gate(original,limits):
 return TimedReachability(VelocityCapModel(original.model,limits),max_joint_step=original.max_joint_step,
  minimum_samples=original.minimum_samples,twist_tolerance=original.twist_tolerance,
  position_tolerance=original.position_tolerance,orientation_tolerance=original.orientation_tolerance)



def endpoint_velocity(q0,v0,q1,dt,acceleration,velocity):
 # Cubic-Hermite acceleration is affine in time. These are its exact two
 # endpoint acceleration inequalities, intersected with terminal speed caps.
 d=(np.asarray(q1)-np.asarray(q0))/dt;v0=np.asarray(v0)
 low=np.maximum.reduce([3*d-2*v0-.5*acceleration*dt,1.5*d-.5*v0-.25*acceleration*dt,-velocity])
 high=np.minimum.reduce([3*d-2*v0+.5*acceleration*dt,1.5*d-.5*v0+.25*acceleration*dt,velocity])
 if np.any(low>high+1e-10):return None
 return np.clip(2*d-v0,low,high)


def position_step(native,robot,q,dt):
 fast=check_sample(native,robot,q,np.zeros_like(q),dt)
 if fast.feasible:return fast
 v=endpoint_velocity(robot.joints,robot.joint_velocities,q,dt,
  np.asarray(native.model.acceleration_limits),np.asarray(native.model.velocity_limits))
 if v is None:return fast
 return native.check_joints(robot,q,v,dt)


def repair_first(native,reference,robot,q0,v0,plan,dt):
 now=float(robot.timestamp);remaining=float(plan.end-now);tail_seconds=remaining-dt
 trials=[]
 if tail_seconds<dt-1e-8:return None,dict(reason='original_deadline_has_no_two_piece_bridge_room',trials=trials)
 q0,v0=np.asarray(q0),np.asarray(v0);qa,va=robot.joints,robot.joint_velocities
 if not np.allclose(native.model.acceleration_limits,6.) or not np.allclose(reference.model.acceleration_limits,4.):
  raise ValueError('unchanged reference4/actual6 contract required')
 a_actual=np.asarray(native.model.acceleration_limits);a_ref=np.asarray(reference.model.acceleration_limits)
 v_actual=np.asarray(native.model.velocity_limits);v_ref=np.asarray(reference.model.velocity_limits)
 # Necessary acceleration and displacement domains; free terminal derivatives
 # are solved analytically and every complete sampled gate remains final.
 low=np.maximum.reduce([qa+va*dt-.5*a_actual*dt**2,q0+v0*dt-.5*a_ref*dt**2,
  qa-v_actual*dt,q0-v_ref*dt,native.model.lower,reference.model.lower])
 high=np.minimum.reduce([qa+va*dt+.5*a_actual*dt**2,q0+v0*dt+.5*a_ref*dt**2,
  qa+v_actual*dt,q0+v_ref*dt,native.model.upper,reference.model.upper])
 if np.any(low>high):return None,dict(reason='actual_reference_position_intersection_empty',trials=trials)
 desired=plan.at(now+dt)[0];closest=np.clip(desired,low,high);actual_near=np.clip(qa+va*dt,low,high)
 original_actual=native.check_joints(robot,plan.q1,plan.v1,remaining)
 if not original_actual.feasible:return None,dict(reason='original_current_actual_full_endpoint_proof_failed',actual_reason=original_actual.reason,trials=trials)
 for fraction in (0.,.25,.5,.75,1.):
  q=(1.-fraction)*closest+fraction*actual_near
  vr=endpoint_velocity(q0,v0,q,dt,a_ref,v_ref)
  actual_first=position_step(native,robot,q,dt)
  if vr is None:
   trials.append(dict(fraction_toward_actual_near=fraction,reference_first='no_four_unit_Hermite_terminal_velocity'));continue
  reference_robot=replace(robot,joints=q0,joint_velocities=v0)
  reference_first=reference.check_joints(reference_robot,q,vr,dt)
  row=dict(fraction_toward_actual_near=fraction,actual_first=actual_first.reason,reference_first=reference_first.reason)
  if not(actual_first.feasible and reference_first.feasible):trials.append(row);continue
  va1=actual_first.joint_velocities
  actual_future=replace(robot,timestamp=now+dt,joints=q,joint_velocities=va1)
  reference_future=replace(robot,timestamp=now+dt,joints=q,joint_velocities=vr)
  actual_tail=offset_gate(native,dt).check_joints(actual_future,plan.q1,plan.v1,tail_seconds)
  reference_tail=offset_gate(reference,dt).check_joints(reference_future,plan.q1,plan.v1,tail_seconds)
  row.update(actual_tail=actual_tail.reason,reference_tail=reference_tail.reason);trials.append(row)
  if not(actual_tail.feasible and reference_tail.feasible):continue
  first=JointTrajectory(now,now+dt,q0.copy(),v0.copy(),q.copy(),vr.copy())
  tail=JointTrajectory(now+dt,plan.end,q.copy(),vr.copy(),plan.q1.copy(),plan.v1.copy())
  return PiecewiseReference(first,tail),dict(reason='complete_dual_branch_first_and_suffix_bridge',trials=trials,
   actual_first=deepcopy(actual_first.evidence),reference_first=deepcopy(reference_first.evidence),
   actual_suffix=deepcopy(actual_tail.evidence),reference_suffix=deepcopy(reference_tail.evidence),
   original_actual_full_endpoint=deepcopy(original_actual.evidence),
   actual_first_terminal_velocity=va1.tolist(),reference_first_terminal_velocity=vr.tolist(),
   suffix_collision_time_offset=dt,original_endpoint=plan.q1.tolist(),original_endpoint_velocity=plan.v1.tolist(),
   original_deadline=float(plan.end),actual_tracking_guaranteed=False)
 return None,dict(reason='no_complete_dual_branch_piecewise_bridge',trials=trials)


def install(actor):
 m,c=actor.continuous_reference_manager,actor.controller
 if m.confirmations or c.feedback_step_calls or c.goal is not None:raise ValueError('install before native execution')
 function=getattr(m.make_brake,'__func__',m.make_brake)
 guards=[cell.cell_contents for cell in (getattr(function,'__closure__',None)or()) if isinstance(cell.cell_contents,NextReferenceStopGuard)]
 if len(guards)!=1:raise ValueError('retain exactly one existing stopping authority')
 guard=guards[0];events=[];previous=m.check_stream
 def stream(instance,robot,duration):
  result=previous(robot,duration)
  if result is None or not result.feasible or instance.pending is None:return result
  old=instance.pending;native=instance.native_gate;reference=instance.reference_reserve_gate
  if instance.plan_kind!='braking':
   prefix=instance.approach_transit.reference_gate
   reference=prefix
   native=capped_gate(native,np.minimum(native.model.velocity_limits,prefix.model.velocity_limits))
  qnext,vnext=np.asarray(old['reference_to_joints']),np.asarray(old['reference_to_velocity'])
  actual=position_step(native,robot,qnext,instance.config.dt)
  if actual.feasible:
   old['exact_next_actual_position_proof']=dict(reason=actual.reason,evidence=deepcopy(actual.evidence),
    actual_terminal_velocity=actual.joint_velocities.tolist(),nominal_derivative_is_separate=True)
   return result
  boundary=instance.continuity_state(robot)
  if boundary is None:return Reachability(False,'position_bridge_requires_confirmed_boundary')
  plan,proof=repair_first(native,reference,robot,boundary[0],boundary[1],instance.plan,instance.config.dt)
  event=dict(time=float(robot.timestamp),original_exact_next_reason=actual.reason,proof=proof,admitted=plan is not None)
  if plan is None:
   events.append(event);instance.pending=None;guard.pending_stop=None
   return Reachability(False,'no_same_deadline_measured_position_bridge',evidence=proof)
  q,v=plan.at(float(robot.timestamp)+instance.config.dt)
  pending=dict(old,reference_to_joints=q.tolist(),reference_to_velocity=v.tolist(),
   reference_gate_evidence=deepcopy(proof['reference_first']),reference_gate_reason='feasible_piecewise_actual_reference_bridge',
   plan_start=plan.start,plan_end=plan.end,measured_position_bridge=deepcopy(proof))
  pending['proposal_id']=digest(dict(time=pending['time'],candidate=pending['candidate_id'],q=q.tolist(),v=v.tolist()))
  stop=None
  if instance.plan_kind!='braking' and np.any(abs(v)>1e-10):
   stop,stop_proof=guard.next_stop(robot,pending)
   if stop is None:
    event.update(admitted=False,offset_stop_rejected=stop_proof);events.append(event);instance.pending=None;guard.pending_stop=None
    return Reachability(False,'position_bridge_has_no_complete_offset_stop')
   pending['next_reference_stop_guard_v274']=dict(complete_nominal_offset_stop=True,
    complete_current_actual_Hermite_to_same_stop=True,proof=deepcopy(stop_proof),stop_at=stop.end,motor_tracking_guaranteed=False)
   guard.pending_stop=dict(proposal_id=pending['proposal_id'],plan=stop,proof=stop_proof)
  else:guard.pending_stop=None
  old['superseded_by_measured_position_bridge']=True
  instance.plan=plan;c.trajectory=plan;instance.pending=pending;instance.reference_events.append(pending)
  event.update(new_proposal_id=pending['proposal_id'],original_goal_phase_epoch_unchanged=True,
   original_absolute_deadline=plan.end);events.append(event)
  return Reachability(True,'feasible_piecewise_actual_reference_bridge',joints=q,joint_velocities=v,evidence=proof)
 m.check_stream=MethodType(stream,m)
 original_metrics=actor.metrics_report
 def metrics(instance):return dict(original_metrics(),receding_position_bridge_v484=dict(
  binding=binding(Path(__file__).resolve().parents[2]),events=deepcopy(events)))
 actor.metrics_report=MethodType(metrics,actor)
 return events
