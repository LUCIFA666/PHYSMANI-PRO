"""Additional low-speed proofs for acquire tracking; original brakes stay intact."""
from copy import deepcopy
from dataclasses import replace
from types import MethodType
from pathlib import Path
import numpy as np
from ..contracts import Reachability
from ..reachability import TimedReachability,hermite
from .evaluation import file_sha


def binding(root):
    return dict(schema='r10_acquire_contact_reference_speed_reserve_v1',
        source_sha256=file_sha(Path(root)/'our_mani/multitask/contact_speed_reserve.py'),
        phases=['acquire'],reference_joint_speed_cap_rad_s=.15,
        cap_rule='half original native acceleration6 times original tick.05',
        complete_candidate_prefix_and_remaining_stream_velocity_proofs=True,
        original_native6_reference4_collision_and_braking_gates_retained=True,
        engineering_reserve_not_actual_motor_tracking_guarantee=True,formal_VAL_TEST=False)


def acquire(manager):
    c=manager.controller
    return c.phase_index<len(c.library.phase_order) and c.library.phase_order[c.phase_index]=='acquire'


class PhaseSpeedModel:
    def __init__(self,original,manager,cap):self.original,self.manager,self.cap=original,manager,cap
    def __getattr__(self,name):return getattr(self.original,name)
    @property
    def velocity_limits(self):
        return np.minimum(self.original.velocity_limits,self.cap) if acquire(self.manager) else self.original.velocity_limits


class CandidateSpeedGate:
    def __init__(self,original,manager,cap,events):self.original,self.manager,self.cap,self.events=original,manager,cap,events
    def __getattr__(self,name):return getattr(self.original,name)
    def check(self,*args,**kwargs):return TimedReachability.check(self,*args,**kwargs)
    def check_joints(self,robot,q,v,duration):
        checked=self.original.check_joints(robot,q,v,duration)
        if not checked.feasible or not acquire(self.manager):return checked
        boundary=self.manager.continuity_state(robot)
        if boundary is None:return Reachability(False,'contact_reserve_requires_confirmed_boundary')
        q0,v0,_=boundary;end_q,end_v=checked.joints,checked.joint_velocities
        speeds=[float(np.max(np.abs(hermite(r.joints,r.joint_velocities,end_q,end_v,duration,21)[1])))
            for r in (robot,replace(robot,joints=q0,joint_velocities=v0))]
        passed=max(speeds)<=self.cap+1e-7
        if len(self.events)<256:self.events.append(dict(time=float(robot.timestamp),event='acquire_full_candidate_speed_reserve',
            passed=passed,peak_actual_state_Hermite_speed=speeds[0],peak_reference_state_Hermite_speed=speeds[1]))
        if not passed:return Reachability(False,'acquire_contact_speed_reserve',evidence=dict(peak_speeds=speeds,cap=self.cap))
        return checked


def install(actor):
    c,m=actor.controller,actor.continuous_reference_manager
    if c.goal is not None or c.feedback_step_calls or m.confirmations:raise ValueError('install before first counted step')
    if not np.allclose(m.native_gate.model.acceleration_limits,6.) or m.config.dt!=.05:
        raise ValueError('original native6/.05s required')
    cap=.5*float(np.min(m.native_gate.model.acceleration_limits))*m.config.dt;events=[]
    m.dual_gate=CandidateSpeedGate(m.dual_gate,m,cap,events)
    c.original_reachability=c.generator.reachability=m.dual_gate
    prefix=m.approach_transit;old_gate=prefix.reference_gate
    prefix.reference_gate=TimedReachability(PhaseSpeedModel(old_gate.model,m,cap),max_joint_step=old_gate.max_joint_step,
        minimum_samples=old_gate.minimum_samples,twist_tolerance=old_gate.twist_tolerance,
        position_tolerance=old_gate.position_tolerance,orientation_tolerance=old_gate.orientation_tolerance)
    previous=m.check_stream
    def stream(instance,robot,duration):
        result=previous(robot,duration)
        if (result is None or not result.feasible or instance.pending is None or not acquire(instance)
            or instance.plan_kind=='braking'):return result
        plan=instance.plan;boundary=instance.continuity_state(robot)
        if plan is None or boundary is None:return result
        q0,v0,_=boundary;remaining=float(plan.end-robot.timestamp)
        if remaining<instance.config.dt-1e-8:return result
        speed=float(np.max(np.abs(hermite(q0,v0,plan.q1,plan.v1,remaining,21)[1])))
        if speed>cap+1e-7:
            instance.pending=None
            if len(events)<256:events.append(dict(time=float(robot.timestamp),event='acquire_remaining_stream_speed_rejected',peak_speed=speed))
            return Reachability(False,'acquire_remaining_stream_contact_speed_reserve')
        return result
    m.check_stream=MethodType(stream,m)
    previous_metrics=actor.metrics_report
    def metrics(instance):return dict(previous_metrics(),contact_speed_reserve=dict(
        binding=binding(Path(__file__).resolve().parents[2]),events=deepcopy(events),
        native_motor_and_measured_window_limits_unchanged=True))
    actor.metrics_report=MethodType(metrics,actor)
    return events
