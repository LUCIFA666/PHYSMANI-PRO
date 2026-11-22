"""Fresh counted safety observations and dense dual-proof stops after timeout."""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import numpy as np
from ..intercept_planner import JointTrajectory
from .block_clip_brake_v315 import SmoothBlockEnvironment as OriginalBoundary
from .block_held_support_v298 import measured_tracker_scope as original_tracker_scope
from .block_full_budget_v176 import finish_counted_budget as original_finish
from .evaluation import file_sha

SOURCES=('our_mani/multitask/block_fresh_tail_v448.py','our_mani/multitask/block_clip_brake_v315.py',
    'our_mani/multitask/block_reference_v197.py','our_mani/multitask/block_full_budget_v176.py')


def binding(root):
    return dict(schema='r10_block_fresh_counted_tail_dense_dual_stop_v448',
        sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},
        after_original_policy_failure='brake only; no task action retry or phase-clock reset',
        tail_observations='one shared current RGBD/proprioception frame per previously unobserved counted safety tick',
        original_native6_reference4_complete_collision_and_exact_next_tick_gates=True,
        original_global10s_and_failure_records_preserved=True,extra_uncounted_ticks=0,
        additional_tail_camera_access_disclosed=True,TEST_read=False,formal_VAL_TEST=False)


class SmoothBlockEnvironment(OriginalBoundary):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.tail_tracker=None;self.tail_frames=[];self.dense_brakes=[]

    def _brake(self,opening,reason):
        now,dt=float(self.env.time),float(self.env.ledger.dt);robot=self.public_robot
        if robot is None or abs(float(robot.timestamp)-now)>1e-8:
            return self._stationary_existing_target('fresh_actual_state_required_for_dense_block_brake')
        actual_gate,reference_gate=self._gates()
        reference=replace(robot,joints=self.reference_q,joint_velocities=self.reference_v)
        candidates=[]
        if self.backup is not None and self.backup.start-1e-8<=now<self.backup.end-1e-8:
            q,v=self.backup.at(now)
            if np.allclose(q,self.reference_q,rtol=0,atol=1e-6) and np.allclose(v,self.reference_v,rtol=0,atol=1e-6):
                candidates.append(('remaining_cached_stop',self.backup))
        lower=max(float(np.max(abs(self.reference_v)/4.)),float(np.max(abs(robot.joint_velocities)/6.)))
        values=sorted({.05,.1,.15,.2,.25,.3,.4,.5,.6,.8,1.1,*(lower*f for f in (1.02,1.25,1.5))})
        for duration in values:
            if duration<dt-1e-8 or now+duration>10.+1e-8 or duration>1.1+1e-8:continue
            end=self.reference_q+.5*self.reference_v*duration
            candidates.append(('fresh_dense_stop',JointTrajectory(now,now+duration,self.reference_q.copy(),
                self.reference_v.copy(),end,np.zeros_like(end))))
        trials=[]
        for kind,plan in candidates:
            duration=float(plan.end-now)
            actual=actual_gate.check_joints(robot,plan.q1,plan.v1,duration)
            checked=reference_gate.check_joints(reference,plan.q1,plan.v1,duration)
            trials.append(dict(kind=kind,duration=duration,actual=actual.reason,reference=checked.reason))
            if not (actual.feasible and checked.feasible):continue
            self.backup=plan;q,v=plan.at(min(now+dt,plan.end))
            self.dense_brakes.append(dict(time=now,kind=kind,trials=trials,current_robot_timestamp=float(robot.timestamp)))
            return self._issue(q,v,opening,dict(kind='fresh_actual6_reference4_dense_continuous_brake',
                reason=reason,actual_gate=actual.reason,reference_gate=checked.reason,full_gate_trials=trials))
        self.dense_brakes.append(dict(time=now,admitted=False,trials=trials))
        return self._stationary_existing_target('no_fresh_actual_reference_dense_block_brake')

    def receipt(self):
        return dict(super().receipt(),additional_camera_queries=len(self.tail_frames),
            additional_camera_query_count_unit='shared aligned four-view observation calls in previously unobserved counted safety ticks',
            additional_uncounted_physics_ticks=0,
            block_fresh_tail_v448=dict(binding=binding(Path(__file__).resolve().parents[2]),
            current_tail_observations=list(self.tail_frames),dense_brakes=list(self.dense_brakes)))


@contextmanager
def measured_tracker_scope(evaluator,boundary):
    with original_tracker_scope(evaluator,boundary):
        previous=evaluator.RedBoxTracker
        def create(*args,**kwargs):
            tracker=previous(*args,**kwargs);boundary.tail_tracker=tracker;return tracker
        evaluator.RedBoxTracker=create
        try:yield
        finally:evaluator.RedBoxTracker=previous


def refresh_tail_frame(boundary):
    now=float(boundary.env.time);robot=boundary.public_robot
    if robot is not None and abs(float(robot.timestamp)-now)<1e-8:return False
    if boundary.tail_tracker is None:raise ValueError('same original measured tracker required for safety tail')
    observation,robot=boundary.observe()
    try:body,metadata=boundary.tail_tracker.observe(observation)
    except ValueError:body=None
    boundary.public_body=body
    dynamics=boundary.local_dynamics
    if body is not None and dynamics is not None and hasattr(dynamics,'observe'):
        dynamics.observe([body])
    boundary.tail_frames.append(dict(time=now,current_RGBD_body_available=body is not None,
        counted_tick=int(boundary.ledger.ticks),shared_existing_tracker=True,task_action_retried=False,
        additional_physics_ticks=0))
    return True


def finish_counted_budget(boundary,record):
    previous=boundary.hold_tick
    def hold():
        refresh_tail_frame(boundary)
        return previous()
    boundary.hold_tick=hold
    try:return original_finish(boundary,record)
    finally:boundary.hold_tick=previous
