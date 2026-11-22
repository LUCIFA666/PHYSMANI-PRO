"""Use the executor's reference bound for every continuation and brake proof."""
from copy import deepcopy
from dataclasses import replace
from types import MethodType
import numpy as np
from ..intercept_planner import JointTrajectory
from .continuous_transit import ConstraintContract
from .reference_stop_guard_v274 import NextReferenceStopGuard, duration_grid, offset_gate
from .evaluation import digest

SCHEMA = 'r10_single_reference_constraint_authority_v1'


class ConsistentStopGuard(NextReferenceStopGuard):
    def reference_gate(self):
        return getattr(self.manager, 'reference_reserve_gate', self.manager.native_gate)

    def durations(self, velocities, available, extra=()):
        limit = np.asarray(self.reference_gate().model.acceleration_limits, float)
        lower = max(float(np.max(np.abs(v)/limit)) for v in velocities)
        return duration_grid(velocities, self.manager.native_gate.model.acceleration_limits,
            available, (*extra, *(lower*f for f in (1.02, 1.25, 1.5))))

    def next_stop(self, robot, pending):
        m = self.manager; self.constraint_contract.verify(m)
        dt, now = float(m.config.dt), float(robot.timestamp)
        q, v = np.asarray(pending['reference_to_joints']), np.asarray(pending['reference_to_velocity'])
        future = replace(robot, timestamp=now+dt, joints=q, joint_velocities=v)
        reference_gate = offset_gate(self.reference_gate(), dt)
        trials = []
        for duration in self.durations((v,), self.available(now)-dt):
            end, zero = q+.5*v*duration, np.zeros_like(v)
            reference = reference_gate.check_joints(future, end, zero, duration)
            actual = m.native_gate.check_joints(robot, end, zero, dt+duration) if reference.feasible else None
            trials.append(dict(duration=duration, offset_reference_reason=reference.reason,
                current_actual_to_same_stop_reason=None if actual is None else actual.reason))
            if reference.feasible and actual is not None and actual.feasible:
                return JointTrajectory(now+dt, now+dt+duration, q.copy(), v.copy(), end, zero), dict(
                    trials=trials, reference_collision_time_offset=dt,
                    offset_reference_full_gate=deepcopy(reference.evidence), actual_full_gate=deepcopy(actual.evidence),
                    reference_acceleration_limits=list(self.constraint_contract.reference_acceleration),
                    actual_acceleration_limits=list(self.constraint_contract.actual_acceleration),
                    actual_proof_is_a_full_Hermite_from_current_state_not_a_motor_tracking_claim=True)
        return None, dict(trials=trials, reference_collision_time_offset=dt)

    def make_brake(self, robot, remaining_seconds, reason):
        m = self.manager; self.constraint_contract.verify(m)
        boundary = m.continuity_state(robot)
        if boundary is None:
            return False
        q0, v0, basis = boundary; now = float(robot.timestamp)
        available, extra = self.available(now, remaining_seconds), []
        if self.confirmed_stop is not None:
            plan = self.confirmed_stop['plan']
            if plan.start-1e-8 <= now < plan.end-1e-8:
                q, v = plan.at(now)
                if np.allclose(q0,q,rtol=0,atol=1e-8) and np.allclose(v0,v,rtol=0,atol=1e-8):
                    extra.append(plan.end-now)
        trials = []
        for duration in self.durations((v0, robot.joint_velocities), available, extra):
            phase = (m.controller.library.phase_order[m.controller.phase_index]
                if m.controller.phase_index < len(m.controller.library.phase_order) else 'verify')
            intent = m.controller.goal if m.controller.goal is not None else m.retired_brake_goal
            same = intent is not None and intent.phase == phase
            m.controller.generator.bind_candidate_scene(m.current_state, m.current_forecast, phase,
                intent.reference_id if same else '', intent.manipulated_ids if same else (), now+duration)
            end, zero = q0+.5*v0*duration, np.zeros_like(v0)
            reference = self.reference_gate().check_joints(replace(robot,joints=q0,joint_velocities=v0),end,zero,duration)
            actual = m.native_gate.check_joints(robot,end,zero,duration)
            trials.append(dict(duration=duration,reference_reason=reference.reason,actual_reason=actual.reason))
            if not (reference.feasible and actual.feasible):
                continue
            m.plan = JointTrajectory(now,now+duration,q0.copy(),v0.copy(),end,zero)
            m.plan_kind = 'braking'
            m.plan_candidate_id = 'consistent_brake:'+digest(dict(time=now,q=q0.tolist(),v=v0.tolist(),duration=duration))[:16]
            m.invalidated = False
            row = dict(time=now,event='consistent_reference_and_actual_brake_staged',reason=reason,
                boundary_basis=basis,trials=trials,
                reference_acceleration_limits=list(self.constraint_contract.reference_acceleration),
                actual_acceleration_limits=list(self.constraint_contract.actual_acceleration),
                reference_full_proof=deepcopy(reference.evidence),actual_full_proof=deepcopy(actual.evidence),
                phase_clock_reset=False,total_budget_extended=False,cached_certificate_reused_without_revalidation=False)
            m.events.append(row); self.record(**deepcopy(row))
            return True
        self.record(time=now,event='no_consistent_reference_and_actual_brake',reason=reason,trials=trials,
            unsafe_hold_not_relabelled_acceleration_limited=True)
        m.events.append(dict(time=now,event='no_feasible_continuous_brake',reason=reason,trials=trials,
            acceleration_limited_stop_verified=False))
        return False


def install(actor):
    m, c = actor.continuous_reference_manager, actor.controller
    if m.confirmations or c.feedback_step_calls:
        raise ValueError('bind reference authority before execution')
    function = getattr(m.make_brake, '__func__', m.make_brake)
    guards = [cell.cell_contents for cell in (getattr(function, '__closure__', None) or ())
        if isinstance(cell.cell_contents, NextReferenceStopGuard)]
    if len(guards) > 1:
        raise ValueError('ambiguous existing stopping service')
    if guards:
        guard = guards[0]
        # The original actor-local stream/confirmation wrappers retain this
        # service object. Replace its planning operations in place.
        for name in ('reference_gate','durations','next_stop','make_brake'):
            setattr(guard,name,MethodType(getattr(ConsistentStopGuard,name),guard))
    else:
        guard = ConsistentStopGuard(m)
        guard.install_stream(); guard.install_confirmation()
        m.make_brake = MethodType(lambda instance,*a,**kw:guard.make_brake(*a,**kw),m)
    guard.constraint_contract = ConstraintContract.from_manager(m)
    previous = actor.metrics_report
    def metrics(instance):
        return dict(previous(), reference_constraint_authority=dict(schema=SCHEMA,
            contract=deepcopy(guard.constraint_contract.__dict__),
            existing_stopping_service_reused=bool(guards), events=deepcopy(guard.events),
            nominal_reference_actual_motor_limits_distinguished=True,
            actual_native_motor_parameters_changed=False))
    actor.metrics_report = MethodType(metrics,actor)
    return guard
