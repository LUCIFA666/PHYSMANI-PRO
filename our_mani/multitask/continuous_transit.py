"""Receding transit segments using the actual executor's constraint authority.

Only TRAIN command geometry is reused. A renewed segment is a new, complete
actual-state and reference-state proof, not an extension of the old deadline.
The existing counted dispatcher remains the sole owner of physical stepping.
"""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np

from .bounded_task_transit import BoundedApproach, SCHEMA as TRANSIT_SCHEMA

SCHEMA = 'r10_continuous_transit_constraint_contract_v1'


@dataclass(frozen=True)
class ConstraintContract:
    actual_acceleration: tuple
    reference_acceleration: tuple
    actual_velocity: tuple
    reference_velocity: tuple
    dt: float

    @classmethod
    def from_manager(cls, manager):
        actual = manager.native_gate.model
        reference = getattr(manager, 'reference_reserve_gate', manager.native_gate).model
        arrays = [np.asarray(x, float) for x in (
            actual.acceleration_limits, reference.acceleration_limits,
            actual.velocity_limits, reference.velocity_limits)]
        if (any(x.shape != arrays[0].shape or x.ndim != 1 or not len(x)
                or not np.isfinite(x).all() or np.any(x <= 0) for x in arrays)
                or np.any(arrays[1] > arrays[0] + 1e-10)
                or np.any(arrays[3] > arrays[2] + 1e-10)
                or not np.isclose(manager.config.dt, .05)):
            raise ValueError('inconsistent actual/reference executor constraints')
        return cls(*(tuple(x.tolist()) for x in arrays), float(manager.config.dt))

    def verify(self, manager):
        if ConstraintContract.from_manager(manager) != self:
            raise ValueError('executor constraint authority changed after binding')


class ContinuousTransit(BoundedApproach):
    """Renew an admissible moving prefix before its old endpoint expires.

The original prefix implementation brakes when motor lag makes reaching its
old endpoint at its old time infeasible. Replanning from the confirmed issued
boundary gives the actual state a fresh full duration without stopping first.
If no fresh segment is feasible, the existing checked braking path takes over.
"""
    def __init__(self, manager, raw=None, *, phases, replan_seconds=.2):
        if (isinstance(replan_seconds, bool) or not np.isfinite(replan_seconds)
                or not .05 <= replan_seconds <= .5
                or not np.isclose(replan_seconds / manager.config.dt,
                                  round(replan_seconds / manager.config.dt))):
            raise ValueError('replanning period must be an integer counted tick interval')
        super().__init__(manager, raw, phases=phases)
        if replan_seconds >= self.config.segment_seconds:
            raise ValueError('renew moving segments before their endpoint')
        self.contract = ConstraintContract.from_manager(manager)
        self.replan_seconds = float(replan_seconds)
        self.renewals = []

    def try_approach(self, robot, remaining_seconds):
        m, c = self.manager, self.manager.controller
        self.contract.verify(m)
        active, plan = self.active, m.plan
        now = float(robot.timestamp)
        can_renew = (active is not None and plan is not None and not m.invalidated
            and m.plan_kind == 'approach_transit' and c.goal is None
            and active['phase_index'] == c.phase_index
            and not m.qualification_failed and not m.braking_active(now)
            and now >= plan.start + self.replan_seconds - 1e-8)
        if not can_renew:
            return super().try_approach(robot, remaining_seconds)
        # Keep the original per-phase deadline in self.deadlines. No goal,
        # phase, source, gripper intent, or simulation time is changed here.
        boundary = m.continuity_state(robot)
        if boundary is None:
            return False
        old_id, deadline = m.plan_candidate_id, active['deadline']
        self.active = None
        m.pending = None
        admitted = super().try_approach(robot, remaining_seconds)
        self.renewals.append(dict(time=now, previous_reference_id=old_id,
            previous_plan_end=float(plan.end), renewed=bool(admitted),
            new_reference_id=m.plan_candidate_id if admitted else None,
            start_reference_joints=boundary[0].tolist(),
            start_reference_velocity=boundary[1].tolist(),
            original_phase_deadline=float(deadline),
            new_phase_deadline=None if self.active is None else float(self.active['deadline']),
            complete_actual_and_reference_proofs_required=True,
            physical_ticks_added=0, phase_clock_reset=False))
        if not admitted:
            # The old trajectory is never resumed after a failed fresh proof.
            # The caller retains its existing fresh collision-checked brake.
            m.invalidate('continuous_transit_renewal_has_no_feasible_fresh_segment', now)
        return admitted

    def metrics(self):
        return dict(super().metrics(), continuous_transit=dict(
            schema=SCHEMA, replan_seconds=self.replan_seconds,
            constraint_contract=deepcopy(self.contract.__dict__),
            renewals=deepcopy(self.renewals), measured_motor_guarantee=False))


def install_transit(actor, *, phases, replan_seconds=.2):
    manager = actor.continuous_reference_manager
    if manager.confirmations or actor.controller.feedback_step_calls:
        raise ValueError('install continuous transit before execution')
    raw = deepcopy(manager.approach_transit.normalized)
    raw['version'] = TRANSIT_SCHEMA
    manager.approach_transit = ContinuousTransit(manager, raw,
        phases=phases, replan_seconds=replan_seconds)
    return manager.approach_transit
