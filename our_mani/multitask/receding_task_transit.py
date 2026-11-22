"""Share continuous renewal while retaining each task's waypoint geometry.

The task adapter supplies its original transit implementation. This wrapper
only renews its time-limited prefix from the confirmed moving boundary. It
does not replace task-specific height, phase, contact, or collision rules.
"""
from copy import deepcopy
from .continuous_transit import ConstraintContract

SCHEMA = 'r10_receding_transit_original_task_geometry_v1'


class RecedingTaskTransit:
    def __init__(self, manager, original, *, replan_seconds=.2):
        dt = float(manager.config.dt)
        if (isinstance(replan_seconds, bool) or not dt <= replan_seconds < original.config.segment_seconds
                or abs(replan_seconds/dt-round(replan_seconds/dt)) > 1e-8):
            raise ValueError('renew before endpoint on counted tick boundaries')
        self.manager, self.original = manager, original
        self.replan_seconds = float(replan_seconds)
        self.contract = ConstraintContract.from_manager(manager)
        self.renewals = []

    def __getattr__(self, name):
        return getattr(self.original, name)

    def try_approach(self, robot, remaining_seconds):
        m, old = self.manager, self.original
        self.contract.verify(m)
        plan, active, now = m.plan, old.active, float(robot.timestamp)
        due = (active is not None and plan is not None and not m.invalidated
            and m.plan_kind == 'approach_transit' and m.controller.goal is None
            and active['phase_index'] == m.controller.phase_index
            and not m.qualification_failed and not m.braking_active(now)
            and now >= plan.start+self.replan_seconds-1e-8)
        if not due:
            return old.try_approach(robot, remaining_seconds)
        boundary = m.continuity_state(robot)
        if boundary is None:
            return False
        original_deadline, old_id = active['deadline'], m.plan_candidate_id
        old.active = None
        m.pending = None
        admitted = old.try_approach(robot, remaining_seconds)
        self.renewals.append(dict(time=now, renewed=bool(admitted),
            previous_reference_id=old_id, previous_plan_end=float(plan.end),
            original_phase_deadline=float(original_deadline),
            new_phase_deadline=None if old.active is None else float(old.active['deadline']),
            start_reference_joints=boundary[0].tolist(), start_reference_velocity=boundary[1].tolist(),
            original_task_waypoint_and_motion_gates=True, additional_physics_ticks=0,
            phase_clock_reset=False))
        if not admitted:
            m.invalidate('receding_task_transit_no_fresh_complete_proof', now)
        return admitted

    def metrics(self):
        return dict(self.original.metrics(), receding_task_transit=dict(
            schema=SCHEMA, replan_seconds=self.replan_seconds,
            original_implementation=type(self.original).__module__+'.'+type(self.original).__name__,
            original_task_waypoint_geometry_preserved=True,
            constraint_contract=deepcopy(self.contract.__dict__), renewals=deepcopy(self.renewals)))


def install(actor, *, replan_seconds=.2):
    m = actor.continuous_reference_manager
    if m.confirmations or actor.controller.feedback_step_calls:
        raise ValueError('install before execution')
    m.approach_transit = RecedingTaskTransit(m, m.approach_transit, replan_seconds=replan_seconds)
    return m.approach_transit
