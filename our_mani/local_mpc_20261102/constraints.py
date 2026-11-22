"""P1 geometric screening and no-step Native qualification, not reachability claims."""
from __future__ import annotations

from copy import deepcopy
import numpy as np


class DiagnosticConstraints:
    """Cheap rollout bounds plus native IK/static collision probes before execution.

    These workspace bounds are conservative diagnostic controller constraints.
    They do not change task physics or replace robot collision detection. A
    feasible pose alone does not certify every intermediate joint trajectory.
    """

    def __init__(self, lower=(-.1, -.65, .76), upper=(.65, .65, 1.65)):
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)

    def __call__(self, state, action, next_state, candidate, absolute_time):
        if not candidate.get("native_pose_qualified", True):
            return float("inf")
        for p in (action["tcp_position"], next_state["tcp_position"]):
            p = np.asarray(p)
            if not np.isfinite(p).all() or np.any(p < self.lower) or np.any(p > self.upper):
                return float("inf")
        return 0.

    def batch(self, state, action, next_state, qualified, absolute_time):
        position, target = next_state["tcp_position"], action["tcp_position"]
        invalid = (~qualified) | np.any((position < self.lower) | (position > self.upper), axis=-1)
        invalid |= np.any((target < self.lower) | (target > self.upper), axis=-1)
        invalid |= ~np.isfinite(position).all(axis=-1) | ~np.isfinite(target).all(axis=-1)
        return np.where(invalid, np.inf, 0.)

    def qualify_candidates(self, backend, state, candidates):
        from .mpc import contact_target, validate_candidate, validate_state
        qualified, receipts = [], []
        phase = (state.get("phase_state") or {}).get("phase", "approach")
        for candidate in candidates:
            c = deepcopy(candidate)
            target = contact_target(validate_state(state), validate_candidate(c),
                                    max(float(state["time"]), float(c["deadline"])),
                                    pregrasp=phase == "approach")
            q = backend.solve_ik(target["position"], target["quaternion"])
            collision = bool(backend.probe_joint_collision(q)) if q is not None else None
            c["native_pose_qualified"] = q is not None and not collision
            receipts.append(dict(id=c["id"], time=state["time"],
                                 target_time=max(state["time"], c["deadline"]),
                                 target_position=target["position"],
                                 IK_available=q is not None, static_arm_collision=collision,
                                 qualified=c["native_pose_qualified"],
                                 whole_path_certified=False,
                                 scope="P1 truth diagnostic current robot IK and frozen static map"))
            qualified.append(c)
        return qualified, receipts

    def qualify_action(self, backend, action):
        p = np.asarray(action["tcp_position"])
        if np.any(p < self.lower) or np.any(p > self.upper):
            return None, "action_outside_diagnostic_workspace"
        q = backend.solve_ik(action["tcp_position"], action["tcp_quaternion"])
        if q is None:
            return None, "action_IK_unavailable"
        if backend.probe_joint_collision(q):
            return None, "action_arm_static_collision"
        current = backend.get_joint_positions()
        # A finite joint-space check is deliberately described as sampling,
        # not a continuous collision certificate. Only two bounded servo ticks
        # execute before the next observation and renewed qualification.
        for alpha in (.125, .25, .375, .5, .625, .75, .875):
            if backend.probe_joint_collision((1 - alpha) * np.asarray(current) + alpha * np.asarray(q)):
                return None, "sampled_joint_path_arm_static_collision"
        return q, None

