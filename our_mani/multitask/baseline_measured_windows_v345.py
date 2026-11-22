"""Measure the common actual6 limit independently of unused baseline planning metadata.

The original legacy PhysMani environment leaves its Python reachability model
at the12 default. V6 RML/direct commands do not use that acceleration array.
This observer keeps it unchanged and applies the declared common6 criterion
only to measured adjacent50ms joint windows. No motor/model setting changes.
"""
from copy import deepcopy
import numpy as np
from .native_window_observer_v191 import CountedWindowObserver as OriginalObserver


class CountedWindowObserver(OriginalObserver):
    def __init__(self, env):
        self.original_planning_acceleration = np.asarray(env.robot_model.acceleration_limits, float).copy()
        if not np.isfinite(self.original_planning_acceleration).all() or np.any(self.original_planning_acceleration <= 0):
            raise ValueError('finite original baseline planning metadata required')
        super().__init__(env)

    def capture(self, before, q0, q1, phase):
        env, dt = self.env, float(self.env.ledger.dt)
        if (env.ledger.ticks != before + 1 or before != len(self.events) or dt != .05
                or q0.shape != q1.shape or not np.isfinite(np.r_[q0, q1]).all()
                or not np.array_equal(env.robot_model.acceleration_limits, self.original_planning_acceleration)):
            raise ValueError('baseline counted window or original planning metadata changed')
        velocity = (q1 - q0) / dt
        acceleration = None if self.previous is None else (velocity - self.previous) / dt
        self.events.append(dict(tick_before=before, tick_after=int(env.ledger.ticks), phase=phase,
            actual_issued_at=before * dt, reference_target_at=(before + 1) * dt,
            measured_q_before=q0.tolist(), measured_q_after=q1.tolist(),
            backward_window_velocity=velocity.tolist(),
            backward_window_acceleration=None if acceleration is None else acceleration.tolist(),
            position_passed=bool(np.all(q1 >= env.robot_model.lower - 1e-7) and np.all(q1 <= env.robot_model.upper + 1e-7)),
            speed_passed=bool(np.all(abs(velocity) <= env.robot_model.velocity_limits + 1e-6)),
            acceleration_passed=None if acceleration is None else bool(np.all(abs(acceleration) <= 6. + 1e-6)),
            actual_window_acceleration_criterion_rad_s2=6., diagnostic_only=True, native_parameters_changed=False))
        self.previous = velocity.copy()

    def receipt(self):
        return dict(super().receipt(), schema='r10_original_baseline_measured6_independent_of_unused_plan_model_v345',
            original_planning_model_acceleration_limits=self.original_planning_acceleration.tolist(),
            original_planning_model_acceleration_unchanged=True,
            actual_window_acceleration_criterion_rad_s2=6.,
            planning_metadata_used_as_actual_acceleration_cap=False)
