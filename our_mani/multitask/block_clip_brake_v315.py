"""Try a fully checked stop before a block reference would be actuator-clipped."""
from pathlib import Path
import numpy as np
from .block_held_support_v298 import SmoothBlockEnvironment as OriginalBoundary
from .block_held_support_v298 import public_decision_scope, measured_tracker_scope
from .block_held_support_v298 import binding as held_binding
from .evaluation import file_sha


def binding(root):
    held = held_binding(root)
    return dict(schema='r10_block_checked_brake_before_actuator_reference_clip_v315',
        held_policy=held, sources_sha256=dict(held['sources_sha256'], **{
            'our_mani/multitask/block_clip_brake_v315.py': file_sha(__file__)}),
        on_unexecutable_reference='original full actual6/reference4 Hermite brake and exact next-tick collision gate before existing explicit unqualified hold',
        original_actuator_velocity_clipping_and_native_parameters_unchanged=True,
        original_collision_and_global10s_unchanged=True,
        recursive_brake_retry=False, actual_motor_motion_guaranteed=False,
        formal_VAL_TEST=False)


class SmoothBlockEnvironment(OriginalBoundary):
    def __init__(self, env, context=None):
        super().__init__(env, context)
        self.clip_brake_active = False

    def _issue(self, q, v, opening, evidence):
        actual = np.asarray(self.env.arm.get_joint_positions(), float)
        would_clip = np.any(abs(q - actual) > self.env.robot_model.velocity_limits * self.env.ledger.dt + 1e-7)
        if would_clip and not self.clip_brake_active:
            self.clip_brake_active = True
            try:
                with self._collision_scope():
                    return self._brake(self.env._requested_opening,
                        'original_native_actuator_cannot_apply_proposal_without_velocity_clipping')
            finally:
                self.clip_brake_active = False
        return super()._issue(q, v, opening, evidence)

    def receipt(self):
        return dict(super().receipt(), block_clip_brake_v315=dict(
            original_actual6_reference4_collision_and_exact_next_tick_gates_retained=True,
            original_native_actuator_clipping_unchanged=True,
            raw_hold_if_no_fresh_original_gate_brake_is_not_relabelled_safe=True))
