"""Retain the original exact next-tick collision check after full path checks."""
from .block_reference_v197 import SmoothBlockEnvironment as OriginalBoundary, public_decision_scope


class SmoothBlockEnvironment(OriginalBoundary):
    def _issue(self, q, v, opening, evidence):
        try:
            with self._collision_scope():
                colliding = self.env.robot_model.collision(q, float(self.env.ledger.dt))
        except (ValueError, KeyError, IndexError):
            return self._stationary_existing_target('fresh_public_exact_next_reference_collision_unavailable')
        if colliding:
            return self._stationary_existing_target('exact_next_reference_collision_rejected')
        return super()._issue(q, v, opening, dict(evidence,
            exact_next_tick_original_collision_gate_passed=True,
            exact_next_tick_collision_uses_fresh_current_public_forecast=True))
