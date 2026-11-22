"""Read-only actual counted-window motion measurements for the original baseline."""
from copy import deepcopy
from pathlib import Path
from types import MethodType
import numpy as np
from scripts.evaluate_multitask_case_r10 import NativeBackend
from .evaluation import file_sha

SCHEMA = 'r10_baseline_actual_counted_joint_window_observer_v191'


def binding(root):
    return dict(schema=SCHEMA,
        sources_sha256={'our_mani/multitask/native_window_observer_v191.py':file_sha(__file__)},
        read_stages=['before_original_counted_tick','after_original_counted_tick'],
        actual_window_acceleration_cap=6., native_motor_or_policy_parameters_changed=False,
        baseline_checkpoint_policy_RML_unchanged=True, additional_camera_or_physics_calls=False,
        diagnostic_values_supplied_to_policy=False)


class CountedWindowObserver:
    def __init__(self, env):
        if env.ledger.ticks != 0: raise ValueError('attach after reset, before original first counted tick')
        self.env, self.events, self.previous = env, [], None
        original_tick, original_receipt = env._tick, env.alignment_receipt
        def tick(instance, phase):
            before = int(instance.ledger.ticks)
            q0 = np.asarray(instance.arm.get_joint_positions(), float)
            original_tick(phase)
            q1 = np.asarray(instance.arm.get_joint_positions(), float)
            self.capture(before, q0, q1, phase)
        def receipt(instance):
            result = original_receipt()
            if len(self.events) != instance.ledger.ticks:
                raise ValueError('every original counted tick needs exactly one observed window')
            return dict(result, all_stage_counted_joint_motion_v191=self.receipt())
        env._tick = MethodType(tick, env)
        env.alignment_receipt = MethodType(receipt, env)

    def capture(self, before, q0, q1, phase):
        env, dt = self.env, float(self.env.ledger.dt)
        if (env.ledger.ticks != before+1 or before != len(self.events) or dt != .05
                or q0.shape != q1.shape or not np.isfinite(np.r_[q0,q1]).all()
                or not np.allclose(env.robot_model.acceleration_limits, 6.)):
            raise ValueError('original one-tick finite joint state and six-unit protocol required')
        velocity = (q1-q0)/dt
        acceleration = None if self.previous is None else (velocity-self.previous)/dt
        self.events.append(dict(tick_before=before, tick_after=int(env.ledger.ticks), phase=phase,
            actual_issued_at=before*dt, reference_target_at=(before+1)*dt,
            measured_q_before=q0.tolist(), measured_q_after=q1.tolist(),
            backward_window_velocity=velocity.tolist(), backward_window_acceleration=None if acceleration is None else acceleration.tolist(),
            position_passed=bool(np.all(q1>=env.robot_model.lower-1e-7) and np.all(q1<=env.robot_model.upper+1e-7)),
            speed_passed=bool(np.all(abs(velocity)<=env.robot_model.velocity_limits+1e-6)),
            acceleration_passed=None if acceleration is None else bool(np.all(abs(acceleration)<=6.+1e-6)),
            diagnostic_only=True, native_parameters_changed=False))
        self.previous = velocity.copy()

    def receipt(self):
        violations = [e['tick_after'] for e in self.events if not e['position_passed'] or not e['speed_passed'] or e['acceleration_passed'] is False]
        return dict(schema=SCHEMA, events=deepcopy(self.events), violations=violations,
            all_stage_actual_window_checks_passed=not violations,
            peak_actual_window_acceleration=max([abs(x) for e in self.events for x in e['backward_window_acceleration'] or []],default=0.),
            extra_physics_ticks=0, extra_camera_queries=0, policy_or_physics_modified=False,
            observed_window_checks_are_not_instantaneous_peak_guarantees=True)


class BaselineWindowBackend(NativeBackend):
    def environment(self, prepared):
        if prepared['args'].method != 'physmani': raise ValueError('original PhysMani baseline observer only')
        expected = binding(Path(__file__).resolve().parents[2])
        if prepared['config'].get('baseline_actual_window_observer_v191') != expected:
            raise ValueError('actual baseline observation profile must be frozen')
        env, descriptions = super().environment(prepared)
        try: self.window_observer = CountedWindowObserver(env)
        except BaseException:
            env.close()
            raise
        return env, descriptions
