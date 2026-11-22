"""Fresh response inverse inside one current static-12mm CEM joint domain.

The progressive desired TCP poses are unchanged from the frozen model study.
Only the numerical inverse's admissible domain is now the actual production
static nominal +/-0.30rad, intersected with original joint bounds at every
step. Projecting a numerical iterate is not clipping a selected policy action.
No simulator or collision qualification is provided by this generator.
"""
from copy import deepcopy
import time
import numpy as np

from .hold_response_feedforward_prototype_20261103 import (
    ModelOnlyHoldResponse, hold_reference_schedule, _pose_residual, _row)
from .hold_orientation_guard_prototype_20261103 import _identity
from .joint_goal_cem import JointCEMConfig
from ..local_mpc_p2_20261103.action_adapter import DLSConfig

VERSION = 'fresh_response_inverse_static12_joint_domain_20261103_v68'
FINITE_DIFFERENCE_RAD = 1e-5


def inverse_in_static_domain(model, x, h, initial_q, static_q, position, quaternion):
    """Original fixed DLS numerics; one fixed static basis for all H8 steps."""
    cfg = DLSConfig()
    initial = np.asarray(initial_q, float)
    static = np.asarray(static_q, float)
    if (initial.shape != (7,) or static.shape != (7,) or
            not np.isfinite(initial).all() or not np.isfinite(static).all()):
        raise ValueError('Finite progressive initial guess and current static q7 required')
    radius = JointCEMConfig().max_perturbation_rad
    lower = np.maximum(model.execution.limits.lower, static - radius)
    upper = np.minimum(model.execution.limits.upper, static + radius)
    if np.any(lower > upper):
        raise ValueError('Nonempty original native/static nominal domain required')
    # Numerical initialization only; selected proposals are not clipped later.
    q = np.minimum(np.maximum(initial, lower), upper)
    best = None; diagnostic = None; trace = []
    for iteration in range(cfg.iterations):
        trials = np.broadcast_to(q, (15, 7)).copy()
        for j in range(7):
            trials[1 + 2*j, j] = min(upper[j], q[j] + FINITE_DIFFERENCE_RAD)
            trials[2 + 2*j, j] = max(lower[j], q[j] - FINITE_DIFFERENCE_RAD)
        pred = model.evaluate(x, h, trials)
        error = _pose_residual(pred, position, quaternion, cfg.rotation_weight)
        losses = np.sum(error * error, axis=1)
        for i in range(len(trials)):
            item = dict(loss=float(losses[i]), joint_goal=trials[i].copy(),
                        prediction=_row(pred, i, len(trials)))
            if diagnostic is None or item['loss'] < diagnostic['loss']:
                diagnostic = item
            if pred['feasible'][i] and (best is None or item['loss'] < best['loss']):
                best = item
        trace.append(dict(iteration=iteration, current_loss=float(losses[0]),
                          current_model_feasible=bool(pred['feasible'][0]),
                          feasible_trials=int(pred['feasible'].sum())))
        jac = np.zeros((6, 7))
        for j in range(7):
            width = trials[1 + 2*j, j] - trials[2 + 2*j, j]
            if width:
                jac[:, j] = (error[1 + 2*j] - error[2 + 2*j]) / width
        if np.linalg.norm(jac) < 1e-12:
            break
        step = -jac.T @ np.linalg.solve(jac @ jac.T + cfg.damping**2 * np.eye(6), error[0])
        length = float(np.linalg.norm(step))
        if length > cfg.max_joint_step:
            step *= cfg.max_joint_step / length
        qnext = np.minimum(np.maximum(q + step, lower), upper)
        if np.linalg.norm(qnext - q) < 1e-12:
            break
        q = qnext
    return dict(selected=best if best is not None else diagnostic,
                model_feasible_proposal=best is not None, iterations=trace,
                numerical_constants=dict(iterations=cfg.iterations, damping=cfg.damping,
                    rotation_weight=cfg.rotation_weight, max_joint_step=cfg.max_joint_step,
                    max_deviation_from_static_IK_rad=radius, finite_difference_rad=FINITE_DIFFERENCE_RAD),
                lower=lower.copy(), upper=upper.copy(),
                selected_action_clipped=False, Native_collision_qualified=False)


def generate_response_seed(execution, goal_helper, actual, static_nominal_q):
    """Generate one full unshifted sequence from this current state+h only."""
    started = time.perf_counter()
    before = _identity(actual)
    static = np.asarray(static_nominal_q, float)
    if (static.shape != (7,) or not np.isfinite(static).all() or
            np.any(static < execution.limits.lower) or np.any(static > execution.limits.upper)):
        raise ValueError('Current in-bounds production static12 nominal q7 required')
    static = static.copy()
    schedule = hold_reference_schedule(actual, progressive=True)
    ik = goal_helper.adapter.solve(np.broadcast_to(np.asarray(actual['q'], float), (8, 7)).copy(),
                                   schedule['positions'], schedule['quaternions'])
    model = ModelOnlyHoldResponse(execution, goal_helper, actual)
    x, h = model.initialize(actual)
    ledger = []; prefix = True
    for k in range(8):
        inverse = inverse_in_static_domain(model, x, h, ik['joint_goal'][k], static,
                                           schedule['positions'][k], schedule['quaternions'][k])
        chosen = inverse['selected']
        if chosen is None:
            raise ValueError('Fixed inverse supplied no finite diagnostic or feasible trial')
        pred = chosen['prediction']; q = chosen['joint_goal']
        feasible = bool(pred['feasible'][0]); prefix &= feasible
        x, h = pred['x'], pred['h']
        domain = bool(np.all(abs(q - static) <= .30 + 1e-12))
        if not domain:
            raise ValueError('Inverse escaped original current static CEM domain')
        ledger.append(dict(step=k + 1, reference_TCP=schedule['positions'][k].copy(),
            reference_R=schedule['quaternions'][k].copy(), progressive_IK_initial_q=ik['joint_goal'][k].copy(),
            proposed_q=q.copy(), model_feasible=feasible, prefix_model_feasible=prefix,
            predicted_TCP=x['tcp_position'][0].copy(), predicted_object=x['object_position'][0].copy(),
            predicted_q=x['q'][0].copy(), predicted_dq=x['dq'][0].copy(),
            reference_tracking_error_m=float(np.linalg.norm(x['tcp_position'][0] - schedule['positions'][k])),
            world_object_goal_distance_m=float(np.linalg.norm(x['object_position'][0] - schedule['global_object_goal'])),
            micro_angles_rad=pred['micro_angles_rad'][0].copy(),
            q_goal_and_actual_bounds=bool(pred['q_bounds'][0] & pred['actual_q_bounds'][0]),
            actual_micro_velocity_bounds=bool(pred['actual_micro_velocity_bounds'][0]),
            static_nominal_perturbation_bound=domain, inverse_iterations=inverse['iterations'],
            Native_collision_qualified=False))
    if _identity(actual) != before or not np.array_equal(static, np.asarray(static_nominal_q)):
        raise ValueError('Actual feedback/h or static nominal changed during seed generation')
    return dict(version=VERSION, feedback_sha256=before, feedback_time=float(actual['time']),
        static_nominal_q=static, schedule=schedule, steps=ledger,
        model_feasible_horizon=prefix, proposed_q_sequence=[step['proposed_q'] for step in ledger] if prefix else None,
        model_calls=dict(batch_calls=model.model_batch_calls, macro_rows=model.modeled_macro_rows,
                         micro_rows=model.modeled_micro_rows),
        generation_wall_seconds=time.perf_counter() - started,
        initialized_from_current_actual_feedback=True, history_warm_used=False, sequence_shift_steps=0,
        Native_collision_qualified=False, virtual_h_reused_as_next_real_feedback=False,
        desired_schedule_only_not_global_cost_changed=True, selected_action_clipped=False)
