"""Current Cube controller bound to the structural execution interface.

The frozen coordinator, CEM, response model and collision qualification are
reused. Their joint_goal is explicitly a smoothed destination; the target
twists remain objectives. This entry does not turn them into velocity commands.
"""
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from ..cube_native_contact_feedback_live_20261104_v1.coordinator import NativeContactCoordinatorV1
from ..local_mpc_control_20261103.response_factory import build_planner
from ..generic_reference_mpc_20261104_v1.reference_mpc import qualify_first_action
from ..local_mpc_20261102.records import append, write_new, quantiles


def run(backend, executor, output, config, *, stop_requested=lambda: False):
    if executor.backend is not backend:
        raise ValueError("executor and controller must share the same backend")
    required = ("latest_robot", "latest_contact", "candidate_mode", "pilot_inference", "candidate_cache")
    if any(not hasattr(backend, name) for name in required):
        raise TypeError("current public Cube observation/contact protocol required")
    if config.get("require_public_goal_completion", False) is not False:
        raise ValueError("only original Native task.success may determine success")
    if executor.config.physics_dt != .05 or executor.config.control_ticks != 2:
        raise ValueError("the frozen CEM response requires two 50ms ticks; single-step control is a different model contract")
    root = Path(__file__).resolve().parents[2]
    output = Path(output)
    core, helper, provenance = build_planner(root, executor.limits, controller="coupled", action_space="joint",
        joint_goal_config={"lower": tuple(config.get("workspace_lower", [-.1, -.65, .755]))},
        terminal_config={"enabled": bool(config.get("terminal_goal_cost", True))},
        reference_frequency=executor.config.reference_frequency)
    if config.get("finger_response_fastpath", False):
        from ..local_mpc_control_20261103.finger_fastpath_integration_20261103 import enable_finger_fastpath
        provenance = enable_finger_fastpath(core.execution_model, provenance)
    coordinator = NativeContactCoordinatorV1(core, helper, backend,
        pi_symmetry_representative=config.get("candidate_pi_symmetry_representative", True))
    write_new(output / "controller_contract.json", dict(executor=executor.capabilities(), controller=provenance,
        planner_action_semantics="smoothed_joint_destination", velocity_reference_role="objective_only",
        Native_contact_feedback_used=True, Native_target_pose_used_by_policy=False,
        episode_success_source="original_PhysMani_Task.success_first_true_latched"))
    cycles, times, stop = 0, [], "budget_exhausted"
    maximum = int(config.get("public_grasp_max_decisions", 80))
    while cycles < maximum and backend.remaining_ticks() > 0:
        if backend.terminal():
            stop = "original_Native_task_success" if backend.native_task_judge_receipt()["native_task_success_latched"] else "Native_terminal_or_budget"
            break
        if stop_requested():
            stop = "administrative_stop"
            break
        started = time.perf_counter()
        robot, h = backend.latest_robot, deepcopy(executor.state())
        if abs(robot.timestamp - backend.clock()) > 1e-8:
            raise ValueError("fresh same-frame robot state required")
        decision = coordinator.decide(robot, h)
        action, qualification, rejected = decision.get("action"), None, None
        if action is None:
            coordinator.warm = None
            goal = robot.q.copy()
            opening = 0. if decision.get("keep_closed") is True or backend.latest_contact["held"] else h["gripper_target"]
            dispatched = dict(joint_goal=goal, opening=opening)
            dispatch_source = "actual_q_hold_keep_native_contact_request"
        elif decision.get("direct_membership_miss_open"):
            allowed, qualification = qualify_first_action(backend, dict(time=robot.timestamp, q=robot.q), action)
            goal, opening = np.asarray(action["joint_goal"]).copy(), 1.
            if not allowed:
                goal, opening, rejected = robot.q.copy(), h["gripper_target"], qualification["reason"]
            dispatched = dict(joint_goal=goal, opening=opening)
            dispatch_source = "actual_completed_close_nonmembership_open" if allowed else "rejected_native_miss_open_hold"
        else:
            goal, rejected, qualification = helper.qualify_action(backend, decision["state"], action, require_predicted_joint_goal=True)
            opening = float(action["opening"])
            if rejected:
                coordinator.warm = None
                goal, opening = robot.q.copy(), h["gripper_target"]
                dispatched = dict(joint_goal=goal, opening=opening)
                dispatch_source = "rejected_model_action_hold_actual_q_keep_request"
            else:
                if not np.array_equal(goal, np.asarray(action["joint_goal"])):
                    raise ValueError("selected model destination must be reused without a second IK solve")
                if backend.latest_contact["held"] and opening != 0.:
                    raise ValueError("visual/model failure cannot release Native held feedback")
                dispatched = deepcopy(action)
                dispatch_source = "exact_selected_modeled_q_and_opening"
        receipt = executor.execute_planner_action(dispatched, velocity_reference_role="objective_only",
            phase="Native_contact_" + decision["mode"], expected_time=robot.timestamp)
        append(output / "decisions.jsonl", dict(index=cycles, time=robot.timestamp, public_robot=robot,
            actuator_state_before=h, decision=decision, action_qualification=qualification,
            action_rejected_reason=rejected, executed_action=dispatched, dispatch_source=dispatch_source,
            executor=receipt, public_robot_after=backend.latest_robot, actuator_state_after=executor.state(),
            native_task_judge=backend.native_task_judge_receipt()))
        if receipt.unexpected_physics_ticks != 0:
            raise RuntimeError("execution advanced unexpected physical ticks")
        cycles += 1
        times.append(time.perf_counter() - started)
        if receipt.feedback.get("native_task_success") is True:
            stop = "original_Native_task_success"
            break
        if receipt.executed_ticks == 0:
            stop = receipt.truncated_reason or "no_actual_tick"
            break
        if cycles % 10 == 0:
            print(f"Cube interface check: {cycles} cycles, sim_time={backend.clock():.3f}", flush=True)
    judge = backend.native_task_judge_receipt()
    return dict(complete=stop != "administrative_stop", cycles=cycles, stop_reason=stop,
        episode_success=judge["native_task_success_latched"], native_task_judge=judge,
        Native_contact_feedback_used=True, Native_target_pose_used_by_policy=False,
        visual_success_gate=False, full_cycle_wall_seconds=quantiles(times),
        candidate_forward_calls=backend.pilot_inference.calls, formal_VAL_TEST=False,
        highspeed_grasp_improvement_claimed=False)
