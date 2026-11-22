"""Minimal original P1 mechanical loop with private exact FREE trajectory."""
from copy import deepcopy
import time,json
import numpy as np
from scipy.spatial.transform import Rotation
from ..local_mpc_20261102.worker import (ROOT,DiagnosticConstraints,GeometricCandidates,
    HoldMetric,plain,write_new,save_rgbd,append_ticks,append,quantiles)
from ..cube_privileged_contact_teacher_20261104_v1.teacher import CanonicalCubeTeacherPolicyView
from ..local_mpc_control_20261103.forecast_control import ForecastJointGoalCEMPlanner
from ..local_mpc_control_20261103.public_grasp import qualify_public_grasp_candidates
from .oracle import forecast_from_private_task,PrivateExactFutureExecutionBinding

class ImmediateExpiryTeacherCandidates(GeometricCandidates):
    def update(self, state):
        now = float(state["time"])
        expired = self.candidates and now > max(c["deadline"] for c in self.candidates) + 1e-8
        if not self.candidates or (expired and not state.get("held", False)):
            reason = "initial_private_teacher_geometry" if not self.candidates else "all_fixed_deadlines_expired_immediate_regeneration"
            self.generation += 1
            self.events.append(dict(time=now, generation=self.generation, reason=reason))
            object_rotation = Rotation.from_quat(state["object_quaternion"])
            down = Rotation.from_quat([0., 1., 0., 0.])
            self.candidates = []
            for yaw_index, yaw in enumerate((0., np.pi / 2, np.pi, 3 * np.pi / 2)):
                world_rotation = Rotation.from_rotvec([0., 0., yaw]) * down
                local_rotation = object_rotation.inv() * world_rotation
                for time_index, delay in enumerate((self.horizon * .625, self.horizon)):
                    self.candidates.append(dict(
                        id=f"geometry_g{self.generation}_yaw{yaw_index}_time{time_index}",
                        object_tcp_position=[0., 0., self.contact_offset],
                        object_tcp_quaternion=local_rotation.as_quat().tolist(),
                        deadline=now + delay,
                        approach_direction=[0., 0., 1.],
                        pregrasp_offset=self.pregrasp_offset,
                        opening=0.,
                        required_opening=1.,
                        required_width_m=.08,
                        allowed_relative_speed=.14,
                        provenance="private_TRAIN_geometric_downward_prior_in_canonical_public_I_frame_not_Transformer_or_public_qualification"))
        return [dict(c) for c in self.candidates]

def qualify_teacher_candidates(backend,state,candidates,helper,*,bound):
    # Original computation ABI names are not information-setting assertions.
    computational_mode='ATTACHED_PUBLIC' if state['held'] else 'FREE_GRU'
    qualified,receipt=qualify_public_grasp_candidates(backend,state,candidates,helper,
        mode=computational_mode,bound=bound)
    receipt=deepcopy(receipt)
    receipt.update(mode='PRIVATE_ACTUAL_ATTACHED_INTERNAL' if state['held'] else 'PRIVATE_EXACT_FUTURE_FREE',
        computation_ABI_mode_only=computational_mode,privileged_TRAIN_teacher_only=True,
        public_bridge_verified=False,GRU_used=False,private_oracle_used=not state['held'],
        Native_schedule_or_actual_held_are_policy_inputs=True,public_qualification_claimed=False)
    for case in receipt['cases']:
        case['target_source']=receipt['mode'];case['public_qualification_claimed']=False
    return qualified,receipt

def run_private_oracle_teacher(backend, executor, output, config, *, capture_rgbd=True, recorder=None):
    if (config.get("response_factory") is not None or config.get("action_space") is not None) and not config.get("use_joint_goal_planner", False):
        raise ValueError("versioned response/action options require use_joint_goal_planner")
    from ..local_mpc_20261102.batch_mpc import BatchMPCPlanner as MPCPlanner
    constraints = DiagnosticConstraints(lower=config.get("workspace_lower", [-.1, -.65, .76]))
    if config.get("terminal_goal_cost", False):
        from ..local_mpc_20261102.terminal_mpc import TerminalBatchMPCPlanner as MPCPlanner
    planner = MPCPlanner(constraint_cost=constraints)
    goal_helper = None
    controller_provenance = None
    if config.get("use_joint_goal_planner", False):
        factory_options = {}
        if config.get("response_factory") == "coupled_joint_20261103_v1":
            from ..local_mpc_control_20261103.response_factory import build_planner
            factory_options["action_space"] = config["action_space"]
        elif config.get("response_factory") is None:
            from ..local_mpc_control_20261103.planner_factory import build_planner
        else:
            raise ValueError("unknown explicitly versioned response factory")
        planner, goal_helper, controller_provenance = build_planner(
            ROOT, executor.limits, controller=config.get("response_controller", "simple"),
            joint_goal_config={"lower": tuple(config.get("workspace_lower", [-.1, -.65, .755]))},
            terminal_config={"enabled": bool(config.get("terminal_goal_cost", False))},
            reference_frequency=executor.config.reference_frequency, **factory_options)
        write_new(output / "controller_provenance.json", controller_provenance)
    geometry = ImmediateExpiryTeacherCandidates(horizon=config["horizon_seconds"])
    if goal_helper is None:raise ValueError('Exact-future teacher requires original joint-goal helper/model')
    core=planner
    previous_motion_mode=None
    metric = HoldMetric(config["hold_seconds"], config["lift_height_m"], config["max_hold_relative_drift_m"])
    initial = backend.read_state()
    metric.update(initial)
    warm, phase_state = None, None
    planning_times, full_times, ages = [], [], []
    native_any = bool(initial.get("native_success", False))
    iteration = 0
    last_generation = -1
    while not backend.terminal() and not native_any:
        if ((output.parent / "STOP").exists() or (output / "STOP").exists()
                or (ROOT / "runtime/framework/local_mpc_20261102/STOP").exists()):
            raise InterruptedError("new local MPC STOP requested")
        wall_start = time.perf_counter()
        state = backend.read_state()
        state["actuator_state"] = plain(executor.state())
        feedback_wall = time.perf_counter()
        state["phase_state"] = phase_state
        candidates = geometry.update(state)
        if geometry.generation != last_generation:
            warm = None
            last_generation = geometry.generation
        rgbd = save_rgbd(backend, output, iteration, recorder) if capture_rgbd else None
        planning_start = time.perf_counter()
        mode='PRIVATE_ACTUAL_ATTACHED' if state['held'] else 'PRIVATE_EXACT_FREE'
        if mode!=previous_motion_mode:warm=None
        previous_motion_mode=mode
        oracle_receipt=None;bound=None
        if state['held']:
            planner=core
        else:
            forecast,oracle_receipt=forecast_from_private_task(backend.task,state)
            bound=PrivateExactFutureExecutionBinding(core.execution_model,forecast,
                anchor_time=state['time'],oracle_receipt=oracle_receipt)
            planner=ForecastJointGoalCEMPlanner(config=core.config,constraint_cost=core.planner.constraint_cost,
                execution_model=bound,goal_helper=goal_helper,joint_config=core.planner.joint_config,
                terminal_config=core.terminal_config)
        candidates,qualification=qualify_teacher_candidates(backend,state,candidates,goal_helper,bound=bound)
        result = planner.plan(state, candidates, warm_start=warm)
        planning_times.append(time.perf_counter() - planning_start)
        action = result["action"]
        action_qualification = None
        if action is None:
            native_q, rejected = None, "no_feasible_candidate"
        elif goal_helper is not None:
            native_q, rejected, action_qualification = goal_helper.qualify_action(
                backend, state, action, require_predicted_joint_goal=True)
        else:
            native_q, rejected = constraints.qualify_action(backend, action)
        age = time.perf_counter() - feedback_wall
        ages.append(age)
        h_before = plain(executor.state()) if hasattr(executor, "state") else None
        if rejected:
            receipt = executor.execute_joint_target(state["q"], opening=state["requested_opening"],
                                                     phase="P1_safe_hold_" + rejected)
        else:
            receipt = executor.execute_joint_target(native_q, opening=action["opening"], phase="P1_local_mpc")
        after = backend.read_state()
        ticks = append_ticks(backend, output, metric)
        native_any |= bool(after.get("native_success", False)) or any(t.get("native_success", False) for t in ticks)
        duration = time.perf_counter() - wall_start
        full_times.append(duration)
        row = dict(index=iteration, information_setting="private_exact_future_TRAIN_teacher_only", before=state,
                   actuator_state_before=h_before, candidates=candidates, plan=result,
                   candidate_qualification=qualification, private_oracle=oracle_receipt,
                   private_oracle_binding=None if bound is None else bound.receipt(), action_rejected_reason=rejected,
                   action_qualification=action_qualification,
                   native_joint_goal=native_q,
                   requested_action=action, executor=receipt, after=after, rgbd=rgbd,
                   actuator_state_after=plain(executor.state()) if hasattr(executor, "state") else None,
                   requested_duration=config["control_dt"], actual_duration=after["time"] - state["time"],
                   observed_native_ticks=len(ticks), planner_wall_seconds=planning_times[-1],
                   feedback_age_at_submit_wall_seconds=age, full_cycle_wall_seconds=duration)
        append(output / "transitions.jsonl", row)
        print(json.dumps(plain(dict(stage="P1", cycle=iteration, time=after["time"],
                              phase=result["phase_state"], held=after["held"],
                              native_success=after.get("native_success", False),
                              planner_wall_seconds=planning_times[-1], full_cycle_wall_seconds=duration))), flush=True)
        full_times[-1] = time.perf_counter() - wall_start
        warm, phase_state = (None if rejected else result["warm_start"]), result["phase_state"]
        iteration += 1
        # Native task.success is the task judge. Hold stability is recorded
        # separately and must not cause additional motion after success.
        if native_any:
            break
        if not ticks:
            raise RuntimeError("executor returned without a physics tick before terminal")
    final = backend.read_state()
    return dict(stage="PRIVATE_TRAIN_EXACT_FUTURE_TEACHER_V2", complete=True,
                private_exact_future_oracle=True,public_policy_score=False,
                private_future_conditional_on_remaining_ungrasped=True, milestone_passed=bool(native_any),
                native_task_success=bool(native_any), task_success=bool(native_any),
                task_success_judge="Native_task.success_any_observed_tick_only",
                native_success_observed=bool(native_any), hold=metric.receipt(),
                hold_metric_is_task_success_gate=False, hold_metric_diagnostic_only=True,
                stop_after_first_native_success="end_of_current_executor_cycle_at_most_two_physics_ticks",
                cycles=iteration,
                initial=initial, final=final, geometric_candidate_events=geometry.events,
                planner_wall_seconds=quantiles(planning_times), full_cycle_wall_seconds=quantiles(full_times),
                feedback_age_at_submit_wall_seconds=quantiles(ages),
                cycle_deadline_miss_fraction=float(np.mean(np.asarray(full_times) > config["control_dt"])) if full_times else None,
                RGBD_every_cycle=capture_rgbd,
                learned_dynamics=config.get("response_controller") in ("learned", "coupled") and goal_helper is not None,
                controller_provenance=controller_provenance, Transformer_used=False,
                GRU_used=False, formal_evaluation=False,
                scenario_motion_events=backend.motion_events,
                timing_scope="full decision incl capture, dispatch and synchronous logs; background compression drained before result",
                dynamics_data="actual counted native transitions, not old HDF5 expert advice",
                native_grasp_semantics="original proximity-checked attachment; no new force attachment; not frictional stability proof")

def run_privileged_teacher(backend,executor,output,config):
    if type(backend)is not CanonicalCubeTeacherPolicyView or executor.backend is not backend:
        raise TypeError('Explicit original private canonical teacher view/executor required')
    if config.get('use_joint_goal_planner')is not True:
        raise ValueError('Original model/joint-goal planner required')
    return run_private_oracle_teacher(backend,executor,output,config,capture_rgbd=False)
