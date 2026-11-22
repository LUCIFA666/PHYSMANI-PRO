"""Serial P0/P1 Native worker. No old TRAIN/VAL/TEST owner is resumed."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
from pathlib import Path
import time
import traceback

import numpy as np

from .candidates import GeometricCandidates
from .constraints import DiagnosticConstraints
from .records import (ROOT, HoldMetric, append, dependency_receipt, freeze_sources,
                      plain, quantiles, utc, verify_sources, write_new, archive_sources)
from .recording import FrameRecorder
from .records import sha


CONFIG = ROOT / "config/framework/local_mpc_20261102_v1.json"
FROZEN_SHA = "1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30"


def environment_proof():
    path = ROOT / "scripts/run_frozen_system_environment_v820_r10.py"
    spec = importlib.util.spec_from_file_location("local_mpc_frozen_os", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.require_environment(module.MANIFEST, FROZEN_SHA)


def make_backend(config, scene, *, budget_ticks=None):
    from .native import MovingBlockScenario, TruthNativeBackend
    velocity = config["uniform_velocity"] if scene != "static" else [0., 0., 0.]
    scenario = MovingBlockScenario(
        name=scene, initial_position=config["initial_target_position"],
        initial_velocity=velocity,
        switch_tick=config["speed_change_at_tick"] if scene == "speed_change" else None,
        changed_velocity=config["changed_velocity"] if scene == "speed_change" else None)
    random.seed(config["seed"])
    return TruthNativeBackend(config["calibration_train_episode"], scenario,
                              budget_ticks=budget_ticks or round(config["simulation_seconds"] / config["physics_dt"]),
                              seed=config["seed"], gripper_velocity=config["gripper_velocity"],
                              preserve_native_height=config.get("preserve_native_height", False))


def native_source_snapshot(config):
    from ..multitask.specs import NATIVE_COMMON_SOURCE_SHA256
    expected = dict(NATIVE_COMMON_SOURCE_SHA256)
    expected["third_party/RLBench/rlbench/tasks/" + config["task"] + ".py"] = config["native_task_source_sha256"]
    expected["third_party/RLBench/rlbench/task_ttms/" + config["task"] + ".ttm"] = config["native_task_scene_sha256"]
    for relative, digest in expected.items():
        if sha(Path(config["native_repo"]) / relative) != digest:
            raise RuntimeError("unchanged native task/common source required: " + relative)
    return expected


def make_executor(backend, config):
    from .executor import ExecutionConfig, JointLimits, ShortHorizonExecutor
    native = backend.joint_limits()
    n = len(native["lower"])
    limits = JointLimits(lower=native["lower"], upper=native["upper"],
                         velocity=native["velocity"],
                         acceleration=np.full(n, config["joint_reference_acceleration"]),
                         jerk=np.full(n, config["joint_reference_jerk"]),
                         provenance=native.get("provenance", {"source": "native joint intervals and velocity", "reference_acceleration_jerk": "new controller configuration"}))
    return ShortHorizonExecutor(backend, limits, ExecutionConfig(),
                                initial_gripper_opening=backend.read_state()["opening"])


def save_rgbd(backend, output, index, recorder=None):
    before = backend.clock()
    observation = backend.observe_rgbd()
    arrays = {"timestamp": np.asarray(observation.get("timestamp", before))}
    for kind in ("images", "depths"):
        for name, array in observation.get("observation/" + kind, {}).items():
            arrays[kind + "_" + name] = np.asarray(array)
    path = output / "rgbd" / f"{index:04d}.npz"
    path.parent.mkdir(exist_ok=True)
    if recorder is None:
        np.savez_compressed(path, **arrays)
    else:
        recorder.submit(path, arrays)
    if abs(backend.clock() - before) > 1e-7:
        raise RuntimeError("RGBD collection advanced the control clock")
    if index == 0:
        calibrations = observation.get("observation/calibrations", {})
        write_new(output / "camera_calibration.json", {
            name: dict(intrinsics=c.intrinsics,
                       camera_to_world=c.camera_to_world.xyzw()) for name, c in calibrations.items()})
        if "images_front" in arrays:
            from PIL import Image
            Image.fromarray(arrays["images_front"].astype(np.uint8)).save(output / "initial_front.png")
    return str(path.relative_to(output))


def append_ticks(backend, output, metric=None):
    ticks = backend.pop_tick_records()
    for tick in ticks:
        append(output / "native_ticks.jsonl", tick)
        if metric is not None:
            metric.update(tick.get("state", tick))
    return ticks


def run_p0(backend, executor, output, config):
    q = np.asarray(backend.get_joint_positions())
    plus, minus = q.copy(), q.copy()
    plus[0] += .02
    minus[0] -= .02
    commands = [("idle_open", q, 1.), ("move_positive", plus, 1.),
                ("replace_in_flight", minus, 1.), ("close_first_cycle", q, 0.),
                ("continue_closing", q, 0.), ("reopen", q, 1.),
                ("one_remaining_tick", q, 1.)]
    before = backend.read_state()
    receipts = []
    for label, target, opening in commands:
        x = backend.read_state()
        receipt = executor.execute_joint_target(target, opening=opening, phase="P0_" + label)
        y = backend.read_state()
        ticks = append_ticks(backend, output)
        row = dict(label=label, before=x, request=dict(q=target, opening=opening),
                   receipt=receipt, after=y, observed_tick_count=len(ticks))
        append(output / "transitions.jsonl", row)
        receipts.append(row)
        print(json.dumps(dict(stage="P0", command=label, ticks=len(ticks), time=backend.clock())), flush=True)
    counts = [r["observed_tick_count"] for r in receipts]
    final = backend.read_state()
    timing = counts == [2, 2, 2, 2, 2, 2, 1] and abs(final["time"] - 13 * .05) < 1e-6
    movement = max(float(np.linalg.norm(np.asarray(r["after"]["q"]) - np.asarray(before["q"]))) for r in receipts)
    close0 = float(receipts[3]["before"]["opening"])
    close1 = float(receipts[3]["after"]["opening"])
    close2 = float(receipts[4]["after"]["opening"])
    grip_continues = close1 < close0 - 1e-5 and close2 <= close1 + .02
    passed = timing and movement > 1e-5 and grip_continues
    return dict(stage="P0", complete=True, passed=passed,
                exact_tick_counts=counts, shared_clock_seconds=final["time"],
                actual_joint_displacement_rad=movement,
                gripper_opening_during_two_close_cycles=[close0, close1, close2],
                timing_passed=timing, persistent_gripper_passed=grip_continues,
                full_pose_IK_and_P1_grasp_qualified=False,
                initial=before, final=final)


def run_fk_canary(backend, output):
    from scipy.spatial.transform import Rotation
    from ..local_mpc_p2_20261103.kinematics import PoEKinematics, export_native_kinematic_snapshot
    snapshot = export_native_kinematic_snapshot(backend.arm, clock=backend._clock_tuple)
    fk = PoEKinematics(snapshot)
    q = backend.get_joint_positions()
    dq = backend.get_joint_velocities()
    targets = np.asarray(backend.arm.get_joint_target_positions())
    clock = backend._clock_tuple()
    bounds = backend.joint_limits()
    rng = np.random.default_rng(20261103)
    probes = np.clip(q + rng.uniform(-.25, .25, (24, 7)), bounds["lower"], bounds["upper"])
    errors = []
    try:
        for sample in probes:
            backend.arm.set_joint_positions(sample.tolist(), disable_dynamics=False)
            actual = np.asarray(backend.arm.get_tip().get_matrix())
            predicted = fk(sample)
            errors.append(dict(position_m=float(np.linalg.norm(actual[:3, 3] - predicted[:3, 3])),
                               rotation_rad=float(np.linalg.norm(Rotation.from_matrix(
                                   actual[:3, :3] @ predicted[:3, :3].T).as_rotvec()))))
    finally:
        backend.arm.set_joint_positions(q.tolist(), disable_dynamics=False)
        backend.arm.set_joint_target_positions(targets.tolist())
        if (backend._clock_tuple() != clock or
                not np.allclose(backend.get_joint_positions(), q, atol=1e-6, rtol=0) or
                not np.array_equal(backend.get_joint_velocities(), dq) or
                not np.allclose(backend.arm.get_joint_target_positions(), targets, atol=1e-6, rtol=0)):
            raise RuntimeError("FK canary changed Native clocks/joints/velocities/servo")
    position_max = max(row["position_m"] for row in errors)
    rotation_max = max(row["rotation_rad"] for row in errors)
    passed = position_max < 1e-5 and rotation_max < 1e-5
    snapshot["Native_FK_validation_passed"] = passed
    snapshot["Native_FK_validation_errors"] = errors
    write_new(output / "kinematics.json", snapshot)
    return dict(stage="P2_known_kinematics_diagnostic", complete=True, passed=passed,
                maximum_TCP_position_error_m=position_max, maximum_rotation_error_rad=rotation_max,
                Native_controlled_physics_ticks=0, random_configurations=len(probes),
                restoration_checked=True, learned_dynamics=False)


def run_native_audit(backend, output):
    from scipy.spatial.transform import Rotation
    from pyrep.backend import sim
    from pyrep.const import ObjectType
    from .collection_audit import collection_objects
    before = backend._clock_tuple()
    members = collection_objects(sim, backend.arm._collision_collection)
    gripper_shapes = {s.get_handle(): s.get_name() for s in
                      backend.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE)}
    if backend._clock_tuple() != before:
        raise RuntimeError("Native collection getter advanced clocks")
    table_shapes = []
    for item in backend.raw.map_service.map:
        if "table" not in item["name"].lower():
            continue
        box, pose = np.asarray(item["bounding_box"]), np.asarray(item["pose"])
        corners = np.asarray([[x, y, z] for x in box[:2] for y in box[2:4] for z in box[4:6]])
        world = Rotation.from_quat(pose[3:]).apply(corners) + pose[:3]
        table_shapes.append(dict(name=item["name"], world_min=world.min(0), world_max=world.max(0)))
    return dict(stage="Native_zero_action_audit", complete=True,
                native_default_target_pose=backend.native_default_target_pose,
                configured_target_position=backend.scenario.initial_position,
                controlled_physics_ticks=backend.raw.ledger.ticks,
                native_target_bbox=backend.task.target_block.get_bounding_box(),
                original_arm_collection_members=members,
                gripper_shapes=gripper_shapes,
                gripper_shapes_missing_from_original_arm_collection={h: n for h, n in gripper_shapes.items() if h not in members},
                table_shapes=table_shapes,
                physics_or_task_sources_edited=False)


def run_p1(backend, executor, output, config, *, capture_rgbd=True, recorder=None):
    if (config.get("response_factory") is not None or config.get("action_space") is not None) and not config.get("use_joint_goal_planner", False):
        raise ValueError("versioned response/action options require use_joint_goal_planner")
    from .batch_mpc import BatchMPCPlanner as MPCPlanner
    constraints = DiagnosticConstraints(lower=config.get("workspace_lower", [-.1, -.65, .76]))
    if config.get("terminal_goal_cost", False):
        from .terminal_mpc import TerminalBatchMPCPlanner as MPCPlanner
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
    geometry = GeometricCandidates(horizon=config["horizon_seconds"])
    metric = HoldMetric(config["hold_seconds"], config["lift_height_m"], config["max_hold_relative_drift_m"])
    initial = backend.read_state()
    metric.update(initial)
    warm, phase_state = None, None
    planning_times, full_times, ages = [], [], []
    native_any = False
    iteration = 0
    last_generation = -1
    while not backend.terminal():
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
        candidates, qualification = (goal_helper.qualify_candidates(backend, state, candidates)
                                     if goal_helper is not None else constraints.qualify_candidates(backend, state, candidates))
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
        row = dict(index=iteration, information_setting="truth_diagnostic_only", before=state,
                   actuator_state_before=h_before, candidates=candidates, plan=result,
                   candidate_qualification=qualification, action_rejected_reason=rejected,
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
        if metric.passed and native_any and after["held"]:
            break
        if not ticks:
            raise RuntimeError("executor returned without a physics tick before terminal")
    final = backend.read_state()
    return dict(stage="P1", complete=True, milestone_passed=bool(metric.passed and native_any and final["held"]),
                native_success_observed=native_any, hold=metric.receipt(), cycles=iteration,
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("p0", "p1", "fk", "audit", "gripper"), required=True)
    parser.add_argument("--scene", choices=("static", "uniform", "speed_change"), default="uniform")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--no-rgbd", action="store_true", help="explicit control-only diagnostic, not a complete perception dataset")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    family = (ROOT / "runtime/framework/local_mpc_20261102").resolve()
    if family not in output.parents:
        parser.error("new local_mpc_20261102 runtime family required")
    output.mkdir(parents=True, exist_ok=False)
    started = utc()
    backend = executor = recorder = None
    frozen = freeze_sources(args.config, args.mode)
    code = 1
    try:
        config = json.loads(args.config.read_text())
        public_modes = ("public_motion_shadow", "public_pregrasp_control", "public_grasp_prototype",
                        "public_grasp_active_visual_v4", "public_grasp_active_visual_v41", "public_grasp_active_visual_v42",
                        "public_grasp_cold_probe_v43", "public_grasp_candidate_transformer_v5")
        if sum(bool(config.get(key,False)) for key in public_modes)>1:
            raise ValueError("one explicit public observer/control protocol at a time required")
        fast=config.get('finger_response_fastpath',False)
        if type(fast) is not bool:
            raise ValueError('explicit boolean finger-response computation modifier required')
        if fast and (args.mode!='p1' or config.get('public_grasp_candidate_transformer_v5') is not True
                     or config.get('response_factory')!='coupled_joint_20261103_v1'
                     or config.get('response_controller')!='coupled'):
            raise ValueError('finger fastpath requires the coupled public candidate p1 route')
        v56_flags=(config.get("candidate_goal_sphere_filter_v56",False),
                   config.get("postconfirmed_position_maintenance_v56",False))
        if any(type(flag) is not bool for flag in v56_flags):
            raise ValueError("explicit boolean candidate-v56 feature modifiers required")
        v56_public=any(v56_flags)
        if v56_public and (args.mode!="p1" or config.get("public_grasp_candidate_transformer_v5") is not True):
            raise ValueError("v56 modifiers require the public candidate p1 route")
        dark=config.get('candidate_darkred_target_v59',False)
        if type(dark) is not bool:
            raise ValueError('explicit boolean candidate dark-red v59 modifier required')
        if dark and (args.mode!='p1' or config.get('public_grasp_candidate_transformer_v5') is not True
                     or config.get('candidate_goal_sphere_filter_v56') is not True
                     or config.get('postconfirmed_position_maintenance_v56') is not True):
            raise ValueError('dark-red v59 requires candidate p1 with goal filtering and position maintenance')
        archive_sources(output, frozen)
        native_sources = native_source_snapshot(config)
        write_new(output / "run_inputs.json", dict(started_utc=started, mode=args.mode, scene=args.scene,
                    config=config, source_sha256=frozen, dependencies=dependency_receipt(),
                    frozen_environment=environment_proof(), pid=os.getpid(),
                    CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    native_sources_sha256=native_sources,
                    old_experiments_resumed=False, TEST_read=False,
                    information_setting=("public_candidate_darkred_v59_development" if dark
                                         else "public_candidate_transformer_v56_development" if v56_public
                                         else "public_candidate_transformer_v5_development" if config.get("public_grasp_candidate_transformer_v5",False)
                                         else "public_cold_probe_v43_development" if config.get("public_grasp_cold_probe_v43",False)
                                         else "public_goal_aware_grasp_v42_development" if config.get("public_grasp_active_visual_v42",False)
                                         else "public_active_visual_grasp_v41_development" if config.get("public_grasp_active_visual_v41",False)
                                         else "public_active_visual_grasp_v4_development" if config.get("public_grasp_active_visual_v4",False)
                                         else "public_estimated_grasp_development_prototype" if config.get("public_grasp_prototype",False)
                                         else "public_estimated_pregrasp_diagnostic" if config.get("public_pregrasp_control",False)
                                         else "truth_diagnostic_only")))
        backend = make_backend(config, args.scene, budget_ticks=13 if args.mode == "p0" else None)
        write_new(output / "initial_snapshot.json", backend.initial_snapshot)
        if config.get("diagnostic_no_proxy_attachment", False):
            if (config.get("counterfactual_not_benchmark") is not True
                    or not (config.get("public_grasp_active_visual_v4") is True
                            or config.get("public_grasp_active_visual_v41") is True
                            or config.get("public_grasp_active_visual_v42") is True
                            or config.get("public_grasp_cold_probe_v43") is True
                            or config.get("public_grasp_candidate_transformer_v5") is True)):
                raise ValueError("proxy-attachment suppression is only an explicit non-benchmark v4 diagnostic")
            from ..local_mpc_control_20261103.no_proxy_attachment import NoProxyAttachmentBackend
            backend = NoProxyAttachmentBackend(backend)
        if config.get("gripper_completion_latch", False):
            from ..local_mpc_control_20261103.completed_gripper import CompletedGripperBackend
            backend = CompletedGripperBackend(backend)
            write_new(output / "gripper_completion_protocol.json", backend.gripper_completion_receipt())
        if config.get("public_grasp_active_visual_v42",False):
            if args.mode!="p1":
                raise ValueError("v42 requires its independent public execution diagnostic")
            from ..local_mpc_control_20261103.public_grasp_v42 import (
                ObservedPublicGraspV42Backend, ActiveVisualGraspV42Config, GoalAwareActiveVisualPublicBridge)
            from ..local_mpc_control_20261103.active_visual_attachment import ActiveVisualConfig
            recorder=FrameRecorder()
            bridge=GoalAwareActiveVisualPublicBridge(active_config=ActiveVisualConfig(**config.get("active_visual_settings",{})))
            backend=ObservedPublicGraspV42Backend(backend,output/"public_shadow",public_bridge=bridge,
                recorder=recorder,save_frames=config.get("public_shadow_save_frames",True),
                grasp_config=ActiveVisualGraspV42Config(**config.get("public_grasp_settings",{})))
            backend.sample()
        elif config.get("public_grasp_candidate_transformer_v5",False):
            if args.mode!="p1":
                raise ValueError("candidate Transformer v5 requires its independent public execution diagnostic")
            from ..local_mpc_control_20261103.public_grasp_candidates_v5 import ObservedPublicCandidateV5Backend
            from ..local_mpc_control_20261103.public_grasp_v41 import ActiveVisualGraspV41Config
            from ..local_mpc_control_20261103.public_grasp_v4 import ActiveVisualPublicBridge
            from ..local_mpc_control_20261103.active_visual_attachment import ActiveVisualConfig
            recorder=FrameRecorder()
            if v56_public:
                if dark:
                    from ..local_mpc_control_20261103.public_candidate_darkred_v59 import ObservedPublicCandidateDarkRedV59Backend as CandidateObserver
                else:
                    from ..local_mpc_control_20261103.public_candidate_goal_maintenance_v56 import ObservedPublicCandidateV56Backend as CandidateObserver
                backend=CandidateObserver(backend,output/"public_shadow",
                    goal_sphere_filter=v56_flags[0],postconfirmed_maintenance=v56_flags[1],
                    active_config=ActiveVisualConfig(**config.get("active_visual_settings",{})),
                    recorder=recorder,save_frames=config.get("public_shadow_save_frames",True),
                    grasp_config=ActiveVisualGraspV41Config(**config.get("public_grasp_settings",{})),
                    checkpoint=ROOT/config["candidate_transformer_checkpoint"],
                    checkpoint_sha256=config["candidate_transformer_checkpoint_sha256"],
                    require_public_goal_completion=config.get("require_public_goal_completion",False))
            else:
                bridge=ActiveVisualPublicBridge(active_config=ActiveVisualConfig(**config.get("active_visual_settings",{})))
                backend=ObservedPublicCandidateV5Backend(backend,output/"public_shadow",public_bridge=bridge,
                    recorder=recorder,save_frames=config.get("public_shadow_save_frames",True),
                    grasp_config=ActiveVisualGraspV41Config(**config.get("public_grasp_settings",{})),
                    checkpoint=ROOT/config["candidate_transformer_checkpoint"],
                    checkpoint_sha256=config["candidate_transformer_checkpoint_sha256"],
                    require_public_goal_completion=config.get("require_public_goal_completion",False))
            backend.sample()
        elif config.get("public_grasp_active_visual_v41",False) or config.get("public_grasp_cold_probe_v43",False):
            if (args.mode!="p1" or config.get("public_grasp_active_visual_v4",False)
                    or config.get("public_motion_shadow",False) or config.get("public_pregrasp_control",False)
                    or config.get("public_grasp_prototype",False)):
                raise ValueError("v41 requires its independent public execution diagnostic")
            from ..local_mpc_control_20261103.public_grasp_v41 import ObservedPublicGraspV41Backend, ActiveVisualGraspV41Config
            from ..local_mpc_control_20261103.public_grasp_v4 import ActiveVisualPublicBridge
            from ..local_mpc_control_20261103.active_visual_attachment import ActiveVisualConfig
            recorder=FrameRecorder()
            bridge=ActiveVisualPublicBridge(active_config=ActiveVisualConfig(**config.get("active_visual_settings",{})))
            backend=ObservedPublicGraspV41Backend(backend,output/"public_shadow",public_bridge=bridge,
                recorder=recorder,save_frames=config.get("public_shadow_save_frames",True),
                grasp_config=ActiveVisualGraspV41Config(**config.get("public_grasp_settings",{})))
            backend.sample()
        elif config.get("public_grasp_active_visual_v4", False):
            if (args.mode != "p1" or config.get("public_motion_shadow",False)
                    or config.get("public_pregrasp_control",False) or config.get("public_grasp_prototype",False)):
                raise ValueError("active visual v4 requires its independently versioned execution diagnostic")
            from ..local_mpc_control_20261103.public_grasp_v4 import (
                ObservedPublicGraspV4Backend, ActiveVisualGraspConfig, ActiveVisualPublicBridge)
            from ..local_mpc_control_20261103.active_visual_attachment import ActiveVisualConfig
            recorder = FrameRecorder()
            bridge = ActiveVisualPublicBridge(active_config=ActiveVisualConfig(**config.get("active_visual_settings",{})))
            backend = ObservedPublicGraspV4Backend(backend, output / "public_shadow",public_bridge=bridge,
                recorder=recorder,save_frames=config.get("public_shadow_save_frames",True),
                grasp_config=ActiveVisualGraspConfig(**config.get("public_grasp_settings",{})))
            backend.sample()
        elif config.get("public_grasp_prototype", False):
            if (args.mode != "p1" or config.get("public_motion_shadow", False)
                    or config.get("public_pregrasp_control", False)):
                raise ValueError("public grasp prototype requires its own explicit execution diagnostic")
            from ..local_mpc_control_20261103.public_grasp import ObservedPublicGraspBackend, PublicGraspConfig
            recorder = FrameRecorder()
            backend = ObservedPublicGraspBackend(backend, output / "public_shadow", recorder=recorder,
                save_frames=config.get("public_shadow_save_frames", True),
                grasp_config=PublicGraspConfig(**config.get("public_grasp_settings",{})))
            backend.sample()
        elif config.get("public_pregrasp_control", False):
            if args.mode != "p1" or config.get("public_motion_shadow", False):
                raise ValueError("public pregrasp controller requires its own explicit P1 execution diagnostic")
            from ..local_mpc_control_20261103.public_pregrasp import ObservedPublicBackend
            recorder = FrameRecorder()
            backend = ObservedPublicBackend(backend, output / "public_shadow", recorder=recorder,
                save_frames=config.get("public_shadow_save_frames", True))
            backend.sample()
        elif config.get("public_motion_shadow", False):
            if args.mode != "p1":
                raise ValueError("public motion shadow is an explicit P1-controller diagnostic")
            from ..local_mpc_control_20261103.public_shadow import PublicShadowBackend
            recorder = FrameRecorder()
            backend = PublicShadowBackend(backend, output / "public_shadow", recorder=recorder,
                save_frames=config.get("public_shadow_save_frames", True))
            backend.sample()
        executor = make_executor(backend, config)
        if args.mode == "gripper":
            from ..local_mpc_control_20261103.gripper_geometry_audit import audit_gripper_geometry
            result = audit_gripper_geometry(backend, output / "gripper_geometry")
            result.update(stage="known_robot_gripper_geometry_calibration",complete=True,
                          formal_evaluation=False,Native_controlled_physics_ticks=0)
        elif args.mode == "audit":
            result = run_native_audit(backend, output)
        elif args.mode == "fk":
            result = run_fk_canary(backend, output)
        elif args.mode == "p0":
            result = run_p0(backend, executor, output, config)
        else:
            if recorder is None and not args.no_rgbd:
                recorder = FrameRecorder()
            if config.get("public_grasp_active_visual_v42",False):
                from ..local_mpc_control_20261103.public_grasp_v42 import run_public_grasp_v42
                result=run_public_grasp_v42(backend,executor,output,config,recorder=recorder)
                append_ticks(backend,output)
            elif config.get("public_grasp_candidate_transformer_v5",False):
                if dark:
                    from ..local_mpc_control_20261103.public_candidate_darkred_v59 import run_public_candidate_darkred_v59
                    result=run_public_candidate_darkred_v59(backend,executor,output,config,recorder=recorder)
                elif v56_public:
                    from ..local_mpc_control_20261103.public_candidate_goal_maintenance_v56 import run_public_candidate_goal_maintenance_v56
                    result=run_public_candidate_goal_maintenance_v56(backend,executor,output,config,recorder=recorder)
                else:
                    from ..local_mpc_control_20261103.public_grasp_candidates_v5 import run_public_grasp_candidates_v5
                    result=run_public_grasp_candidates_v5(backend,executor,output,config,recorder=recorder)
                append_ticks(backend,output)
            elif config.get("public_grasp_cold_probe_v43",False):
                from ..local_mpc_control_20261103.public_grasp_cold_probe_v43 import run_public_grasp_cold_probe_v43
                result=run_public_grasp_cold_probe_v43(backend,executor,output,config,recorder=recorder)
                append_ticks(backend,output)
            elif config.get("public_grasp_active_visual_v41",False):
                from ..local_mpc_control_20261103.public_grasp_v41 import run_public_grasp_v41
                result=run_public_grasp_v41(backend,executor,output,config,recorder=recorder)
                append_ticks(backend,output)
            elif config.get("public_grasp_active_visual_v4", False):
                from ..local_mpc_control_20261103.public_grasp_v4 import run_public_grasp_v4
                result = run_public_grasp_v4(backend,executor,output,config,recorder=recorder)
                append_ticks(backend,output)
            elif config.get("public_grasp_prototype", False):
                from ..local_mpc_control_20261103.public_grasp import run_public_grasp
                result = run_public_grasp(backend, executor, output, config, recorder=recorder)
                append_ticks(backend, output)
            elif config.get("public_pregrasp_control", False):
                from ..local_mpc_control_20261103.public_pregrasp import run_public_pregrasp
                result = run_public_pregrasp(backend, executor, output, config, recorder=recorder)
                # Privileged tick records are evaluation data, flushed only
                # after the public controller has returned its result.
                append_ticks(backend, output)
            else:
                result = run_p1(backend, executor, output, config, capture_rgbd=not args.no_rgbd, recorder=recorder)
            if config.get("public_motion_shadow", False):
                result["public_motion_shadow"] = backend.receipt()
            if recorder is not None:
                recorder.close()
                recorder = None
        if native_source_snapshot(config) != native_sources:
            raise RuntimeError("original Native source changed during diagnostic")
        verify_sources(frozen)
        if config.get("gripper_completion_latch", False):
            result["gripper_completion_protocol"] = backend.gripper_completion_receipt()
            result["finger_response_requalified_for_runtime_protocol"] = False
        if config.get("diagnostic_no_proxy_attachment", False):
            result["counterfactual_no_proxy_attachment"] = backend.no_proxy_attachment_receipt()
            result["positive_benchmark_score_eligible"] = False
        result.update(started_utc=started, finished_utc=utc(), source_unchanged=True)
        write_new(output / "result.json", result)
        code = 0
    except BaseException as error:
        if backend is not None:
            append_ticks(backend, output)
            write_new(output / "last_native_query_receipts.json", dict(
                IK=getattr(backend, "last_ik_receipt", None),
                collision=getattr(backend, "last_collision_probe", None)))
        if executor is not None and executor.last_receipt is not None:
            write_new(output / "last_execution_receipt.json", executor.last_receipt)
        write_new(output / "failure.json", dict(started_utc=started, finished_utc=utc(),
                    type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
                    source_sha256=frozen, stage=args.mode, formal_evaluation=False))
        traceback.print_exc()
    finally:
        if recorder is not None:
            recorder.close()
        if backend is not None:
            backend.close()
        write_new(output / "worker_exit.json", dict(finished_utc=utc(), exit_code=code,
                    result_recorded=(output / "result.json").exists(), native_closed=backend is not None))
    return code

