"""Public RGB-D tracking and the unchanged frozen v671 GRU interface.

Semantic role masks are learned predictions, never Native instance IDs. Each
body keeps its own [23,20] forecast and role identity; bodies are not silently
flattened into the existing single-body candidate interface. Missing geometry,
unknown manipulation mode and insufficient causal history remain unavailable.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import time

import h5py
import numpy as np
import torch

from ..benchmark_perception import RedBoxTracker
from ..contracts import Pose
from ..perception import CameraCalibration
from ..multitask.context_se3 import MultiBodyContextDynamics, HISTORY_DIM, STATE_DIM
from ..multitask.factory import artifact_path, load_rigid_layout
from ..multitask.perception import (
    BodyTemplate, GeometryComponent, MultiBodySE3Tracker, PriorProvenance,
    RegistrationConfig,
)
from ..multitask.rotation_equivalence import VerifiedFiniteSymmetry
from ..multitask.specs import TASK_SPECS
from ..multitask.training_geometry import _check_asset
from .data import CAMERAS, DT, TASK_FAMILIES, decode_metric_depth, file_sha256, source_stat


ROLE_IDS = {"frame": 1, "peg0": 2, "peg1": 3, "peg2": 4, "ring": 5}
FORECAST_SHA256 = "ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef"
FORECAST_RELATIVE = "artifacts/framework/multitask_r10/context_joint16_after_executor_v671/dynamics.pt"
FROZEN_CONFIGURATIONS = {
    "insert_onto_rotating_peg": dict(
        path="artifacts/framework/multitask_r10/post_training_qualification_v680_setup/insert_onto_rotating_peg_ours_configuration.json",
        sha256="daab94d36b1d83e47852809de8e31a561a29c1ae94ca304489b1a52a63061dd4"),
    "insert_onto_rotating_peg_high_speed": dict(
        path="artifacts/framework/multitask_r10/post_training_qualification_v680_setup/insert_onto_rotating_peg_high_speed_ours_configuration.json",
        sha256="2524afccc1c1df52d5b37090e18eefc55f373df3f329c28c57a6eb86f26c1222"),
}


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "receipt"):
        return _plain(value.receipt())
    return value


def validate_public_observation(observation):
    allowed = {"timestamp", "observation/images", "observation/depths",
               "observation/calibrations", "observation/camera_timestamps"}
    required = allowed - {"observation/camera_timestamps"}
    if set(observation) - allowed or not required <= set(observation):
        raise ValueError("only public RGB-D/calibration/time channels are accepted")
    if not np.isfinite(float(observation["timestamp"])):
        raise ValueError("finite observation timestamp required")
    images, depths, calibrations = (observation[name] for name in (
        "observation/images", "observation/depths", "observation/calibrations"))
    if set(images) != set(CAMERAS) or set(depths) != set(CAMERAS) or set(calibrations) != set(CAMERAS):
        raise ValueError("the four declared public camera streams must be aligned")
    for camera in CAMERAS:
        rgb, depth = np.asarray(images[camera]), np.asarray(depths[camera])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or depth.shape != rgb.shape[:2]:
            raise ValueError("aligned original-resolution RGB uint8 and metric depth required")
        if not isinstance(calibrations[camera], CameraCalibration):
            raise ValueError("data-only CameraCalibration required")
        now = float(observation["timestamp"])
        camera_time = float(observation.get("observation/camera_timestamps", {}).get(camera, now)) + calibrations[camera].timestamp_offset
        if not np.isfinite(camera_time) or camera_time > now + 1e-9 or now - camera_time > .035:
            raise ValueError("public camera timestamp is future or stale")


def semantic_templates_from_asset(asset, orientation_observability, finite_symmetries):
    """Merge visual components in their existing body frame, one ID per role."""
    _check_asset(asset)
    if set(asset["bodies"]) != set(ROLE_IDS) or set(orientation_observability) != set(ROLE_IDS):
        raise ValueError("exact frozen frame/three-peg/ring CAD roles required")
    templates = []
    for role, role_id in ROLE_IDS.items():
        body = asset["bodies"][role]
        if body["empty_visual_role"] or not body["components"]:
            raise ValueError("semantic masks cannot invent a missing CAD visual surface")
        points = np.concatenate([np.asarray(component["points_body"], dtype=float)
                                 for component in body["components"]])
        provenance = PriorProvenance(
            "declared_cad", "frozen TRAIN native mesh asset:" + asset["asset_sha256"],
            body["frame_definition"], True, "train", (asset["source_group"],), asset["asset_sha256"])
        templates.append(BodyTemplate(
            role, (GeometryComponent("merged_role_visual_surface", (role_id,), points),),
            provenance, orientation_observability=orientation_observability[role],
            finite_rotation_symmetry=finite_symmetries.get(role)))
    return tuple(templates)


def build_insert_tracker(root, task):
    if task not in FROZEN_CONFIGURATIONS:
        raise ValueError("only the two declared insertion tasks use this CAD backend")
    root = Path(root).resolve()
    configuration_pointer = FROZEN_CONFIGURATIONS[task]
    config = json.loads(artifact_path(root, configuration_pointer).read_text())
    references = config["learned_artifacts"][task]
    asset = json.loads(artifact_path(root, references["geometry"]).read_text())
    symmetry_pointer = config["finite_rotation_symmetries"][task]["ring"]
    symmetry = VerifiedFiniteSymmetry.load(root, symmetry_pointer)
    if symmetry.certificate["task"] != task or symmetry.certificate["CAD_asset_sha256"] != asset["asset_sha256"]:
        raise ValueError("ring D4 evidence differs from the selected frozen TRAIN CAD")
    orientation = config["body_orientation_observability"][task]
    if orientation["ring"] != "none":
        raise ValueError("ring full rotation must remain explicitly unobserved")
    templates = semantic_templates_from_asset(asset, orientation, {"ring": symmetry})
    rigid = load_rigid_layout(artifact_path(root, references["rigid_layout"]), task)
    tracker = MultiBodySE3Tracker(templates, task_spec=TASK_SPECS[task], rigid_children=rigid,
                                config=RegistrationConfig(**config.get("registration_config", {})))
    receipt = dict(configuration=configuration_pointer, TRAIN_CAD=references["geometry"],
                   TRAIN_rigid_layout=references["rigid_layout"], ring_D4=symmetry_pointer,
                   source_group=asset["source_group"], TRAIN_asset_internal_sha256=asset["asset_sha256"],
                   semantic_role_ids=ROLE_IDS, components_merged_in_original_body_frames=True,
                   registration_config=config.get("registration_config", {}),
                   orientation_observability=orientation,
                   ring_full_orientation_observable=False,
                   ring_representative="original tracker ICP/D4 representative; no learned-skill grasp orientation selected",
                   ring_D4_scope="geometric quotient only, not a physical grasp or action symmetry guarantee",
                   legacy_task_policy_installed=False, native_runtime_ids_used=False,
                   symmetry_replay=_plain(symmetry.replay_receipt))
    return tracker, receipt


def audit_canonical_public_priors(root):
    """Audit both frozen TRAIN variants before choosing one fixed common prior.

    The common normal prior is a predeclared geometry-only choice for every
    input scene. Numerical CAD/layout differences are reported, not used to
    relax any tracker threshold or select a source-dependent runtime branch.
    """
    root = Path(root).resolve()
    sources = []
    for task, configuration in FROZEN_CONFIGURATIONS.items():
        config = json.loads(artifact_path(root, configuration).read_text())
        geometry = config["learned_artifacts"][task]["geometry"]
        layout = config["learned_artifacts"][task]["rigid_layout"]
        asset = json.loads(artifact_path(root, geometry).read_text())
        _check_asset(asset)
        rigid = json.loads(artifact_path(root, layout).read_text())
        sources.append((task, config, asset, rigid, dict(
            configuration=configuration, geometry=geometry, rigid_layout=layout,
            geometry_asset_sha256=asset["asset_sha256"])))
    first, second = sources
    geometry_differences = {}
    for role in ROLE_IDS:
        a, b = first[2]["bodies"][role], second[2]["bodies"][role]
        if [part["name"] for part in a["components"]] != [part["name"] for part in b["components"]]:
            raise ValueError("frozen variant CAD component identities differ")
        visual_a = np.concatenate([part["points_body"] for part in a["components"]])
        visual_b = np.concatenate([part["points_body"] for part in b["components"]])
        vertices_a, vertices_b = np.asarray(a["collision_vertices_body"]), np.asarray(b["collision_vertices_body"])
        if visual_a.shape != visual_b.shape or vertices_a.shape != vertices_b.shape or not np.array_equal(a["collision_triangles"], b["collision_triangles"]):
            raise ValueError("frozen variant CAD topology differs from the audited common-prior scope")
        geometry_differences[role] = dict(
            surface_maximum_absolute_difference_m=float(np.abs(visual_a - visual_b).max()),
            collision_vertex_maximum_absolute_difference_m=float(np.abs(vertices_a - vertices_b).max()),
            surface_exactly_equal=bool(np.array_equal(visual_a, visual_b)),
            triangles_exactly_equal=True)
    layouts = []
    if [(row["parent"], row["child"]) for row in first[3]["accepted"]] != [(row["parent"], row["child"]) for row in second[3]["accepted"]]:
        raise ValueError("frozen variant rigid role relationships differ")
    for a, b in zip(first[3]["accepted"], second[3]["accepted"]):
        layouts.append(dict(parent=a["parent"], child=a["child"],
            maximum_position_difference_m=float(np.abs(np.asarray(a["child_to_parent_position"]) - b["child_to_parent_position"]).max()),
            maximum_rotation_matrix_difference=float(np.abs(np.asarray(a["child_to_parent_rotation"]) - b["child_to_parent_rotation"]).max())))
    equal_config = dict(
        registration=first[1].get("registration_config", {}) == second[1].get("registration_config", {}),
        orientation=first[1]["body_orientation_observability"][first[0]] == second[1]["body_orientation_observability"][second[0]],
        forecast=first[1].get("forecast_config", {}) == second[1].get("forecast_config", {}))
    if not all(equal_config.values()):
        raise ValueError("canonical prior configuration differs from the audited scope")
    return dict(schema="blind_shared_TRAIN_prior_geometry_audit_20261030_v1",
                frozen_insert_sources={source[0]: source[4] for source in sources},
                per_role_CAD_differences=geometry_differences, rigid_layout_differences=layouts,
                configuration_equality=equal_config,
                fixed_pick_prior="unchanged RedBoxTracker default BoxPrior, shared by both speeds",
                fixed_insert_prior="insert_onto_rotating_peg frozen normal TRAIN geometry/layout/D4",
                source_task_or_speed_selects_frontend=False,
                selection_basis="fixed common TRAIN geometry prior after numerical CAD/layout audit; no outcome selection",
                registration_thresholds_changed=False, full_orientation_promoted=False,
                native_VAL_or_TEST_used=False)


class FrozenGRUForecaster:
    """Freeze v671 and admit only fresh bodies with known public motion modes."""
    def __init__(self, checkpoint, roles, *, expected_sha256=FORECAST_SHA256):
        self.path = Path(checkpoint)
        if file_sha256(self.path) != expected_sha256:
            raise ValueError("frozen motion checkpoint fingerprint differs")
        self.predictor = MultiBodyContextDynamics(self.path)
        self.predictor.model.requires_grad_(False)
        if HISTORY_DIM != 21 or STATE_DIM != 19 or self.predictor.maximum_validated_horizon < 1.1:
            raise ValueError("v671 21-history/19-state/22-step contract required")
        self.roles = tuple(roles)
        if not self.roles or len(self.roles) != len(set(self.roles)):
            raise ValueError("unique declared forecast roles required")
        self.last_timestamp = None

    def reset(self):
        self.predictor.reset()
        self.last_timestamp = None

    @torch.inference_mode()
    def observe_and_predict(self, objects, metadata, modes, timestamp):
        now = float(timestamp)
        if not np.isfinite(now) or self.last_timestamp is not None and now <= self.last_timestamp:
            raise ValueError("strictly increasing public timestamps required")
        self.last_timestamp = now
        current = {obj.object_id: obj for obj in objects}
        if len(current) != len(objects) or set(current) - set(self.roles):
            raise ValueError("unique known public body roles required")
        result, status = {}, {}
        bodies = metadata.get("bodies", {})
        for role in self.roles:
            obj, diagnostic = current.get(role), bodies.get(role, {})
            reason = None
            if obj is None or not obj.visible or not diagnostic.get("registration_valid", False):
                reason = "no_fresh_public_pose"
            elif abs(obj.timestamp - now) > 1e-8:
                reason = "public_pose_time_mismatch"
            elif modes.get(role, "unknown") != "external":
                reason = "motion_mode_unknown_or_not_supported_by_frozen_external_forecast"
            if reason:
                self.predictor.history.pop(role, None)
                status[role] = dict(available=False, reason=reason, history_frames=0,
                                    motion_mode=modes.get(role, "unknown"))
                continue
            observable = bool(diagnostic.get("orientation_observable", False))
            self.predictor.observe([obj], modes={role: "external"},
                                   orientation_observability={role: observable})
            count = len(self.predictor.history[role])
            status[role] = dict(available=False, reason="fewer_than_five_causal_valid_frames",
                                history_frames=count, motion_mode="external",
                                full_orientation_observable=observable)
            if count < 5:
                continue
            history, lengths, _, cutoff = self.predictor.inputs(role)
            forecast = self.predictor.predict([obj], horizon=1.1, dt=.05)
            positions = torch.tensor(forecast.positions[:, 0], dtype=torch.float32)
            rotations = torch.tensor(forecast.rotations[:, 0], dtype=torch.float32)
            velocities = torch.tensor(forecast.linear_velocities[:, 0], dtype=torch.float32)
            angular = torch.tensor(forecast.angular_velocities[:, 0], dtype=torch.float32)
            normalized = self.predictor.model.normalized(
                positions, rotations, velocities, angular, torch.zeros(23, dtype=torch.long),
                torch.full((23,), observable, dtype=torch.bool))
            relative_times = np.arange(23, dtype=np.float32) * np.float32(.05)
            tokens = np.concatenate((normalized.numpy(), relative_times[:, None]), axis=-1)
            if tokens.shape != (23, 20) or not np.isfinite(tokens).all():
                raise ValueError("frozen GRU returned an invalid real trajectory")
            result[role] = dict(trajectory=tokens.astype(np.float32),
                                positions=forecast.positions[:, 0].astype(np.float32),
                                rotations=forecast.rotations[:, 0].astype(np.float32),
                                history=history[0].numpy(), history_length=int(lengths[0]),
                                observed_until=float(cutoff), full_orientation_observable=observable)
            status[role].update(available=True, reason=None)
        return dict(forecasts=result, status=status,
                    checkpoint_sha256=self.predictor.checkpoint_sha256,
                    forecast_steps=22, dt_seconds=.05,
                    native_or_expert_poses_used=False, trajectories_fabricated=False)


class PublicMotionPipeline:
    """Episode-scoped public tracker; no stage/held/goal label input exists.

    The HDF5 public channels contain no contact or held identity. After the first
    observed closed gripper, a graspable body's external/attached/released mode
    cannot be resolved and stays unknown until reset. An open gripper alone
    cannot retrospectively certify which of those transitions happened.
    """
    def __init__(self, root, task, *, segment_checkpoint=None, segmenter=None,
                 motion_checkpoint=None):
        if task not in TASK_FAMILIES:
            raise ValueError("only the declared four task versions are supported")
        self.root, self.task = Path(root).resolve(), task
        self.is_pick = task.startswith("pick_moving_target_on_the_table")
        if self.is_pick:
            self.tracker = RedBoxTracker()
            self.roles = ("target",)
            self.segmenter = None
            self.prior_receipt = dict(backend="unchanged_RedBoxTracker", prior=self.tracker.prior.__dict__,
                                      known_world_aligned_shape_prior=True, full_rotation_observed=False)
        else:
            if segmenter is None:
                if segment_checkpoint is None or not Path(segment_checkpoint).is_file():
                    raise FileNotFoundError("a fitted public semantic segmentation checkpoint is required")
                from .segmentation import PublicSemanticSegmenter
                segmenter = PublicSemanticSegmenter(segment_checkpoint, device="cpu")
            self.segmenter = segmenter
            self.tracker, self.prior_receipt = build_insert_tracker(self.root, task)
            self.roles = tuple(ROLE_IDS)
        self.forecaster = FrozenGRUForecaster(
            self.root / FORECAST_RELATIVE if motion_checkpoint is None else motion_checkpoint,
            self.roles)
        self.manipulated_role = "target" if self.is_pick else "ring"
        self.manipulated_mode_unresolved = False
        self.last_timestamp = None

    def reset(self):
        if self.is_pick:
            self.tracker = RedBoxTracker(prior=self.tracker.prior)
        else:
            self.tracker.reset()
        self.forecaster.reset()
        self.manipulated_mode_unresolved = False
        self.last_timestamp = None

    @property
    def checkpoint_sha256(self):
        return self.forecaster.predictor.checkpoint_sha256

    @property
    def checkpoint_paths(self):
        return (self.forecaster.path,)

    def observe(self, observation, robot):
        validate_public_observation(observation)
        robot = np.asarray(robot, dtype=float)
        if robot.shape != (15,) or not np.isfinite(robot).all() or robot[7] not in (0., 1.):
            raise ValueError("measured TCP7/gripper_open/joints7 robot vector required")
        now = float(observation["timestamp"])
        if self.last_timestamp is not None and now <= self.last_timestamp:
            raise ValueError("reset tracking pipeline at episode boundaries")
        self.last_timestamp = now
        if robot[7] == 0.:
            self.manipulated_mode_unresolved = True
        modes = {role: "external" for role in self.roles}
        if self.manipulated_mode_unresolved:
            modes[self.manipulated_role] = "unknown"
        segmentation = None
        if self.is_pick:
            try:
                obj, details = self.tracker.observe(observation)
                objects = [obj]
                metadata = dict(bodies={"target": dict(details, registration_valid=True,
                                                       orientation_observable=False,
                                                       rotation_gauge="known-world-aligned box prior; full orientation not measured")})
            except ValueError as error:
                # An occluded body is not continued as a current measurement.
                self.tracker.history.clear()
                self.tracker.last_velocity[:] = 0
                objects = []
                metadata = dict(bodies={"target": dict(registration_valid=False,
                                                       orientation_observable=False, failure=str(error))})
        else:
            segmentation = self.segmenter.predict(observation)
            masks = segmentation["semantic_maps"]
            if set(masks) != set(CAMERAS):
                raise ValueError("semantic role masks lost a public camera")
            for camera in CAMERAS:
                mask = np.asarray(masks[camera])
                if mask.shape != np.asarray(observation["observation/depths"][camera]).shape or mask.dtype.kind not in "iu" or np.any((mask < -1) | (mask > 5)):
                    raise ValueError("predicted role masks must be aligned integer -1..5")
            tracker_observation = dict(observation, **{"observation/instance_masks": masks})
            objects, metadata = self.tracker.observe(tracker_observation, modes=modes)
        for obj in objects:
            if not obj.visible:
                modes[obj.object_id] = "unknown"
        motion = self.forecaster.observe_and_predict(objects, metadata, modes, now)
        return dict(timestamp=now, objects=objects, tracking=metadata, modes=modes,
                    motion=motion, segmentation=segmentation,
                    manipulated_mode_unresolved=self.manipulated_mode_unresolved,
                    mode_evidence="measured open gripper before any observed close; immutable non-graspable CAD roles",
                    missing_held_feedback=True)


class BlindPublicMotionPipeline:
    """Always run both fixed public frontends; no task/speed argument exists."""
    def __init__(self, root, segment_checkpoint=None, *, segmenter=None, motion_checkpoint=None):
        if segmenter is None and (segment_checkpoint is None or not Path(segment_checkpoint).is_file()):
            raise FileNotFoundError("blind tracking requires a fitted public segmenter on every scene")
        self.canonical_prior_audit = audit_canonical_public_priors(root)
        self.frontends = {
            "pick": PublicMotionPipeline(root, "pick_moving_target_on_the_table", motion_checkpoint=motion_checkpoint),
            "insert": PublicMotionPipeline(root, "insert_onto_rotating_peg", segment_checkpoint=segment_checkpoint,
                                           segmenter=segmenter, motion_checkpoint=motion_checkpoint),
        }
        self.roles = ("target", *ROLE_IDS)
        self.role_prior_ids = {role: 0 if role == "target" else 1 for role in self.roles}
        self.prior_receipt = dict(
            mode="blind_parallel_public_frontends", canonical_prior_audit=self.canonical_prior_audit,
            frontends={key: pipeline.prior_receipt for key, pipeline in self.frontends.items()},
            stable_roles=list(self.roles), role_prior_ids=self.role_prior_ids,
            all_frontends_run_on_every_scene=True, task_id_or_description_input=False,
            wrong_scene_frontends_not_filtered_using_labels=True)

    @property
    def checkpoint_sha256(self):
        values = {pipeline.checkpoint_sha256 for pipeline in self.frontends.values()}
        if values != {FORECAST_SHA256}:
            raise ValueError("blind frontends must share the original frozen GRU")
        return FORECAST_SHA256

    @property
    def checkpoint_paths(self):
        return tuple(path for pipeline in self.frontends.values() for path in pipeline.checkpoint_paths)

    def reset(self):
        for pipeline in self.frontends.values():
            pipeline.reset()

    def observe(self, observation, robot):
        # Both calls are unconditional; source task, stage and task-classifier
        # predictions cannot suppress a frontend or alter the geometry prior.
        results = {key: pipeline.observe(observation, robot) for key, pipeline in self.frontends.items()}
        bodies, modes, forecasts, status, objects = {}, {}, {}, {}, []
        for result in results.values():
            bodies.update(result["tracking"]["bodies"])
            modes.update(result["modes"])
            forecasts.update(result["motion"]["forecasts"])
            status.update(result["motion"]["status"])
            objects.extend(result["objects"])
        return dict(timestamp=float(observation["timestamp"]), objects=objects,
                    tracking=dict(bodies=bodies, input_information="blind public RGBD and fixed common TRAIN priors",
                                  source_task_or_speed_used=False), modes=modes,
                    motion=dict(forecasts=forecasts, status=status, checkpoint_sha256=self.checkpoint_sha256,
                                forecast_steps=22, dt_seconds=.05, native_or_expert_poses_used=False,
                                trajectories_fabricated=False),
                    segmentation=results["insert"]["segmentation"],
                    frontend_modes={key: result["manipulated_mode_unresolved"] for key, result in results.items()},
                    all_frontends_evaluated=("pick", "insert"), missing_held_feedback=True)


def iter_public_hdf5_frames(row, *, max_frames=None):
    """Read original 256 RGB-D and matching public K; no label dataset opened."""
    split, task = row.get("split"), row.get("task")
    if split not in ("train", "validation") or task not in TASK_FAMILIES:
        raise ValueError("only declared TRAIN/validation source cases are readable")
    path = Path(row["source_path"]).resolve()
    package = "train_package_compressed" if split == "train" else "val_package_compressed"
    permitted = Path("/home/ubuntu/桌面/phsimani/PhysMani/data/rmt/physmani_bench") / package
    if not path.is_relative_to(permitted.resolve()) or path.name != "data.h5":
        raise ValueError("HDF5 source is outside the exact declared split package")
    before = source_stat(path)
    if row.get("source_file_stat") is not None and before != row["source_file_stat"]:
        raise ValueError("source changed since audited scene preparation")
    with h5py.File(path, "r") as source:
        if str(source.attrs["task"]) != task:
            raise ValueError("source task attribute differs")
        n = int(source.attrs["num_frames"])
        if n != row["frames"]:
            raise ValueError("source frame count differs")
        pose = source["low_dim_obs/gripper_pose"][()]
        opening = source["low_dim_obs/gripper_open"][()]
        joints = source["low_dim_obs/joint_positions"][()]
        robot = np.column_stack((pose, opening, joints)).astype(np.float32)
        if robot.shape != (n, 15) or not np.isfinite(robot).all():
            raise ValueError("invalid public robot history")
        calibrations, rgb, depth = {}, {}, {}
        for camera in CAMERAS:
            parameters = source[f"cam_params/{camera}"]
            extrinsic = parameters["extrinsics"][()]
            calibrations[camera] = CameraCalibration(parameters["intrinsics"][()], Pose(extrinsic[:3, 3], extrinsic[:3, :3]))
            rgb[camera] = source[f"observations/{camera}/rgb"][()]
            encoded_depth = source[f"observations/{camera}/depth"][()]
            if rgb[camera].shape != (n, 256, 256, 3) or encoded_depth.shape != rgb[camera].shape:
                raise ValueError("source RGBD must remain at original 256 resolution with its K")
            near, far = float(parameters["near"][()]), float(parameters["far"][()])
            depth[camera] = np.stack([decode_metric_depth(frame, near, far) for frame in encoded_depth])
        count = n if max_frames is None else min(n, int(max_frames))
        if count < 1:
            raise ValueError("positive development frame limit required")
        for frame in range(count):
            yield dict(timestamp=frame * DT,
                       **{"observation/images": {cam: rgb[cam][frame] for cam in CAMERAS},
                          "observation/depths": {cam: depth[cam][frame] for cam in CAMERAS},
                          "observation/calibrations": calibrations}), robot[frame]
    if source_stat(path) != before:
        raise ValueError("source changed during public tracking replay")


def collect_motion_case(pipeline, frames):
    """Return explicit per-body observations, forecasts, masks and diagnostics."""
    roles = pipeline.roles
    observations, forecasts, valid, receipts, timestamps, frame_seconds = [], [], [], [], [], []
    failures = Counter()
    for observation, robot in frames:
        frame_started = time.perf_counter()
        result = pipeline.observe(observation, robot)
        frame_seconds.append(time.perf_counter() - frame_started)
        by_role = {obj.object_id: obj for obj in result["objects"]}
        observed = np.full((len(roles), 20), np.nan, np.float32)
        tokens = np.full((len(roles), 23, 20), np.nan, np.float32)
        mask = np.zeros((len(roles), 23), bool)
        for index, role in enumerate(roles):
            obj = by_role.get(role)
            if obj is not None and obj.visible:
                observed[index] = np.r_[obj.pose.position, obj.pose.rotation[:, :2].reshape(-1),
                                        obj.linear_velocity, obj.angular_velocity, obj.confidence,
                                        float(result["tracking"]["bodies"][role].get("orientation_observable", False)),
                                        0. if result["modes"][role] == "external" else -1., float(obj.visible), obj.timestamp]
            status = result["motion"]["status"][role]
            if status["available"]:
                tokens[index] = result["motion"]["forecasts"][role]["trajectory"]
                mask[index] = True
            else:
                failures[f"{role}:{status['reason']}"] += 1
        observations.append(observed)
        forecasts.append(tokens)
        valid.append(mask)
        timestamps.append(result["timestamp"])
        receipts.append(dict(timestamp=result["timestamp"], modes=result["modes"],
                             tracking_seconds=frame_seconds[-1],
                             motion_status=result["motion"]["status"],
                             bodies={role: {key: _plain(value) for key, value in diagnostic.items()
                                            if key not in ("axis_evidence", "rotation_equivalence_evidence")}
                                     for role, diagnostic in result["tracking"]["bodies"].items()},
                             segmentation_quality=_plain(result["segmentation"].get("quality")) if result["segmentation"] else None))
    if not forecasts:
        raise ValueError("no public frames collected")
    arrays = dict(timestamps=np.asarray(timestamps, np.float64), role_names=np.asarray(roles),
                  role_prior_ids=np.asarray([getattr(pipeline, "role_prior_ids", {}).get(role, 0) for role in roles], np.int64),
                  observed_public_state=np.stack(observations),
                  predicted_trajectory=np.stack(forecasts), trajectory_mask=np.stack(valid))
    available = arrays["trajectory_mask"].all(-1)
    summary = dict(frames=len(timestamps), role_valid_frames={role: int(available[:, i].sum()) for i, role in enumerate(roles)},
                   role_valid_ratio={role: float(available[:, i].mean()) for i, role in enumerate(roles)},
                   any_role_valid_ratio=float(available.any(-1).mean()),
                   all_roles_valid_ratio=float(available.all(-1).mean()), unavailable_reasons=dict(failures),
                   prior_receipt=_plain(pipeline.prior_receipt),
                   checkpoint_sha256=pipeline.checkpoint_sha256,
                   tracking_seconds_total=float(np.sum(frame_seconds)),
                   tracking_seconds_per_frame_mean=float(np.mean(frame_seconds)),
                   tracking_seconds_per_frame_median=float(np.median(frame_seconds)),
                   tracking_seconds_per_frame_max=float(np.max(frame_seconds)),
                   original_gru_weights_unchanged=True, history_dim=21, state_dim=19,
                   forecast_steps=22, dt_seconds=.05, missing_forecasts_are_nan_and_masked=True,
                   multi_body_trajectories_kept_separate=True, Native_started=False,
                   expert_stage_or_target_inputs=False, native_instance_ids_used=False)
    return arrays, summary, receipts
