"""Small source-bound TRAIN capture with RGBD inputs and separate mask labels.

Nothing in this module imports the native runtime until ``capture_case``.  The
capture uses newly generated native motion, never replays or joins HDF5 actions.
Rendered instance handles are resolved only by the offline annotation writer.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


SCHEMA = "hierarchy_visual_train_capture_v1_20261030"
TASKS = ("insert_onto_rotating_peg", "insert_onto_rotating_peg_high_speed")
CAMERAS = ("front", "8", "16", "36")
ROLES = ("frame", "peg0", "peg1", "peg2", "ring")
ROLE_IDS = {role: i + 1 for i, role in enumerate(ROLES)}
HOLD_TICKS = 20
SAMPLE_EVERY = 5
DT = 0.05
ARTIFACTS = Path("artifacts/framework/multitask_r10/hierarchy_20261030/visual_train_v1")
RUNTIME = Path("runtime/framework/hierarchy_20261030/visual_train_v1")
TRAIN_MANIFEST = Path("artifacts/framework/protocol/train_manifest.json")
CAD_MANIFEST = Path("artifacts/framework/multitask_r10/frozen_TRAIN_geometry_manifest_v11.json")
ENV_MANIFEST = Path("artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/hybrid_environment_manifest_v820.json")
CODE_INPUTS = (
    "our_mani/hierarchy_20261030/native_capture.py",
    "scripts/collect_hierarchy_visual_train_20261030.py",
    "our_mani/multitask/simulation.py", "our_mani/multitask/maps.py",
    "our_mani/multitask/native_tasks.py", "our_mani/multitask/specs.py",
    "our_mani/multitask/training_geometry.py", "our_mani/multitask/component_flags.py",
    "our_mani/multitask/safe_native_mesh.py", "our_mani/multitask/protocol.py",
    "our_mani/aligned/protocol.py", "our_mani/sensor_alignment/author_rgbd.py",
    "scripts/run_frozen_system_environment_v820_r10.py", "scripts/run_local_sim.sh",
)
SHARED_LOCKS = (
    "runtime/framework/fixed_TRAIN104_geometry_native.lock",
    "runtime/framework/TRAIN_postcomplete_diagnostics_v1106.lock",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pointer(root, path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path.relative_to(root)), "sha256": sha(path)}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _checked(root, ref):
    path = (root / ref["path"]).resolve(strict=True)
    if not path.is_relative_to(root) or sha(path) != ref["sha256"]:
        raise ValueError("bound project input changed: " + str(path))
    return path


def _process_identity():
    stat = Path(f"/proc/{os.getpid()}/stat").read_text()
    return {"pid": os.getpid(), "start_ticks": int(stat.rsplit(")", 1)[1].split()[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


@contextmanager
def resource_locks(root):
    """Serialize this namespace and take existing cross-project native locks.

    Existing shared lock files are opened read-only; their contents and old
    STOP files are untouched. A failed nonblocking flock never starts native.
    """
    runtime = root / RUNTIME
    runtime.mkdir(parents=True, exist_ok=True)
    handles, evidence = [], []
    paths = [(runtime / "capture.lock", "a+b")]
    paths.extend((root / name, "rb") for name in SHARED_LOCKS)
    try:
        for path, mode in paths:
            handle = path.open(mode)
            handles.append(handle)
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("native resource lock already held: " + str(path)) from error
            stat = os.fstat(handle.fileno())
            evidence.append({"path": str(path.relative_to(root)), "device": stat.st_dev,
                             "inode": stat.st_ino, "fd": handle.fileno()})
        yield {"process": _process_identity(), "locks": evidence}
    finally:
        for handle in reversed(handles):
            handle.close()


def build_plan(root):
    """Read only the TRAIN roster, camera metadata and frozen TRAIN geometry."""
    root = Path(root).resolve()
    manifest = json.loads((root / TRAIN_MANIFEST).read_text())
    if manifest.get("split") != "train":
        raise ValueError("TRAIN-only source manifest required")
    cad = json.loads((root / CAD_MANIFEST).read_text())
    if cad.get("TRAIN_only") is not True or cad.get("VAL_TEST_used") is not False:
        raise ValueError("frozen TRAIN-only CAD manifest required")
    native_repo = Path("/home/ubuntu/桌面/phsimani/PhysMani").resolve(strict=True)
    train_root = native_repo / "data/rmt/physmani_bench/train_package_compressed"
    cases = []
    for task in TASKS:
        # Preserve the declared roster order; no outcome-dependent selection.
        selected = [row for row in manifest["episodes"] if row["task"] == task][:2]
        if len(selected) != 2:
            raise ValueError("need the first two fixed TRAIN sources for " + task)
        asset_ref = cad["assets"][task]
        asset_path = _checked(root, asset_ref)
        asset = json.loads(asset_path.read_text())
        if asset.get("split") != "train" or asset.get("task") != task or set(asset["bodies"]) != set(ROLES):
            raise ValueError("insert TRAIN CAD role inventory differs")
        for row in selected:
            path = Path(row["path"]).resolve(strict=True)
            episode = row["raw_episode_id"]
            expected = train_root / task / "all_variations/episodes" / episode / "data.h5"
            if path != expected or Path(episode).name != episode:
                raise ValueError("source is outside its declared TRAIN task/episode")
            before = path.stat()
            source_sha = sha(path)
            with h5py.File(path, "r") as source:
                variation = int(source.attrs["variation"])
                for camera in CAMERAS:
                    for key, shape in (("intrinsics", (3, 3)), ("extrinsics", (4, 4))):
                        value = source[f"cam_params/{camera}/{key}"][...]
                        if value.shape != shape or not np.isfinite(value).all():
                            raise ValueError("invalid source camera calibration")
                    near = float(np.asarray(source[f"cam_params/{camera}/near"][...]).item())
                    far = float(np.asarray(source[f"cam_params/{camera}/far"][...]).item())
                    if not 0 <= near < far:
                        raise ValueError("invalid source camera clipping planes")
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError("source changed during planning")
            cases.append({"case_key": task + "__" + episode, "task": task,
                          "raw_episode_id": episode, "variation": variation,
                          "source": {"task": task, "raw_episode_id": episode, "path": str(path)},
                          "source_sha256": source_sha, "source_bytes": after.st_size,
                          "geometry": {"path": asset_ref["path"], "sha256": asset_ref["sha256"]}})
    from our_mani.multitask.native_tasks import verify_native_sources
    native_sources = {task: verify_native_sources(native_repo, task) for task in TASKS}
    protocol = {"dt": DT, "simulation_seconds": 10.0, "gripper_velocity": 0.2,
                "baseline_repo": str(native_repo), "current_instance_masks": True,
                "information_setting": "TRAIN annotation capture only; policy inputs restricted to RGBD and calibration"}
    return {"schema": SCHEMA, "split": "train", "tasks": list(TASKS), "cases": cases,
            "selection": "first two sources per task in the frozen TRAIN manifest order",
            "hold_ticks": HOLD_TICKS, "sample_every_ticks": SAMPLE_EVERY,
            "sample_ticks": list(range(0, HOLD_TICKS + 1, SAMPLE_EVERY)),
            "physical_budget_ticks": 200, "protocol": protocol,
            "model_input_channels": ["rgb", "depth_m", "camera_intrinsics", "camera_to_world", "timestamp"],
            "offline_label_channels": ["rendered_instance_ids", "semantic_role_ids", "role_component_handle_bindings"],
            "native_motion_is_new_source_seeded_generation": True, "HDF5_motion_replayed": False,
            "expert_debug_decoded": False, "hold_states_joined_to_HDF5_actions": False,
            "policy_or_teacher_executed": False, "native_source_hashes": native_sources,
            "source_manifest": pointer(root, root / TRAIN_MANIFEST),
            "CAD_manifest": pointer(root, root / CAD_MANIFEST),
            "frozen_environment": pointer(root, root / ENV_MANIFEST),
            "code_inputs": {name: sha(root / name) for name in CODE_INPUTS}}


def verify_plan(root, plan_path):
    root, plan_path = Path(root).resolve(), Path(plan_path).resolve(strict=True)
    if not plan_path.is_relative_to(root / ARTIFACTS):
        raise ValueError("plan must be inside the new visual_train_v1 artifact namespace")
    plan = json.loads(plan_path.read_text())
    expected = build_plan(root)
    if plan != expected:
        raise ValueError("capture plan or its bound TRAIN inputs changed; use a new version")
    return plan


def check_plan(root, plan_path):
    root, plan_path = Path(root).resolve(), Path(plan_path).resolve()
    if not plan_path.is_relative_to(root / ARTIFACTS):
        raise ValueError("plan must be inside the new artifact namespace")
    with resource_locks(root) as resources:
        plan = build_plan(root)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("existing immutable capture plan differs")
        else:
            atomic_json(plan_path, plan)
        receipt = {"schema": SCHEMA + "_CPU_check", "checked_utc": utc(),
                   "plan": pointer(root, plan_path), "case_keys": [x["case_key"] for x in plan["cases"]],
                   "native_started": False, "resources": resources}
        destination = root / RUNTIME / "CPU_check.json"
        if destination.with_suffix(".json.partial").exists():
            raise ValueError("unfinished CPU check receipt requires inspection")
        atomic_json(destination, receipt)
        return receipt


def frame_payloads(observation, bindings, tick):
    """Pure data conversion: inputs contain no mask, handle, pose GT or action."""
    if set(bindings) != set(ROLES):
        raise ValueError("all five TRAIN label roles required")
    id_to_role = {}
    for role, components in bindings.items():
        for values in components.values():
            for identity in values:
                if int(identity) in id_to_role:
                    raise ValueError("one current instance ID cannot annotate two components")
                id_to_role[int(identity)] = ROLE_IDS[role]
    rgb, depth, intrinsics, extrinsics, instances, semantic = [], [], [], [], [], []
    for camera in CAMERAS:
        image = np.asarray(observation["observation/images"][camera])
        distance = np.asarray(observation["observation/depths"][camera], dtype=np.float32)
        encoded = np.asarray(observation["observation/instance_masks"][camera])
        if image.dtype != np.uint8 or image.shape != (256, 256, 3) or distance.shape != (256, 256):
            raise ValueError("original aligned 256x256 RGBD required")
        if not np.isfinite(distance).all() or np.any(distance < 0):
            raise ValueError("nonfinite or negative metric depth")
        if encoded.shape == (256, 256, 3):
            if not np.isfinite(encoded).all() or np.any(encoded < 0) or np.any(encoded > 1):
                raise ValueError("normalized native RGB24 mask expected")
            channels = np.rint(encoded * 255).astype(np.uint32)
            labels = channels[..., 0] + 256 * channels[..., 1] + 65536 * channels[..., 2]
        elif encoded.shape == (256, 256) and encoded.dtype.kind in "ui":
            labels = encoded.astype(np.uint32)
        else:
            raise ValueError("expected current rendered RGB24 or integer instance mask")
        role_labels = np.zeros((256, 256), dtype=np.uint8)
        for identity, role in id_to_role.items():
            role_labels[labels == identity] = role
        calibration = observation["observation/calibrations"][camera]
        if not np.isfinite(calibration.depth_scale) or calibration.depth_scale <= 0:
            raise ValueError("finite positive public depth scale required")
        distance = distance * float(calibration.depth_scale)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = calibration.camera_to_world.rotation
        matrix[:3, 3] = calibration.camera_to_world.position
        intrinsic = np.asarray(calibration.intrinsics, dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or not np.isfinite(matrix).all():
            raise ValueError("finite public camera matrices required")
        rgb.append(image.copy()); depth.append(distance.copy())
        intrinsics.append(intrinsic.copy())
        extrinsics.append(matrix); instances.append(labels); semantic.append(role_labels)
    timestamp = float(observation["timestamp"])
    if abs(timestamp - tick * DT) > 1e-8:
        raise ValueError("capture timestamp differs from counted tick")
    inputs = {"rgb": np.stack(rgb), "depth_m": np.stack(depth),
              "camera_intrinsics": np.stack(intrinsics), "camera_to_world": np.stack(extrinsics),
              "camera_names": np.asarray(CAMERAS), "timestamp": np.asarray(timestamp),
              "counted_tick": np.asarray(tick, dtype=np.int64)}
    labels = {"rendered_instance_ids": np.stack(instances), "semantic_role_ids": np.stack(semantic),
              "role_names": np.asarray(ROLES), "role_ids": np.asarray(list(ROLE_IDS.values()), dtype=np.uint8),
              "timestamp": np.asarray(timestamp), "counted_tick": np.asarray(tick, dtype=np.int64)}
    return inputs, labels


def _write_npz(path, arrays):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(arrays) or any(not np.array_equal(loaded[key], value) for key, value in arrays.items()):
            raise ValueError("capture NPZ round-trip changed arrays")


def capture_case(root, plan_path, case_key):
    """Run exactly one fresh-process TRAIN hold case, only when explicitly called."""
    root = Path(root).resolve()
    plan = verify_plan(root, plan_path)
    selected = [case for case in plan["cases"] if case["case_key"] == case_key]
    if len(selected) != 1:
        raise ValueError("select exactly one planned TRAIN case key")
    if not os.environ.get("DISPLAY"):
        raise ValueError("explicit independent DISPLAY / xvfb-run is required")
    # Verify the existing environment wrapper before importing native modules.
    from scripts.run_frozen_system_environment_v820_r10 import namespace_context
    environment = namespace_context(plan["frozen_environment"]["sha256"])
    case = selected[0]
    output = root / ARTIFACTS / "cases" / case_key
    runtime = root / RUNTIME / "cases" / case_key
    stop = root / RUNTIME / "STOP"
    if stop.exists():
        raise RuntimeError("new visual capture namespace STOP exists")
    if output.exists() or runtime.exists():
        raise ValueError("fresh case namespace required; no automatic overwrite or retry")
    with resource_locks(root) as resources:
        from our_mani.multitask.simulation import AlignedDevelopmentSimulation
        from our_mani.multitask.training_geometry import declared_instance_bindings
        output.mkdir(parents=True)
        runtime.mkdir(parents=True)
        for folder in ("inputs", "labels"):
            (output / folder).mkdir()
        atomic_json(runtime / "launch.json", {"schema": SCHEMA, "started_utc": utc(),
                    "case_key": case_key, "plan": pointer(root, plan_path),
                    "resources": resources, "frozen_environment_context": environment,
                    "display": os.environ["DISPLAY"], "behavior": "hold_only_20_counted_ticks"})
        env = None
        record = None
        failure = None
        frames = []
        terminal = {"schema": SCHEMA, "case_key": case_key, "complete": False,
                    "natural_20_tick_capture": False, "administrative_stop": False}
        try:
            env = AlignedDevelopmentSimulation(case["task"], case["source"]["path"],
                    source=case["source"], split="train", protocol=plan["protocol"], seed=0, replica=0)
            env.reset_development(variation=case["variation"])
            asset = json.loads(_checked(root, case["geometry"]).read_text())
            # Annotation-only mapping: this helper reads names/handles, no poses.
            before = (env.physics_time(), float(env.task.t), env.ledger.ticks)
            bindings = declared_instance_bindings(env, asset, shared_permission=True)
            if before != (env.physics_time(), float(env.task.t), env.ledger.ticks):
                raise RuntimeError("label binding advanced the physical clock")
            atomic_json(output / "labels" / "instance_bindings.json", {
                "schema": SCHEMA + "_offline_labels", "current_bindings": bindings,
                "role_ids": ROLE_IDS, "role_ids_zero_means": "background_or_unlabelled",
                "frozen_TRAIN_instance_ids_used_as_current_ids": False,
                "native_body_poses_read_by_label_writer": False, "model_input_allowed": False})
            while True:
                tick = env.ledger.ticks
                if tick % SAMPLE_EVERY == 0:
                    clock = (env.physics_time(), float(env.task.t), tick)
                    observation, _ = env.observe()
                    inputs, labels = frame_payloads(observation, bindings, tick)
                    if clock != (env.physics_time(), float(env.task.t), env.ledger.ticks):
                        raise RuntimeError("sensor capture advanced physics outside counted hold")
                    name = f"frame_{tick:05d}.npz"
                    input_path, label_path = output / "inputs" / name, output / "labels" / name
                    _write_npz(input_path, inputs); _write_npz(label_path, labels)
                    frames.append({"tick": tick, "inputs": pointer(root, input_path),
                                   "labels": pointer(root, label_path)})
                if tick == HOLD_TICKS or env._native_terminal_latched:
                    break
                if stop.exists():
                    terminal["administrative_stop"] = True
                    break
                # hold_tick alone advances and accounts one physical/task step.
                env.hold_tick()
                if env.ledger.ticks != tick + 1:
                    raise RuntimeError("hold did not account exactly one tick")
            terminal.update(counted_ticks=env.ledger.ticks, clock=env.ledger.receipt(),
                            native_terminal_observed=bool(env._native_terminal_latched),
                            natural_20_tick_capture=env.ledger.ticks == HOLD_TICKS,
                            complete=env.ledger.ticks == HOLD_TICKS and not terminal["administrative_stop"])
            record = {"schema": SCHEMA, "split": "train", "case": case,
                      "plan": pointer(root, plan_path), "frames": frames,
                      "source_use": "identity/seed and camera calibration; newly generated hold motion",
                      "inputs_directory": "inputs", "offline_supervision_directory": "labels",
                      "HDF5_expert_actions_used": False, "native_poses_used_as_policy_inputs": False,
                      "teacher_or_policy_executed": False, "terminal": terminal}
        except BaseException as error:
            failure = error
            terminal.update(error_type=type(error).__name__, error=str(error), complete=False)
        finally:
            if env is not None:
                try:
                    env.close()
                except BaseException as error:
                    terminal.update(cleanup_error=repr(error), complete=False)
                    if failure is None:
                        failure = error
            terminal.update(finished_utc=utc(), frames=len(frames),
                            infrastructure_failed=failure is not None, process=_process_identity())
            atomic_json(runtime / "terminal.json", terminal)
            if record is not None:
                atomic_json(output / "case.json", record)
        if failure is not None:
            raise failure
        return terminal
