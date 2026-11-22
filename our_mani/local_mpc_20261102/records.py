"""Append-only native transitions and auditable P0/P1 run provenance."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def utc():
    return datetime.now(timezone.utc).isoformat()


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "INFEASIBLE_POSITIVE_INFINITY" if value > 0 else "NONFINITE"
    return value


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(plain(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(plain(value), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def freeze_sources(config_path, stage=None):
    initial_config=json.loads(Path(config_path).read_text())
    dark=initial_config.get('candidate_darkred_target_v59',False)
    if type(dark) is not bool:
        raise ValueError('explicit boolean candidate dark-red v59 modifier required')
    if dark and (stage!='p1' or initial_config.get('public_grasp_candidate_transformer_v5') is not True
                 or initial_config.get('candidate_goal_sphere_filter_v56') is not True
                 or initial_config.get('postconfirmed_position_maintenance_v56') is not True):
        raise ValueError('dark-red v59 requires candidate p1 with goal filtering and position maintenance')
    fast=initial_config.get('finger_response_fastpath',False)
    if type(fast) is not bool:
        raise ValueError('explicit boolean finger-response computation modifier required')
    if fast and (stage!='p1' or initial_config.get('public_grasp_candidate_transformer_v5') is not True
                 or initial_config.get('response_factory')!='coupled_joint_20261103_v1'
                 or initial_config.get('response_controller')!='coupled'):
        raise ValueError('finger fastpath requires the coupled public candidate p1 route')
    paths = sorted((ROOT / "our_mani/local_mpc_20261102").glob("*.py"))
    if stage == "p1":
        paths += sorted((ROOT / "our_mani/local_mpc_p2_20261103").glob("*.py"))
        paths += sorted((ROOT / "our_mani/local_mpc_p2_20261103_v2").glob("*.py"))
        paths += [ROOT / "our_mani/local_mpc_control_20261103" / p for p in
                  ("__init__.py", "joint_goals.py", "planner_factory.py")
                  if (ROOT / "our_mani/local_mpc_control_20261103" / p).is_file()]
        config = json.loads(Path(config_path).read_text())
        v56_flags=(config.get("candidate_goal_sphere_filter_v56",False),
                   config.get("postconfirmed_position_maintenance_v56",False))
        if any(type(flag) is not bool for flag in v56_flags):
            raise ValueError("explicit boolean candidate-v56 feature modifiers required")
        if any(v56_flags) and config.get("public_grasp_candidate_transformer_v5") is not True:
            raise ValueError("v56 modifiers require the public candidate route")
        active_visual = any(config.get(key, False) for key in (
            "public_grasp_active_visual_v4", "public_grasp_active_visual_v41", "public_grasp_active_visual_v42",
            "public_grasp_cold_probe_v43", "public_grasp_candidate_transformer_v5"))
        if config.get("response_factory") == "coupled_joint_20261103_v1":
            paths += [ROOT / "our_mani/local_mpc_control_20261103" / p for p in
                      ("response_factory.py", "coupled_response.py", "joint_goal_cem.py")]
            # The factory loads these exact external payloads as well as its
            # Python sources. Archive them prospectively; older source120
            # runs retain their original, explicitly narrower archive scope.
            runtime_payloads={
                "artifacts/framework/local_mpc_20261102/P2_servo_training_v2/best.pt":
                    "25a0e0cbd49e0e0c9e8a4671649f96a1bb565013011d728e88b2c2f3eaca0073",
                "runtime/framework/local_mpc_20261102/p2_fk_canary_v1/result.json":
                    "2a3d301dbd0926760b74ba6b68fc1de85f448c402c215d0f6877e7059278600a",
                "artifacts/framework/local_mpc_20261102/P2_data_v2/manifest.json":
                    "bf7187fcb47b7e1392d762915bc3a577b0d472be124c8a6ecc7dbe65bec2af1d",
                "artifacts/framework/local_mpc_20261102/P2_servo_training_v2/source_archive.json":
                    "6e004e03e3c3829a4504b5fb43b3200839721e3c190000ce7ba00f9a7bcdf4ff"}
            for relative,expected in runtime_payloads.items():
                path=ROOT/relative
                if sha(path)!=expected:raise ValueError("factory runtime/training evidence payload changed: "+relative)
                paths.append(path)
            if config.get("response_controller") == "coupled":
                paths.append(ROOT / "artifacts/framework/local_mpc_20261102/P2_coupled_response_v1/B_parameters_frozen_20261103.json")
        if (config.get("public_motion_shadow", False) or config.get("public_pregrasp_control", False)
                or config.get("public_grasp_prototype", False) or active_visual):
            paths += [ROOT / p for p in (
                "our_mani/local_mpc_control_20261103/public_motion_bridge.py",
                "our_mani/local_mpc_control_20261103/public_shadow.py",
                "our_mani/benchmark_perception.py", "our_mani/goal_perception.py",
                "our_mani/context_dynamics.py", "our_mani/multitask/context_se3.py",
                "artifacts/framework/multitask_r10/context_joint16_after_executor_v671/dynamics.pt",
                "artifacts/framework/local_policy/goal_marker_calibration.json")]
        if config.get("public_pregrasp_control", False):
            paths += [ROOT / "our_mani/local_mpc_control_20261103" / p for p in
                      ("public_pregrasp.py", "public_state.py", "forecast_control.py")]
            if config.get("candidate_generation") == "immediate_expiry":
                paths.append(ROOT / "our_mani/local_mpc_control_20261103/temporal_candidates.py")
        if config.get("public_grasp_prototype", False) or active_visual:
            paths += [ROOT / "our_mani/local_mpc_control_20261103" / p for p in
                      ("public_grasp.py", "public_grasp_state.py", "public_attachment.py",
                       "public_state.py", "forecast_control.py", "temporal_candidates.py", "joint_goal_cem.py")]
            if (config.get("public_grasp_settings",{}).get("verification_probe_enabled",False)
                    or active_visual):
                paths.append(ROOT / "our_mani/local_mpc_control_20261103/public_grasp_probe.py")
        if active_visual:
            paths += [ROOT / "our_mani/local_mpc_control_20261103" / p for p in
                      ("active_visual_attachment.py", "public_grasp_v4.py")]
        if (config.get("public_grasp_active_visual_v41",False) or config.get("public_grasp_active_visual_v42",False)
                or config.get("public_grasp_cold_probe_v43",False) or config.get("public_grasp_candidate_transformer_v5",False)):
            paths.append(ROOT/"our_mani/local_mpc_control_20261103/public_grasp_v41.py")
        if config.get("public_grasp_active_visual_v42",False):
            paths += [ROOT / "our_mani/local_mpc_control_20261103" / p for p in
                      ("public_grasp_v42.py", "goal_aware_binding_v42.py", "goal_aware_perception.py")]
        if config.get("public_grasp_cold_probe_v43",False):
            paths.append(ROOT/"our_mani/local_mpc_control_20261103/public_grasp_cold_probe_v43.py")
        if config.get("public_grasp_candidate_transformer_v5",False):
            paths.append(ROOT/"our_mani/local_mpc_control_20261103/public_grasp_candidates_v5.py")
            if dark:
                paths += [ROOT/"our_mani/local_mpc_control_20261103"/name for name in
                          ("darkred_goal_perception_v59.py", "goal_aware_binding_v59.py", "public_candidate_darkred_v59.py")]
            if fast:
                paths += [ROOT/"our_mani/local_mpc_control_20261103"/name for name in
                          ("finger_response_fastpath_20261103.py", "finger_fastpath_integration_20261103.py")]
            if any(v56_flags):
                paths += [ROOT/"our_mani/local_mpc_control_20261103"/name for name in
                          ("public_candidate_goal_maintenance_v56.py", "postconfirmed_maintenance_candidate_20261103.py",
                           "public_grasp_v42.py", "goal_aware_binding_v42.py", "goal_aware_perception.py")]
            paths += [ROOT/"our_mani/local_mpc_p4_20261103"/name for name in
                      ("__init__.py", "model.py", "data.py", "candidates.py", "runtime.py")]
            paths += [ROOT/"scripts"/name for name in
                      ("train_p4_candidate_20261103.py", "compile_p4_public_dataset_20261103.py")]
            if config.get("candidate_pi_symmetry_representative",False):
                paths.append(ROOT/"our_mani/local_mpc_p4_20261103/symmetry_runtime.py")
            if config.get("require_public_goal_completion",False):
                paths.append(ROOT/"our_mani/local_mpc_control_20261103/public_grasp_goal_completion_v52.py")
            if config.get("cache_fixed_probe_nominal_ik",False):
                paths.append(ROOT/"our_mani/local_mpc_control_20261103/probe_nominal_cache_v53.py")
            if config.get("probe_terminal_at_fixed_deadline",False):
                if config.get("cache_fixed_probe_nominal_ik") is not True:
                    raise ValueError("fixed probe deadline requires explicit cached nominal protocol")
                paths.append(ROOT/"our_mani/local_mpc_control_20261103/probe_deadline_v54.py")
            if config.get("probe_onpolicy_joint_response",False):
                if (config.get("cache_fixed_probe_nominal_ik") is not True
                        or config.get("probe_terminal_at_fixed_deadline") is not True):
                    raise ValueError("on-policy probe response requires fixed nominal and absolute deadline protocols")
                from ..local_mpc_control_20261103.probe_onpolicy_response_v55 import probe_response_sources
                for source in probe_response_sources():
                    path=Path(source).resolve()
                    path.relative_to(ROOT)
                    paths.append(path)
            paths.append(ROOT/config["candidate_transformer_checkpoint"])
            if config.get("candidate_model_freeze_manifest"):
                manifest_path=ROOT/config["candidate_model_freeze_manifest"]
                manifest=json.loads(manifest_path.read_text())
                if (manifest.get("schema")!="P4_model_freeze_before_newdevVAL_v1"
                        or manifest.get("training_complete") is not True or manifest.get("fit_split")!="train"
                        or manifest.get("new_devVAL_used_for_fit") is not False or manifest.get("formal_TEST_used") is not False
                        or manifest.get("checkpoint")!=config["candidate_transformer_checkpoint"]
                        or manifest.get("files_sha256",{}).get(manifest["checkpoint"])!=config["candidate_transformer_checkpoint_sha256"]):
                    raise ValueError("exact prior TRAIN-only candidate model freeze required")
                paths.append(manifest_path)
                for relative,expected in manifest["files_sha256"].items():
                    relative_path=Path(relative)
                    if relative_path.is_absolute() or ".." in relative_path.parts or sha(ROOT/relative_path)!=expected:
                        raise ValueError("candidate freeze evidence changed: "+relative)
                    paths.append(ROOT/relative_path)
    elif stage == "fk":
        paths += [ROOT / "our_mani/local_mpc_p2_20261103" / p
                  for p in ("__init__.py", "kinematics.py")]
    elif stage == "gripper":
        paths += [ROOT / "our_mani/local_mpc_control_20261103" / p for p in
                  ("__init__.py", "gripper_geometry_audit.py")]
    paths += [ROOT / p for p in (
        "our_mani/aligned/simulation.py", "our_mani/aligned/protocol.py",
        "our_mani/pyrep_env.py", "our_mani/reachability.py", "our_mani/contracts.py",
        "our_mani/perception.py", "our_mani/multitask/specs.py")]
    if json.loads(Path(config_path).read_text()).get("gripper_completion_latch",False):
        paths.append(ROOT / "our_mani/local_mpc_control_20261103/completed_gripper.py")
    if json.loads(Path(config_path).read_text()).get("diagnostic_no_proxy_attachment",False):
        paths.append(ROOT / "our_mani/local_mpc_control_20261103/no_proxy_attachment.py")
    if stage == "p0":
        paths = [p for p in paths if p.name != "mpc.py"]
    paths += [ROOT / "scripts/run_local_mpc_20261102.py",
              ROOT / "scripts/launch_local_mpc_20261102.py",
              ROOT / "scripts/run_local_mpc_20261102.sh", Path(config_path).resolve()]
    return {str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(paths)}


def archive_sources(output, frozen):
    for relative, expected in frozen.items():
        source = ROOT / relative
        target = Path(output) / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            with source.open("rb") as original:
                shutil.copyfileobj(original, stream)
        if sha(target) != expected:
            raise RuntimeError("source raced during archival: " + relative)


def verify_sources(frozen):
    changed = [p for p, digest in frozen.items() if sha(ROOT / p) != digest]
    if changed:
        raise RuntimeError("source changed during diagnostic run: " + repr(changed))


def dependency_receipt():
    import cffi
    import _cffi_backend
    import scipy
    import pyrep
    return dict(python=sys.executable, version=sys.version,
                numpy=np.__version__, scipy=scipy.__version__, cffi=cffi.__version__,
                cffi_backend=_cffi_backend.__file__, pyrep=pyrep.__file__)


def quantiles(values):
    return dict(count=len(values), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)), max=float(max(values))) if values else dict(count=0)


class HoldMetric:
    """Separate diagnostic hold criterion; never replaces the Native result."""

    def __init__(self, seconds=.5, height=.04, max_relative_drift=.02):
        self.seconds, self.height, self.max_relative_drift = seconds, height, max_relative_drift
        self.start_time = None
        self.first_relative = None
        self.initial_height = None
        self.longest_seconds = 0.
        self.max_lift = 0.
        self.passed = False
        self.drops = 0
        self.was_held = False

    def update(self, state):
        p = np.asarray(state["object_position"])
        if self.initial_height is None:
            self.initial_height = float(p[2])
        self.max_lift = max(self.max_lift, float(p[2]) - self.initial_height)
        held = bool(state["held"])
        self.drops += int(self.was_held and not held)
        self.was_held = held
        if not held or float(p[2]) - self.initial_height < self.height - 1e-9:
            self.start_time, self.first_relative = None, None
            return
        from scipy.spatial.transform import Rotation
        relative = Rotation.from_quat(state["tcp_quaternion"]).inv().apply(
            p - np.asarray(state["tcp_position"]))
        if self.start_time is None:
            self.start_time = float(state["time"])
            self.first_relative = relative
        drift = float(np.linalg.norm(relative - self.first_relative))
        if drift > self.max_relative_drift:
            self.start_time = float(state["time"])
            self.first_relative = relative
        duration = float(state["time"]) - self.start_time
        self.longest_seconds = max(self.longest_seconds, duration)
        self.passed |= duration + 1e-9 >= self.seconds

    def receipt(self):
        return dict(passed=self.passed, longest_held_seconds=self.longest_seconds,
                    maximum_lift_m=self.max_lift, drops=self.drops,
                    required_seconds=self.seconds, required_lift_m=self.height,
                    max_relative_drift_m=self.max_relative_drift,
                    native_success_replaced=False)

