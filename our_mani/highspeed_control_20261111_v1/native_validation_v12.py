"""Bounded high-speed Native checks with the repaired executor and native judge."""
from pathlib import Path
from copy import deepcopy
import json
import numpy as np
from ..execution_interface_20261111_v1.native_validation import AuditedNativeBackend,make_executor,boundary_checks
from ..local_mpc_20261102.records import append,write_new

def run_case(spec, output, stop_requested):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    backend = None
    closed, error, result = False, None, None
    try:
        from ..cube_native_contact_feedback_live_20261104_v1.backend import NativeContactCubeBackendV1
        # Retain partial construction for shutdown if initialization raises.
        backend = NativeContactCubeBackendV1.__new__(NativeContactCubeBackendV1)
        NativeContactCubeBackendV1.__init__(backend, spec["task"], spec["calibration_episode"],
            seed=spec["seed"], variation=0, budget_ticks=160,
            gripper_velocity=spec["config"]["gripper_velocity"], execute_native=True)
        write_new(output / "initial_snapshot.json", backend.initial_snapshot)
        if spec["kind"] == "pipeline":
            from ..local_mpc_control_20261103.completed_gripper import CompletedGripperBackend
            from ..cube_native_contact_feedback_live_20261104_v1.observer import ObservedNativeContactCubeBackendV1
            root = Path(__file__).resolve().parents[2]
            c = spec["config"]
            backend = CompletedGripperBackend(backend)
            backend = ObservedNativeContactCubeBackendV1(backend, output / "public_observer",
                candidate_mode="matched_candidate_with_corrected_forecast",
                checkpoint=root / c["candidate_transformer_checkpoint"],
                checkpoint_sha256=c["candidate_transformer_checkpoint_sha256"],
                forecast_checkpoint=root / c["corrected_forecast_checkpoint"],
                forecast_checkpoint_sha256=c["corrected_forecast_checkpoint_sha256"],
                refiner_checkpoint=root / c["position_refiner_checkpoint"],
                refiner_checkpoint_sha256=c["position_refiner_checkpoint_sha256"], save_frames=False)
            backend.sample()
        backend = AuditedNativeBackend(backend, output)
        executor = make_executor(backend, spec["config"])
        write_new(output / "interface_capabilities.json", executor.capabilities())
        if spec["kind"] == "boundary":
            result = boundary_checks(backend, executor, output)
        else:
            from .producer_v12 import run
            result = run(backend, executor, output, spec["config"], stop_requested=stop_requested)
            result.update(actual_ticks=len(backend.ticks), servo_writes=len(backend.commands),
                maximum_servo_command_difference_rad=max((r["maximum_difference_rad"] for r in backend.commands), default=0.),
                Native_task_or_physics_modified=False)
        if len(backend.commands) != len(backend.ticks):
            raise AssertionError("servo write / actual tick counts differ")
        result.update(task=spec["task"], seed=spec["seed"], kind=spec["kind"],
                      dataset_role="generated_TRAIN_development", formal_VAL_TEST=False)
        write_new(output / "result.json", result)
        print(json.dumps(dict(case=spec["id"], complete=result["complete"], actual_ticks=result["actual_ticks"],
                              episode_success=result.get("episode_success")), ensure_ascii=False), flush=True)
        return result
    except BaseException as exception:
        error = type(exception).__name__ + ": " + str(exception)
        raise
    finally:
        if backend is not None:
            backend.close()
            closed = bool(getattr(backend, "_closed", False))
        write_new(output / "worker_exit.json", dict(native_closed=closed, error=error, result_recorded=result is not None))
