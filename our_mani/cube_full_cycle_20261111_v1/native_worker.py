"""Scoped Cube collection/evaluation; private TRAIN labels never enter the observer."""
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
        from .backends import SourceContactCubeBackend, ArchiveContactCubeBackend
        kind = ArchiveContactCubeBackend if spec['source_split']=='test' else SourceContactCubeBackend
        backend = kind.__new__(kind)
        kind.__init__(backend,spec,execute_native=True)
        write_new(output / "initial_snapshot.json", backend.initial_snapshot)
        if spec["kind"] == "pipeline":
            from ..local_mpc_control_20261103.completed_gripper import CompletedGripperBackend
            from .evaluation_observer import ObservedCycleCubeBackend as ObservedNativeContactCubeBackendV1
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
                refiner_checkpoint_sha256=c["position_refiner_checkpoint_sha256"], save_frames=True)
            backend.sample()
            write_new(output / "public_sphere.json", backend.tracker.sphere)
        if spec['role'] in ('fit', 'selection'):
            backend = TrainingLabelRecorder(backend, output / 'private_TRAIN_labels.jsonl', spec['episode_id'])
        backend = AuditedNativeBackend(backend, output)
        executor = make_executor(backend, spec["config"])
        write_new(output / "interface_capabilities.json", executor.capabilities())
        if spec["kind"] == "boundary":
            result = boundary_checks(backend, executor, output)
        else:
            from ..perception_repair_20261111_v1.producer import run
            result = run(backend, executor, output, spec["config"], stop_requested=stop_requested)
            result.update(actual_ticks=len(backend.ticks), servo_writes=len(backend.commands),
                maximum_servo_command_difference_rad=max((r["maximum_difference_rad"] for r in backend.commands), default=0.),
                Native_task_or_physics_modified=False)
        if len(backend.commands) != len(backend.ticks):
            raise AssertionError("servo write / actual tick counts differ")
        result.update(task=spec["task"], seed=spec["seed"], kind=spec["kind"],
                      dataset_role=spec["role"], parent_id=spec["parent_id"], formal_VAL_TEST=False, generated_heldout=spec['source_split']!='test',
                      official_source_id=spec['raw_episode_id'],source_split=spec['source_split'],variation=spec['variation'],
                      official_TEST_reset=spec['source_split']=='test',paper_10s_protocol_comparable=False)
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


class TrainingLabelRecorder:
    """Independent post-tick sink; the public producer receives no label tensors."""
    KEYS = ('time', 'physics_time', 'task_time', 'object_position', 'tcp_position',
            'tcp_quaternion', 'held', 'native_success', 'requested_opening')

    def __init__(self, backend, destination, episode_id):
        self.backend, self.destination, self.episode_id = backend, Path(destination), episode_id
        if self.destination.exists(): raise FileExistsError(self.destination)
        self.previous_time = None
        self.record()

    def __getattr__(self, name): return getattr(self.backend, name)

    def record(self):
        before = float(self.backend.clock())
        raw = self.backend.read_state()
        row = {key: deepcopy(raw[key]) for key in self.KEYS}
        row['episode_id'] = self.episode_id
        if abs(float(row['time']) - before) > 1e-8 or float(self.backend.clock()) != before:
            raise RuntimeError('Private supervision read advanced or misaligned physics')
        if self.previous_time is not None and abs(before-self.previous_time-.05) > 1e-6:
            raise RuntimeError('Private TRAIN labels require one row per actual50ms')
        if self.previous_time is None and abs(before) > 1e-8:
            raise RuntimeError('Initial private TRAIN label must be actual origin')
        append(self.destination, row)
        self.previous_time = before

    def tick(self, phase):
        state = self.backend.tick(phase)
        self.record()
        return state
