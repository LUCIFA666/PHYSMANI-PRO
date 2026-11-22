import argparse
from pathlib import Path
import time

from .common import ROOT, OUT, read, write, append, pointer, stopped, sources, verify, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stopped()
    case = read(args.case)
    calibration_lock = read(OUT/'calibration_lock.json')
    calibration = str(Path(case['calibration_episode']).resolve(strict=True))
    if calibration_lock.get(calibration) != sha(calibration):
        raise ValueError('Fixed TRAIN camera calibration source changed')
    if case['role'] not in ('fit', 'selection', 'validation', 'test', 'TRAIN_diagnostic'):
        raise ValueError('Explicit prospective small-cycle role required')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = sources()
    write(output / 'request.json', dict(case=case, case_file=pointer(args.case), source_sha256=frozen,
                                       generated_reset=True, archived_VAL_TEST=False,
                                       old_queues_resumed=False))
    from ..multitask_executor_cycle_20261112_v6.backend import TaskExecutionBackend
    from ..multitask_executor_cycle_20261112_v6.runtime import PublicContinuousHistoryPolicy
    from .controller import run
    from ..execution_interface_20261111_v1.executor import PhysManiExecutor
    from ..local_mpc_20261102.executor import JointLimits
    backend = None
    started = time.perf_counter()
    try:
        backend = TaskExecutionBackend(case['task'], case['calibration_episode'], seed=case['seed'],
                                       variation=case['variation'], task_time_limit_seconds=60,
                                       execute_native=True, dataset_use='train' if case['role'] in
                                       ('fit', 'selection', 'TRAIN_diagnostic') else 'development',
                                       renderer_profile='official_all_GL')
        backend.arm.receipt_sink = lambda value: append(output/'atomic_path_queries.jsonl', value)
        native = backend.joint_limits()
        limits = JointLimits(lower=native['lower'], upper=native['upper'], velocity=native['velocity'],
                              acceleration=12., jerk=1000., provenance=native.get('provenance', {}))
        executor = PhysManiExecutor(backend, limits, initial_gripper_opening=backend.requested_opening)
        policy = PublicContinuousHistoryPolicy(case.get('checkpoint'), case.get('checkpoint_sha256'))
        import hashlib
        import numpy as np
        def observe_frame(packet,frame_index):
            policy.observe(packet,frame_index)
            arrays={k:v for k,v in packet.items()if isinstance(v,np.ndarray)}
            evidence=dict(frame_index=frame_index,timestamp=packet['timestamp'],
                          array_sha256={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()for k,v in arrays.items()},
                          saved_RGBD=None)
            if frame_index%20==0:
                folder=output/'continuous_public_frames';folder.mkdir(exist_ok=True)
                path=folder/f'{frame_index:04d}.npz'
                with path.open('xb')as stream:np.savez_compressed(stream,timestamp=np.asarray(packet['timestamp']),**arrays)
                evidence['saved_RGBD']=str(path)
            append(output/'continuous_public_frame_provenance.jsonl',evidence)
        backend.on_public_frame=observe_frame
        backend.public_sample()
        write(output / 'initial.json', dict(backend=backend.provenance,
                                             instructions=backend.descriptions, executor=executor.capabilities(),
                                             policy_checkpoint_sha256=policy.checkpoint_sha256))
        result = run(backend, executor, policy, output)
        result.update(native_path_queries=backend.arm.query_count,
                      temporary_q_changes_restored=backend.arm.restored_changed_q_queries,
                      last_path_query=backend.arm.last_query_receipt)
        result.update(public_actual50ms_frames=policy.observed_frames,
                      inference_device='cuda_BF16',GPU_fitting_during_Native=False)
        verify(frozen)
        result.update(task=case['task'], seed=case['seed'], role=case['role'],
                      source_unchanged=True, wall_seconds=time.perf_counter()-started)
        backend.close()
        result['native_closed'] = True
        write(output / 'result.json', result)
        print(result, flush=True)
        return 0
    except BaseException as error:
        write(output / 'failure.json', dict(complete=False, error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        if backend is not None:
            backend.close()


if __name__ == '__main__':
    raise SystemExit(main())
