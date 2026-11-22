"""One fixed TRAIN interface validation, using real Native after --execute only.

This checks the shared interface, measured feedback, and actuator scheduling.
It does not run a learned task policy or report a benchmark success rate.
"""
import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def plain(value):
    if is_dataclass(value): return plain(asdict(value))
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    if hasattr(value, 'tolist'): return value.tolist()
    return value


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(plain(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify(plan_path, digest):
    if sha(plan_path) != digest: raise ValueError('External frozen validation plan SHA required')
    plan = read(plan_path)
    source_map = plan['method_sources']
    if sha(source_map['path']) != source_map['sha256']: raise ValueError('Validation source map changed')
    sources = read(source_map['path'])['source_sha256']
    for path, expected in sources.items():
        if sha(path) != expected: raise ValueError('Frozen validation source changed: ' + path)
    return plan, sources


def execute(plan_path, digest, case_id, output):
    plan, sources = verify(plan_path, digest)
    case = next(c for c in plan['cases'] if c['id'] == case_id)
    calibration_stat = Path(case['calibration_episode']).stat()
    actual_stat = dict(size=calibration_stat.st_size, mtime_ns=calibration_stat.st_mtime_ns,
        inode=calibration_stat.st_ino)
    if actual_stat != case['calibration_stat']: raise ValueError('Fixed TRAIN calibration file identity changed')
    if output.resolve() != (ROOT / case['output']).resolve(): raise ValueError('Fixed fresh case output required')
    if output.exists(): raise FileExistsError('No retry or output overwrite')
    # Use the same isolated file environment as the working Cube runs.
    from scripts.run_frozen_system_environment_v820_r10 import require_environment, MANIFEST
    environment = require_environment(MANIFEST, plan['environment_manifest_sha256'])
    import numpy as np
    from our_mani.physmani_feedback_interface_20261104_v1.backend import PhysManiFeedbackBackendV1
    from our_mani.physmani_feedback_interface_20261104_v1.facade import FeedbackFacadeV1
    from our_mani.physmani_feedback_interface_20261104_v1.observation import legacy_policy_packet
    output.mkdir(parents=True, exist_ok=False)
    (output / 'public_frames').mkdir()
    write_new(output / 'request.json', dict(case=case, source_sha256=sources,
        plan=dict(path=str(plan_path.resolve()), sha256=digest), environment=environment,
        scope='generated TRAIN interface mechanics validation; not learned task policy'))
    backend = facade = None
    result = dict(case_id=case_id, task=case['task'], complete=False,
        interface_validation_passed=False, native_closed=False, cycles=[],
        task_policy_evaluated=False, formal_VAL_TEST=False, NN_forward_calls=0,
        source_unchanged=False)
    begin = time.monotonic()
    exit_code = 0

    def stopped():
        paths = (output / 'STOP', output.parent / 'STOP', ROOT / 'runtime/framework/STOP')
        if any(p.exists() for p in paths): raise InterruptedError('Administrative STOP')

    def snapshot(index):
        stopped()
        before = backend.clock()
        value = facade.read_snapshot()
        if backend.clock() != before: raise RuntimeError('Observation read advanced physics')
        packet = legacy_policy_packet(value)
        path = output / 'public_frames' / f'{index:03d}.npz'
        np.savez_compressed(path, **{k: v for k, v in value.items() if isinstance(v, np.ndarray)})
        metadata = {k: v for k, v in value.items() if not isinstance(v, np.ndarray)}
        metadata.update(file=path.name, sha256=sha(path),
            legacy_policy_robot14=packet['robot14'], gripper_scalar_observed=False,
            observed_physics_steps=0)
        write_new(path.with_suffix('.json'), metadata)
        return value

    try:
        stopped()
        backend = PhysManiFeedbackBackendV1(case['task'], case['calibration_episode'],
            seed=case['seed'], variation=case['variation'], budget_ticks=16,
            execute_native=True, dataset_use='train',
            renderer_profile=plan['renderer_profile'])
        facade = FeedbackFacadeV1(backend)
        result['provenance'] = backend.provenance
        initial = snapshot(0)
        parked = initial['q'].copy()
        limits = backend.joint_limits()
        retarget = parked.copy()
        retarget[6] += .005 if parked[6] + .005 <= limits['upper'][6] else -.005
        # Exercise stale commands and unsupported budgets without any physics.
        before = backend.clock()
        pose_plan = facade.submit_target(initial['tcp_pose'][:3], initial['tcp_pose'][3:],
            phase='same_TCP_pose_zero_step_submission')
        if backend.clock() != before: raise RuntimeError('TCP planning advanced physics')
        result['same_pose_IK'] = plain(backend.last_ik_receipt)
        result['superseded_TCP_plan_id'] = pose_plan
        stale = facade.submit_joint_target(parked, phase='superseded_hold')
        newest = facade.submit_joint_target(parked, phase='initial_hold')
        stale_result = facade.advance(stale)
        if stale_result.get('status') != 'stale_plan' or backend.clock() != before:
            raise RuntimeError('Stale plan caused execution or was not reported')
        try:
            facade.advance(newest, max_ticks=1)
        except ValueError:
            pass
        else:
            raise RuntimeError('Unsupported v1 physical budget was silently accepted')
        if backend.clock() != before: raise RuntimeError('Rejected budget advanced physics')
        result['zero_step_checks'] = dict(TCP_submission=True, stale_plan_rejected=True,
            unsupported_budget_rejected=True)
        schedule = ('KEEP', 'KEEP', 'CLOSE', 'KEEP', 'KEEP', 'OPEN', 'KEEP', 'KEEP')
        all_accepted = True
        goal = parked
        for cycle, request in enumerate(schedule):
            stopped()
            if backend.terminal():
                result['truncated_before_cycle'] = cycle
                break
            if cycle == 1: goal = retarget
            before = backend.clock()
            plan_id = newest if cycle == 0 else facade.submit_joint_target(
                goal, gripper_request=request, phase=f'interface_cycle_{cycle}_{request}')
            if backend.clock() != before: raise RuntimeError('Submit advanced physics')
            feedback = facade.advance(plan_id)
            after = backend.clock()
            ticks = feedback.get('executed_ticks')
            if type(ticks) is not int or ticks < 0 or ticks > 2:
                raise RuntimeError('Actual physical tick budget was not returned')
            if abs(after - before - ticks * .05) > 2e-5:
                raise RuntimeError('Unexpected physical advancement outside reported ticks')
            row = dict(cycle=cycle, command=request, plan_id=plan_id,
                clock_before=before, clock_after=after, feedback=feedback,
                private_native_metrics=facade.metric_receipt())
            result['cycles'].append(plain(row))
            write_new(output / f'cycle{cycle:02d}.json', row)
            snapshot(cycle + 1)
            if feedback.get('status') not in ('cycle_completed', 'truncated'):
                all_accepted = False
                result['command_rejected_reason'] = feedback.get('status')
                break
        result.update(complete=True, interface_validation_passed=all_accepted,
            actual_postreset_ticks=backend._ticks, actual_physical_duration=backend.clock(),
            full_16tick_actuator_schedule_completed=backend._ticks == 16,
            private_native_metrics=facade.metric_receipt(),
            final_actuator_state=facade.executor.state())
    except BaseException as error:
        exit_code = 1
        result['error'] = dict(type=type(error).__name__, message=str(error))
    finally:
        if backend is not None:
            try:
                backend.close()
                result['native_closed'] = True
            except BaseException as error:
                exit_code = 1
                result['shutdown_error'] = dict(type=type(error).__name__, message=str(error))
        try:
            verify(plan_path, digest)
            result['source_unchanged'] = True
        except BaseException as error:
            exit_code = 1
            result['source_error'] = str(error)
        result['wall_seconds'] = time.monotonic() - begin
        write_new(output / 'result.json', result)
        write_new(output / 'worker_exit.json', dict(actual_worker_returncode=exit_code,
            native_closed=result['native_closed'], result_recorded=True,
            source_unchanged=result['source_unchanged']))
    print(json.dumps({k: v for k, v in result.items() if k not in ('cycles', 'provenance')}, default=plain))
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--case-id')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    plan, sources = verify(args.plan, args.plan_sha256)
    if not args.execute:
        print(json.dumps(dict(complete=True, source_files=len(sources), cases=len(plan['cases']),
            Native_started=False, NN_forward_calls=0)))
        return 0
    if not args.case_id or args.output is None: parser.error('Explicit fixed case and fresh output required')
    return execute(args.plan, args.plan_sha256, args.case_id, args.output)


if __name__ == '__main__': raise SystemExit(main())
