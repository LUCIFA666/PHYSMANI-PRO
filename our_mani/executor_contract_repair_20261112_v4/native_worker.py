"""Executor-only qualification with robot-FK goals, without a policy network."""
import argparse
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation

from .common import OUT, read, write, append, pointer, stopped, sources, verify, sha
from .pose_executor import WorldPoseGoal, PoseGoalExecutor


def pose_error(actual, desired):
    actual = np.asarray(actual, float)
    return dict(position_m=float(np.linalg.norm(actual[:3] - desired[:3, 3])),
                rotation_rad=float((Rotation.from_quat(actual[3:]).inv() *
                                    Rotation.from_matrix(desired[:3, :3])).magnitude()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stopped()
    case = read(args.case)
    if case['role'] != 'TRAIN_executor_diagnostic':
        raise ValueError('This is an executor qualification, not a task evaluation')
    calibration = str(Path(case['calibration_episode']).resolve(strict=True))
    if read(OUT/'calibration_lock.json').get(calibration) != sha(calibration):
        raise ValueError('Fixed TRAIN camera calibration changed')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = sources()
    write(output/'request.json', dict(case=case, case_file=pointer(args.case), source_sha256=frozen,
                                    model_forward_calls=0, optimizer_steps=0,
                                    target_source='read_only_robot_FK_not_task_object_state'))
    from .backend import TaskExecutionBackend
    from ..execution_interface_20261111_v1.executor import PhysManiExecutor
    from ..local_mpc_20261102.executor import JointLimits
    from ..local_mpc_p2_20261103.kinematics import export_native_kinematic_snapshot, PoEKinematics
    backend = None
    started = time.perf_counter()
    try:
        backend = TaskExecutionBackend(case['task'], calibration, seed=case['seed'], variation=0,
                                       task_time_limit_seconds=60, execute_native=True,
                                       dataset_use='train', renderer_profile='official_all_GL')
        native = backend.joint_limits()
        limits = JointLimits(lower=native['lower'], upper=native['upper'], velocity=native['velocity'],
                             acceleration=12., jerk=1000., provenance=native.get('provenance', {}))
        executor = PhysManiExecutor(backend, limits, initial_gripper_opening=backend.requested_opening)
        dispatcher = PoseGoalExecutor(backend, executor)
        snapshot = export_native_kinematic_snapshot(backend.arm, clock=backend._clock_tuple)
        fk = PoEKinematics(snapshot)
        q0 = np.asarray(backend.get_joint_positions(), float)
        if not limits.lower[0] + .3 < q0[0] < limits.upper[0] - .3:
            raise ValueError('Qualification requires room for the two specified joint-0 goals')
        export_error = pose_error(backend.arm.get_tip().get_pose(), fk(q0))
        write(output/'robot_kinematics.json', dict(snapshot=snapshot, initial_fk_error=export_error))
        write(output/'initial.json', dict(backend=backend.provenance, executor=executor.capabilities(),
                                        task_success_is_not_the_probe_acceptance=True))
        phases = []
        rapid=case.get('probe_pattern')=='rapid_retargets'
        specifications=[(.4,0.,1),(-.4,0.,1),(.4,1.,1),(-.4,0.,1),(.2,0.,1),(-.2,1.,1),(0.,1.,5)]if rapid else[(.2,0.,5),(-.15,1.,5)]
        actual_path=[q0.copy()]
        for index, (offset, opening, chunk_count) in enumerate(specifications):
            q_goal = q0.copy()
            q_goal[0] += offset
            target = fk(q_goal)
            quat = Rotation.from_matrix(target[:3, :3]).as_quat()
            phase_start = backend._ticks
            phase_clock = backend.clock()
            chunks = []
            for chunk in range(chunk_count):
                stopped()
                if backend.terminal():
                    break
                action = WorldPoseGoal(backend.clock(), target[:3, 3], quat, opening)
                receipt = dispatcher.execute(action, stop_requested=lambda: (stopped() or False))
                append(output/'chunks.jsonl', dict(phase=index, chunk=chunk, **receipt))
                for item in receipt['execution']:
                    append(output/'execution.jsonl', dict(probe_phase=index, chunk=chunk, **item))
                    actual_path.append(np.asarray(item['feedback']['q']))
                chunks.append({k:v for k,v in receipt.items() if k != 'execution'})
                if receipt['status'] != 'chunk_complete':
                    break
            error = pose_error(backend.arm.get_tip().get_pose(), target)
            phase = dict(index=index, desired_pose=np.r_[target[:3, 3], quat].tolist(),
                         known_reachable_joint_goal=q_goal.tolist(),
                         initial_clock=phase_clock, final_clock=backend.clock(),
                         actual_ticks=backend._ticks-phase_start, opening=opening,
                         expected_ticks=chunk_count*10,settling_phase=chunk_count==5,
                         actual_tcp_pose=backend.arm.get_tip().get_pose(), error=error,
                         gripper_done=executor.gripper_done,
                         measured_finger_open_amounts=backend.gripper.get_open_amount(), chunks=chunks)
            phases.append(phase)
            write(output/f'phase{index}.json', phase)
            print(f"{case['task']} phase{index}: {phase['actual_ticks']} ticks, "
                  f"TCP error {error['position_m']:.6f}m/{error['rotation_rad']:.6f}rad, "
                  f"gripper_done={executor.gripper_done}", flush=True)
        servo_error = backend.servo_readback_max_error_rad
        from ..executor_contract_repair_20261112_v2.audit import audit_execution
        reference_audit=audit_execution(output/'execution.jsonl')
        settled=[p for p in phases if p['settling_phase']]
        travel=float(np.linalg.norm(np.diff(np.asarray(actual_path),axis=0),axis=1).sum())
        checks = dict(all_phases_expected_actual_ticks=all(p['actual_ticks']==p['expected_ticks']for p in phases)and len(phases)==len(specifications),
                      actual_command_finite_differences_bounded=reference_audit['all_reference_differences_within_limits'],
                      settled_TCP_position_within_5mm=bool(settled)and all(p['error']['position_m']<.005 for p in settled),
                      settled_TCP_rotation_within_002rad=bool(settled)and all(p['error']['rotation_rad']<.02 for p in settled),
                      settled_gripper_completed=all(p['gripper_done'] is True for p in settled),
                      robot_actually_moved=travel>(.5 if rapid else .1),
                      no_prohibited_hardware_table_contacts=backend.prohibited_contact_frames==0,
                      exact_servo_reference_readback=servo_error<=1e-6,
                      clock_matches_ticks=abs(backend.clock()-backend._ticks*.05)<2e-5,
                      FK_initial_matches_native=export_error['position_m']<1e-6 and export_error['rotation_rad']<1e-5)
        verify(frozen)
        result = dict(complete=True, probe_passed=all(checks.values()), checks=checks, phases=phases,
                      probe_pattern='rapid_retargets'if rapid else'settled_targets',actual_joint_travel_L2_rad=travel,
                      hardware_contact_audit=backend.hardware_contact_audit(),
                      issued_reference_audit=reference_audit,
                      actual_ticks=backend._ticks, actual_seconds=backend.clock(),
                      task=case['task'], seed=case['seed'], role=case['role'],
                      original_task_metric=backend.native_task_judge_receipt(),
                      source_unchanged=True, model_forward_calls=0, optimizer_steps=0,
                      formal_VAL_TEST=False, task_ability_evaluated=False,
                      servo_reference_readback_max_error_rad=servo_error,
                      exact_servo_writes=backend.servo_writes,
                      wall_seconds=time.perf_counter()-started)
        backend.close()
        result['native_closed'] = True
        write(output/'result.json', result)
        print(dict(task=case['task'], probe_passed=result['probe_passed'], checks=checks), flush=True)
        return 0
    except BaseException as error:
        write(output/'failure.json', dict(complete=False, error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        if backend is not None:
            backend.close()


if __name__ == '__main__':
    raise SystemExit(main())
