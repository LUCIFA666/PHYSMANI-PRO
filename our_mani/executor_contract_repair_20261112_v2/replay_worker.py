"""Same precomputed servo stream, isolating IK queries and no-op restorations."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import OUT, read, write, append, pointer, sources, stopped, sha, verify


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();stopped();case=read(args.case)
    if case['role']!='TRAIN_executor_replay_diagnostic':raise ValueError('TRAIN replay diagnostic only')
    mode=case['query_mode']
    if mode not in ('none','noop_restore','native_ik'):raise ValueError('Explicit query comparison mode required')
    calibration=str(Path(case['calibration_episode']).resolve(strict=True))
    if read(OUT/'calibration_lock.json').get(calibration)!=sha(calibration):raise ValueError('Calibration changed')
    reference_path=Path(case['references_path']);decisions_path=Path(case['decisions_path'])
    reference_rows=[json.loads(v)for v in reference_path.read_text().splitlines()]
    decisions=[json.loads(v)for v in decisions_path.read_text().splitlines()]
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    frozen=sources();frozen[str(reference_path.resolve())]=case['references_sha256']
    frozen[str(decisions_path.resolve())]=case['decisions_sha256'];verify(frozen)
    write(output/'request.json',dict(case=case,case_file=pointer(args.case),source_sha256=frozen,
                                    model_forward_calls=0,recorded_position_stream_replay=True))
    from ..multitask_executor_cycle_20261112_v4.backend import TaskExecutionBackend
    from ..execution_interface_20261111_v1.executor import PhysManiExecutor
    from ..execution_interface_20261111_v1.commands import JointTrajectory
    from ..local_mpc_20261102.executor import JointLimits
    from pyrep.backend import sim
    backend=None
    try:
        backend=TaskExecutionBackend(case['task'],calibration,seed=case['seed'],variation=0,
                    task_time_limit_seconds=60,execute_native=True,dataset_use='train',renderer_profile='official_all_GL')
        native=backend.joint_limits()
        limits=JointLimits(lower=native['lower'],upper=native['upper'],velocity=native['velocity'],acceleration=12.,jerk=1000.)
        executor=PhysManiExecutor(backend,limits,initial_gripper_opening=backend.requested_opening)
        initial_q=np.asarray(backend.get_joint_positions());initial_dq=np.asarray(backend.get_joint_velocities())
        first=reference_rows[0]
        if (np.max(abs(initial_q-np.asarray(first['initial_measured_positions'])))>1e-6 or
                np.max(abs(initial_dq-np.asarray(first['initial_measured_velocities'])))>1e-6):
            raise RuntimeError('Replay initial robot state differs; no robot teleport allowed')
        write(output/'initial_motor_state.json',dict(modes=[x.name for x in backend.arm.get_joint_modes()],
            motor_enabled=[int(j.is_motor_enabled())for j in backend.arm.joints],
            control_loop_enabled=[int(j.is_control_loop_enabled())for j in backend.arm.joints],
            max_forces=[sim.simGetJointMaxForce(j.get_handle())for j in backend.arm.joints]))
        query_calls=0;skipped=0;contact_samples=0
        for index,row in enumerate(reference_rows):
            stopped()
            if backend.terminal():break
            if index%10==0 and mode!='none':
                before=backend._clock_tuple()
                q=np.asarray(backend.get_joint_positions());dq=np.asarray(backend.get_joint_velocities())
                servo=np.asarray(backend.arm.get_joint_target_positions());ik=np.asarray(backend.arm._ik_target.get_pose())
                if np.any(q<limits.lower)or np.any(q>limits.upper):
                    skipped+=1
                elif mode=='noop_restore':
                    backend.arm.set_joint_positions(q.tolist(),disable_dynamics=False)
                    backend.arm.set_joint_target_positions(servo.tolist());backend.arm._ik_target.set_pose(ik.tolist())
                    query_calls+=1
                else:
                    ref=decisions[index//10]['reference']
                    backend.solve_ik(ref['position'],ref['quaternion']);query_calls+=1
                assert backend._clock_tuple()==before
                append(output/'queries.jsonl',dict(tick=index,mode=mode,
                    q_error=float(np.max(abs(np.asarray(backend.get_joint_positions())-q))),
                    dq_error=float(np.max(abs(np.asarray(backend.get_joint_velocities())-dq))),
                    servo_error=float(np.max(abs(np.asarray(backend.arm.get_joint_target_positions())-servo)))))
            tick=row['ticks'][0]
            receipt=executor.execute_command(JointTrajectory(start_time=backend.clock(),
                    positions=[tick['reference_positions']],times_from_start=[.05],opening=tick['gripper_target']),
                    phase='recorded_position_reference_replay')
            append(output/'execution.jsonl',receipt.to_dict())
            error=max(abs(v)for v in receipt.ticks[-1]['tracking_error_positions'])
            if index%10==0 or error>.5 and index%5==0:
                before=backend._clock_tuple()
                contacts=sim.simGetContactInfo(sim.sim_handle_all,True)
                named=[]
                for contact in contacts:
                    item=dict(contact)
                    item['contact_names']=[sim.simGetObjectName(h)for h in item['contact_handles']]
                    named.append(item)
                append(output/'motor_contact_diagnostics.jsonl',dict(tick=backend._ticks,clock=backend.clock(),
                    max_tracking_error_rad=error,measured_q=backend.get_joint_positions(),
                    servo=backend.arm.get_joint_target_positions(),torques=backend.arm.get_joint_forces(),contacts=named))
                assert backend._clock_tuple()==before;contact_samples+=1
        from .audit import audit_execution
        audit=audit_execution(output/'execution.jsonl');verify(frozen)
        result=dict(complete=True,task=case['task'],query_mode=mode,actual_ticks=backend._ticks,actual_seconds=backend.clock(),
            replay_reference_source=pointer(reference_path),issued_reference_audit=audit,
            task_success=backend.native_task_judge_receipt()['native_task_success_latched'],
            native_task_judge=backend.native_task_judge_receipt(),query_calls=query_calls,skipped_out_of_bounds_queries=skipped,
            contact_samples=contact_samples,model_forward_calls=0,optimizer_steps=0,source_unchanged=True,
            exact_servo_readback_max_error_rad=backend.servo_readback_max_error_rad)
        backend.close();result['native_closed']=True;write(output/'result.json',result)
        print({k:v for k,v in result.items()if k!='native_task_judge'},flush=True);return 0
    except BaseException as error:
        write(output/'failure.json',dict(complete=False,error_type=type(error).__name__,error=str(error)))
        if hasattr(error,'receipt'):write(output/'failed_execution_receipt.json',error.receipt.to_dict())
        raise
    finally:
        if backend is not None:backend.close()


if __name__=='__main__':raise SystemExit(main())
