"""Single-attempt true-clock TRAIN observation collector. Root owns Native."""
import argparse
from copy import deepcopy
from pathlib import Path
import json,traceback
import numpy as np

from .contracts import ROOT,SCHEMA,validate_plan,source_map,check_freeze
from ..local_mpc_20261102.records import write_new,append,archive_sources,verify_sources,utc


def check(plan_path,parent_id,*,freeze=None,freeze_sha256=None,strict=False):
    plan=validate_plan(plan_path);parent=next((p for p in plan['parents']if p['parent_id']==parent_id),None)
    if parent is None:raise ValueError('Exact predeclared parent required')
    missing=[]
    if plan.get('preparation_only')is not False or plan.get('execution_authorized')is not True:missing.append('Root_final_plan')
    if freeze is None or freeze_sha256 is None:missing.append('final_source_freeze')
    if not missing:check_freeze(plan_path,freeze,freeze_sha256)
    if strict and missing:raise ValueError('Capture execution is not ready: '+', '.join(missing))
    return dict(schema=SCHEMA,parent_id=parent_id,episode_id=parent['episode_id'],role=parent['role'],
        ready=not missing,missing=missing,Native_started=False,NN_forward_calls=0,
        contact_candidate_supervision_ready=False,raw_HDF_arrays_read=False)


def collect(plan_path,parent_id,output,*,freeze,freeze_sha256):
    check(plan_path,parent_id,freeze=freeze,freeze_sha256=freeze_sha256,strict=True)
    plan=validate_plan(plan_path);parent=next(p for p in plan['parents']if p['parent_id']==parent_id)
    sources,native=check_freeze(plan_path,freeze,freeze_sha256)
    output=Path(output).resolve();family=ROOT/'runtime/framework/cube_motion_TRAIN_20261104_v2'
    if family not in output.parents or output.exists():raise ValueError('Fresh independent motion pass only; no resume/retry')
    stops=(output/'STOP',output.parent/'STOP',family/'STOP',ROOT/'runtime/framework/STOP',ROOT/'runtime/framework/local_mpc_20261102/STOP')
    if any(p.exists()for p in stops):raise ValueError('Administrative STOP preserved')
    from scripts.feedback_current_environment_20261104_v2 import current_environment
    environment=current_environment();output.mkdir(parents=True,exist_ok=False)
    write_new(output/'request.json',dict(started_utc=utc(),plan=str(Path(plan_path).resolve()),parent=parent,
        source_sha256=sources,native_files_sha256=native,environment=environment,
        teacher_policy='initial_measured_q_hold_OPEN_not_contact_teacher',GT_policy_inputs=False))
    backend=observer=executor=None;code=1;label_recorder=None
    try:
        archive_sources(output,sources)
        from .backend import CubeMotionObservationBackendV2
        from .collector import MotionObservationRecorder
        from ..local_mpc_control_20261103.completed_gripper import CompletedGripperBackend
        from ..local_mpc_control_20261103.public_candidate_pixel_centre_reverification_v70 import PublicCandidatePixelCentreReverificationBridgeV70
        from ..local_mpc_20261102.executor import JointLimits,ExecutionConfig
        from ..generic_reference_mpc_20261104_v1.train_labels import TRAINLabelRecorder,TRAINRecordingExecutor
        from ..generic_reference_mpc_20261104_v1.reference_mpc import qualify_first_action
        backend=CubeMotionObservationBackendV2(parent['task'],parent['calibration_episode'],
            seed=parent['seed'],variation=0,budget_ticks=160,execute_native=True)
        write_new(output/'initial_snapshot_PRIVATE.json',backend.initial_snapshot)
        bridge=PublicCandidatePixelCentreReverificationBridgeV70(
            refiner_checkpoint=ROOT/plan['position_checkpoint']['path'],
            refiner_checkpoint_sha256=plan['position_checkpoint']['sha256'])
        observer=MotionObservationRecorder(CompletedGripperBackend(backend),output/'public_motion',
            bridge=bridge,parent=parent,episode_id=parent['episode_id'])
        def labels(row):
            append(output/'private_labels.jsonl',dict(row,episode_id=parent['episode_id'],
                parent_id=parent_id,role=parent['role'],pass_kind='keep_open_motion',
                timestamp=row['actual_relative_physics_timestamp'],physics_time=row['physics_time_absolute'],
                task_time=row['task_time_absolute'],reset_boundary_id=parent['episode_id'],attempt_id=None,
                close_requested=False,native_velocity_or_acceleration_setpoint_issued=False,
                q_reference_is_native_target=True,dq_ddq_are_controller_references_not_Native_motor_setpoints=True))
        label_recorder=TRAINLabelRecorder(backend,labels)
        limits=backend.joint_limits()
        executor=TRAINRecordingExecutor(observer,JointLimits(lower=limits['lower'],upper=limits['upper'],
            velocity=limits['velocity'],acceleration=12.,jerk=1000.),ExecutionConfig(),
            initial_gripper_opening=1.,recorder=label_recorder)
        observer.sample();label_recorder.record_initial(executor.state())
        hold_q=np.asarray(backend.get_joint_positions()).copy();cycles=0;reason='budget_exhausted'
        while backend.remaining_ticks()>=2 and not backend.terminal():
            if any(p.exists()for p in stops):raise InterruptedError('Administrative STOP during attempt')
            actual=dict(time=backend.clock(),q=backend.get_joint_positions())
            allowed,qualification=qualify_first_action(backend,actual,{'joint_goal':hold_q})
            append(output/'qualification.jsonl',dict(time=actual['time'],allowed=allowed,qualification=qualification))
            if not allowed:
                reason='safe_initial_q_qualification_rejected';break  # retain partial; never substitute another action
            receipt=executor.execute_joint_target(hold_q,opening=1.,phase='TRAIN_keep_open_motion_observation')
            append(output/'executor.jsonl',receipt.to_dict());cycles+=1
        if backend.terminal()and backend.remaining_ticks()>0:reason='Native_terminal_or_fault'
        if observer.public_rows!=backend._ticks+1 or label_recorder.rows!=observer.public_rows:
            raise RuntimeError('Exactly one public frame and one committed private label per actual tick, including initial row, required')
        verify_sources(sources);check_freeze(plan_path,freeze,freeze_sha256)
        write_new(output/'result.json',dict(complete=True,attempt_complete=True,parent=parent,
            episode_id=parent['episode_id'],pass_kind='keep_open_motion',cycles=cycles,
            actual_physics_ticks=backend._ticks,public_rows=observer.public_rows,private_label_rows=label_recorder.rows,
            full160tick_schedule=backend._ticks==160,stop_reason=reason,source_unchanged=True,
            actual_physical_duration=backend.clock(),actual_physical_clock=True,
            target_UNKNOWN_rows_retained=True,contact_candidate_supervision_ready=False,
            candidate_inference_calls=0,teacher_uses_Native_body_as_action_input=False,
            original_Native_task_conditions_unchanged=True,formal_task_success_evaluated=False,
            private_metrics=deepcopy(backend._last_native_result),
            gripper_completion=observer.backend.gripper_completion_receipt()))
        code=0
    except BaseException as error:
        write_new(output/'failure.json',dict(type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc(),partial_public_rows=None if observer is None else observer.public_rows,
            partial_private_rows=None if label_recorder is None else label_recorder.rows,
            no_retry=True,failed_parent_retained=True))
        traceback.print_exc()
    finally:
        closed=False
        try:
            if backend is not None:backend.close();closed=True
        finally:write_new(output/'worker_exit.json',dict(actual_worker_returncode=code,native_closed=closed,
            result_recorded=(output/'result.json').exists(),partial_rows_retained=True,finished_utc=utc()))
    return code


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--parent-id');p.add_argument('--output',type=Path)
    p.add_argument('--source-freeze',type=Path);p.add_argument('--source-freeze-sha256')
    p.add_argument('--collect',action='store_true');p.add_argument('--print-source-freeze',action='store_true');a=p.parse_args(argv)
    if a.print_source_freeze:
        if a.collect:p.error('Source inspection cannot execute')
        s,n=source_map(a.plan);print(json.dumps(dict(schema='Cube_trueclock_motion_capture_execution_freeze_v2',
            files_sha256=s,native_files_sha256=n),allow_nan=False));return 0
    if not a.collect:
        plan=validate_plan(a.plan);parents=[x for x in plan['parents']if a.parent_id in (None,x['parent_id'])]
        print(json.dumps(dict(complete=True,parents=[check(a.plan,x['parent_id'],freeze=a.source_freeze,
            freeze_sha256=a.source_freeze_sha256)for x in parents],Native_started=False,NN_forward_calls=0),allow_nan=False));return 0
    if not all((a.parent_id,a.output,a.source_freeze,a.source_freeze_sha256)):
        p.error('--collect requires exact parent, fresh output and external final source freeze/SHA')
    return collect(a.plan,a.parent_id,a.output,freeze=a.source_freeze,freeze_sha256=a.source_freeze_sha256)


if __name__=='__main__':raise SystemExit(main())
