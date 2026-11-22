"""Private TRAIN teacher attempt. Default is source-only; Root owns Native."""
import argparse,json,traceback
from pathlib import Path
from copy import deepcopy
from .contracts import ROOT,SCHEMA,REFERENCE,read,validate_plan,source_map,check_freeze
from ..local_mpc_20261102.records import write_new,append,archive_sources,verify_sources,utc


def check(plan_path,parent_id,freeze=None,freeze_sha256=None,strict=False):
    plan=validate_plan(plan_path);parent=next((p for p in plan['parents']if p['parent_id']==parent_id),None)
    if parent is None:raise ValueError('Only the12 predeclared fit/selection teacher parents, never heldout')
    missing=[]
    if plan.get('preparation_only')is not False or plan.get('execution_authorized')is not True:missing.append('Root_final_plan')
    if freeze is None or freeze_sha256 is None:missing.append('final_teacher_source_freeze')
    if not missing:check_freeze(plan_path,freeze,freeze_sha256)
    if strict and missing:raise ValueError('Teacher not admitted: '+', '.join(missing))
    return dict(parent_id=parent_id,episode_id=parent['episode_id'],role=parent['role'],ready=not missing,
        missing=missing,Native_started=False,NN_forward_calls=0,privileged_teacher=True,formal_public_task_score=False)


def collect(plan_path,parent_id,output,*,freeze,freeze_sha256):
    check(plan_path,parent_id,freeze,freeze_sha256,strict=True)
    plan=validate_plan(plan_path);parent=next(p for p in plan['parents']if p['parent_id']==parent_id)
    sources,native=check_freeze(plan_path,freeze,freeze_sha256)
    output=Path(output).resolve();family=ROOT/'runtime/framework/cube_exact_future_teacher_TRAIN_20261104_v2'
    if family not in output.parents or output.exists():raise ValueError('Fresh teacher episode only; no resumption')
    stops=(output/'STOP',output.parent/'STOP',family/'STOP',ROOT/'runtime/framework/STOP',ROOT/'runtime/framework/local_mpc_20261102/STOP')
    if any(p.exists()for p in stops):raise ValueError('Administrative STOP preserved')
    from scripts.feedback_current_environment_20261104_v2 import current_environment
    environment=current_environment();output.mkdir(parents=True,exist_ok=False)
    write_new(output/'request.json',dict(plan=str(Path(plan_path).resolve()),parent=parent,
        environment=environment,source_sha256=sources,native_files_sha256=native,
        policy_uses_privileged_current_Cube_centre_velocity_held_goal=True,
        policy_uses_privileged_conditional_Native_future_schedule=True,
        formal_public_task_score=False,teacher_planner_reference='canonical_Cube_I; rawNativeR unchanged'))
    raw=observer=executor=None;recorder=None;code=1
    try:
        archive_sources(output,sources)
        from ..cube_motion_capture_recovery_20261104_v2.backend import CubeMotionObservationBackendV2
        from .observer import TeacherObservationRecorder
        from .teacher import CanonicalCubeTeacherPolicyView,run_privileged_teacher
        from ..local_mpc_control_20261103.completed_gripper import CompletedGripperBackend
        from ..local_mpc_control_20261103.public_candidate_pixel_centre_reverification_v70 import PublicCandidatePixelCentreReverificationBridgeV70
        from ..generic_reference_mpc_20261104_v1.train_labels import TRAINLabelRecorder,TRAINRecordingExecutor
        from ..local_mpc_20261102.executor import JointLimits,ExecutionConfig
        raw=CubeMotionObservationBackendV2(parent['task'],parent['calibration_episode'],seed=parent['seed'],
            variation=0,budget_ticks=160,execute_native=True)
        write_new(output/'initial_snapshot_PRIVATE.json',raw.initial_snapshot)
        bridge=PublicCandidatePixelCentreReverificationBridgeV70(
            refiner_checkpoint=ROOT/plan['position_checkpoint']['path'],
            refiner_checkpoint_sha256=plan['position_checkpoint']['sha256'])
        completed=CompletedGripperBackend(raw)
        observer=TeacherObservationRecorder(completed,output/'public_motion',bridge=bridge,parent=parent,episode_id=parent['episode_id'])
        last={'close':False,'attempt':0}
        def labels(row):
            close=row['requested_opening']<.5
            if close and not last['close']:last['attempt']+=1
            last['close']=close
            append(output/'private_labels.jsonl',dict(row,episode_id=parent['episode_id'],parent_id=parent_id,
                role=parent['role'],pass_kind='private_exact_future_contact_teacher_v2',timestamp=row['actual_relative_physics_timestamp'],
                physics_time=row['physics_time_absolute'],task_time=row['task_time_absolute'],
                reset_boundary_id=parent['episode_id'],attempt_id=parent['episode_id']+'_attempt'+str(last['attempt'])if close else None,
                close_requested=close,collecting_policy_uses_Native_current_state=True,
                collecting_policy_uses_private_exact_future_schedule=True,
                planner_reference='canonical_Cube_I_only_in_policyview',raw_Native_quaternion_retained=True,
                native_velocity_or_acceleration_setpoint_issued=False))
        recorder=TRAINLabelRecorder(raw,labels)  # never bind the canonical planner view
        view=CanonicalCubeTeacherPolicyView(observer);limits=raw.joint_limits()
        executor=TRAINRecordingExecutor(view,JointLimits(lower=limits['lower'],upper=limits['upper'],
            velocity=limits['velocity'],acceleration=12.,jerk=1000.),ExecutionConfig(),initial_gripper_opening=1.,recorder=recorder)
        observer.sample();recorder.record_initial(executor.state())
        config=deepcopy(read(REFERENCE));config.update(task=parent['task'],seed=parent['seed'],simulation_seconds=8.,
            calibration_train_episode=parent['calibration_episode'],dataset_use='train',formal_evaluation=False)
        result=run_privileged_teacher(view,executor,output,config)
        if observer.public_rows!=raw._ticks+1 or recorder.rows!=observer.public_rows:
            raise RuntimeError('One initial row plus one public/private row per real tick required')
        verify_sources(sources);check_freeze(plan_path,freeze,freeze_sha256)
        result.update(parent=parent,episode_id=parent['episode_id'],pass_kind='private_exact_future_contact_teacher_v2',
            actual_physics_ticks=raw._ticks,actual_physical_duration=raw.clock(),
            public_rows=observer.public_rows,private_label_rows=recorder.rows,
            source_unchanged=True,attempt_complete=True,privileged_teacher=True,
            original_P1_strict_dwell_not_V72_comparison=True,
            private_exact_future_oracle=True,public_policy_score=False,all_deadline_expiry_immediate=True,public_task_success_rate_evaluated=False,
            actual_Native_quaternion_label_unchanged=True,raw_private_snapshot_function_unchanged=True,
            gripper_completion=completed.gripper_completion_receipt(),finished_utc=utc())
        write_new(output/'result.json',result);code=0
    except BaseException as error:
        write_new(output/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            partial_public_rows=None if observer is None else observer.public_rows,
            partial_private_rows=None if recorder is None else recorder.rows,no_retry=True))
        traceback.print_exc()
    finally:
        closed=False
        try:
            if raw is not None:raw.close();closed=True
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
        s,n=source_map(a.plan);print(json.dumps(dict(schema='Cube_private_exact_future_teacher_execution_freeze_v2',files_sha256=s,native_files_sha256=n),allow_nan=False));return 0
    if not a.collect:
        plan=validate_plan(a.plan);parents=[p for p in plan['parents']if a.parent_id in (None,p['parent_id'])]
        if a.parent_id is not None and not parents:raise ValueError('Heldout or unknown teacher parent forbidden')
        print(json.dumps(dict(complete=True,parents=[check(a.plan,x['parent_id'],a.source_freeze,a.source_freeze_sha256)for x in parents],Native_started=False,NN_forward_calls=0),allow_nan=False));return 0
    if not all((a.parent_id,a.output,a.source_freeze,a.source_freeze_sha256)):
        p.error('Collect requires predeclared parent/fresh output/external freezeSHA')
    return collect(a.plan,a.parent_id,a.output,freeze=a.source_freeze,freeze_sha256=a.source_freeze_sha256)


if __name__=='__main__':raise SystemExit(main())
