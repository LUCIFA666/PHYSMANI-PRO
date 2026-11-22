"""TRAIN oracle-input executor diagnostic. Never a learned-policy benchmark."""
import argparse
import contextlib
import importlib
import io
from pathlib import Path
import numpy as np
from .common import OUT,read,write,append,pointer,sha,sources,verify,stopped

class StageProxy:
    def __init__(self,task):self._task=task;self.stage=task.stage
    def __getattr__(self,name):return getattr(self._task,name)

def query(backend,proxy,expert):
    before=backend._clock_tuple()
    q=np.asarray(backend.get_joint_positions());dq=np.asarray(backend.get_joint_velocities())
    servo=np.asarray(backend.arm.get_joint_target_positions());native_stage=backend.task.stage
    with contextlib.redirect_stdout(io.StringIO()):info=expert(proxy,bool_return_path=False)
    errors=dict(q=float(np.max(abs(np.asarray(backend.get_joint_positions())-q))),
                dq=float(np.max(abs(np.asarray(backend.get_joint_velocities())-dq))),
                servo=float(np.max(abs(np.asarray(backend.arm.get_joint_target_positions())-servo))))
    if (before!=backend._clock_tuple()or max(errors.values())>1e-9 or backend.task.stage!=native_stage
            or backend.task._bool_expert):raise RuntimeError('Oracle read changed Native state')
    proxy.stage=info['stage']
    return np.asarray(info['trajectory'])[0,0].copy(),errors

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();stopped();case=read(args.case)
    if case['role']!='TRAIN_online_expert_executor_diagnostic':raise ValueError('Explicit TRAIN oracle diagnostic required')
    if case['task']not in ('place_cups_on_rotating_frame','place_cups_on_rotating_frame_high_speed','remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed','beat_the_rotating_buzz_high_speed','insert_onto_rotating_peg_high_speed','beat_the_rotating_buzz','insert_onto_rotating_peg'):raise ValueError('Reviewed source goal tasks only')
    calibration=str(Path(case['calibration_episode']).resolve(strict=True))
    if read(OUT/'calibration_lock.json').get(calibration)!=sha(calibration):raise ValueError('TRAIN calibration changed')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False);frozen=sources();frozen[calibration]=sha(calibration)
    from .backend import TaskExecutionBackend
    from .pose_executor import PoseGoalExecutor,WorldPoseGoal
    from ..execution_interface_20261111_v1.executor import PhysManiExecutor,ExecutionConfig
    from ..local_mpc_20261102.executor import JointLimits
    from ..executor_contract_repair_20261112_v2.audit import audit_execution
    import torch
    torch.set_num_threads(2)
    backend=None
    try:
        backend=TaskExecutionBackend(case['task'],calibration,seed=case['seed'],variation=0,
                    task_time_limit_seconds=60,execute_native=True,dataset_use='train',renderer_profile='official_all_GL')
        module=importlib.import_module(type(backend.task).__module__)
        frozen[str(Path(module.__file__).resolve())]=sha(module.__file__)
        write(output/'request.json',dict(case=case,case_file=pointer(args.case),source_sha256=frozen,
            NN_forward_calls=0,learned_policy_task_ability_evaluated=False,oracle_dynamic_state_used=True,
            original_expert_called_with_bool_return_path=False,Native_task_stage_not_modified=True,future_rotation_helper_is_pure=True))
        backend.on_hardware_contact=lambda r:append(output/'hardware_table_contacts.jsonl',r)
        stats=[]
        def sample():
            before=backend._clock_tuple()
            row=dict(tick=backend._ticks,time=backend.clock(),tcp_pose=backend.arm.get_tip().get_pose(),
                held=bool(backend.gripper.get_grasped_objects()),open_amount=backend.gripper.get_open_amount(),
                joint_positions=backend.get_joint_positions(),joint_velocities=backend.get_joint_velocities(),
                joint_targets=backend.arm.get_joint_target_positions())
            if case.get('audit_grasp_rigidity',False):
                row['TRAIN_rigidity_labels_never_used_by_controller']=dict(
                    tcp_matrix=backend.arm.get_tip().get_matrix(),attach_matrix=backend.gripper._attach_point.get_matrix(),
                    bodies=[dict(handle=obj.get_handle(),name=obj.get_name(),matrix=obj.get_matrix(),
                                 held=obj in backend.gripper.get_grasped_objects())for obj in backend.task.get_graspable_objects()])
            if case['task'].startswith('moving_basketball'):
                row['TRAIN_private_diagnostic_only']=dict(ball_pose=backend.task.ball.get_pose(),
                    grasp_waypoint=backend.task.wp1.get_pose(),hoop_waypoint=backend.task.wp3.get_pose())
            elif case['task'].startswith('insert_onto'):
                row['TRAIN_private_diagnostic_only']=dict(ring_pose=backend.task._square_ring.get_pose(),
                    grasp_waypoint=backend.task.w1.get_pose(),insert_waypoint=backend.task.w4.get_pose())
            if before!=backend._clock_tuple():raise RuntimeError('Diagnostics advanced physics')
            stats.append(row);append(output/'mechanical_progress.jsonl',row)
        backend.on_diagnostic_tick=sample
        from ..executor_grasp_frame_20261113_v1.oracle import instrument,proxy as make_proxy
        from ..executor_timed_tasks_20261112_v1.commands import TimedWorldTarget
        expert=instrument(module.get_expert_info)
        native=backend.joint_limits();limits=JointLimits(native['lower'],native['upper'],native['velocity'],acceleration=12.,jerk=1000.)
        core=PhysManiExecutor(backend,limits,ExecutionConfig(reference_frequency=case['reference_frequency']),
                             initial_gripper_opening=backend.requested_opening)
        dispatcher=PoseGoalExecutor(backend,core,control_ticks=case['control_ticks'],barrier_rate=case['barrier_rate'],
                                    refine_ik=case.get('refine_ik',False),
                                    capture_lookahead_seconds=case.get('capture_lookahead_seconds'),
                                    commit_release=case.get('commit_release',False),
                                    release_position_tolerance=case.get('release_position_tolerance',.002),
                                    release_angle_tolerance=case.get('release_angle_tolerance',.015),
                                    feedback_mode=case.get('feedback_mode','actual'),
                                    native_reference_speed=case.get('native_reference_speed',False),
                                    joint_margin=case.get('joint_margin',0.),
                                    joint_avoidance=case.get('joint_avoidance','centre'),
                                    compensate_grasp_frame=case.get('compensate_grasp_frame',False),
                                    taskspace_objective=case.get('taskspace_objective',True),
                                    event_driven_gripper=case.get('event_driven_gripper',True),
                                    grasp_reference_world_offset=case.get('grasp_reference_world_offset',[0.,0.,0.]),
                                    align_grasp_before_descent=case.get('align_grasp_before_descent',False),
                                    latch_held_contact=case.get('latch_held_contact',False),
                                    aligned_hover_approach=case.get('aligned_hover_approach',False),
                                    servo_tracking_recovery=case.get('servo_tracking_recovery',False),
                                    capture_position_tolerance=case.get('capture_position_tolerance'),
                                    capture_rotation_tolerance=case.get('capture_rotation_tolerance'))
        write(output/'hardware_static_geometry.json',dispatcher.clearance.export())
        proxy=make_proxy(backend.task)
        if case.get('source_nominal_grasp_frame',False):
            from ..executor_grasp_frame_20261113_v1.nominal_robot import Robot
            proxy.robot=Robot(backend.task.robot,lambda:dispatcher.grasp_delta)
        chunks=0;rejected=0;reason='budget_exhausted'
        while not backend.terminal()and backend.remaining_ticks():
            if backend.clock()>=case.get('diagnostic_horizon_seconds',60.):
                reason='declared_TRAIN_diagnostic_horizon';break
            stopped();target,errors=query(backend,proxy,expert)
            receipt=dispatcher.execute(TimedWorldTarget(
                WorldPoseGoal(backend.clock(),target[:3],target[3:7],float(target[7]>=.5)),
                expert.last_timing['lookahead_seconds'],expert.last_timing['timing_source']))
            if receipt['status']in ('IK_failed','reference_rejected','measured_start_outside_bounds','hardware_clearance_rejected'):
                rejection=dict(receipt,IK_diagnostic=getattr(backend,'last_ik_receipt',None));rejected+=1
                receipt=dispatcher.advance_retained_goal(stop_requested=lambda:(stopped()or False));receipt['rejected_pose']=rejection
            append(output/'decisions.jsonl',dict(index=chunks,oracle_goal=target.tolist(),oracle_stage=proxy.stage,
                read_only_query_errors=errors,source_goal_timing=expert.last_timing,**{k:v for k,v in receipt.items()if k!='execution'}))
            for row in receipt['execution']:append(output/'execution.jsonl',row)
            chunks+=1
            if receipt['status']not in ('chunk_complete','retained_goal_chunk_complete'):
                reason=receipt['status'];break
        judge=backend.native_task_judge_receipt()
        if judge['native_task_success_latched']:reason='original_Native_task_success'
        elif (judge.get('original_native_metric',{}).get('native_first_terminal')or{}).get('terminal_failure'):
            reason='original_Native_terminal_failure'
        audit=audit_execution(output/'execution.jsonl');verify(frozen)
        result=dict(complete=True,task=case['task'],seed=case['seed'],profile=case['profile'],stop_reason=reason,
            original_native_success=judge['native_task_success_latched'],native_task_judge=judge,
            actual_ticks=backend._ticks,actual_seconds=backend.clock(),issued_reference_audit=audit,
            hardware_contact_audit=backend.hardware_contact_audit(),source_unchanged=True,
            NN_forward_calls=0,optimizer_steps=0,learned_policy_task_success_claimed=False,
            oracle_dynamic_state_used=True,formal_VAL_TEST=False,rejected_pose_chunks=rejected,
            ever_held=any(r['held']for r in stats),
            first_held_time=next((r['time']for r in stats if r['held']),None),
            source_expert_branch_logic_preserved=True,
            future_rotation_computed_without_native_setters=True)
        backend.close();result['native_closed']=True;write(output/'result.json',result)
        print({k:v for k,v in result.items()if k not in ('native_task_judge','issued_reference_audit')},flush=True);return 0
    except BaseException as error:
        write(output/'failure.json',dict(complete=False,error_type=type(error).__name__,error=str(error)))
        raise
    finally:
        if backend is not None:backend.close()

if __name__=='__main__':raise SystemExit(main())
