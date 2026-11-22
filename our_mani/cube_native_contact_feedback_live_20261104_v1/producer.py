"""Explicit Native-contact execution profile, not the old public-proof policy."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from ..local_mpc_control_20261103.public_grasp_v41 import build_response_planner,ROOT
from ..local_mpc_20261102.records import append,write_new,quantiles
from ..generic_reference_mpc_20261104_v1.reference_mpc import qualify_first_action
from .observer import ObservedNativeContactCubeBackendV1
from .coordinator import NativeContactCoordinatorV1
from .sources import native_contact_source_map,validate_native_success_protocol


def run_native_contact_cube_v1(backend,executor,output,config,*,recorder=None):
    if type(backend)is not ObservedNativeContactCubeBackendV1 or executor.backend is not backend:
        raise TypeError('Exact new Native-contact observer must own every actual50ms executor tick')
    if config.get('native_contact_feedback_v1')is not True or config.get('Native_contact_feedback_used')is not True:
        raise ValueError('Explicit Native contact execution input required; this is not public-only P4 qualification')
    if config.get('require_public_goal_completion',False)is not False:
        raise ValueError('Visual goal/hold/probe may not gate original Native task success')
    if float(config.get('physics_dt',.05))!=.05 or float(config.get('control_dt',.1))!=.1:
        raise ValueError('Original2actual50ms receding cycle required')
    validate_native_success_protocol();sources=native_contact_source_map()
    if backend.samples==0:backend.sample()
    core,helper,provenance=build_response_planner(ROOT,executor.limits,controller='coupled',action_space='joint',
        joint_goal_config={'lower':tuple(config.get('workspace_lower',[-.1,-.65,.755]))},
        terminal_config={'enabled':bool(config.get('terminal_goal_cost',True))},reference_frequency=executor.config.reference_frequency)
    if config.get('finger_response_fastpath',False):
        from ..local_mpc_control_20261103.finger_fastpath_integration_20261103 import enable_finger_fastpath
        provenance=enable_finger_fastpath(core.execution_model,provenance)
    coordinator=NativeContactCoordinatorV1(core,helper,backend,pi_symmetry_representative=config.get('candidate_pi_symmetry_representative',True))
    output=Path(output)
    write_new(output/'Native_contact_control_protocol.json',dict(source_sha256=sources,controller=provenance,
        Native_contact_feedback_used=True,Native_object_pose_used_by_policy=False,
        episode_success='original_PhysMani_Task.success_first_true_latched',held_alone_is_success=False,
        visual_probe_hold_or_goal_used_as_success_or_release_gate=False,
        old_public_attachment_and_natural_witness_modules_not_used=True,
        actual_close_done_and_membership_false_is_miss=True,
        carry_lift_tcp_m=.10,carry_stage_transition_TCP_distance_m=.015,
        fixed_lift_and_delivery_keep_close=True,original_sampled_collision_and_executor_retained=True))
    maximum=int(config.get('public_grasp_max_decisions',np.ceil(config.get('simulation_seconds',8.)/.1)))
    cycles=0;times=[];stop='budget_exhausted'
    while cycles<maximum and backend.remaining_ticks()>0:
        if backend.terminal():
            stop='original_Native_task_success'if backend.native_task_judge_receipt()['native_task_success_latched']else'Native_terminal_or_budget';break
        if any(p.exists()for p in(output/'STOP',output.parent/'STOP',ROOT/'runtime/framework/STOP',ROOT/'runtime/framework/local_mpc_20261102/STOP')):
            stop='administrative_stop';break
        started=time.perf_counter();robot=backend.latest_robot;h=deepcopy(executor.state())
        if abs(robot.timestamp-backend.clock())>1e-8:raise ValueError('Fresh actual public robot frame required')
        decision=coordinator.decide(robot,h);action=decision.get('action');qualification=None;rejected=None
        if action is None:
            coordinator.warm=None;goal=robot.q.copy()
            opening=0. if decision.get('keep_closed')is True or backend.latest_contact['held'] else h['gripper_target']
            executed=dict(joint_goal=goal.copy(),opening=opening,source='actual_q_hold_keep_native_contact_request')
        elif decision.get('direct_membership_miss_open'):
            allowed,qualification=qualify_first_action(backend,dict(time=robot.timestamp,q=robot.q),action)
            goal=np.asarray(action['joint_goal']).copy();opening=1.
            if not allowed:goal=robot.q.copy();opening=h['gripper_target'];rejected=qualification['reason']
            executed=dict(joint_goal=goal.copy(),opening=opening,source='actual_completed_close_nonmembership_open'if allowed else'rejected_native_miss_open_hold')
        else:
            goal,rejected,qualification=helper.qualify_action(backend,decision['state'],action,require_predicted_joint_goal=True)
            opening=float(action['opening'])
            if rejected:
                goal=robot.q.copy();opening=h['gripper_target'];coordinator.warm=None
                executed=dict(joint_goal=goal.copy(),opening=opening,source='rejected_model_action_hold_actual_q_keep_request')
            else:
                if not np.array_equal(goal,np.asarray(action['joint_goal'])):raise ValueError('Selected modeled q must be dispatched exactly')
                if backend.latest_contact['held']and opening!=0.:raise ValueError('Visual/model failure cannot release actual Native membership')
                executed=dict(joint_goal=goal.copy(),opening=opening,source='exact_selected_modeled_q_and_opening')
        receipt=executor.execute_joint_target(goal,opening=opening,phase='Native_contact_'+decision['mode'])
        after=backend.latest_robot
        append(output/'public_candidate_native_contact_v1_decisions.jsonl',dict(index=cycles,time=robot.timestamp,public_robot=robot,
            actuator_state_before=h,decision=decision,action_qualification=qualification,action_rejected_reason=rejected,
            executed_action=executed,executor=receipt,public_robot_after=after,actuator_state_after=deepcopy(executor.state()),
            Native_contact_feedback_used=True,Native_object_pose_used_by_policy=False,
            native_task_judge=backend.native_task_judge_receipt()))
        cycles+=1;times.append(time.perf_counter()-started)
        if backend.native_task_judge_receipt()['native_task_success_latched']:
            stop='original_Native_task_success';break
        if after.timestamp<=robot.timestamp+1e-8:stop='no_actual_tick';break
    judge=backend.native_task_judge_receipt()
    return dict(stage='Cube_native_contact_feedback_live_v1',complete=True,cycles=cycles,canary_stop_reason=stop,
        episode_success=judge['native_task_success_latched'],native_task_judge=judge,
        Native_task_success_claimed=judge['native_task_success_latched'],Native_success_exit_used=True,
        Native_contact_feedback_used=True,Native_object_pose_used_by_policy=False,
        visual_success_gate=False,visual_failure_may_release_actual_Native_held=False,
        Transformer_used=backend.pilot_inference.calls>0,candidate_transformer=backend.pilot_inference.receipt(),
        candidate_events=coordinator.geometry.events,candidate_public_cache=backend.candidate_cache.receipt(),
        mode_events=coordinator.mode_events,optional_public_producer=backend.receipt(),
        full_cycle_wall_seconds=quantiles(times),full_cycle_wall_seconds_samples=times,
        geometry_fallback_used=False,force_sensor_contact_claimed=False,formal_qualification=False,
        source_sha256=sources,real_time_guaranteed=False)
