"""Drive tracking references through the repaired interface and original gates."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from ..local_mpc_control_20261103.response_factory import build_planner
from ..local_mpc_20261102.records import append,write_new
from ..generic_reference_mpc_20261104_v1.reference_mpc import qualify_first_action
from ..execution_interface_20261111_v1.commands import JointTrajectory
from .coordinator_v2 import PositionResponseTrackingCoordinator as TrackingCoordinator


def run(backend,executor,output,config,*,stop_requested=lambda:False):
    root=Path(__file__).resolve().parents[2];output=Path(output)
    core,helper,provenance=build_planner(root,executor.limits,controller='coupled',action_space='joint',
        joint_goal_config={'lower':tuple(config.get('workspace_lower',[-.1,-.65,.755]))},
        terminal_config={'enabled':bool(config.get('terminal_goal_cost',True))},reference_frequency=executor.config.reference_frequency)
    coordinator=TrackingCoordinator(core,helper,backend,tracking_config=config.get('tracking_config'),response_parameters=config['response_parameters'],response_sha256=config['response_sha256'],
        pi_symmetry_representative=config.get('candidate_pi_symmetry_representative',True))
    write_new(output/'controller_contract.json',dict(executor=executor.capabilities(),controller=provenance,
        tracking_config=vars(coordinator.reference_planner.config),response_parameters=coordinator.reference_planner.joint_response.evidence,free_control_period=.05,
        held_carry_control_period=.1,NN_input_or_weights_changed=False,Native_target_pose_used_by_policy=False,
        target_velocity_used_in_actual_reference=True,episode_success_source='original_PhysMani_Task.success_only',
        reference_joint_bounds_velocity_acceleration_jerk_and_collision_gates_preserved=True))
    cycles=0;stop='budget_exhausted'
    while backend.remaining_ticks()>0 and cycles<160:
        if backend.terminal():stop='original_Native_task_success' if backend.native_task_success_latched else 'Native_terminal';break
        if stop_requested():stop='administrative_stop';break
        robot=backend.latest_robot;h=deepcopy(executor.state())
        decision=coordinator.decide(robot,h);action=decision.get('action');qualification=None;rejected=None
        direct=action is not None and action.get('execution_semantics')=='sampled_joint_reference'
        if action is None:
            opening=0. if decision.get('keep_closed') or backend.latest_contact['held'] else h['gripper_target']
            dispatched=dict(joint_goal=robot.q.copy(),opening=opening)
        elif decision.get('direct_membership_miss_open'):
            allowed,qualification=qualify_first_action(backend,dict(time=robot.timestamp,q=robot.q),action)
            dispatched=dict(joint_goal=np.asarray(action['joint_goal']) if allowed else robot.q.copy(),opening=1. if allowed else h['gripper_target'])
            if not allowed:rejected=qualification['reason']
        else:
            goal,rejected,qualification=helper.qualify_action(backend,decision['state'],action,require_predicted_joint_goal=True)
            if rejected:
                dispatched=dict(joint_goal=robot.q.copy(),opening=h['gripper_target']);direct=False
            else:
                if not np.array_equal(goal,np.asarray(action['joint_goal'])):raise ValueError('Selected reference changed before dispatch')
                dispatched=deepcopy(action)
        if backend.latest_contact['held'] and dispatched['opening']!=0.:raise ValueError('Held feedback cannot be released by tracking')
        if direct:
            receipt=executor.execute_command(JointTrajectory(robot.timestamp,[dispatched['joint_goal']],[.05],
                opening=dispatched['opening'],joint_names=executor.joint_names),phase=decision['mode'])
        else:
            receipt=executor.execute_joint_target(dispatched['joint_goal'],opening=dispatched['opening'],
                ticks=1 if not backend.latest_contact['held'] else 2,expected_time=robot.timestamp,phase=decision['mode'])
        coordinator.observe_execution(receipt)
        append(output/'decisions.jsonl',dict(index=cycles,time=robot.timestamp,public_robot=robot,actuator_state_before=h,
            decision=decision,executed_action=dispatched,trajectory_mode=direct,action_qualification=qualification,
            action_rejected_reason=rejected,executor=receipt,public_robot_after=backend.latest_robot,
            native_task_judge=backend.native_task_judge_receipt()))
        if receipt.unexpected_physics_ticks!=0:raise RuntimeError('Unexpected physical step')
        cycles+=1
        if receipt.feedback.get('native_task_success') is True:stop='original_Native_task_success';break
        if receipt.executed_ticks==0:stop='no_actual_tick';break
        if cycles%20==0:print(f"Tracking check: {cycles} cycles, sim_time={backend.clock():.3f}, mode={decision['mode']}",flush=True)
    judge=backend.native_task_judge_receipt()
    return dict(complete=stop!='administrative_stop',cycles=cycles,stop_reason=stop,
        episode_success=judge['native_task_success_latched'],native_task_judge=judge,
        Native_contact_feedback_used=True,Native_target_pose_used_by_policy=False,
        visual_success_gate=False,candidate_forward_calls=backend.pilot_inference.calls,formal_VAL_TEST=False)
