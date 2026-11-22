"""Compose a real selected world reference with generic100ms robot mechanics.

This contains no trained world-command policy loader. A real source-bound
BC model/feature adapter must supply references later; no dummy weights or
known-task e labels are substituted to claim a closed-loop task baseline.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path
import numpy as np
from . import VERSION
from .contracts import WorldTCPReference, retained_gripper_command
from .public_observation import validate_policy_packet
from .reference_mpc import GenericReferenceMPC, qualify_first_action
from ..multitask_execution_pipeline_20261104_v1.registry import TASKS, inventory

ROOT = Path(__file__).resolve().parents[2]


def source_map():
    names = ['__init__.py','contracts.py','public_observation.py','native_backend.py','reference_mpc.py','runner.py','train_labels.py']
    paths = [Path(__file__).with_name(n) for n in names]
    paths += [ROOT / p for p in (
        'scripts/run_generic_reference_task_20261104_v1.py',
        'our_mani/local_mpc_20261102/native.py','our_mani/local_mpc_20261102/executor.py',
        'our_mani/local_mpc_p2_20261103/reference.py','our_mani/local_mpc_p2_20261103/kinematics.py',
        'our_mani/local_mpc_p2_20261103/action_adapter.py',
        'our_mani/local_mpc_control_20261103/coupled_response.py',
        'our_mani/local_mpc_control_20261103/completed_gripper.py',
        'our_mani/local_mpc_control_20261103/finger_response_fastpath_20261103.py',
        'our_mani/local_mpc_control_20261103/planner_factory.py',
        'our_mani/local_mpc_p2_20261103_v2/servo.py',
        'our_mani/local_mpc_p2_20261103_v2/servo_adapter.py',
        'our_mani/multitask_execution_pipeline_20261104_v1/registry.py',
        'our_mani/multitask_execution_pipeline_20261104_v1/native_adapter.py')]
    paths += [ROOT / p for p in ('our_mani/multitask/specs.py','our_mani/multitask/native_tasks.py')]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def check(task=None):
    if task is not None and task not in TASKS: raise ValueError('Exact16task identifier required')
    actual = inventory(ROOT)
    return dict(version=VERSION,method='current_network_reference_tracking_pretrain_baseline',
        tasks=[r['task'] for r in actual['tasks'] if task is None or r['task']==task],
        implemented=dict(task_dispatch=True,generated_development_native_reset=True,
            four_camera_public_RGBD_K_T=True,policy_q_TCP_only_masked_gripper=True,
            original100ms_two50ms_executor=True,original_gripper_completion_latch=True,
            legal_per_task_native_graspable_dispatch=True,original_task_success_metric_isolated=True,
            explicit_TRAIN_only_true_clock_private_multibody_labels=True,
            initial_actual_q_HoldTeacher_without_NN=True,
            robot_only_reference_CEM=True,full_scene_endpoint_plus7_collision_samples=True),
        blockers=['Actual trained world-command BC checkpoint/runtime loader not yet supplied',
            'Actual feature extractor selected e checkpoint must be bound; no assumption e-v1 is good',
            'Root must explicitly select caller index or highest_demo_likelihood; no implicit closest pose',
            'Opening intent is pending only: real arm-to-gripper sequencing/closing-time supervision is absent',
            'Official TEST archive/reset and whole-suite admission not bound; generated reset is development only',
            'New baseline has not been Native validated; static FK/old robot response does not qualify task contact'],
        native_execution_ready=False,closed_loop_task_support_demonstrated=False,
        old_P4_local_object_handoff_reused=False,physical_handoff_tau_assigned=False,
        opening_intent_is_immediate_actuator_command=False,
        gripper_sequence='retain_actual_persistent_target; future_intent_unissued',
        executor_completion_is_arm_arrival_or_object_alignment=False,
        Native_held_or_task_body_pose_in_model_inputs=False,
        task_or_operation_labels_in_policy_packet=False,
        optional_TRAIN_labels_are_separate_from_policy=True,
        motion_clock_source='actual_Native_sim_and_task_readbacks; tick_times_DT_is_diagnostic_only',
        old_finger_model_conditioning='both legacy0/1 hypotheses; neither is observed contact',
        source_sha256=source_map(),Native_started=False,GPU_used=False,
        model_constructed=False,model_forward_calls=0,training_started=False,raw_TEST_files_opened=0)


def execute_reference_cycle(backend,executor,planner,reference):
    """Functional execution port; only Root calls this with actual artifacts.

    The policy receives public_sample separately, never this private return.
    A rejected/unknown action preserves the prior gripper command.
    """
    if type(reference) is not WorldTCPReference or type(planner) is not GenericReferenceMPC:
        raise TypeError('Actual generic planner and selected current world reference required')
    state=backend.mechanical_state(executor.state())
    retained_opening=retained_gripper_command(state)
    if executor.gripper_target != retained_opening:
        raise ValueError('Actual gripper command and actuator snapshot disagree')
    if abs(reference.timestamp-state['time'])>1e-9:raise ValueError('Stale/future command cannot execute')
    plan=planner.plan(state,reference)
    action=plan.get('action');qualification=None
    if action is not None:
        accepted,qualification=qualify_first_action(backend,state,action)
    else:accepted=False
    if accepted:
        if action.get('opening') != retained_opening:
            raise ValueError('Pending BC gripper intent cannot change the actual motor target')
        receipt=executor.execute_joint_target(action['joint_goal'],opening=retained_opening,phase='world_reference_tracking')
    else:
        receipt=executor.execute_joint_target(state['q'],opening=executor.gripper_target,phase='reference_unknown_or_rejected_actual_q_hold')
    return dict(reference=asdict(reference),planner=plan,action_accepted=accepted,
        opening_intent_issued=False,actual_gripper_target=retained_opening,
        opening_intent_condition=reference.opening_intent_condition,
        qualification=qualification,execution=receipt.to_dict(),private_Native_metrics=backend.metric_receipt(),
        policy_input_is_only_separate_public_packet=True,task_success_qualified=False)


def run_bound_reference_policy(backend,executor,planner,predict_reference,*,log_private):
    """Caller binds a REAL trained policy/runtime and exact artifacts upstream.

    This is an integration function, not a certificate for arbitrary callables.
    No environment/body/held/metric/cache reference is passed to the policy.
    """
    cycles=0
    while not backend.terminal():
        packet=validate_policy_packet(backend.public_sample())
        reference=predict_reference(packet)
        if type(reference) is not WorldTCPReference:
            raise TypeError('Real current policy must return an explicitly selected WorldTCPReference')
        private=execute_reference_cycle(backend,executor,planner,reference)
        log_private(private);cycles+=1
    return dict(version=VERSION,cycles=cycles,native_metrics=backend.metric_receipt(),
        native_success_authority='task.success()',public_attachment_or_force_proof_provided=False,
        task_success_guaranteed=False,real_time_guaranteed=False)
