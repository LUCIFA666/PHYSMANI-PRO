"""Explicit adapter from existing world-command proposals to executor goals."""
from dataclasses import asdict
from .pose_executor import WorldPoseGoal, PoseGoalExecutor
from .common import append, stopped,write,pointer


def execute_reference(dispatcher, reference):
    # Existing opening head is a binary classifier. This is a declared fixed
    # schedule for its desired intent, not a learned contact-time prediction.
    action = WorldPoseGoal(reference.timestamp, reference.position, reference.quaternion,
                           float(reference.opening_intent >= .5))
    return dispatcher.execute(action, stop_requested=lambda: (stopped() or False))


def execute_ranked_references(dispatcher,references):
    if not references:raise ValueError('At least one declared network proposal required')
    attempts=[];receipt=None;reference=references[0]
    for candidate in references:
        reference=candidate
        receipt=execute_reference(dispatcher,candidate)
        attempts.append(dict(selected_index=candidate.selected_index,status=receipt['status'],
            executed_ticks=receipt['executed_ticks'],hardware_clearance=receipt.get('hardware_clearance')))
        # Only a read-only, candidate-specific rejection may try another pose.
        # After any physical step, the old observation timestamp is consumed.
        if receipt['executed_ticks']or receipt['status']not in ('IK_failed','reference_rejected','hardware_clearance_rejected'):
            break
    return reference,receipt,attempts


def run(backend, executor, policy, output, *, max_chunks):
    if type(max_chunks) is not int or not 1 <= max_chunks <= 120:
        raise ValueError('Explicit diagnostic horizon of 1..120 half-second chunks required')
    dispatcher = PoseGoalExecutor(backend, executor)
    write(output/'hardware_static_geometry.json',dispatcher.clearance.export())
    chunks = []
    reason = 'diagnostic_horizon'
    while not backend.terminal() and len(chunks) < max_chunks:
        stopped()
        packet=backend.public_sample()
        references=policy.predict_candidates(packet)if hasattr(policy,'predict_candidates')else[policy.predict_reference(packet)]
        reference,receipt,attempts=execute_ranked_references(dispatcher,references)
        if receipt['status']in ('IK_failed','reference_rejected','measured_start_outside_bounds','hardware_clearance_rejected'):
            rejection=dict(receipt,IK_diagnostic=getattr(backend,'last_ik_receipt',None))
            receipt=dispatcher.advance_retained_goal(stop_requested=lambda: (stopped()or False))
            receipt['rejected_pose']=rejection
        append(output/'decisions.jsonl', dict(index=len(chunks), reference=asdict(reference),
               network_proposals=[asdict(r)for r in references],
               proposal_attempts=attempts,proposal_selection='highest_demo_likelihood_with_executor_admissibility',
               inference=policy.last_diagnostic,
               opening_conversion='binary_classifier_at_0.5_scheduled_on_tenth_tick',
               learned_grasp_time_claimed=False, **{k:v for k,v in receipt.items() if k!='execution'}))
        for item in receipt['execution']:
            append(output/'execution.jsonl', item)
        chunks.append({k:v for k,v in receipt.items() if k!='execution'})
        if receipt['status'] not in ('chunk_complete','retained_goal_chunk_complete'):
            reason=receipt['status']
            break
    judge = backend.native_task_judge_receipt()
    if judge['native_task_success_latched']:
        reason='original_Native_task_success'
    elif backend.remaining_ticks()==0:
        reason='budget_exhausted'
    elif backend.terminal():
        reason='original_Native_terminal_failure'
    return dict(complete=True, episode_success=judge['native_task_success_latched'], stop_reason=reason,
                chunks=chunks, actual_ticks=backend._ticks, actual_seconds=backend.clock(),
                native_task_judge=judge, model_forward_calls=policy.calls, optimizer_steps=0,
                observed_public_frames=policy.observed_frames,
                rejected_pose_chunks=sum('rejected_pose'in c for c in chunks),
                rejected_reference_chunks=sum(c.get('rejected_pose',{}).get('status')=='reference_rejected'for c in chunks),
                rejected_hardware_clearance_chunks=sum(c.get('rejected_pose',{}).get('status')=='hardware_clearance_rejected'for c in chunks),
                hardware_static_geometry=pointer(output/'hardware_static_geometry.json'),
                hardware_FK_validation_errors=dispatcher.clearance.native_pose_errors(backend),
                servo_reference_readback_max_error_rad=backend.servo_readback_max_error_rad,
                exact_servo_writes=backend.servo_writes, formal_VAL_TEST=False,
                all_model_weights_fixed=True, exact_unmodified_author_executor=False,
                motion_mode='coupled_measured_feedback_governor_with_static_table_barrier',
                timing='nine_arm_ticks_then_gripper_on_tenth_tick_nonblocking',
                whole_path_or_gripper_completion_does_not_block_next_model_query=True)
