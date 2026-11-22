"""Execute public desired poses through exact original RRT/RML samples."""
from dataclasses import asdict
import hashlib
import numpy as np
from pyrep.errors import ConfigurationPathError, ConfigurationError, IKError

from .common import append, stopped
from .sequencing import SequenceConfig, sample_native_rml_path, requested_opening_after_arrival, retime_reference_path
from ..execution_interface_20261111_v1.commands import JointTrajectory
from ..local_mpc_20261102.executor import ReferenceLimitError
from ..multitask_executor_cycle_20261112_v4.joint_recovery import mechanical_recovery_reference
from .sequencing import RMLSamplingUnavailable


def execute_samples(backend, executor, points, *, phase, opening=None, log_path):
    for start in range(0, len(points), 2):
        stopped()
        if backend.terminal():
            return
        selected = points[start:start + 2]
        command = JointTrajectory(start_time=backend.clock(), positions=selected,
                                  times_from_start=[.05 * (i+1) for i in range(len(selected))], opening=opening)
        receipt = executor.execute_command(command, phase=phase,
                                            planner_contract={'reference_source': 'explicitly_retimed_RRT_RML_50ms_polyline',
                                                              'prepared_samples_sent_unchanged': True})
        append(log_path, receipt.to_dict())
        if receipt.executed_ticks == 0:
            return


def run(backend, executor, policy, output, *, config=SequenceConfig()):
    decisions = output / 'decisions.jsonl'
    execution = output / 'execution.jsonl'
    queries, arrivals, opening_changes, rejected = 0, 0, 0, 0
    recovery_ticks, consecutive_recovery = 0, 0
    reason = 'budget_exhausted'
    while not backend.terminal():
        stopped()
        actual_q = backend.get_joint_positions()
        prior = executor._last_direct
        previous_reference = prior[1] if prior is not None else backend.arm.get_joint_target_positions()
        recovery = mechanical_recovery_reference(actual_q, previous_reference, executor.limits.lower, executor.limits.upper)
        if recovery is not None:
            if consecutive_recovery >= 20:
                reason = 'joint_limit_servo_recovery_timeout'
                break
            goal, recovery_audit = recovery
            receipt = executor.execute_command(JointTrajectory(start_time=backend.clock(), positions=[goal],
                                                               times_from_start=[.05]), phase='counted_joint_limit_recovery')
            append(execution, receipt.to_dict())
            append(output/'joint_limit_recovery.jsonl', dict(recovery_audit, clock_after=backend.clock(),
                                                            executed_ticks=receipt.executed_ticks))
            consecutive_recovery += receipt.executed_ticks
            recovery_ticks += receipt.executed_ticks
            continue
        consecutive_recovery = 0
        packet = backend.public_sample()
        arrays = {k: v for k, v in packet.items() if isinstance(v, np.ndarray)}
        frame_provenance = dict(query_index=queries, timestamp=packet['timestamp'],
                                array_sha256={k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                                              for k, v in arrays.items()}, saved_RGBD=None)
        if queries % 10 == 0:
            folder = output / 'public_frames'
            folder.mkdir(exist_ok=True)
            path = folder / f'{queries:04d}.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream, timestamp=np.asarray(packet['timestamp']), **arrays)
            frame_provenance['saved_RGBD'] = str(path)
        append(output / 'public_frame_provenance.jsonl', frame_provenance)
        reference = policy.predict_reference(packet)
        diagnostic = dict(index=queries, timestamp=backend.clock(), reference=asdict(reference),
                          inference=policy.last_diagnostic, pending_gripper_intent=True)
        queries += 1
        try:
            points = sample_native_rml_path(backend, reference, config)
        except (ConfigurationPathError, ConfigurationError, IKError, RMLSamplingUnavailable) as error:
            diagnostic.update(rejected='unreachable_public_model_reference', error=str(error))
            append(decisions, diagnostic)
            rejected += 1
            # The returned model target is unreachable. Do not invent a target,
            # a task phase or an opening command; end with an explicit outcome.
            reason = 'model_reference_unreachable'
            break
        diagnostic['path_samples'] = len(points)
        diagnostic['native_RML_sampling_audit'] = getattr(backend, 'last_rml_sampling_audit', None)
        path_values=np.asarray(points,float)
        if np.any(path_values < executor.limits.lower) or np.any(path_values > executor.limits.upper):
            diagnostic.update(rejected='planned_joint_path_outside_native_bounds',
                              planned_reference_min=path_values.min(0).tolist(),
                              planned_reference_max=path_values.max(0).tolist(),
                              position_values_clipped=False)
            append(decisions,diagnostic)
            rejected+=1
            reason='planned_joint_path_outside_native_bounds'
            break
        try:
            points, diagnostic['reference_retiming'] = retime_reference_path(
                points, executor, backend.get_joint_positions(), backend.get_joint_velocities(), backend.clock())
            execute_samples(backend, executor, points, phase='public_world_arm_path', log_path=execution)
        except ReferenceLimitError as error:
            diagnostic.update(rejected='RML_reference_limits', error=str(error))
            append(decisions, diagnostic)
            rejected += 1
            reason = 'RML_reference_limits'
            break
        if backend.terminal():
            append(decisions, diagnostic)
            break
        # A completed reference is not measured arrival. Hold the final arm
        # reference, retain gripper intent, and observe the real public TCP.
        target = points[-1]
        arrival = None
        for _ in range(int(config.max_settle_seconds / .05) + 1):
            opening, arrival = requested_opening_after_arrival(
                reference, backend.arm.get_tip().get_pose(), executor.gripper_target, config)
            if arrival['measured_arm_arrived'] or backend.terminal():
                break
            execute_samples(backend, executor, [target], phase='arm_arrival_settle', log_path=execution)
        diagnostic['arrival'] = arrival
        if backend.terminal():
            append(decisions, diagnostic)
            break
        if not arrival['measured_arm_arrived']:
            diagnostic['gripper_intent_issued'] = False
            append(decisions, diagnostic)
            reason = 'measured_arm_arrival_timeout'
            break
        arrivals += 1
        changed = opening != executor.gripper_target
        opening_changes += int(changed)
        # Native opening/closing is advanced until done; no same-target restart.
        while not backend.terminal():
            execute_samples(backend, executor, [target], phase='arrived_native_gripper',
                            opening=opening, log_path=execution)
            if executor.gripper_done:
                break
        diagnostic.update(gripper_intent_issued=True, opening_changed=changed,
                          gripper_completion=executor._gripper_driver.gripper_completion_receipt())
        # Match the author's original OPEN drop-settling interval, counted and
        # first-native-terminal truncated, not a success condition.
        if changed and opening == 1. and not backend.terminal():
            execute_samples(backend, executor, [target] * 10, phase='native_open_drop_settle', log_path=execution)
        append(decisions, diagnostic)
        if queries % 5 == 0:
            print(f'{backend.task_name}: queries={queries} sim={backend.clock():.2f}s', flush=True)
    judge = backend.native_task_judge_receipt()
    if judge['native_task_success_latched']:
        reason = 'original_Native_task_success'
    elif judge['original_native_metric']['native_first_terminal'] is not None:
        reason = 'original_Native_terminal_failure'
    return dict(complete=True, episode_success=judge['native_task_success_latched'],
                stop_reason=reason, actual_ticks=backend._ticks, actual_seconds=backend.clock(),
                native_task_judge=judge, model_forward_calls=policy.calls, pose_queries=queries,
                measured_arrivals=arrivals, gripper_target_changes=opening_changes, rejected_targets=rejected,
                servo_reference_readback_max_error_rad=backend.servo_readback_max_error_rad,
                exact_servo_writes=backend.servo_writes, sequence_config=asdict(config),
                counted_joint_limit_recovery_ticks=recovery_ticks,
                formal_VAL_TEST=False, visual_success_gate=False,
                Native_target_pose_or_instance_masks_in_NN=False)
