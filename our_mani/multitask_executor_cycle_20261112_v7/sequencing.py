from ..multitask_executor_cycle_20261112_v1.sequencing import arrival_error

def author_sequence_gripper_request(reference,measured_tcp_pose,retained_opening,*,arm_sequence_completed,terminal,config):
    position,angle=arrival_error(reference,measured_tcp_pose)
    authorized=bool(arm_sequence_completed and not terminal)
    return (float(reference.opening_intent>.5)if authorized else float(retained_opening)),dict(
        arm_reference_sequence_completed=bool(arm_sequence_completed),
        measured_arm_arrived=position<=config.arrival_position_m and angle<=config.arrival_rotation_rad,
        position_error_m=position,rotation_error_rad=angle,
        measured_arrival_is_diagnostic_not_gripper_gate=True,
        gripper_intent_authorized_by='original_MoveArmThenGripper_arm_sequence_then_gripper'if authorized else None,
        object_alignment_proven=False,task_success_inferred=False,
        physical_handoff_time_supervised=False)
