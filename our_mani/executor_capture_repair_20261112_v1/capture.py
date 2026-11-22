def qualify(*,graspables,detected,stationary,position_error,rotation_error,require_moving_precision=False):
    if not graspables:return True,'operation_has_no_native_graspables'
    if not detected:return False,'wait_for_original_gripper_detection'
    if stationary and(position_error>.0015 or rotation_error>.005):
        return False,'wait_for_precise_stationary_grasp_pose'
    if require_moving_precision and not stationary and(position_error>.02 or rotation_error>.1):
        return False,'wait_for_motion_compensated_capture_pose'
    return True,'native_capture_window_and_pose_qualified'
