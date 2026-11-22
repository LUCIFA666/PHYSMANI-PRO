"""Explicit non-benchmark negative control for visual attachment tests.

The original Native gripper motor routine still runs, but its proximity-based
parent-attachment attempt is suppressed during that routine. This intervention
is never used for positive benchmark scores. It reads no object pose/held label
and introduces no physical step or permanent method mutation.
"""


class NoProxyAttachmentBackend:
    def __init__(self, backend):
        self.backend = backend
        self.suppressed = 0

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def actuate_gripper(self, opening):
        before = float(self.backend.clock())
        gripper = self.backend.gripper
        original = gripper.grasp
        had_instance_override = 'grasp' in vars(gripper)
        original_override = vars(gripper).get('grasp')

        def suppressed_attempt(*args, **kwargs):
            self.suppressed += 1
            return False

        gripper.grasp = suppressed_attempt
        try:
            result = self.backend.actuate_gripper(opening)
        finally:
            if had_instance_override:
                gripper.grasp = original_override
            else:
                del gripper.grasp
        if abs(float(self.backend.clock())-before)>1e-9:
            raise RuntimeError('negative-control actuation advanced physical time')
        return result

    def no_proxy_attachment_receipt(self):
        return dict(protocol='negative_control_suppress_proximity_parent_attachment_only',
            suppressed_grasp_attempts=self.suppressed,motor_routine_preserved=True,
            original_method_restored=True,physics_steps_added=0,
            object_pose_or_held_read=False,standard_benchmark_protocol=False,
            positive_success_score_eligible=False)
