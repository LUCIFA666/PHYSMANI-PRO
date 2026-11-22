import numpy as np
from ..executor_release_retreat_20261113_v17.pose_executor import PoseGoalExecutor as RetreatExecutor,WorldPoseGoal
from .settle import VerticalReleaseSettle

class PoseGoalExecutor(RetreatExecutor):
    def __init__(self,*args,pre_release_settle=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.release_settle=VerticalReleaseSettle()if pre_release_settle else None

    def _sample_target(self):
        p,q,twist,motion=super()._sample_target()
        settle=self.release_settle
        if settle is not None:
            if not self.backend.gripper.get_grasped_objects():settle.reset()
            elif settle.pending:
                p=np.asarray(p,float).copy();twist=np.asarray(twist,float).copy()
                p[2]=settle.height;twist[2]=0.
                motion=dict(motion,pre_release_settle=dict(active=True,held_height_m=settle.height,
                    actual_clock_only=True,Native_object_pose_used=False))
        return p,q,twist,motion

    def _release_receipt(self):
        receipt=super()._release_receipt();settle=self.release_settle
        if settle is None:return receipt
        actual=np.asarray(self.backend.arm.get_tip().get_pose())
        if receipt['allowed']and not settle.pending:
            settle.start(actual[2])
            return dict(receipt,allowed=False,reason='start_physical_vertical_settle_before_opening',pre_release_settle=True)
        if not settle.pending:return receipt
        q=np.asarray(self.backend.get_joint_positions());dq=np.asarray(self.backend.get_joint_velocities());state=self._state()
        measured_vz=float(self.clearance.fk.geometric_jacobian(q)[2]@dq)
        reference_vz=float(self.clearance.fk.geometric_jacobian(state.position)[2]@state.velocity)
        gap=float(self.clearance.fk(state.position)[2,3]-actual[2])
        ready=settle.qualify(self.backend.clock(),measured_vz,reference_vz,gap)
        return dict(receipt,allowed=bool(receipt['allowed']and ready),
            reason='settled_physical_release'if receipt['allowed']and ready else'wait_for_physical_vertical_settle',
            pre_release_settle=dict(ready=ready,measured_vz_m_s=measured_vz,reference_vz_m_s=reference_vz,
                reference_height_gap_m=gap,stable_duration_required_seconds=.10),
            original_release_alignment_still_required=True,Native_object_pose_used=False)
