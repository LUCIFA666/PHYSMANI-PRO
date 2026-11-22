"""Apply the calibrated trajectory interface to actual-held lift/delivery too."""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from ..contracts import Pose
from .coordinator_v2 import PositionResponseTrackingCoordinator


class ControlledPoseReference:
    """Desired carry reference, not a predicted/observed free object trajectory."""
    def __init__(self,pose):self.pose=pose
    def at(self,object_id,time):return self.pose,np.zeros(3),np.zeros(3)


def carry_reference(tcp_position,estimated_object_pose,object_goal,hold_tcp_quaternion):
    # The estimated relative transform comes only from the existing public
    # contact-time anchor and measured TCP propagation, never Native body pose.
    local=estimated_object_pose.rotation.T@(np.asarray(tcp_position)-estimated_object_pose.position)
    local_R=estimated_object_pose.rotation.T@Rotation.from_quat(hold_tcp_quaternion).as_matrix()
    candidate=dict(object_tcp_position=local,object_tcp_quaternion=Rotation.from_matrix(local_R).as_quat(),
        approach_direction=np.array([0.,0.,1.]),pregrasp_offset=0.)
    return ControlledPoseReference(Pose(object_goal,estimated_object_pose.rotation)),candidate


class CompleteTaskTrackingCoordinator(PositionResponseTrackingCoordinator):
    def decide(self,robot,h):
        b=self.backend;now=robot.timestamp
        if not b.latest_contact['held']:
            return super().decide(robot,h)
        carry=b.latest_carry;pose=carry.get('estimated_pose')
        if pose is None or b.goal_pose is None:
            return super().decide(robot,h)
        if not self._last_held:
            self.carry_stage='lift';self.lift_TCP=robot.tcp_position+np.array([0.,0.,.10])
            self.lift_object=pose.position+np.array([0.,0.,.10]);self.hold_quaternion=robot.tcp_quaternion.copy()
            self.phase=self.warm=None
        if self.carry_stage=='lift' and np.linalg.norm(robot.tcp_position-self.lift_TCP)<.015:
            self.carry_stage='deliver'
        goal=self.lift_object if self.carry_stage=='lift' else b.goal_pose.position
        self.phase=dict(phase='hold',dwell=0.,lost_hold=0.,updated_at=now,lift_goal_position=goal.copy(),hold_tcp_quaternion=self.hold_quaternion.copy())
        state=self._state(robot,h,pose,carry['linear_velocity'],carry['angular_velocity'],held=True,goal=goal,mode='NATIVE_CARRY')
        self._last_held=True;self._mode('NATIVE_TRACKED_'+self.carry_stage.upper(),now)
        if state is None:return dict(mode=self.mode,action=None,keep_closed=True,reason='actual_robot_twist_unavailable')
        reference,candidate=carry_reference(robot.tcp_position,pose,goal,self.hold_quaternion)
        planned=self.reference_planner.plan(state,candidate,reference,pregrasp=False)
        action=deepcopy(planned.get('action'))
        if action is not None:action.update(opening=0.,execution_semantics='sampled_joint_reference',candidate_id=self.selected_id)
        return dict(mode=self.mode,state=state,action=action,plan=planned,phase_state=deepcopy(self.phase),carry=carry,
            Native_contact_feedback_used=True,Native_object_state_used=False,visual_attachment_gate_used=False,
            actual_held_drives_carry=True,original_lift_height_and_stage_threshold_preserved=True)
