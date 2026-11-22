"""Public candidate tracking, actual-contact carry, unchanged Native success."""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation

from ..cube_native_contact_feedback_live_20261104_v1.coordinator import NativeContactCoordinatorV1
from ..local_mpc_control_20261103.public_grasp import qualify_public_grasp_candidates
from ..local_mpc_control_20261103.forecast_control import BoundForecastExecutionModel
from ..local_mpc_control_20261103.free_phase_dwell_v72 import update_phase
from ..local_mpc_p4_20261103.runtime import ProposalUnavailable
from .tracking import TrackingReferencePlanner,TrackingConfig


class TrackingCoordinator(NativeContactCoordinatorV1):
    def __init__(self,*args,tracking_config=None,**kwargs):
        super().__init__(*args,**kwargs)
        transition=self.core.execution_model.transition_batch
        self.reference_planner=TrackingReferencePlanner(self.helper.kinematics,transition.joint_response,
            self.helper.limits,TrackingConfig(**(tracking_config or {})))
        self.selected_id=None

    def observe_execution(self,receipt):self.reference_planner.observe_execution(receipt)

    def decide(self,robot,h):
        b=self.backend;now=robot.timestamp
        held=b.latest_contact['held']
        # The original actual-contact path owns lift/delivery and completed
        # close misses. It never releases a true held target due to vision.
        if held or self._last_held or (h['gripper_target']==0. and h['gripper_done'] is True):
            return super().decide(robot,h)
        result=b.latest;obj=result.get('object_observation')
        if b.goal_pose is None or not result.get('free_forecast_available') or obj is None:
            return super().decide(robot,h)
        pose,v,w=result['forecast'].at('target',now)
        state=self._state(robot,h,obj.pose,v,w,held=False,goal=b.goal_pose.position,mode='FREE_GRU')
        if state is None:return dict(mode='UNAVAILABLE',action=None,reason='current_robot_twist_unavailable')
        pending=h['gripper_target']==0.
        try:self.candidates=self.geometry.update(state)
        except ProposalUnavailable as error:
            return dict(mode='UNAVAILABLE',action=None,reason=str(error),keep_closed=pending)
        bound=BoundForecastExecutionModel.from_public_bridge(self.core.execution_model,result)
        candidates,qualification=qualify_public_grasp_candidates(b,state,deepcopy(self.candidates),self.helper,mode='FREE_GRU',bound=bound)
        available=[c for c in candidates if c.get('native_pose_qualified',False) and c['deadline']>=now-1e-9 and c['deadline']<=result['forecast'].times[-1]+1e-9]
        candidate=next((c for c in available if c['id']==self.selected_id),None)
        if candidate is None and available:
            def score(c):
                target=pose.apply(c['object_tcp_position']+c['approach_direction']*c['pregrasp_offset'])
                R=pose.rotation@Rotation.from_quat(c['object_tcp_quaternion']).as_matrix()
                angle=Rotation.from_matrix(R@Rotation.from_quat(robot.tcp_quaternion).as_matrix().T).magnitude()
                return float(np.linalg.norm(target-robot.tcp_position)+.1*angle)
            candidate=min(available,key=score)
        if candidate is None:
            return dict(mode='UNAVAILABLE',state=state,action=None,reason='no_current_qualified_candidate',keep_closed=pending,qualification=qualification)
        if candidate['id']!=self.selected_id and self.phase is not None:
            self.phase=deepcopy(self.phase);self.phase.update(dwell=0.,updated_at=None)
        self.selected_id=candidate['id']
        if pending:
            self.phase=dict(phase='close',dwell=0.,lost_hold=0.,updated_at=now,close_entered_at=now,reopening=False)
        else:
            self.phase=update_phase(state,candidate,self.phase,self.core.config)
        state['phase_state']=deepcopy(self.phase)
        planned=self.reference_planner.plan(state,candidate,result['forecast'],pregrasp=self.phase['phase']=='approach')
        action=deepcopy(planned.get('action'))
        opening=0. if pending or self.phase['phase']=='close' else 1.
        if action is not None:
            action.update(opening=opening,candidate_id=candidate['id'],execution_semantics='sampled_joint_reference')
        self._mode('NATIVE_PENDING_TWIST_TRACKING' if pending else 'FREE_TWIST_TRACKING',now)
        return dict(mode=self.mode,state=state,action=action,plan=planned,phase_state=deepcopy(self.phase),
            selected_candidate=deepcopy(candidate),candidate_pool=deepcopy(self.geometry.last_snapshot),
            qualification=qualification,Native_contact_feedback_used=True,Native_object_state_used=False,
            visual_attachment_gate_used=False,position_orientation_speed_deadline_gates_unchanged=True)
