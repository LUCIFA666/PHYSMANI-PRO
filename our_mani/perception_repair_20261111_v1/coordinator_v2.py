"""Use the same candidate pregrasp geometry in execution and phase gates."""
from copy import deepcopy
import numpy as np
from ..highspeed_control_20261111_v1.coordinator_v8 import PublicMotionCompleteTrackingCoordinator
from ..highspeed_control_20261111_v1.hybrid_tracking import HybridTrackingPlanner,HybridTrackingConfig
from ..highspeed_control_20261111_v1.tracking_v4 import StoppableTrackingReferencePlanner
from ..highspeed_control_20261111_v1.qp_tracking import QPTrackingPlanner


class CandidateConsistentTrackingPlanner(HybridTrackingPlanner):
    def plan(self,state,candidate,forecast,*,pregrasp):
        working=deepcopy(candidate)
        if not hasattr(forecast,'times'):
            result=StoppableTrackingReferencePlanner.plan(self,state,working,forecast,pregrasp=pregrasp)
            result['controller_regime']='controlled_carry_response_tracking'
            return result
        pose,_,_=forecast.at('target',state['time'])
        local=np.asarray(working['object_tcp_position']).copy()
        if pregrasp:local+=working['approach_direction']*working['pregrasp_offset']
        distance=np.linalg.norm(pose.apply(local)-state['tcp_position'])
        if distance>self.config.local_mpc_radius:
            result=StoppableTrackingReferencePlanner.plan(self,state,working,forecast,pregrasp=pregrasp)
            result['controller_regime']='global_response_tracking'
        else:
            result=QPTrackingPlanner.plan(self,state,working,forecast,pregrasp=pregrasp)
            result['controller_regime']='local_public_future_MPC'
        result.update(reference_pregrasp_offset=working['pregrasp_offset'],phase_gate_candidate_unchanged=True,
            pregrasp_reference_matches_phase_candidate=True)
        return result


class CandidateConsistentPerceptionCoordinator(PublicMotionCompleteTrackingCoordinator):
    def __init__(self,*args,tracking_config=None,**kwargs):
        super().__init__(*args,tracking_config=tracking_config,**kwargs)
        old=self.reference_planner
        self.reference_planner=CandidateConsistentTrackingPlanner(old.kinematics,old.joint_response,old.limits,
            config=HybridTrackingConfig(**(tracking_config or {})))
