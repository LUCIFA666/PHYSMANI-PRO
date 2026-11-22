"""Global response tracking followed by local future-trajectory MPC."""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from .qp_tracking import QPTrackingPlanner,PredictiveTrackingConfig
from .tracking_v4 import StoppableTrackingReferencePlanner


@dataclass(frozen=True)
class HybridTrackingConfig(PredictiveTrackingConfig):
    horizon_ticks: int = 4
    orientation_weight: float = 2.
    terminal_velocity_zero: bool = False
    local_mpc_radius: float = .16
    execution_pregrasp_offset: float = .04


class HybridTrackingPlanner(QPTrackingPlanner):
    def __init__(self,*args,config=None,**kwargs):
        super().__init__(*args,config=config or HybridTrackingConfig(),**kwargs)

    def plan(self,state,candidate,forecast,*,pregrasp):
        working=deepcopy(candidate)
        if pregrasp:working['pregrasp_offset']=min(working['pregrasp_offset'],self.config.execution_pregrasp_offset)
        if not hasattr(forecast,'times'):
            result=StoppableTrackingReferencePlanner.plan(self,state,working,forecast,pregrasp=pregrasp)
            result['controller_regime']='controlled_carry_response_tracking'
            return result
        pose,_,_=forecast.at('target',state['time'])
        offset=np.asarray(working['object_tcp_position']).copy()
        if pregrasp:offset+=working['approach_direction']*working['pregrasp_offset']
        distance=np.linalg.norm(pose.apply(offset)-state['tcp_position'])
        if distance>self.config.local_mpc_radius:
            result=StoppableTrackingReferencePlanner.plan(self,state,working,forecast,pregrasp=pregrasp)
            result['controller_regime']='global_response_tracking'
        else:
            result=QPTrackingPlanner.plan(self,state,working,forecast,pregrasp=pregrasp)
            result['controller_regime']='local_public_future_MPC'
        result['reference_pregrasp_offset']=working['pregrasp_offset']
        result['phase_gate_candidate_unchanged']=True
        return result
