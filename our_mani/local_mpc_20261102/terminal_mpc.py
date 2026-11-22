"""Independent endpoint geometry; no learned value or contact prediction.

Default enabled only for this new controller version. Original frozen planners
are untouched. Set TerminalCostConfig(enabled=False) for legacy equivalence.
"""
from copy import deepcopy
from dataclasses import dataclass, asdict
import numpy as np

from our_mani.local_mpc_20261102 import mpc as scalar
from our_mani.local_mpc_20261102.batch_mpc import norm,rotate,qmul,qinv,qlog,unit


@dataclass(frozen=True)
class TerminalCostConfig:
    enabled: bool = True
    weight: float = 2.
    grasp_position: float = 1.
    grasp_orientation: float = .2
    grasp_relative_speed: float = .2
    held_lift: float = 2.
    held_orientation: float = .2
    held_opening: float = .3
    lost_hold: float = 2.

    def __post_init__(self):
        if type(self.enabled) is not bool:raise ValueError('explicit terminal-cost bool flag required')
        for key,value in asdict(self).items():
            if key!='enabled' and (not np.isfinite(value) or value<0):raise ValueError('finite nonnegative terminal weight required: '+key)


def _endpoint(endpoint,n,c,initial):
    sizes=dict(tcp_position=3,tcp_quaternion=4,tcp_velocity=3,object_position=3,object_quaternion=4,
               object_linear_velocity=3,object_angular_velocity=3)
    value={}
    for key,size in sizes.items():
        value[key]=np.asarray(endpoint[key],float)
        if value[key].shape!=(n,size) or not np.isfinite(value[key]).all():raise ValueError('finite terminal '+key+' required')
    for key in ('held','opening','time'):
        value[key]=np.asarray(endpoint[key])
        if value[key].shape!=(n,):raise ValueError('aligned terminal '+key+' required')
    if value['held'].dtype!=np.bool_:raise ValueError('terminal held must be explicit actual/predicted bool')
    if not np.isfinite(value['time']).all() or not np.allclose(value['time'],initial['time']+c.horizon*c.dt,atol=1e-8,rtol=0):
        raise ValueError('terminal TCP and object must use the same H8 endpoint absolute time')
    if not np.isfinite(value['opening']).all() or np.any((value['opening']<0)|(value['opening']>1)):
        raise ValueError('valid actual endpoint aperture required')
    for key in ('tcp_quaternion','object_quaternion'):
        if not np.allclose(norm(value[key]),1.,atol=1e-8,rtol=0):raise ValueError('terminal unit quaternions required')
    if not initial['held'] and value['held'].any():raise ValueError('terminal cost cannot invent attachment')
    return value


def terminal_cost_batch(initial,endpoint,candidates,modes,phase,*,config=None,terminal_config=None):
    """Evaluate phi(x_H) against an independently defined final objective.

    Free mode always targets the real grasp pose, regardless of approach/align/
    close phase labels. Actual held at the planning anchor selects the hold/lift
    objective; release remains penalized, and a closed command never grants
    held credit. Grasp and object target are compared at the same future time.
    Squared metric residuals give progress credit even when full reach is not
    possible within this horizon. No threshold, stage bonus or success claim.
    """
    c=config or scalar.MPCConfig();tc=terminal_config or TerminalCostConfig()
    modes=np.asarray(modes,dtype=int);n=len(modes)
    if not tc.enabled:return np.zeros(n),{}
    s=_endpoint(endpoint,n,c,initial)
    r=np.asarray([candidates[m]['object_tcp_position'] for m in modes])
    rq=np.asarray([candidates[m]['object_tcp_quaternion'] for m in modes])
    limits=np.asarray([candidates[m]['allowed_relative_speed'] for m in modes])
    lever=rotate(s['object_quaternion'],r)
    grasp_position=s['object_position']+lever
    grasp_quaternion=unit(qmul(s['object_quaternion'],rq))
    # This lever excludes the pregrasp approach offset by design.
    from our_mani.local_mpc_20261102.batch_mpc import cross
    grasp_velocity=s['object_linear_velocity']+cross(s['object_angular_velocity'],lever)
    distance=norm(s['tcp_position']-grasp_position)
    angle=norm(qlog(qmul(grasp_quaternion,qinv(s['tcp_quaternion']))))
    relative=norm(s['tcp_velocity']-grasp_velocity)
    hold=bool(initial['held'])
    goal=np.asarray(phase.get('lift_goal_position',initial.get('lift_goal_position',
                    initial['object_position']+np.array([0.,0.,c.lift_height]))),float)
    hold_q=np.asarray(phase.get('hold_tcp_quaternion',initial['tcp_quaternion']),float)
    parts=dict(grasp_position=np.zeros(n),grasp_orientation=np.zeros(n),grasp_relative_speed=np.zeros(n),
               held_lift=np.zeros(n),held_orientation=np.zeros(n),held_opening=np.zeros(n),lost_hold=np.zeros(n))
    if hold:
        parts['held_lift']=np.sum(((s['object_position']-goal)/c.position_scale)**2,axis=-1)
        parts['held_orientation']=(norm(qlog(qmul(hold_q,qinv(s['tcp_quaternion']))))/c.orientation_scale)**2
        parts['held_opening']=s['opening']**2
        parts['lost_hold']=np.where(s['held'],0.,25.)
    else:
        parts['grasp_position']=(distance/c.position_scale)**2
        parts['grasp_orientation']=(angle/c.orientation_scale)**2
        parts['grasp_relative_speed']=(relative/limits)**2
    value=tc.weight*sum(getattr(tc,key)*item for key,item in parts.items())
    if not np.isfinite(value).all():raise ValueError('nonfinite terminal geometry cannot be filtered')
    return value,parts


def terminal_cost(initial,endpoint,candidate,phase,*,config=None,terminal_config=None):
    batch={key:np.asarray(value)[None] for key,value in endpoint.items() if key in (
        'tcp_position','tcp_quaternion','tcp_velocity','object_position','object_quaternion',
        'object_linear_velocity','object_angular_velocity','opening','held','time')}
    value,parts=terminal_cost_batch(initial,batch,[candidate],[0],phase,config=config,terminal_config=terminal_config)
    return float(value[0]),{key:float(item[0]) for key,item in parts.items()}


class TerminalScalarMPCPlanner(scalar.MPCPlanner):
    def __init__(self,*args,terminal_config=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.terminal_config=terminal_config if isinstance(terminal_config,TerminalCostConfig) else TerminalCostConfig(**(terminal_config or {}))
    def _rollout(self,state,candidate,controls,previous_phase,initial_h,committed_id,fixed_sequence=None):
        result=super()._rollout(state,candidate,controls,previous_phase,initial_h,committed_id,fixed_sequence)
        if self.terminal_config.enabled:
            value,parts=terminal_cost(state,result['states'][-1],candidate,previous_phase,
                config=self.config,terminal_config=self.terminal_config)
            result['cost']+=value
            result['cost_terms']['terminal_phi']=value
            for key,part in parts.items():result['cost_terms']['terminal_'+key]=part
        return result


from our_mani.local_mpc_p2_20261103.batch_mpc_pluggable import BatchMPCPlanner


class TerminalBatchMPCPlanner(BatchMPCPlanner):
    def __init__(self,*args,terminal_config=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.terminal_config=terminal_config if isinstance(terminal_config,TerminalCostConfig) else TerminalCostConfig(**(terminal_config or {}))
    def _rollout_batch(self,initial,candidates,modes,controls,old,committed,fixed_sequence):
        scores,parts,states,actions=super()._rollout_batch(initial,candidates,modes,controls,old,committed,fixed_sequence)
        if self.terminal_config.enabled:
            endpoint={key:value[:,-1] for key,value in states.items()}
            value,terms=terminal_cost_batch(initial,endpoint,candidates,modes,old,
                config=self.config,terminal_config=self.terminal_config)
            scores=scores+value
            parts['terminal_phi']=value
            for key,part in terms.items():parts['terminal_'+key]=part
        return scores,parts,states,actions
