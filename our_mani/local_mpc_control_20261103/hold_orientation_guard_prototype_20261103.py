"""Unwired hold-only predicted-microstep attitude feasibility research.

No Native, observation, target ownership, contact or physical safety assertion.
Only the original model's feasibility mask is narrowed; predicted state/h and
the action are neither edited nor substituted by measured future feedback.
"""
from copy import deepcopy
import hashlib
import json
import numpy as np

from ..local_mpc_20261102.batch_mpc import norm,qinv,qlog,qmul
from ..local_mpc_20261102.records import plain

VERSION='unwired_hold_predicted_micro_orientation_guard_20261103_v1'


def _quaternion(value):
    q=np.asarray(value,float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1.)>1e-4:
        raise ValueError('existing finite unit hold TCP quaternion required')
    q=q.copy();q.setflags(write=False);return q


def _identity(actual):
    if not isinstance(actual,dict) or not isinstance(actual.get('actuator_state'),dict):
        raise TypeError('current actual state and executor snapshot required')
    now=actual.get('time');h=actual['actuator_state'];clock=h.get('reference_clock')
    if (isinstance(now,(bool,np.bool_)) or not np.isscalar(now) or not np.isfinite(now)
        or isinstance(clock,(bool,np.bool_)) or not np.isscalar(clock) or not np.isfinite(clock)
        or abs(float(clock)-float(now))>1e-8 or h.get('predicted',False) is not False):
        raise ValueError('fresh actual actuator snapshot, never predicted/stale h')
    phase=actual.get('phase_state',{})
    if phase.get('phase')!='hold' or 'hold_tcp_quaternion' not in phase:
        raise ValueError('bind only actual hold phase with its existing fixed holdR')
    for name in ('q','dq'):
        value=np.asarray(actual.get(name),float)
        if value.shape!=(7,) or not np.isfinite(value).all():raise ValueError('actual seven joint '+name+' required')
    payload=dict(time=now,q=actual['q'],dq=actual['dq'],actuator_state=h,
                 hold_tcp_quaternion=phase['hold_tcp_quaternion'])
    return hashlib.sha256(json.dumps(plain(payload),sort_keys=True,allow_nan=False).encode()).hexdigest()


class HoldMicroOrientationGuard:
    """Bind one current hold feedback; delegate all arithmetic except feasible.

    ``maximum_rotation_rad`` is mandatory and belongs to a future explicit
    research protocol. This class supplies no tuned/default hold threshold.
    The closed-development replay uses the existing probe's .04 rad solely as
    a declared reference. Other phases return the base step tuple unchanged.
    """
    requires_joint_state=True
    requires_joint_goal=True

    def __init__(self,base_execution,actual_hold_state,*,maximum_rotation_rad):
        if (getattr(base_execution,'requires_joint_state',None) is not True
            or getattr(base_execution,'requires_joint_goal',None) is not True
            or not callable(getattr(base_execution,'fk_batch',None))):
            raise TypeError('joint response and its original public calibrated FK are required')
        if (isinstance(maximum_rotation_rad,(bool,np.bool_)) or not np.isscalar(maximum_rotation_rad)
            or not np.isfinite(maximum_rotation_rad) or maximum_rotation_rad<=0 or maximum_rotation_rad>np.pi):
            raise ValueError('explicit positive finite angle bound required; no default/tuning here')
        self.base=base_execution;self.config=base_execution.config
        self.maximum_rotation_rad=float(maximum_rotation_rad)
        self.hold_tcp_quaternion=_quaternion(actual_hold_state['phase_state']['hold_tcp_quaternion'])
        self.feedback_sha256=_identity(actual_hold_state)
        self.guard_calls=[]

    def initialize_batch(self,actual,batch_size):
        if _identity(actual)!=self.feedback_sha256:
            raise ValueError('new real q/dq/h/holdR needs a fresh independently bound wrapper')
        return self.base.initialize_batch(actual,batch_size)

    def step_batch(self,x,u,h,dt=.1):
        n=len(x['time']);phase=np.asarray(u.get('phase_index'))
        if phase.shape!=(n,) or phase.dtype.kind not in 'iu' or phase.dtype.kind=='b' or np.any((phase<0)|(phase>3)):
            raise ValueError('explicit per-row original MPC phase_index[N] required')
        out=self.base.step_batch(x,u,h,dt)
        hold=phase==3
        if not np.any(hold):return out
        nxt,hn,info=out
        feasible=np.asarray(info.get('feasible'))
        micro=np.asarray(info.get('predicted_actual_q'))
        if (info.get('microsteps')!=2 or feasible.shape!=(n,) or feasible.dtype!=np.bool_
            or micro.shape!=(n,2,7) or not np.isfinite(micro).all()):
            raise ValueError('two disclosed finite actual-joint predictions and original bool feasibility required')
        fk=self.base.fk_batch(micro.reshape(n*2,7).copy())
        q=np.asarray(fk.get('tcp_quaternion'),float)
        if q.shape!=(2*n,4) or not np.isfinite(q).all() or np.any(abs(norm(q)-1.)>1e-4):
            raise ValueError('original FK must return unit finite microstep quaternions')
        angles=norm(qlog(qmul(q,qinv(self.hold_tcp_quaternion)))).reshape(n,2)
        macro=norm(qlog(qmul(np.asarray(nxt['tcp_quaternion']),qinv(self.hold_tcp_quaternion))))
        inside=np.all(angles<=self.maximum_rotation_rad+1e-10,axis=1)&(macro<=self.maximum_rotation_rad+1e-10)
        narrowed=feasible&((~hold)|inside)
        receipt=dict(version=VERSION,hold_rows=int(hold.sum()),original_feasible=int(feasible.sum()),
            guarded_feasible=int(narrowed.sum()),newly_rejected=int(np.sum(feasible&hold&(~inside))),
            maximum_rotation_rad=self.maximum_rotation_rad,hold_micro_angles_rad=angles[hold].copy(),
            state_and_h_not_edited=True,action_not_edited=True,cost_or_budget_changed=False,
            discrete_model_predictions_only=True,continuous_or_real_attitude_guaranteed=False,
            Native_steps=0,production_integrated=False)
        self.guard_calls.append(deepcopy(receipt))
        guarded=dict(info,feasible=narrowed,hold_micro_orientation_guard=receipt)
        return nxt,hn,guarded


def bounded_object_nominal_destination(current_object_position,world_object_goal,*,maximum_step_m):
    """Pure proposal helper, deliberately unconnected and with no chosen step.

    A future nominal-IK-only variant may use this destination while retaining
    the original world goal in running/terminal costs and fresh real feedback.
    It is not a cached virtual state or executed motion/reference certificate.
    """
    current=np.asarray(current_object_position,float);goal=np.asarray(world_object_goal,float)
    if (current.shape!=(3,) or goal.shape!=(3,) or not np.isfinite(current).all() or not np.isfinite(goal).all()
        or isinstance(maximum_step_m,(bool,np.bool_)) or not np.isscalar(maximum_step_m)
        or not np.isfinite(maximum_step_m) or maximum_step_m<=0):
        raise ValueError('finite current/goal positions and explicitly predeclared positive step required')
    delta=goal-current;distance=float(np.linalg.norm(delta))
    return current+delta*min(1.,float(maximum_step_m)/distance) if distance else current.copy()
