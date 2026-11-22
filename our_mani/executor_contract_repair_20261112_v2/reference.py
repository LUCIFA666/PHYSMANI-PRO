"""Position commands with bounds on their actual discrete differences.

Targets can change without reseeding from measured q/dq. The state represents
only dispatched references; it never claims to predict real robot motion.
"""
from dataclasses import dataclass
import numpy as np

DT=.05

@dataclass(frozen=True)
class ReferenceState:
    position: object
    velocity: object
    acceleration: object

    def __post_init__(self):
        arrays=[np.asarray(v,float).copy()for v in (self.position,self.velocity,self.acceleration)]
        if any(a.ndim!=1 or a.shape!=arrays[0].shape or not np.isfinite(a).all()for a in arrays):
            raise ValueError('Finite equal-sized reference state required')
        for name,a in zip(('position','velocity','acceleration'),arrays):
            a.setflags(write=False);object.__setattr__(self,name,a)

class ReferenceInfeasible(ValueError):
    pass

def step(state,goal,limits,*,frequency=6.):
    p,v,a=state.position,state.velocity,state.acceleration
    goal=np.asarray(goal,float)
    lo,hi=np.asarray(limits.lower),np.asarray(limits.upper)
    vmax,amax,jmax=map(np.asarray,(limits.effective_velocity,limits.acceleration,limits.jerk))
    if (goal.shape!=p.shape or not np.isfinite(goal).all()or np.any(goal<lo)or np.any(goal>hi)):
        raise ValueError('Unmodified finite destination within native bounds required')
    # v_next=(p_next-p)/dt, a_next=(v_next-v)/dt, j_next=(a_next-a)/dt.
    # In particular, the position bound uses dt**2, not the continuous-time
    # trapezoid formula that would give different finite differences.
    low=np.maximum.reduce((-amax,a-jmax*DT,(-vmax-v)/DT,(lo-p-v*DT)/DT**2))
    high=np.minimum.reduce((amax,a+jmax*DT,(vmax-v)/DT,(hi-p-v*DT)/DT**2))
    if np.any(low>high+1e-9):raise ReferenceInfeasible('No bounded continuous next reference')
    desired=frequency**2*(goal-p)-2*frequency*v
    next_a=np.minimum(high,np.maximum(low,desired))
    next_v=v+next_a*DT
    next_p=p+next_v*DT
    return ReferenceState(next_p,next_v,next_a)

def preview(state,goal,limits,*,ticks=200,frequency=6.):
    current=state
    for index in range(ticks):
        try:current=step(current,goal,limits,frequency=frequency)
        except ReferenceInfeasible:
            return dict(feasible=False,failed_reference_preview_tick=index+1,physics_steps=0)
    return dict(feasible=True,preview_ticks=ticks,physics_steps=0,actual_robot_prediction=False)
