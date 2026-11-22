"""Coupled reference governor with measured-state tabletop barrier constraints."""
import numpy as np
from scipy.optimize import minimize,Bounds,LinearConstraint
from ..executor_contract_repair_20261112_v2.reference import ReferenceState,ReferenceInfeasible

DT=.05


def step(state,desired_velocity,limits,*,actual_q,barrier_A=None,barrier_b=None):
    p,v,a=state.position,state.velocity,state.acceleration
    desired_velocity=np.asarray(desired_velocity,float);actual=np.asarray(actual_q,float)
    lo,hi=np.asarray(limits.lower),np.asarray(limits.upper)
    vmax,amax,jmax=map(np.asarray,(limits.effective_velocity,limits.acceleration,limits.jerk))
    if desired_velocity.shape!=p.shape or not np.isfinite(desired_velocity).all():
        raise ValueError('Finite desired reference velocity required')
    # Slow approach to a joint bound before the one-step position interval
    # becomes empty. This constrains references, never clips measured q/goals.
    braking_velocity_lower=np.maximum(-vmax,-2.*(p-lo))
    braking_velocity_upper=np.minimum(vmax,2.*(hi-p))
    low=np.maximum.reduce((-amax,a-jmax*DT,(braking_velocity_lower-v)/DT,(lo-p-v*DT)/DT**2))
    high=np.minimum.reduce((amax,a+jmax*DT,(braking_velocity_upper-v)/DT,(hi-p-v*DT)/DT**2))
    if np.any(low>high+1e-9):raise ReferenceInfeasible('Joint reference constraints infeasible')
    error=float(np.max(abs(p-actual)))
    progress=float(np.clip((.15-error)/.10,0,1))
    desired=(desired_velocity-v)/DT
    constraints=[dict(type='ineq',fun=lambda x:12.**2-np.dot(x,x),jac=lambda x:-2*x),
                 dict(type='ineq',fun=lambda x:3.**2-np.dot(v+x*DT,v+x*DT),jac=lambda x:-2*DT*(v+x*DT))]
    A=np.asarray(barrier_A if barrier_A is not None else np.zeros((0,len(p))),float)
    b=np.asarray(barrier_b if barrier_b is not None else np.zeros(0),float)
    if len(A):constraints.append(LinearConstraint(A,b,np.inf))
    def feasible(x,tolerance=1e-7):
        return (np.all(x>=low-tolerance)and np.all(x<=high+tolerance)and np.linalg.norm(x)<=12.+tolerance and
                np.linalg.norm(v+x*DT)<=3.+tolerance and (not len(A)or np.all(A@x>=b-tolerance)))
    clip=lambda x:np.minimum(high,np.maximum(low,x))
    radial=clip(desired*min(1.,12./max(np.linalg.norm(desired),1e-12)))
    seed=radial
    if not feasible(seed):
        candidates=[radial,clip(np.zeros(len(p))),clip(a),
                    clip(-v*min(1./DT,12./max(np.linalg.norm(v),1e-12)))]
        valid=[x for x in candidates if feasible(x)]
        initial=min(valid,key=lambda x:np.linalg.norm(x-desired))if valid else radial
        scale=max(1.,np.linalg.norm(desired))
        result=minimize(lambda x:.5*np.dot(x-desired,x-desired)/scale,initial,jac=lambda x:(x-desired)/scale,
                        bounds=Bounds(low,high),constraints=constraints,method='SLSQP',
                        options=dict(maxiter=120,ftol=1e-10))
        seed=result.x
        if not feasible(seed)and valid:seed=initial
        if not feasible(seed):
            error=ReferenceInfeasible('Coupled actuator/tabletop barrier infeasible: '+result.message)
            error.diagnostic=dict(p=p.tolist(),v=v.tolist(),a=a.tolist(),desired_velocity=desired_velocity.tolist(),actual=actual.tolist(),
                A=A.tolist(),b=b.tolist(),low=low.tolist(),high=high.tolist(),candidate=seed.tolist())
            raise error
    # Only absorb floating-point optimizer error into a commanded acceleration;
    # destinations and measured robot state are never clipped or relabelled.
    next_a=np.minimum(high,np.maximum(low,seed))
    if not feasible(next_a):raise ReferenceInfeasible('Projected reference failed independent constraint check')
    next_v=v+next_a*DT;next_p=p+next_v*DT
    return ReferenceState(next_p,next_v,next_a),dict(tracking_progress_scale=progress,
        tracking_error_before_rad=error,reference_velocity_norm=float(np.linalg.norm(next_v)),
        reference_acceleration_norm=float(np.linalg.norm(next_a)),
        minimum_barrier_residual=None if not len(A)else float(np.min(A@next_a-b)))
