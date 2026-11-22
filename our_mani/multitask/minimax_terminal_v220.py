"""TRAIN-only source-balanced minimax terminal translation, without dropping rows."""
from collections import Counter
import numpy as np
from scipy.optimize import minimize


def enclosing_center(points,weights=None):
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]!=3 or not len(p) or not np.isfinite(p).all():
        raise ValueError('finite terminal translations required')
    origin=np.average(p,axis=0,weights=weights);x=p-origin
    radius2=float(np.max(np.sum(x*x,axis=1)))
    if radius2<1e-24:return origin.copy(),dict(radius_m=0.,points=len(p),solver_success=True)
    initial=np.r_[np.zeros(3),radius2]
    result=minimize(lambda z:z[3],initial,jac=lambda z:np.array([0.,0.,0.,1.]),
        method='SLSQP',bounds=[(None,None)]*3+[(0.,None)],
        constraints=[dict(type='ineq',fun=lambda z:z[3]-np.sum((x-z[:3])**2,axis=1),
            jac=lambda z:np.c_[2.*(x-z[:3]),np.ones(len(x))])],
        options=dict(ftol=1e-14,maxiter=500))
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('terminal enclosing-ball optimization did not converge: '+str(result.message))
    center=result.x[:3]+origin
    measured=float(np.max(np.linalg.norm(p-center,axis=1)))
    if measured**2>result.x[3]+1e-8:raise ValueError('solver omitted a terminal constraint')
    return center,dict(radius_m=measured,points=len(p),solver_success=True,iterations=int(result.nit))


def fit_terminal(samples):
    terminal=[r for r in samples if r.get('terminal_event')]
    if not terminal:raise ValueError('original completed terminal rows required')
    counts=Counter(r['source'] for r in terminal)
    weights=np.asarray([1./counts[r['source']] for r in terminal])
    center,audit=enclosing_center([r['position'] for r in terminal],weights)
    return center,dict(audit,independent_parents=len(counts),parent_sources=sorted(counts),
        every_original_terminal_row_constrained=True,task_outcomes_used_for_estimator=False)
