"""Receding-horizon position-servo MPC using the public future trajectory.

Eight 50ms references are optimized jointly. Joint dynamics are the calibrated
position-input response. Configuration/velocity/acceleration/jerk constraints
are imposed on the actual reference sequence, with a stoppable first action.
The original caller still qualifies the first reference against Native static
collisions before dispatch. Future object poses come only from public GRU.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import LinearConstraint
from scipy import sparse
import osqp
from itertools import product
from scipy.spatial.transform import Rotation
from .tracking_v4 import StoppableTrackingReferencePlanner,StoppableTrackingConfig,reference_window


@dataclass(frozen=True)
class PredictiveTrackingConfig(StoppableTrackingConfig):
    horizon_ticks: int = 8
    position_scale: float = .03
    orientation_scale: float = .2
    relative_speed_weight: float = .2
    orientation_weight: float = .2
    terminal_velocity_zero: bool = True
    optimizer_iterations: int = 80

    def __post_init__(self):
        super().__post_init__()
        if type(self.horizon_ticks) is not int or not 4<=self.horizon_ticks<=16:raise ValueError('4..16 predicted 50ms ticks required')


class ReservedQPTrackingPlanner(StoppableTrackingReferencePlanner):
    def __init__(self,*args,config=None,**kwargs):
        super().__init__(*args,config=config or PredictiveTrackingConfig(),**kwargs)

    def plan(self,state,candidate,forecast,*,pregrasp):
        began=time.perf_counter();c=self.config;dt=.05;H=c.horizon_ticks;n=H*7
        q=np.asarray(state['q']);dq=np.asarray(state['dq']);r,v,a=self._history(state)
        low0,high0=reference_window(r,v,a,self.limits,margin=c.joint_limit_margin)
        if np.any(low0>high0+1e-10):return dict(action=None,reason='no_stoppable_reference')
        coeff=self.joint_response.coefficients
        gain,momentum,bias=coeff[:,0],coeff[:,1],coeff[:,2]
        # q_pred[k] = maps[k] @ (qref_sequence-r) + offsets[k].
        maps=[];offsets=[];previous_map=np.zeros((7,n));before_map=np.zeros((7,n))
        previous_q=q.copy();before_q=q-dq*dt
        for k in range(H):
            F=(1-gain+momentum)[:,None]*previous_map-momentum[:,None]*before_map
            F[:,k*7:(k+1)*7]+=np.diag(gain)
            b=(1-gain+momentum)*previous_q-momentum*before_q+gain*r+bias
            maps.append(F);offsets.append(b)
            before_map,previous_map=previous_map,F;before_q,previous_q=previous_q,b
        maps=np.asarray(maps);offsets=np.asarray(offsets)
        J=self.kinematics.geometric_jacobian(q);Jp,Jw=J[:3],J[3:]
        previous_map=np.zeros((7,n));previous_offset=q.copy()
        velocity_maps=[];velocity_offsets=[]
        M=[];rhs=[]
        local=np.asarray(candidate['object_tcp_position']).copy()
        if pregrasp:local+=candidate['approach_direction']*candidate['pregrasp_offset']
        actual_R=Rotation.from_quat(state['tcp_quaternion']).as_matrix()
        center=.5*(np.asarray(self.limits.lower)+np.asarray(self.limits.upper))
        for k in range(H):
            pose,target_v,target_w=forecast.at('target',state['time']+(k+1)*dt)
            target=pose.apply(local)
            target_R=pose.rotation@Rotation.from_quat(candidate['object_tcp_quaternion']).as_matrix()
            rot=Rotation.from_matrix(target_R@actual_R.T).as_rotvec()
            F,b=maps[k],offsets[k]
            V=(F-previous_map)/dt;vb=(b-previous_offset)/dt
            velocity_maps.append(Jp@V);velocity_offsets.append(Jp@vb)
            weight=.95**k
            M.extend([weight*(Jp@F)/c.position_scale,
                weight*np.sqrt(c.orientation_weight)*(Jw@F)/c.orientation_scale,
                weight*np.sqrt(c.relative_speed_weight)*(Jp@V)/.14,
                .02*F])
            rhs.extend([weight*(target-state['tcp_position']-Jp@(b-q))/c.position_scale,
                weight*np.sqrt(c.orientation_weight)*(rot-Jw@(b-q))/c.orientation_scale,
                weight*np.sqrt(c.relative_speed_weight)*(target_v-Jp@vb)/.14,
                .02*(center-b)])
            previous_map,previous_offset=F,b
        M=np.vstack(M+[c.regularization*np.eye(n)]);rhs=np.concatenate(rhs+[np.zeros(n)])
        P=M.T@M;linear=-(M.T@rhs)
        # Finite differences of absolute references. Initial history is actual
        # previously issued reference data, not a virtual warm trajectory.
        D1=np.eye(H);D2=np.eye(H);D3=np.eye(H)
        for k in range(H):
            if k>=1:D1[k,k-1]=-1.;D2[k,k-1]=-2.;D3[k,k-1]=-3.
            if k>=2:D2[k,k-2]=1.;D3[k,k-2]=3.
            if k>=3:D3[k,k-3]=-1.
        hist=[r,r-v*dt,r-2*v*dt+a*dt*dt]
        def differences(order,matrix):
            weights={1:[1,-1],2:[1,-2,1],3:[1,-3,3,-1]}[order]
            const=[]
            for k in range(H):
                val=np.zeros(7)
                for j,w in enumerate(weights):
                    if k-j>=0:val+=w*r
                    else:val+=w*hist[-(k-j)-1]
                const.extend(val/dt**order)
            return np.kron(matrix,np.eye(7))/dt**order,np.asarray(const)
        constraints=[];Crows=[];L=[];U=[]
        for order,D,limit in [(1,D1,np.asarray(self.limits.effective_velocity)-1e-4),
                              (2,D2,np.asarray(self.limits.acceleration)-1e-3),
                              (3,D3,np.asarray(self.limits.jerk)-1e-2)]:
            C,b=differences(order,D);Crows.append(C);lim=np.tile(limit,H)
            L.extend(-lim-b);U.extend(lim-b)
        Crows.append(maps.reshape(H*7,n));L.extend(np.tile(self.limits.lower,H)-offsets.reshape(-1));U.extend(np.tile(self.limits.upper,H)-offsets.reshape(-1))
        if c.terminal_velocity_zero:
            final=np.zeros((7,n));final[:,-7:]=np.eye(7);final[:,-14:-7]=-np.eye(7)
            Crows.append(final);L.extend(np.zeros(7));U.extend(np.zeros(7))
        C=np.vstack(Crows);L=np.asarray(L);U=np.asarray(U)
        equal=np.abs(U-L)<1e-12
        constraints.append(LinearConstraint(C[~equal],L[~equal],U[~equal]))
        if equal.any():constraints.append(LinearConstraint(C[equal],L[equal],U[equal]))
        lower=np.tile(np.asarray(self.limits.lower)+c.joint_limit_margin-r,H)
        upper=np.tile(np.asarray(self.limits.upper)-c.joint_limit_margin-r,H)
        lower[:7]=np.maximum(lower[:7],low0-r+1e-8);upper[:7]=np.minimum(upper[:7],high0-r-1e-8)
        if np.any(lower>upper):return dict(action=None,reason='empty_predictive_reference_bounds')
        # Start from bounded braking, rather than an infeasible instantaneous
        # stop. It also provides an explicit fallback when optimization fails.
        rr,vv,aa=r.copy(),v.copy(),a.copy();start=[]
        for k in range(H):
            lo,hi=reference_window(rr,vv,aa,self.limits,margin=c.joint_limit_margin)
            p=np.minimum(hi,np.maximum(lo,rr));nv=(p-rr)/dt;na=(nv-vv)/dt
            start.extend(p-r);rr,vv,aa=p,nv,na
        start=np.minimum(upper,np.maximum(lower,np.asarray(start)))
        vm=np.asarray(velocity_maps);vo=np.asarray(velocity_offsets)
        def speed_constraint(z):
            velocities=np.einsum('kij,j->ki',vm,z)+vo
            return c.maximum_tcp_speed**2-np.sum(velocities**2,axis=1)
        def speed_jac(z):
            velocities=np.einsum('kij,j->ki',vm,z)+vo
            return -2*np.einsum('ki,kij->kj',velocities,vm)
        directions=np.array([v for v in product((-1.,0.,1.),repeat=3) if any(v)])
        directions/=np.linalg.norm(directions,axis=1)[:,None]
        # This symmetric 26-facet polytope lies inside the original .7m/s
        # sphere. Its furthest unit-bound vertex has the norm below.
        factor=np.sqrt(1.+(np.sqrt(2.)-1.)**2+(np.sqrt(3.)-np.sqrt(2.))**2)
        speed_bound=c.maximum_tcp_speed/factor-1e-5
        velC=np.concatenate([directions@vm[k] for k in range(H)],axis=0)
        velb=np.concatenate([directions@vo[k] for k in range(H)])
        Ac=np.vstack([C,np.eye(n),velC])
        lc=np.r_[L,lower,-speed_bound-velb];uc=np.r_[U,upper,speed_bound-velb]
        scales=1./np.maximum(np.max(np.abs(Ac),axis=1),1.)
        objective_scale=max(float(np.max(np.abs(np.diag(P)))),1.)
        solver=osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(.5*(P+P.T)/objective_scale),q=linear/objective_scale,
            A=sparse.csc_matrix(Ac*scales[:,None]),l=lc*scales,u=uc*scales,
            verbose=False,eps_abs=1e-10,eps_rel=1e-10,max_iter=30000,polishing=True)
        solver.warm_start(x=start)
        result=solver.solve(raise_error=False)
        if result.x is None or result.info.status_val not in (1,2):
            return dict(action=None,reason='predictive_QP_not_solved',optimizer_status=int(result.info.status_val),optimizer_message=result.info.status)
        z=result.x;cz=C@z
        # Above, tiny numerical reserves tightened the original derivative
        # constraints. Acceptance is checked against the ORIGINAL limits,
        # rather than rejecting a feasible point for using that reserve.
        reserve=np.r_[np.full(H*7,1e-4),np.full(H*7,1e-3),np.full(H*7,1e-2),np.zeros(len(L)-3*H*7)]
        violation=max(0.,float(np.max(L-reserve-cz)),float(np.max(cz-U-reserve)),float(np.max(lower-z)),float(np.max(z-upper)),float(-np.min(speed_constraint(z))))
        if not np.isfinite(z).all() or violation>1e-7:
            return dict(action=None,reason='predictive_reference_infeasible',optimizer_status=int(result.info.status_val),maximum_violation=violation,violation_components=dict(lower_constraints=float(np.max(L-reserve-cz)),upper_constraints=float(np.max(cz-U-reserve)),lower_bounds=float(np.max(lower-z)),upper_bounds=float(np.max(z-upper)),speed=float(-np.min(speed_constraint(z)))))
        ref=r+z[:7];ref_v=(ref-r)/dt;ref_a=(ref_v-v)/dt
        prediction=self.joint_response.predict_joint(q[None],dq[None],ref[None],ref_v[None],ref_a[None])
        predicted_pose=self.kinematics(prediction['q'][0]);command_pose=self.kinematics(ref)
        return dict(action=dict(joint_goal=ref,tcp_position=command_pose[:3,3],tcp_quaternion=Rotation.from_matrix(command_pose[:3,:3]).as_quat(),joint_goal_feasible=True),
            reference=dict(q=ref,v=ref_v,a=ref_a),reference_sequence=r+z.reshape(H,7),
            predicted_q=prediction['q'][0],predicted_dq=prediction['dq'][0],predicted_TCP=predicted_pose,
            predicted_linear_velocity=(predicted_pose[:3,3]-state['tcp_position'])/dt,
            horizon_ticks=H,future_public_forecast_used=hasattr(forecast,'times'),
            optimizer_status=int(result.info.status_val),optimizer_iterations=int(result.info.iter),maximum_violation=violation,wall_seconds=time.perf_counter()-began,solver='OSQP',
            Native_object_state_used=False,dt=dt)
