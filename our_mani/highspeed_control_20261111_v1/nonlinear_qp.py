"""Sequential QP with FK/Jacobians recomputed along the predicted robot path."""
from copy import deepcopy
import time
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
from itertools import product
import osqp
from .qp_tracking_v2 import ReservedQPTrackingPlanner
from .tracking_v4 import StoppableTrackingReferencePlanner,reference_window
from .hybrid_tracking import HybridTrackingConfig


class NonlinearHybridTrackingPlanner(ReservedQPTrackingPlanner):
    def __init__(self,*args,config=None,workspace_lower=(-.1,-.65,.755),workspace_upper=(.65,.65,1.65),**kwargs):
        self.workspace_lower=np.asarray(workspace_lower);self.workspace_upper=np.asarray(workspace_upper)
        super().__init__(*args,config=config or HybridTrackingConfig(),**kwargs)

    def plan(self,state,candidate,forecast,*,pregrasp):
        c=self.config;working=deepcopy(candidate)
        if pregrasp:working['pregrasp_offset']=min(working['pregrasp_offset'],c.execution_pregrasp_offset)
        if not hasattr(forecast,'times'):
            return StoppableTrackingReferencePlanner.plan(self,state,working,forecast,pregrasp=pregrasp)
        pose,_,_=forecast.at('target',state['time'])
        local=np.asarray(working['object_tcp_position']).copy()
        if pregrasp:local+=working['approach_direction']*working['pregrasp_offset']
        distance=np.linalg.norm(pose.apply(local)-state['tcp_position'])
        if distance>c.local_mpc_radius:
            return StoppableTrackingReferencePlanner.plan(self,state,working,forecast,pregrasp=pregrasp)
        return self.preview(state,working,forecast,pregrasp=pregrasp)

    def preview(self,state,candidate,forecast,*,pregrasp):
        began=time.perf_counter();c=self.config;dt=.05;H=c.horizon_ticks;n=7*H
        q=np.asarray(state['q']);dq=np.asarray(state['dq']);r,v,a=self._history(state)
        lo0,hi0=reference_window(r,v,a,self.limits,margin=c.joint_limit_margin)
        if np.any(lo0>hi0+1e-10):return dict(action=None,reason='no_stoppable_reference')
        gain,momentum,bias=self.joint_response.coefficients.T
        maps=[];offsets=[];prevF=np.zeros((7,n));beforeF=prevF.copy();prevq=q.copy();beforeq=q-dq*dt
        for k in range(H):
            F=(1-gain+momentum)[:,None]*prevF-momentum[:,None]*beforeF
            F[:,7*k:7*k+7]+=np.diag(gain)
            b=(1-gain+momentum)*prevq-momentum*beforeq+gain*r+bias
            maps.append(F);offsets.append(b);beforeF,prevF=prevF,F;beforeq,prevq=prevq,b
        maps=np.asarray(maps);offsets=np.asarray(offsets)
        Crows=[];L=[];U=[]
        history=[r,r-v*dt,r-2*v*dt+a*dt*dt]
        for order,limit in [(1,np.asarray(self.limits.effective_velocity)-1e-4),
                            (2,np.asarray(self.limits.acceleration)-1e-3),
                            (3,np.asarray(self.limits.jerk)-1e-2)]:
            weights={1:[1,-1],2:[1,-2,1],3:[1,-3,3,-1]}[order]
            D=np.zeros((H,H));const=[]
            for k in range(H):
                b=np.zeros(7)
                for j,w in enumerate(weights):
                    if k-j>=0:D[k,k-j]+=w;b+=w*r
                    else:b+=w*history[j-k-1]
                const.extend(b/dt**order)
            Crows.append(np.kron(D,np.eye(7))/dt**order)
            L.extend(-np.tile(limit,H)-const);U.extend(np.tile(limit,H)-const)
        Crows.append(maps.reshape(7*H,n));L.extend(np.tile(self.limits.lower,H)-offsets.ravel());U.extend(np.tile(self.limits.upper,H)-offsets.ravel())
        if c.terminal_velocity_zero:
            D=np.zeros((7,n));D[:,-7:]=np.eye(7);D[:,-14:-7]=-np.eye(7)
            Crows.append(D);L.extend(np.zeros(7));U.extend(np.zeros(7))
        C=np.vstack(Crows);L=np.asarray(L);U=np.asarray(U)
        lower=np.tile(np.asarray(self.limits.lower)+c.joint_limit_margin-r,H)
        upper=np.tile(np.asarray(self.limits.upper)-c.joint_limit_margin-r,H)
        lower[:7]=np.maximum(lower[:7],lo0-r+1e-8);upper[:7]=np.minimum(upper[:7],hi0-r-1e-8)
        if np.any(lower>upper):return dict(action=None,reason='empty_nonlinear_reference_bounds')
        initial=ReservedQPTrackingPlanner.plan(self,state,candidate,forecast,pregrasp=pregrasp)
        if initial.get('reference_sequence') is not None:z=(initial['reference_sequence']-r).ravel()
        else:
            rr,vv,aa=r.copy(),v.copy(),a.copy();refs=[]
            for k in range(H):
                lo,hi=reference_window(rr,vv,aa,self.limits,margin=c.joint_limit_margin)
                nr=np.minimum(hi,np.maximum(lo,rr));nv=(nr-rr)/dt;na=(nv-vv)/dt
                refs.append(nr-r);rr,vv,aa=nr,nv,na
            z=np.asarray(refs).ravel()
        local=np.asarray(candidate['object_tcp_position']).copy()
        if pregrasp:local+=candidate['approach_direction']*candidate['pregrasp_offset']
        directions=np.array([v for v in product((-1.,0.,1.),repeat=3) if any(v)])
        directions/=np.linalg.norm(directions,axis=1)[:,None]
        factor=np.sqrt(1.+(np.sqrt(2.)-1.)**2+(np.sqrt(3.)-np.sqrt(2.))**2)
        speed_bound=(c.maximum_tcp_speed-.01)/factor
        total_iterations=0
        for iteration in range(3):
            nominal=np.einsum('kij,j->ki',maps,z)+offsets
            M=[];rhs=[];velC=[];velb=[];workC=[];workL=[];workU=[]
            previous_p=np.asarray(state['tcp_position']);previous_F=np.zeros((3,n));previous_b=previous_p.copy()
            center=.5*(np.asarray(self.limits.lower)+np.asarray(self.limits.upper))
            for k in range(H):
                pose,velocity,_=forecast.at('target',state['time']+(k+1)*dt)
                target=pose.apply(local);targetR=pose.rotation@Rotation.from_quat(candidate['object_tcp_quaternion']).as_matrix()
                F,b=maps[k],offsets[k];T=self.kinematics(nominal[k]);J=self.kinematics.geometric_jacobian(nominal[k]);Jp,Jw=J[:3],J[3:]
                position_F=Jp@F;position_b=T[:3,3]+Jp@(b-nominal[k])
                V=(position_F-previous_F)/dt;vb=(position_b-previous_b)/dt
                rot=Rotation.from_matrix(targetR@T[:3,:3].T).as_rotvec()
                weight=.95**k
                M.extend([weight*position_F/c.position_scale,weight*np.sqrt(c.orientation_weight)*(Jw@F)/c.orientation_scale,
                    weight*np.sqrt(c.relative_speed_weight)*V/.14,.02*F])
                rhs.extend([weight*(target-position_b)/c.position_scale,
                    weight*np.sqrt(c.orientation_weight)*(rot-Jw@(b-nominal[k]))/c.orientation_scale,
                    weight*np.sqrt(c.relative_speed_weight)*(velocity-vb)/.14,.02*(center-b)])
                velC.append(directions@V);velb.extend(directions@vb)
                # The same diagnostic workspace is also imposed on reference
                # FK, rather than repeatedly rejecting overshoot afterwards.
                reference=r+z[7*k:7*k+7];Tr=self.kinematics(reference);Jr=self.kinematics.geometric_jacobian(reference)[:3]
                Wr=np.zeros((3,n));Wr[:,7*k:7*k+7]=Jr;br=Tr[:3,3]+Jr@(r-reference)
                workC.append(Wr);workL.extend(self.workspace_lower+.001-br);workU.extend(self.workspace_upper-.001-br)
                previous_p,previous_F,previous_b=T[:3,3],position_F,position_b
            M=np.vstack(M+[c.regularization*np.eye(n)]);rhs=np.concatenate(rhs+[np.zeros(n)])
            P=M.T@M;linear=-(M.T@rhs);velb=np.asarray(velb)
            Ac=np.vstack([C,np.eye(n),np.vstack(velC),np.vstack(workC)])
            lc=np.r_[L,lower,-speed_bound-velb,workL];uc=np.r_[U,upper,speed_bound-velb,workU]
            row_scale=1./np.maximum(np.max(np.abs(Ac),axis=1),1.);obj_scale=max(np.max(np.abs(np.diag(P))),1.)
            solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(.5*(P+P.T)/obj_scale),q=linear/obj_scale,
                A=sparse.csc_matrix(Ac*row_scale[:,None]),l=lc*row_scale,u=uc*row_scale,
                verbose=False,eps_abs=1e-10,eps_rel=1e-10,max_iter=30000,polishing=True)
            solver.warm_start(x=z);result=solver.solve(raise_error=False)
            if result.x is None or result.info.status_val not in (1,2):
                return dict(action=None,reason='nonlinear_QP_not_solved',optimizer_message=result.info.status)
            z=result.x;total_iterations+=result.info.iter
        reserve=np.r_[np.full(H*7,1e-4),np.full(H*7,1e-3),np.full(H*7,1e-2),np.zeros(len(L)-3*H*7)]
        violation=max(0.,np.max(L-reserve-C@z),np.max(C@z-U-reserve),np.max(lower-z),np.max(z-upper))
        predicted=np.einsum('kij,j->ki',maps,z)+offsets
        poses=[self.kinematics(qk) for qk in predicted]
        speeds=np.linalg.norm(np.diff(np.vstack([state['tcp_position'],[T[:3,3] for T in poses]]),axis=0)/dt,axis=1)
        if not np.isfinite(z).all() or violation>1e-7 or np.max(speeds)>c.maximum_tcp_speed+1e-7:
            return dict(action=None,reason='nonlinear_prediction_infeasible',maximum_violation=float(violation),maximum_predicted_tcp_speed=float(np.max(speeds)))
        ref=r+z[:7];nv=(ref-r)/dt;na=(nv-v)/dt;T=self.kinematics(ref)
        return dict(action=dict(joint_goal=ref,tcp_position=T[:3,3],tcp_quaternion=Rotation.from_matrix(T[:3,:3]).as_quat(),joint_goal_feasible=True),
            reference=dict(q=ref,v=nv,a=na),reference_sequence=r+z.reshape(H,7),predicted_q=predicted[0],
            predicted_TCP=poses[0],predicted_linear_velocity=(poses[0][:3,3]-state['tcp_position'])/dt,
            future_public_forecast_used=True,horizon_ticks=H,maximum_violation=float(violation),
            maximum_predicted_tcp_speed=float(np.max(speeds)),optimizer_iterations=int(total_iterations),
            controller_regime='sequential_QP_with_future_FK_relinearization',solver='OSQP',
            Native_object_state_used=False,wall_seconds=time.perf_counter()-began,dt=dt)
