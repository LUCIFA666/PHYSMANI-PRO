"""Qualify a reference and its bounded stopping continuation before dispatch.

Only robot q/dq, actual reference history, the robot response calibration and
the original zero-step static collision oracle are used. No object forecast
or Native target body state is accepted here.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from .tracking_v4 import reference_window


class RobustCollisionStoppingGuard:
    def __init__(self,planner,helper,backend):
        self.planner,self.helper,self.backend=planner,helper,backend

    def action(self,ref,opening):
        pose=self.planner.kinematics(ref)
        return dict(joint_goal=np.asarray(ref).copy(),tcp_position=pose[:3,3],
            tcp_quaternion=Rotation.from_matrix(pose[:3,:3]).as_quat(),
            joint_goal_feasible=True,execution_semantics='sampled_joint_reference',opening=opening)

    def certify(self,state,ref,opening):
        p=self.planner;dt=.05
        r,v,a=p._history(state)
        q=np.asarray(state['q']).copy();dq=np.asarray(state['dq']).copy()
        ref=np.asarray(ref).copy();probes=[]
        # Original amax=12 can stop the maximum 3rad/s in five samples.
        # Two extra zero-velocity samples check the calibrated plant settling.
        steps=int(np.ceil(np.max(np.asarray(p.limits.effective_velocity)/np.asarray(p.limits.acceleration))/dt))+3
        stopped=0
        for k in range(steps):
            lo,hi=reference_window(r,v,a,p.limits,margin=p.config.joint_limit_margin)
            if np.any(ref<lo-1e-8) or np.any(ref>hi+1e-8):
                return False,dict(reason='stopping_reference_bounds',step=k,probes=probes)
            action=self.action(ref,opening)
            _,reason,receipt=self.helper.qualify_action(self.backend,dict(state,q=q),action,require_predicted_joint_goal=True)
            probes.append(dict(step=k,reference=ref.copy(),qualification=receipt,reason=reason))
            if reason:return False,dict(reason=reason,step=k,probes=probes)
            nv=(ref-r)/dt;na=(nv-v)/dt
            predicted=p.joint_response.predict_joint(q[None],dq[None],ref[None],nv[None],na[None])
            nq,ndq=predicted['q'][0],predicted['dq'][0]
            # The motor's momentum can move outside the q -> qref segment.
            predicted_action=self.action(nq,opening)
            _,reason,pred_receipt=self.helper.qualify_action(self.backend,dict(state,q=q),predicted_action,require_predicted_joint_goal=True)
            probes[-1].update(predicted_q=nq.copy(),predicted_qualification=pred_receipt,predicted_reason=reason)
            if reason:return False,dict(reason='predicted_robot_'+reason,step=k,probes=probes)
            # Empirical robot-response residuals are nonzero. Check nearby
            # configurations before accepting a stopping path on a surface.
            margin_probes=[]
            for anchor,margin in ((ref,.005),(nq,.01)):
                for joint in range(7):
                    for sign in (-1.,1.):
                        near=anchor.copy();near[joint]+=sign*margin
                        if np.any(near<np.asarray(p.limits.lower)) or np.any(near>np.asarray(p.limits.upper)):
                            collision=True
                        else:collision=bool(self.backend.probe_joint_collision(near))
                        margin_probes.append(dict(joint=joint,sign=sign,margin_rad=margin,collision=collision))
                        if collision:
                            probes[-1]['response_margin_probes']=margin_probes
                            return False,dict(reason='robot_response_margin_static_collision',step=k,probes=probes)
            probes[-1]['response_margin_probes']=margin_probes
            q,dq=nq,ndq;r,v,a=ref,nv,na
            stopped=stopped+1 if np.max(np.abs(v))<1e-7 else 0
            if stopped>=2:
                return True,dict(reason=None,probes=probes,stopping_continuation_qualified=True,
                    whole_path_certified=False,physical_steps=0,Native_object_state_used=False)
            lo,hi=reference_window(r,v,a,p.limits,margin=p.config.joint_limit_margin)
            if np.any(lo>hi+1e-10):return False,dict(reason='no_stopping_continuation',probes=probes)
            ref=np.minimum(hi,np.maximum(lo,r))
        return False,dict(reason='stopping_continuation_not_stationary',probes=probes)

    def select(self,state,proposed,opening):
        p=self.planner;r,v,a=p._history(state)
        lo,hi=reference_window(r,v,a,p.limits,margin=p.config.joint_limit_margin)
        if np.any(lo>hi+1e-10):raise ValueError('No bounded reference in collision guard')
        brake=np.minimum(hi,np.maximum(lo,r))
        candidates=[(0.,brake)] if proposed is None else [(weight,brake+weight*(np.asarray(proposed['joint_goal'])-brake)) for weight in (1.,.75,.5,.25,0.)]
        attempts=[]
        for weight,ref in candidates:
            allowed,receipt=self.certify(state,ref,opening)
            attempts.append(dict(proposal_weight=weight,allowed=allowed,receipt=receipt))
            if allowed:
                action=deepcopy(proposed) if weight==1. and proposed is not None else self.action(ref,opening)
                return action,dict(proposal_weight=weight,attempts=attempts,stopping_continuation_qualified=True,
                    original_collision_gates_preserved=True,physical_steps=0)
        raise RuntimeError('No collision-qualified bounded stopping continuation: '+str([(x['proposal_weight'],x['receipt']['reason']) for x in attempts]))
