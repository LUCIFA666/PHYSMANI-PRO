"""Position-servo tracking with stoppable references and joint-limit posture."""
from dataclasses import dataclass
import numpy as np
from .tracking import TrackingConfig


@dataclass(frozen=True)
class StoppableTrackingConfig(TrackingConfig):
    joint_limit_margin: float = .08
    posture_gain: float = 2.
    posture_weight: float = .04


def reference_window(r,v,a,limits,*,margin=.08,dt=.05):
    """Intersect original reference limits with a braking-distance envelope.

    At the original jerk=1000 and acceleration=12, a full acceleration sign
    reversal is possible in one 50ms sample. The stopping envelope reserves
    space for subsequent bounded deceleration, rather than reaching a joint
    position limit at nonzero reference velocity.
    """
    lo,hi=np.asarray(limits.lower),np.asarray(limits.upper)
    amax,jmax=np.asarray(limits.acceleration),np.asarray(limits.jerk)
    if np.any(jmax*dt<2*amax):raise ValueError('This stopping envelope requires acceleration sign reversal within one tick')
    reserve=np.minimum(margin,.1*(hi-lo))
    left=np.maximum(r-(lo+reserve),0.)
    right=np.maximum((hi-reserve)-r,0.)
    negative=np.maximum(0.,np.sqrt((amax*dt)**2+2*amax*left)-amax*dt)
    positive=np.maximum(0.,np.sqrt((amax*dt)**2+2*amax*right)-amax*dt)
    lower=np.maximum.reduce([lo,r-np.asarray(limits.effective_velocity)*dt,
        r+(v-amax*dt)*dt,r+(v+(a-jmax*dt)*dt)*dt,r-negative*dt])
    upper=np.minimum.reduce([hi,r+np.asarray(limits.effective_velocity)*dt,
        r+(v+amax*dt)*dt,r+(v+(a+jmax*dt)*dt)*dt,r+positive*dt])
    return lower,upper

from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
from .tracking import limited

class StoppableTrackingReferencePlanner:
    def __init__(self,kinematics,joint_response,limits,config=None):
        self.kinematics,self.joint_response,self.limits=kinematics,joint_response,limits
        self.config=config or StoppableTrackingConfig()
        self.last_applied=None

    def observe_execution(self,receipt):
        # A prediction never becomes actual next-plan reference history.
        rows=[r for r in receipt.ticks if r['tick_completed']]
        if rows:
            r=rows[-1]
            self.last_applied=dict(time=r['clock_after'],q=np.asarray(r['reference_positions']),
                v=np.asarray(r['reference_velocities']),a=np.asarray(r['reference_accelerations']))

    def _history(self,state):
        if self.last_applied is not None:
            if abs(self.last_applied['time']-state['time'])>2e-5:
                raise ValueError('only the actual previous applied reference can seed tracking')
            return tuple(self.last_applied[k].copy() for k in ('q','v','a'))
        h=state.get('actuator_state',{})
        if h.get('reference_positions') is not None and h.get('reference_clock') is not None and abs(h['reference_clock']-state['time'])<=2e-5:
            return tuple(np.asarray(h[k],float).copy() for k in ('reference_positions','reference_velocities','reference_accelerations'))
        return np.asarray(state['q']).copy(),np.asarray(state['dq']).copy(),np.zeros(7)

    def plan(self,state,candidate,forecast,*,pregrasp):
        dt=.05;c=self.config;limits=self.limits
        q=np.asarray(state['q'],float);dq=np.asarray(state['dq'],float)
        r,v,a=self._history(state)
        lower,upper=reference_window(r,v,a,limits,margin=c.joint_limit_margin)
        if np.any(lower>upper+1e-10):
            return dict(action=None,reason='no_bounded_trajectory_reference')
        pose,velocity,angular=forecast.at('target',state['time'])
        local=np.asarray(candidate['object_tcp_position']).copy()
        if pregrasp:local+=np.asarray(candidate['approach_direction'])*candidate['pregrasp_offset']
        target=pose.apply(local)
        target_R=pose.rotation@Rotation.from_quat(candidate['object_tcp_quaternion']).as_matrix()
        target_v=velocity+np.cross(angular,pose.rotation@local)
        error=target-np.asarray(state['tcp_position'])
        rotation_error=Rotation.from_matrix(target_R@Rotation.from_quat(state['tcp_quaternion']).as_matrix().T).as_rotvec()
        desired_v=limited(c.feedforward_scale*target_v+c.position_gain*error,c.maximum_tcp_speed)
        actual_v=np.asarray(state['tcp_velocity'])
        desired_v=actual_v+limited(desired_v-actual_v,c.maximum_tcp_acceleration*dt)
        desired_w=limited(c.feedforward_scale*angular+c.rotation_gain*rotation_error,c.maximum_angular_speed)
        actual_w=np.asarray(state['tcp_angular_velocity'])
        desired_w=actual_w+limited(desired_w-actual_w,c.maximum_angular_acceleration*dt)

        # The frozen response is affine in qref, vref and aref. Expressing the
        # latter as exact discrete reference differences gives a 7x7 affine
        # map of the position reference, inverted under all reference limits.
        references=np.repeat(r[None],8,axis=0)
        references[1:]+=np.eye(7)*1e-4
        vref=(references-r)/dt;aref=(vref-v)/dt
        prediction=self.joint_response.predict_joint(np.repeat(q[None],8,axis=0),np.repeat(dq[None],8,axis=0),references,vref,aref)
        base=prediction['q'][0]
        A=(prediction['q'][1:]-base).T/1e-4
        J=self.kinematics.geometric_jacobian(q)
        weights=np.array([1.,1.,1.,c.rotation_length_scale,c.rotation_length_scale,c.rotation_length_scale])
        matrix=(J@A/dt)*weights[:,None]
        rhs=(np.r_[desired_v,desired_w]-J@(base-q)/dt)*weights
        # A small joint-space objective uses kinematic redundancy to stay away
        # from hard stops. It does not relax Cartesian capture or speed gates.
        center=.5*(np.asarray(limits.lower)+np.asarray(limits.upper))
        span=np.asarray(limits.upper)-np.asarray(limits.lower)
        posture_v=c.posture_gain*(center-q)/span
        posture_weight=c.posture_weight/(.15+np.minimum(q-np.asarray(limits.lower),np.asarray(limits.upper)-q))
        matrix=np.vstack([matrix,(A/dt)*posture_weight[:,None]])
        rhs=np.r_[rhs,posture_weight*(posture_v-(base-q)/dt)]
        matrix=np.vstack([matrix,c.regularization*np.eye(7)])
        rhs=np.r_[rhs,np.zeros(7)]
        lo,hi=lower-r,upper-r
        fixed=(hi-lo)<=1e-12
        delta=np.zeros(7);delta[fixed]=.5*(lo[fixed]+hi[fixed])
        free=~fixed
        if free.any():
            solve=lsq_linear(matrix[:,free],rhs-matrix[:,fixed]@delta[fixed],bounds=(lo[free],hi[free]),tol=1e-10,max_iter=100)
            if not solve.success:return dict(action=None,reason='bounded_response_inverse_failed')
            delta[free]=solve.x
        ref=r+delta
        ref_v=(ref-r)/dt;ref_a=(ref_v-v)/dt
        predicted=self.joint_response.predict_joint(q[None],dq[None],ref[None],ref_v[None],ref_a[None])
        predicted_pose=self.kinematics(predicted['q'][0])
        command_pose=self.kinematics(ref)
        return dict(action=dict(joint_goal=ref,tcp_position=command_pose[:3,3],
                tcp_quaternion=Rotation.from_matrix(command_pose[:3,:3]).as_quat(),joint_goal_feasible=True),
            reference=dict(q=ref,v=ref_v,a=ref_a),desired_twist=np.r_[desired_v,desired_w],
            target_position=target,target_velocity=target_v,position_error=error,rotation_error=rotation_error,
            predicted_q=predicted['q'][0],predicted_dq=predicted['dq'][0],predicted_TCP=predicted_pose,
            predicted_linear_velocity=(predicted_pose[:3,3]-state['tcp_position'])/dt,
            motor_interface='unchanged_native_position_servo',target_twist_used_in_reference=True,
            Native_object_state_used=False,dt=dt)

    def brake(self,state):
        r,v,a=self._history(state)
        lo,hi=reference_window(r,v,a,self.limits,margin=self.config.joint_limit_margin)
        if np.any(lo>hi+1e-10):raise ValueError('No bounded stoppable reference remains')
        ref=np.minimum(hi,np.maximum(lo,r))
        pose=self.kinematics(ref)
        return dict(joint_goal=ref,tcp_position=pose[:3,3],tcp_quaternion=Rotation.from_matrix(pose[:3,:3]).as_quat(),
            joint_goal_feasible=True,execution_semantics='sampled_joint_reference',braking_reference=True)
