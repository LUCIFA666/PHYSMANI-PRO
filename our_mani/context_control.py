"""Pipeline v1 local MPC follows context-conditioned latent forecasts."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .contracts import Pose
from .control import VisualMPC
from .local_capture import LocalCaptureController


class ContextVisualMPC(VisualMPC):
    name = 'context_simnorm_forecast_mpc_v1'

    def __init__(self, dynamics, **kwargs):
        super().__init__(**kwargs)
        self.dynamics = dynamics

    def command(self, robot, obj, local_grasp_pose):
        forecast = self.dynamics.predict([obj], self.horizon*self.dt, self.dt)
        desired = np.array([forecast.at(obj.object_id,obj.timestamp+(i+1)*self.dt)[0].apply(local_grasp_pose.position)
                            for i in range(self.horizon)])
        n, dt = self.horizon, self.dt
        lower = np.tril(np.ones((n,n)))
        difference = np.eye(n)-np.eye(n,k=-1)
        def states(flat):
            u = flat.reshape(n,3)
            du = difference@u
            du[0] -= robot.linear_velocity
            error = robot.pose.position+dt*lower@u-desired
            return u,du,error
        def objective(flat):
            _,du,error = states(flat)
            return self.position_weight*np.square(error).sum()+self.smoothness_weight*np.square(du).sum()
        def gradient(flat):
            _,du,error = states(flat)
            return (2*self.position_weight*dt*lower.T@error+2*self.smoothness_weight*difference.T@du).ravel()
        def constraint(flat):
            u,du,_ = states(flat)
            return np.r_[self.max_speed**2-np.square(u).sum(1),
                         (self.max_acceleration*dt)**2-np.square(du).sum(1)]
        def jacobian(flat):
            u,du,_ = states(flat)
            speed = np.einsum('ij,ik->ijk',np.eye(n),-2*u).reshape(n,3*n)
            acceleration = np.einsum('ij,ik->ijk',difference,-2*du).reshape(n,3*n)
            return np.vstack([speed,acceleration])
        seed = np.asarray(robot.linear_velocity).copy()
        seed *= min(1.,self.max_speed/max(np.linalg.norm(seed),1e-9))
        result = minimize(objective,np.tile(seed,(n,1)).ravel(),jac=gradient,method='SLSQP',
            bounds=[(-self.max_speed,self.max_speed)]*(3*n),
            constraints=[{'type':'ineq','fun':constraint,'jac':jacobian}],
            options={'maxiter':120,'ftol':1e-7})
        if not np.isfinite(result.fun) or np.min(constraint(result.x)) < -1e-5:
            raise RuntimeError(f'context MPC infeasible: {result.message}')
        velocity = result.x.reshape(n,3)[0]
        desired_rotation = obj.pose.rotation@local_grasp_pose.rotation
        angular = 4.*Rotation.from_matrix(desired_rotation@robot.pose.rotation.T).as_rotvec()+obj.angular_velocity
        angular *= min(1.,self.max_angular_speed/max(np.linalg.norm(angular),1e-9))
        pose = Pose(robot.pose.position+velocity*dt,Rotation.from_rotvec(angular*dt).as_matrix()@robot.pose.rotation)
        return pose,velocity,angular,{'controller':self.name,'cost':float(result.fun),'iterations':int(result.nit),
            'dynamics_context_encoder_sha256':self.dynamics.encoder_sha256,'forecast_ground_truth_used':False}


class ContextCaptureController(LocalCaptureController):
    name = 'context_simnorm_capture_and_visual_goal_v1'

    def __init__(self, goal_pose, config, dynamics):
        super().__init__(goal_pose,config)
        self.mpc = ContextVisualMPC(dynamics,dt=config.dt,max_speed=config.max_speed,
                                  max_acceleration=config.max_acceleration)


def controller_sha256(checkpoint, config, servo_lead=1.4):
    root = Path(__file__).resolve().parents[1]
    names = ('local_capture.py','control.py','benchmark_perception.py','goal_perception.py','pyrep_env.py',
             'context_control.py','context_dynamics.py')
    digest = hashlib.sha256(b''.join((root/'our_mani'/name).read_bytes() for name in names))
    digest.update(hashlib.sha256(Path(checkpoint).read_bytes()).digest())
    digest.update(json.dumps({'controller_config':asdict(config),'servo_lead':servo_lead},sort_keys=True).encode())
    return digest.hexdigest()
