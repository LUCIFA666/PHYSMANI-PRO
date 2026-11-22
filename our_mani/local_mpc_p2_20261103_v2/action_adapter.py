"""Pure NumPy batch pose-to-joint targets for virtual P2 rollouts.

This uses the exported world PoE chain only. It never invokes Native IK,
collision checks, setters or stepping. Convergence certifies algebraic pose
agreement and joint position bounds, not collision freedom or reachability in
100 ms. Native execution still needs the independent first-action safety gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .kinematics import PoEKinematics
from ..local_mpc_20261102.batch_mpc import norm, qlog, qmul, qinv, unit


def matrix_quaternion(matrix):
    """Rigid rotation [...,3,3] to xyzw; stable near 180 degrees."""
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape[-2:] != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("finite rotation[...,3,3] required")
    shape = rotation.shape[:-2]
    r = rotation.reshape(-1, 3, 3)
    # Largest diagonal branch avoids dividing by a vanishing trace branch.
    squared = np.stack((1+r[:,0,0]-r[:,1,1]-r[:,2,2],
                        1-r[:,0,0]+r[:,1,1]-r[:,2,2],
                        1-r[:,0,0]-r[:,1,1]+r[:,2,2],
                        1+r[:,0,0]+r[:,1,1]+r[:,2,2]), axis=-1)
    index = np.argmax(squared, axis=-1)
    result = np.empty((len(r), 4), dtype=np.float64)
    for branch in range(4):
        use = index == branch
        selected = r[use]
        if not len(selected):
            continue
        value = .5*np.sqrt(np.maximum(squared[use, branch], 0.))
        denominator = np.maximum(4*value, 1e-30)
        if branch == 0:
            parts = (value, (selected[:,0,1]+selected[:,1,0])/denominator,
                     (selected[:,0,2]+selected[:,2,0])/denominator,
                     (selected[:,2,1]-selected[:,1,2])/denominator)
        elif branch == 1:
            parts = ((selected[:,0,1]+selected[:,1,0])/denominator, value,
                     (selected[:,1,2]+selected[:,2,1])/denominator,
                     (selected[:,0,2]-selected[:,2,0])/denominator)
        elif branch == 2:
            parts = ((selected[:,0,2]+selected[:,2,0])/denominator,
                     (selected[:,1,2]+selected[:,2,1])/denominator, value,
                     (selected[:,1,0]-selected[:,0,1])/denominator)
        else:
            parts = ((selected[:,2,1]-selected[:,1,2])/denominator,
                     (selected[:,0,2]-selected[:,2,0])/denominator,
                     (selected[:,1,0]-selected[:,0,1])/denominator, value)
        result[use] = np.stack(parts, axis=-1)
    result = unit(result).reshape(*shape, 4)
    return np.where(result[..., 3, None] < 0, -result, result)


@dataclass(frozen=True)
class DLSConfig:
    iterations: int = 12
    damping: float = .02
    position_tolerance: float = .01
    angle_tolerance: float = .08
    rotation_weight: float = .25
    max_joint_step: float = .25

    def __post_init__(self):
        if type(self.iterations) is not int or self.iterations <= 0:
            raise ValueError("positive integer DLS iteration budget required")
        for name in ("damping", "position_tolerance", "angle_tolerance",
                     "rotation_weight", "max_joint_step"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError("finite positive " + name + " required")


class BatchPoseJointAdapter:
    """Damped least squares with fixed bounded search, independently per row.

    Input seed is x.q, the current measured/predicted actual joint state.
    Reference h.q_ref is never substituted for actual joint feedback. Invalid
    seeds fail closed; nonconverged rows return finite bounded joint targets
    with joint_goal_feasible=False. The execution-model wrapper must retain
    that mask as infinite cost. This module supplies no contact inference.
    """
    def __init__(self, kinematics, limits, config=None):
        if not isinstance(kinematics, PoEKinematics):
            raise TypeError("explicit PoEKinematics required")
        self.kinematics = kinematics
        self.lower = np.asarray(limits.lower, dtype=np.float64).copy()
        self.upper = np.asarray(limits.upper, dtype=np.float64).copy()
        if (self.lower.shape != (kinematics.dof,) or self.upper.shape != self.lower.shape
                or not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all()
                or np.any(self.lower >= self.upper)):
            raise ValueError("finite ordered joint bounds match kinematic chain")
        self.config = config if isinstance(config, DLSConfig) else DLSConfig(**(config or {}))
        self.metadata = dict(method="batched_world_PoE_geometric_Jacobian_DLS",
                             Native_IK_called=False, collision_qualified=False,
                             physical_reachability_qualified=False,
                             seed="current_actual_or_virtual_predicted_q_not_qref")

    def _error(self, q, position, quaternion):
        transform = self.kinematics.forward(q)
        dp = position-transform[..., :3, 3]
        dr = qlog(qmul(quaternion, qinv(matrix_quaternion(transform[..., :3, :3]))))
        error = np.concatenate((dp, self.config.rotation_weight*dr), axis=-1)
        return error, norm(dp), norm(dr)

    def solve(self, q, position, quaternion):
        q = np.asarray(q, dtype=np.float64)
        position = np.asarray(position, dtype=np.float64)
        quaternion = np.asarray(quaternion, dtype=np.float64)
        if (q.ndim != 2 or q.shape[1] != self.kinematics.dof
                or position.shape != (len(q),3) or quaternion.shape != (len(q),4)
                or not np.isfinite(q).all() or not np.isfinite(position).all()
                or not np.isfinite(quaternion).all() or np.any(norm(quaternion) < 1e-12)):
            raise ValueError("aligned finite q[N,J] and world pose[N,3/4] required")
        quaternion = unit(quaternion)
        seed_valid = np.all((q >= self.lower-1e-8)&(q <= self.upper+1e-8), axis=-1)
        target = np.clip(q, self.lower, self.upper)
        iterations = np.zeros(len(q), dtype=np.int64)
        config = self.config
        for iteration in range(config.iterations):
            error, distance, angle = self._error(target, position, quaternion)
            active = seed_valid&((distance > config.position_tolerance)|(angle > config.angle_tolerance))
            if not np.any(active):
                break
            rows = np.flatnonzero(active)
            joints = target[rows]
            jacobian = self.kinematics.geometric_jacobian(joints)
            jacobian[:,3:,:] *= config.rotation_weight
            regularized = jacobian@np.swapaxes(jacobian,-1,-2)+config.damping**2*np.eye(6)
            delta = (np.swapaxes(jacobian,-1,-2)@np.linalg.solve(regularized,error[rows,...,None]))[...,0]
            delta *= np.minimum(1.,config.max_joint_step/np.maximum(norm(delta),1e-30))[:,None]
            best = joints.copy()
            best_cost = np.sum(error[rows]**2,axis=-1)
            # Bounded line search is optimizer projection, never a fabricated
            # measured state. Targets still need the independent Native gate.
            for fraction in (1.,.5,.25):
                proposal = np.clip(joints+fraction*delta,self.lower,self.upper)
                proposed_error,_,_ = self._error(proposal,position[rows],quaternion[rows])
                cost = np.sum(proposed_error**2,axis=-1)
                better = cost < best_cost
                best[better],best_cost[better] = proposal[better],cost[better]
            target[rows] = best
            iterations[rows] += 1
        _,distance,angle = self._error(target,position,quaternion)
        feasible = seed_valid&(distance <= config.position_tolerance)&(angle <= config.angle_tolerance)
        return dict(joint_goal=target,joint_goal_feasible=feasible,
                    ik_position_error=distance,ik_angle_error=angle,
                    ik_iterations=iterations,ik_seed_valid=seed_valid)

    def __call__(self, x, u, h, context):
        if "q" not in x:
            raise ValueError("batch pose-to-joint adapter requires actual/predicted q")
        return self.solve(x["q"],u["tcp_position"],u["tcp_quaternion"])
