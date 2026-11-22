"""World-frame product-of-exponentials FK from a Native reference snapshot.

No simulator or GPU imports, setters, collision calls or physical steps exist
here. Snapshot validity against the real scene remains a separate Native
canary requirement; passing the CPU algebra tests is not that qualification.
Spatial screw order is [v_x,v_y,v_z,w_x,w_y,w_z].
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np


SCHEMA = "local_mpc_p2_world_space_poe_20261103_v1"


def _array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"finite {name} with shape {shape} required")
    return result.copy()


def _transform(value, name):
    matrix = _array(value, (4, 4), name)
    if (not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-7, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-5, rtol=0)):
        raise ValueError(f"{name} must be a rigid SE3 transform")
    return matrix


def skew(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def twist_exp(screw, theta):
    """Unit revolute or unit prismatic screw exponential, with batched theta."""
    v, w = np.asarray(screw, dtype=float)[:3], np.asarray(screw, dtype=float)[3:]
    theta = np.asarray(theta, dtype=float)
    if not np.isfinite(theta).all():
        raise ValueError("finite joint displacements required")
    result = np.broadcast_to(np.eye(4), (*theta.shape, 4, 4)).copy()
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        result[..., :3, 3] = theta[..., None] * v
        return result
    if not np.isclose(norm, 1., atol=1e-6, rtol=0):
        raise ValueError("unit revolute axes required")
    wx = skew(w)
    a = theta * np.sinc(theta / np.pi)
    b = .5 * theta**2 * np.sinc(theta / (2*np.pi))**2
    c = np.where(np.abs(theta) < 1e-4,
                 theta**3 * (1/6 - theta**2/120 + theta**4/5040), theta - np.sin(theta))
    result[..., :3, :3] = np.eye(3) + a[..., None, None]*wx + b[..., None, None]*(wx@wx)
    result[..., :3, 3] = (theta[..., None]*v + b[..., None]*(wx@v)
                          + c[..., None]*(wx@wx@v))
    return result


class PoEKinematics:
    def __init__(self, snapshot):
        if snapshot.get("schema") != SCHEMA or snapshot.get("twist_order") != "v_omega":
            raise ValueError("explicit world-space v/omega PoE snapshot required")
        self.q_ref = np.asarray(snapshot["q_ref"], dtype=float)
        if self.q_ref.ndim != 1 or not len(self.q_ref) or not np.isfinite(self.q_ref).all():
            raise ValueError("finite nonempty reference joint vector required")
        self.q_ref = self.q_ref.copy()
        self.dof = len(self.q_ref)
        self.screws = _array(snapshot["space_screws_vw"], (self.dof, 6), "world screws")
        self.reference_tcp = _transform(snapshot["tcp_reference_matrix"], "reference world TCP")
        axes = np.linalg.norm(self.screws[:, 3:], axis=-1)
        if not np.all(np.isclose(axes, 1., atol=1e-6, rtol=0)):
            raise ValueError("this exported Panda chain requires unit revolute axes")

    def _joints(self, q):
        q = np.asarray(q, dtype=float)
        if q.ndim < 1 or q.shape[-1] != self.dof or not np.isfinite(q).all():
            raise ValueError("finite joints[...,dof] required")
        return q

    def forward(self, q):
        q = self._joints(q)
        value = np.broadcast_to(np.eye(4), (*q.shape[:-1], 4, 4)).copy()
        for i in range(self.dof):
            value = value @ twist_exp(self.screws[i], q[..., i] - self.q_ref[i])
        return value @ self.reference_tcp

    __call__ = forward

    def space_jacobian(self, q):
        """Spatial twist Jacobian [v_space, omega]; v_space is NOT TCP p_dot."""
        q = self._joints(q)
        value = np.broadcast_to(np.eye(4), (*q.shape[:-1], 4, 4)).copy()
        columns = []
        for i in range(self.dof):
            v, w = self.screws[i, :3], self.screws[i, 3:]
            world_w = (value[..., :3, :3] @ w[..., None])[..., 0]
            world_v = ((value[..., :3, :3] @ v[..., None])[..., 0]
                       + np.cross(value[..., :3, 3], world_w))
            columns.append(np.concatenate((world_v, world_w), axis=-1))
            value = value @ twist_exp(self.screws[i], q[..., i] - self.q_ref[i])
        return np.stack(columns, axis=-1)

    def geometric_jacobian(self, q):
        """World TCP [linear velocity p_dot, angular velocity] Jacobian."""
        jacobian = self.space_jacobian(q)
        tcp_position = self.forward(q)[..., :3, 3]
        angular = np.swapaxes(jacobian[..., 3:, :], -1, -2)
        offset = np.cross(angular, tcp_position[..., None, :])
        result = jacobian.copy()
        result[..., :3, :] += np.swapaxes(offset, -1, -2)
        return result


def _same_clock(left, right):
    if isinstance(left, dict):
        return set(left) == set(right) and all(_same_clock(left[key], right[key]) for key in left)
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def export_native_kinematic_snapshot(arm, *, clock=None, expected_dof=7):
    """Read current Native joint axes/origins and exact arm-tip transform.

    Uses only get_joint_positions/velocities/targets, Joint.get_matrix/type,
    arm.get_tip().get_matrix and IK-target.get_pose. In particular this does NOT
    call arm.get_jacobian, simCheckIkGroup, FK probes or any setter. A caller
    should supply a clock callback returning physics/task/ledger clocks; no
    clock-invariance qualification is claimed when no callback is supplied.
    Joint matrices and tip matrix are absolute world transforms by PyRep API.
    """
    before_clock = deepcopy(clock()) if clock is not None else None
    joints = tuple(arm.joints)
    count = len(joints)
    if expected_dof is not None and count != expected_dof:
        raise ValueError("native joint count differs from the declared Panda chain")
    q_ref = _array(arm.get_joint_positions(), (count,), "reference joints")
    dq = _array(arm.get_joint_velocities(), (count,), "native joint velocities")
    servo = _array(arm.get_joint_target_positions(), (count,), "native servo targets")
    ik = _array(arm._ik_target.get_pose(), (7,), "native IK target xyzw")
    axes, origins, types = [], [], []
    for joint in joints:
        kind = joint.get_joint_type()
        name = getattr(kind, "name", str(kind))
        if str(name).lower() != "revolute":
            raise ValueError("only Native revolute joints are supported by this Panda exporter")
        matrix = _transform(joint.get_matrix(), "native world joint matrix")
        axis = matrix[:3, 2]
        axes.append(axis / np.linalg.norm(axis))
        origins.append(matrix[:3, 3])
        types.append(str(name))
    axes, origins = np.asarray(axes), np.asarray(origins)
    tcp = _transform(arm.get_tip().get_matrix(), "native world TCP matrix")
    if (not np.array_equal(np.asarray(arm.get_joint_positions()), q_ref)
            or not np.array_equal(np.asarray(arm.get_joint_velocities()), dq)
            or not np.array_equal(np.asarray(arm.get_joint_target_positions()), servo)
            or not np.array_equal(np.asarray(arm._ik_target.get_pose()), ik)):
        raise RuntimeError("Native joint/servo/velocity/IK state changed during kinematic export")
    after_clock = deepcopy(clock()) if clock is not None else None
    if clock is not None and not _same_clock(before_clock, after_clock):
        raise RuntimeError("kinematic getter export advanced or raced Native clocks")
    return dict(schema=SCHEMA, q_ref=q_ref.tolist(),
                space_screws_vw=np.concatenate((-np.cross(axes, origins), axes), axis=-1).tolist(),
                tcp_reference_matrix=tcp.tolist(), joint_origins_world=origins.tolist(),
                joint_axes_world=axes.tolist(), joint_types=types, twist_order="v_omega",
                frame="world", position_units="metres", joint_units="radians",
                tcp_frame="exact_native_arm_get_tip_not_assumed_standard_URDF_tool",
                getters_only=True, setters_called=False, physical_step_called=False,
                clock_callback_supplied=clock is not None, clocks_before=before_clock, clocks_after=after_clock,
                native_state_preserved=True, Native_FK_validation_passed=False,
                validation_scope="export_only_requires_separate_restoring_Native_FK_probe_canary")
