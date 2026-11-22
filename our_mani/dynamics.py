"""Forecast adapters. NGFF's training coordinates are never silently treated as SI."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import Forecast, array


def check_states(states):
    if not states or len({s.object_id for s in states}) != len(states):
        raise ValueError("missing/duplicate object states")
    timestamp = states[0].timestamp
    if any(abs(s.timestamp-timestamp) > 1e-6 for s in states):
        raise ValueError("dynamics inputs must share a timestamp")
    if any(not s.visible for s in states):
        raise ValueError("cannot predict an unobserved target as a fresh observation")
    return timestamp


class ConstantVelocityDynamics:
    name = "constant_velocity"

    def predict(self, states, horizon=0.6, dt=0.05):
        timestamp = check_states(states)
        if dt <= 0 or horizon < dt:
            raise ValueError("invalid horizon")
        offsets = np.arange(int(np.ceil(horizon/dt))+1)*dt
        positions = np.stack([s.pose.position + offsets[:, None]*s.linear_velocity for s in states], axis=1)
        rotations = np.stack([Rotation.from_rotvec(offsets[:, None]*s.angular_velocity).as_matrix() @ s.pose.rotation for s in states], axis=1)
        velocity = np.broadcast_to([s.linear_velocity for s in states], positions.shape).copy()
        angular = np.broadcast_to([s.angular_velocity for s in states], positions.shape).copy()
        return Forecast(tuple(s.object_id for s in states), timestamp+offsets, positions, rotations, velocity, angular,
                        metadata={"model": self.name, "causal": True})


@dataclass(frozen=True)
class NGFFFrame:
    """Explicit positive scale/offset mapping; z must remain the gravity axis.

    model_xyz = (world_xyz - world_origin) / metres_per_model_unit.
    model_time = seconds / seconds_per_model_unit.
    No undocumented object-size normalization or gravity cancellation is applied.
    """
    metres_per_model_unit: float
    seconds_per_model_unit: float
    world_origin: np.ndarray
    boundary_model: np.ndarray
    label: str

    def __post_init__(self):
        if self.metres_per_model_unit <= 0 or self.seconds_per_model_unit <= 0 or not self.label:
            raise ValueError("NGFF requires an explicit coordinate calibration")
        object.__setattr__(self, "world_origin", array(self.world_origin, (3,)))
        object.__setattr__(self, "boundary_model", array(self.boundary_model, (6,)))
        if np.any(self.boundary_model[[0, 2, 4]] <= self.boundary_model[[1, 3, 5]]):
            raise ValueError("boundary order must be xmax,xmin,ymax,ymin,zmax,zmin")


class NGFFDynamics:
    name = "ngff_frozen"

    def __init__(self, checkpoint, frame: NGFFFrame, *, device="cuda", keypoints=None):
        import torch
        from dynamic_models import NGFFobj
        self.torch, self.device, self.frame = torch, torch.device(device), frame
        self.checkpoint = Path(checkpoint).resolve()
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        saved = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        c = saved["config"]
        self.keypoints = keypoints or c["keypoints_per_object"]
        self.model = NGFFobj(input_dim=3, hidden_dim=c["hidden_dim"], output_dim=3,
                             num_layers=c["num_layers"], num_keypoints=self.keypoints, k=8,
                             mass=0.1, dt=c["sampled_dt"], ode_method="euler",
                             step_size=c["ode_step_size"], threshold=0.05).to(self.device).eval()
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.model.requires_grad_(False)
        self.model.boundary = torch.tensor(frame.boundary_model.copy(), dtype=torch.float32, device=self.device)
        self.trained_update = saved["training_state"]["updates"]

    def _sample(self, points):
        if len(points) >= self.keypoints:
            return points[np.linspace(0, len(points)-1, self.keypoints, dtype=int)]
        # Explicit replication; point count is recorded in metadata, not hidden.
        return points[np.arange(self.keypoints) % len(points)]

    def predict(self, states, horizon=0.6, dt=0.05):
        timestamp = check_states(states)
        if dt <= 0 or horizon < dt:
            raise ValueError("invalid horizon")
        torch, f = self.torch, self.frame
        count = int(np.ceil(horizon/dt))
        sampled = np.stack([self._sample(s.points_world) for s in states])
        centers = sampled.mean(1)
        tensor = lambda x: torch.as_tensor(np.asarray(x).copy(), device=self.device, dtype=torch.float32).unsqueeze(0)
        points = tensor((sampled-f.world_origin)/f.metres_per_model_unit)
        com = tensor((centers-f.world_origin)/f.metres_per_model_unit)
        # Tracker pose origins need not equal sampled visible-cloud centroids.
        # Convert the rigid velocity at the origin to the centroid's velocity.
        centroid_velocity = [s.linear_velocity+np.cross(s.angular_velocity, c-s.pose.position)
                             for s, c in zip(states, centers)]
        velocity = tensor(np.asarray(centroid_velocity)*f.seconds_per_model_unit/f.metres_per_model_unit)
        zeros = torch.zeros_like(com)
        angular = tensor([s.angular_velocity*f.seconds_per_model_unit for s in states])
        padding = torch.ones(points.shape[:-1], device=self.device)
        knn = torch.zeros(*points.shape[:-1], 8, device=self.device, dtype=torch.long)
        self.model.dt = dt/f.seconds_per_model_unit
        self.model.step_size = min(0.02, self.model.dt)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.no_grad():
            # Initial geometry is already rotated in world coordinates. Zero
            # Euler angles prevent the original NGFF API from rotating it twice.
            local, com_t, angles = self.model(points, com, velocity, zeros, angular, padding, knn, pred_len=count)
            from utils.transformation_utils import euler_xyz_to_matrix
            delta_r = euler_xyz_to_matrix(angles)[0].cpu().numpy()
            local = local[0].cpu().numpy()
            com_t = com_t[0].cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter()-start
        world_points = (local @ np.swapaxes(delta_r, -1, -2) + com_t[:, :, None])*f.metres_per_model_unit + f.world_origin
        rotations = delta_r @ np.stack([s.pose.rotation for s in states])
        # Preserve an object frame origin distinct from a sampled cloud centroid.
        origin_offsets = np.stack([s.pose.position-c for s, c in zip(states, centers)])
        positions = com_t*f.metres_per_model_unit + f.world_origin + np.einsum("toij,oj->toi", delta_r, origin_offsets)
        velocities = np.gradient(positions, dt, axis=0)
        angular_v = np.empty_like(velocities)
        for o in range(len(states)):
            increments = rotations[1:, o] @ rotations[:-1, o].transpose(0, 2, 1)
            angular_v[1:, o] = Rotation.from_matrix(increments).as_rotvec()/dt
            angular_v[0, o] = states[o].angular_velocity
            velocities[0, o] = states[o].linear_velocity
        if not np.allclose(world_points[0], sampled, atol=3e-6):
            raise RuntimeError("NGFF coordinate round trip failed")
        return Forecast(tuple(s.object_id for s in states), timestamp+np.arange(count+1)*dt,
                        positions, rotations, velocities, angular_v, world_points,
                        {"model": self.name, "checkpoint_sha256": self.checkpoint_sha256,
                         "trained_update": self.trained_update, "frame": f.label,
                         "keypoints_per_object": self.keypoints, "latency_seconds": elapsed,
                         "gravity_world_m_s2": float(self.model.gravity.item())*f.metres_per_model_unit/f.seconds_per_model_unit**2,
                         "domain_adapted_to_physmani": False, "causal": True})
