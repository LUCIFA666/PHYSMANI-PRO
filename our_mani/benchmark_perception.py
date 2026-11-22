"""Causal RGB-D red-box perception for the two released moving-pick tasks.

This is a task-specific, known-shape development backend. No simulator object
pose/mask, expert label, future image, or world-model prediction is an input.
The world-aligned box frame and dimensions are explicit task priors, not a
general pose estimator for arbitrary objects.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

from .contracts import ObjectState, Pose


@dataclass(frozen=True)
class BoxPrior:
    dimensions: tuple[float, float, float] = (.05, .05, .05)
    floor_z: float = .75
    workspace_min: tuple[float, float, float] = (-.05, -.6, .74)
    workspace_max: tuple[float, float, float] = (.5, .6, 1.3)
    minimum_red: int = 170
    red_ratio: float = 3.5


def box_surface(pose, dimensions, points=256):
    """Deterministic complete box surface, using a declared known-shape prior."""
    count = int(np.ceil(np.sqrt(points/6)))
    axis = np.linspace(-.5, .5, count)
    u, v = np.meshgrid(axis, axis)
    uv = np.c_[u.ravel(), v.ravel()]
    result = []
    for normal in range(3):
        tangents = [i for i in range(3) if i != normal]
        for sign in (-.5, .5):
            face = np.zeros((len(uv), 3))
            face[:, normal] = sign
            face[:, tangents] = uv
            result.append(face)
    geometry = np.concatenate(result)*np.asarray(dimensions)
    indices = np.linspace(0, len(geometry)-1, points, dtype=int)
    return pose.apply(geometry[indices])


class RedBoxTracker:
    def __init__(self, prior=None, *, object_id='target', history=5, complete_geometry=True, points=256):
        self.prior = prior or BoxPrior()
        self.object_id = object_id
        self.history = deque(maxlen=history)
        self.complete_geometry, self.points = complete_geometry, points
        self.last_velocity = np.zeros(3)

    def observe(self, observation):
        p = self.prior
        clouds, colors, views = [], [], []
        for camera, rgb in observation['observation/images'].items():
            image = np.asarray(rgb, dtype=float)
            mask = (image[..., 0] >= p.minimum_red) & (image[..., 0] > p.red_ratio*image[..., 1]) & (image[..., 0] > p.red_ratio*image[..., 2])
            camera_points = observation['observation/calibrations'][camera].backproject(observation['observation/depths'][camera], mask)
            valid_depth = np.isfinite(observation['observation/depths'][camera]) & (observation['observation/depths'][camera] > 0)
            camera_colors = image[mask & valid_depth]
            inside = ((camera_points >= p.workspace_min) & (camera_points <= p.workspace_max)).all(1)
            clouds.append(camera_points[inside])
            colors.append(camera_colors[inside, 0])
            views.extend([camera]*int(inside.sum()))
        cloud = np.concatenate(clouds)
        red = np.concatenate(colors)
        views = np.asarray(views)
        if len(cloud) < 12:
            raise ValueError('red target not sufficiently visible')
        # Limit association cost before building connected components.
        if len(cloud) > 5000:
            indices = np.linspace(0, len(cloud)-1, 5000, dtype=int)
            cloud, red, views = cloud[indices], red[indices], views[indices]
        graph = cKDTree(cloud).sparse_distance_matrix(cKDTree(cloud), .028, output_type='coo_matrix')
        count, labels = connected_components(graph, directed=False)
        candidates = []
        for label in range(count):
            chosen = labels == label
            if chosen.sum() < 12:
                continue
            points = cloud[chosen]
            center = (np.quantile(points, .01, axis=0)+np.quantile(points, .99, axis=0))/2
            # The top surface is visible in all four released camera views.
            center[2] = np.quantile(points[:, 2], .98)-p.dimensions[2]/2
            # The released scene also has a red goal marker around z=1.0.
            # Initialize the moving target on the known table, then associate
            # causally. Do not follow the larger stationary elevated marker.
            if not self.history and abs(center[2]-(p.floor_z+p.dimensions[2]/2)) > .08:
                continue
            prior_penalty = 0.
            if self.history:
                dt = observation['timestamp']-self.history[-1][0]
                if dt <= 0:
                    raise ValueError('non-monotonic target frame')
                predicted = self.history[-1][1]+self.last_velocity*dt
                if np.linalg.norm(center-predicted) > .045+1.0*dt:
                    continue
                prior_penalty = 80*np.linalg.norm(center-predicted)
            # Maroon distractors can contribute many bright specular pixels.
            # Their lower mean red reflectance and oversized fused geometry
            # must outweigh a raw pixel-count advantage.
            extent = np.quantile(points, .99, axis=0)-np.quantile(points, .01, axis=0)
            oversize = np.maximum(extent/np.asarray(p.dimensions)-1.3, 0.).sum()
            score = .5*np.log(chosen.sum())+8.*float(red[chosen].mean())/255-2.*oversize-prior_penalty
            candidates.append((score, chosen, center))
        if not candidates:
            raise ValueError('target geometry association failed')
        _, chosen, center = max(candidates, key=lambda x: x[0])
        measured = cloud[chosen]
        visible_views = len(set(views[chosen]))
        timestamp = float(observation['timestamp'])
        if self.history and timestamp-self.history[-1][0] > .25:
            self.history.clear()
        samples = list(self.history)+[(timestamp, center)]
        velocity = np.zeros(3)
        if len(samples) > 1:
            times = np.array([s[0] for s in samples])
            times -= times.mean()
            positions = np.array([s[1] for s in samples])
            velocity = (times[:, None]*(positions-positions.mean(0))).sum(0)/(times@times)
        self.history.append((timestamp, center.copy()))
        self.last_velocity = velocity.copy()
        pose = Pose(center)
        geometry = box_surface(pose, p.dimensions, self.points) if self.complete_geometry else measured
        confidence = min(1., visible_views/3)*min(1., len(measured)/60)
        state = ObjectState(self.object_id, timestamp, pose, geometry, velocity, np.zeros(3), confidence)
        return state, {'backend': 'rgbd_red_known_axis_aligned_box', 'measured_points': len(measured),
                       'visible_views': visible_views, 'geometry_completion': self.complete_geometry,
                       'simulator_pose_used': False, 'future_observations_used': False,
                       'known_dimensions_m': list(p.dimensions)}
