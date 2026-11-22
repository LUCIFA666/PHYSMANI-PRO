"""RGB-D localization of the released moving-pick task's visible red goal.

The marker's rendered sphere is displaced from the task's goal frame. That
fixed offset is calibrated against TRAIN expert lift labels, never simulator
poses at inference. This is a declared task-specific visual backend.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

from .contracts import Pose


class RedGoalMarker:
    def __init__(self, marker_to_goal=(0., 0., .015)):
        self.offset = np.asarray(marker_to_goal, dtype=float)

    def observe(self, observation):
        clouds = []
        for camera, rgb in observation['observation/images'].items():
            image = np.asarray(rgb, dtype=float)
            mask = (image[..., 0] > 170) & (image[..., 0] > 3.5*image[..., 1]) & (image[..., 0] > 3.5*image[..., 2])
            cloud = observation['observation/calibrations'][camera].backproject(observation['observation/depths'][camera], mask)
            inside = ((cloud >= [-.05, -.6, .91]) & (cloud <= [.5, .6, 1.15])).all(1)
            clouds.append(cloud[inside])
        cloud = np.concatenate(clouds)
        if len(cloud) < 25:
            raise ValueError('visible goal marker unavailable')
        if len(cloud) > 4000:
            cloud = cloud[np.linspace(0, len(cloud)-1, 4000, dtype=int)]
        tree = cKDTree(cloud)
        count, labels = connected_components(tree.sparse_distance_matrix(tree, .025, output_type='coo_matrix'), directed=False)
        choices = []
        for label in range(count):
            points = cloud[labels == label]
            if len(points) < 25:
                continue
            initial = np.r_[np.median(points, axis=0), .038]
            fit = least_squares(lambda z: np.linalg.norm(points-z[:3], axis=1)-z[3], initial,
                                bounds=([-.1, -.65, .9, .025], [.55, .65, 1.15, .055]),
                                loss='soft_l1', f_scale=.002, max_nfev=40)
            residual = float(np.median(np.abs(fit.fun)))
            if fit.success and residual < .006:
                choices.append((len(points)/(1+residual*1000), fit.x, residual))
        if not choices:
            raise ValueError('goal marker sphere fit failed')
        _, fit, residual = max(choices, key=lambda row: row[0])
        return Pose.from_xyzw(np.r_[fit[:3]+self.offset, [0., 1., 0., 0.]]), {
            'backend': 'rgbd_red_goal_sphere_train_calibrated_frame',
            'sphere_radius_m': float(fit[3]), 'median_fit_residual_m': residual,
            'marker_to_goal_m': self.offset.tolist(), 'simulator_pose_used': False}
