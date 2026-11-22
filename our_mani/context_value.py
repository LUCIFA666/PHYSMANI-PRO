"""Controller-specific candidate success probabilities conditioned on context."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .contracts import Pose
from .success_prediction import SuccessProbability


class ContextCandidateSuccessValue:
    def __init__(self, checkpoint, dynamics, *, controller_sha256, clock, capture_budget=3.):
        self.value = SuccessProbability.load(checkpoint,controller_sha256=controller_sha256)
        if self.value.context_dim != 16 or self.value.metadata['context_encoder_sha256'] != dynamics.encoder_sha256:
            raise ValueError('v1 success model requires the same frozen 16D history encoder')
        self.dynamics, self.clock, self.capture_budget = dynamics, clock, capture_budget
        path = Path(checkpoint).with_name('value_support.json')
        self.support = json.loads(path.read_text())
        if self.support['value_checkpoint_sha256'] != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
            raise ValueError('success input support belongs to different weights')
        self.support_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.tree = cKDTree(np.asarray(self.support['normalized_fit_features']))

    def __call__(self, *, region, object_pose, object_velocity, object_angular_velocity,
                 robot_pose, robot_velocity, robot_angular_velocity):
        try:
            snapshot = self.dynamics.context_snapshot(region.object_id)
        except ValueError:
            return float('nan')
        age = self.clock()-snapshot['observed_until']
        if age < -1e-8 or age > .1:
            return float('nan')
        if Rotation.from_matrix(object_pose.rotation.copy()).magnitude() > .1 or np.linalg.norm(object_angular_velocity) > .1:
            return float('nan')
        orientation = Rotation.from_matrix(robot_pose.rotation@Pose.from_xyzw([0,0,0,0,1,0,0]).rotation.T).as_rotvec()
        local = np.r_[robot_pose.position-object_pose.position,robot_velocity-object_velocity,
                      object_velocity,object_angular_velocity,orientation,1.,0.,self.capture_budget]
        combined = np.r_[local,snapshot['values']]
        accepted = (np.isfinite(combined).all() and np.all(combined>=self.support['lower'])
                    and np.all(combined<=self.support['upper'])
                    and self.tree.query((combined-self.support['mean'])/self.support['scale'])[0]
                        <= self.support['maximum_nearest_fit_distance'])
        if not accepted:
            return float('nan')
        return float(self.value(local,context=snapshot['values'],context_encoder_sha256=snapshot['encoder_sha256']))
