"""Forecast contract for the frozen-NGFF driver/support adapter and its ablation."""
from collections import deque
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import Forecast
from .dynamics_adapter import LoadedDriverAdapter


class CausalDriverDynamics:
    maximum_validated_horizon = .6

    def __init__(self, checkpoint, frozen_ngff=None):
        self.adapter = LoadedDriverAdapter(checkpoint)
        report_path = Path(checkpoint).parent/'report.json'
        if not report_path.exists():
            raise ValueError('driver adapter validation report required for runtime use')
        validation = json.loads(report_path.read_text())
        self.maximum_validated_horizon = validation.get('maximum_forecast_horizon_seconds', .6)
        self.frozen_ngff = frozen_ngff
        if self.adapter.mode == 'ngff_driver_adapter' and frozen_ngff is None:
            raise ValueError('NGFF driver adapter requires the frozen NGFF model')
        self.name = self.adapter.mode
        self.checkpoint_sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        self._history, self._lock = {}, threading.RLock()

    def reset(self):
        with self._lock:
            self._history.clear()

    def observe(self, objects):
        with self._lock:
            for obj in objects:
                history = self._history.setdefault(obj.object_id, deque(maxlen=16))
                if history and obj.timestamp < history[-1][0]:
                    raise ValueError('non-monotonic driver history; reset at episode boundary')
                if history and obj.timestamp == history[-1][0]:
                    continue
                history.append((obj.timestamp, obj.pose.position.copy()))

    def predict(self, objects, horizon, dt):
        start = time.perf_counter()
        if horizon > self.maximum_validated_horizon+1e-8:
            raise ValueError(f'driver adapter validated horizon is {self.maximum_validated_horizon} seconds')
        if len(objects) != 1:
            raise ValueError('moving-pick adapter was trained for one driven box')
        obj = objects[0]
        if Rotation.from_matrix(obj.pose.rotation.copy()).magnitude() > .1 or np.linalg.norm(obj.angular_velocity) > .1:
            raise ValueError('rotating-object transfer is outside the driver adapter training domain')
        with self._lock:
            history = [entry for entry in self._history.get(obj.object_id, ()) if entry[0] <= obj.timestamp][-5:]
        if len(history) != 5 or history[-1][0] != obj.timestamp or history[-1][0]-history[0][0] > .35:
            raise ValueError('driver adapter requires five fresh causal observations')
        offsets = np.arange(int(round(horizon/dt))+1)*dt
        if offsets[-1] > self.maximum_validated_horizon+1e-8:
            raise ValueError('forecast grid exceeds validated driver horizon')
        times, positions = np.array([r[0] for r in history]), np.array([r[1] for r in history])
        base, base_metadata = None, {}
        if self.adapter.mode == 'ngff_driver_adapter':
            frozen = self.frozen_ngff.predict(objects, horizon, dt)
            base, base_metadata = frozen.positions[:, 0], frozen.metadata
        predicted = self.adapter.predict(times, positions, obj.linear_velocity, offsets, base)
        velocities = np.gradient(predicted, dt, axis=0)
        velocities[0] = obj.linear_velocity
        rotations = np.broadcast_to(obj.pose.rotation, (len(offsets), 1, 3, 3)).copy()
        points = obj.pose.inverse_apply(obj.points_world)
        clouds = (points@obj.pose.rotation.T)[None, None]+predicted[:, None, None]
        return Forecast((obj.object_id,), obj.timestamp+offsets, predicted[:, None], rotations,
                        velocities[:, None], np.zeros((len(offsets), 1, 3)), clouds,
                        {'method': self.name, 'latency_seconds': time.perf_counter()-start,
                         'checkpoint_sha256': self.checkpoint_sha256, 'ngff_weights_updated': False,
                         'rotation_model': 'declared world-aligned box prior', 'base': base_metadata,
                         'validated_horizon_seconds': self.maximum_validated_horizon,
                         'causal_history_frames': 5})
