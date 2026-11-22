"""Measure public perception/forecast at each real 50ms tick, in shadow.

This adapter does not replace the truth-state diagnostic controller. Public
predictions and privileged evaluation labels have separate files and paths.
No inferred intermediate frame, future observation, held flag or task success
is passed to the public bridge. Native camera calls stay on the owner thread.
"""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np

from .public_motion_bridge import PublicMotionBridge, PublicRobotSensors, PUBLIC_OBSERVATION_KEYS
from ..local_mpc_20261102.records import append, write_new, quantiles


def sanitized_robot(state):
    """Whitelist robot sensors before invoking any public perception code."""
    return PublicRobotSensors(timestamp=float(state['time']),
        tcp_position=state['tcp_position'], tcp_quaternion=state['tcp_quaternion'],
        q=state['q'], dq=state['dq'], finger_open_amounts=state['finger_open_amounts'])


class PublicShadowBackend:
    """Delegate actual execution; run one public observer after each tick.

    ``bridge`` is injectable for CPU contract tests. The tick method never adds
    a physical tick. A strict 50ms observer period is verified by the bridge.
    Records include an initial real frame at t=0, followed by actual tick ends.
    """
    def __init__(self, backend, output, *, bridge=None, recorder=None, save_frames=True,
                 controller_uses_public_forecast=False, observer_role=None, attachment_estimator_used=False):
        self.backend = backend
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.bridge = bridge if bridge is not None else PublicMotionBridge()
        self.recorder = recorder
        self.save_frames = bool(save_frames)
        if type(controller_uses_public_forecast) is not bool:
            raise ValueError('explicit public controller-use scope required')
        self.controller_uses_public_forecast = controller_uses_public_forecast
        self.observer_role = observer_role or ('pregrasp' if controller_uses_public_forecast else 'shadow')
        if (self.observer_role not in ('shadow','pregrasp','grasp_prototype')
                or (self.observer_role=='shadow')==controller_uses_public_forecast
                or type(attachment_estimator_used) is not bool):
            raise ValueError('explicit coherent observer and attachment-estimator scope required')
        self.attachment_estimator_used = attachment_estimator_used
        self.samples = 0
        self.times = []
        self._cached_observation = None
        self._cached_time = None
        self.latest = None
        write_new(self.output / 'scope.json', dict(
            mode=('public_perception_GRU_'+self.observer_role+'_control_diagnostic' if controller_uses_public_forecast
                  else 'public_perception_GRU_shadow_only_truth_controller_unchanged'),
            controller_uses_public_forecast=controller_uses_public_forecast, formal_TEST_used=False,
            public_inputs=['current_RGBD', 'calibration', 'timestamp', 'robot_TCP_q_dq_two_fingers'],
            object_GT_held_success_allowed_in_public_bridge=False,
            labels_file='evaluation_labels.jsonl', actual_sensor_dt=.05,
            orientation_observed=False, public_attachment_estimator_used=self.attachment_estimator_used))

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def sample(self, state=None):
        before = float(self.backend.clock())
        if self._cached_time is not None and abs(before - self._cached_time) < 1e-9:
            raise ValueError('one public observer sample per unique real timestamp')
        started = time.perf_counter()
        state = self.backend.read_state() if state is None else state
        robot = sanitized_robot(state)
        raw = self.backend.observe_rgbd()
        observation = {key: raw[key] for key in PUBLIC_OBSERVATION_KEYS}
        if abs(float(observation['timestamp']) - before) > 1e-8:
            raise ValueError('public RGBD timestamp differs from actual counted time')
        close_requested = float(state['requested_opening']) < .5
        result = self.bridge.observe(observation, robot, close_requested=close_requested)
        self._cached_observation, self._cached_time = raw, before
        relative = None
        if self.save_frames:
            directory = self.output / 'rgbd'
            directory.mkdir(exist_ok=True)
            path = directory / f'{self.samples:04d}.npz'
            arrays = {'timestamp': np.asarray(before)}
            for kind in ('images', 'depths'):
                arrays.update({kind+'_'+name: np.asarray(value) for name,value in observation['observation/'+kind].items()})
            if self.recorder is None:
                with path.open('xb') as stream:
                    np.savez_compressed(stream, **arrays)
            else:
                self.recorder.submit(path, arrays)
            relative = str(path.relative_to(self.output))
        if self.samples == 0:
            write_new(self.output / 'camera_calibration.json', {
                name:dict(intrinsics=c.intrinsics,camera_to_world=c.camera_to_world.xyzw())
                for name,c in observation['observation/calibrations'].items()})
        # Labels are appended only after the public result has been computed.
        # They are not fed back to the bridge, tracker or frozen GRU.
        append(self.output / 'public_outputs.jsonl', dict(index=self.samples,
            result=result, public_robot=robot, close_requested=close_requested, rgbd=relative))
        append(self.output / 'evaluation_labels.jsonl', {key:state[key] for key in
            ('time','object_position','object_quaternion','object_linear_velocity',
             'object_angular_velocity','held','native_success')})
        if abs(float(self.backend.clock()) - before) > 1e-9:
            raise RuntimeError('public shadow observation advanced physical time')
        duration = time.perf_counter() - started
        self.times.append(duration)
        self.samples += 1
        self.latest = result
        return result

    def tick(self, phase='local_mpc'):
        state = self.backend.tick(phase)
        self.sample(state)
        return state

    def observe_rgbd(self):
        # The controller's optional image logging reuses this exact current
        # sensor sample; it never changes the observer's history or frame rate.
        if self._cached_time is not None and abs(self.backend.clock()-self._cached_time) < 1e-9:
            return self._cached_observation
        return self.backend.observe_rgbd()

    def receipt(self):
        return dict(public_perception_GRU_shadow_only=not self.controller_uses_public_forecast,
            controller_uses_public_forecast=self.controller_uses_public_forecast, formal_TEST_used=False,
            frames=self.samples, real_sensor_period_seconds=.05,
            observer_wall_seconds=quantiles(self.times),
            attachment_estimator_used=self.attachment_estimator_used,observer_role=self.observer_role,
            labels_only_used_for_offline_evaluation=True)
