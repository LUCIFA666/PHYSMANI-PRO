"""Shared author-domain RGB-D sensors for both r07 methods.

Only the front renderer changes to OPENGL. The existing uint8 image channel
remains available to trackers/video; the additional native RGB channel uses
the released PhysMani front-floor/numeric-float convention for both methods.
All reads reuse the already rendered buffers and never advance simulation.
"""
import numpy as np

from ..aligned.audit import public_initial_conditions as original_public_initial_conditions
from ..aligned.simulation import (
    AlignedLocalSimulation as OriginalLocalSimulation,
    AlignedOfficialSimulation as OriginalOfficialSimulation,
    physics_time,
)


class SensorProtocolInvariantError(Exception):
    """A sensor protocol failure is infrastructure, not a task outcome."""


PROTOCOL = 'shared_author_opengl_rgbd_r07'
RGB_SPEC = {
    'observation/images': 'all cameras retain common round(capture_rgb*255) uint8 for trackers/video',
    'observation/raw_rgb_255': {
        'front': 'author clip((capture_rgb*255).astype(uint8),0,255)',
        '8': 'native capture_rgb float32*255, no quantization',
        '16': 'native capture_rgb float32*255, no quantization',
        '36': 'native capture_rgb float32*255, no quantization',
    },
    'buffer_access': 'capture_rgb from the frame rendered by super.observe; no additional handle_explicitly or physics',
    'depth': 'existing metric depth unchanged',
    'access': 'both methods receive both RGB channels and identical sensor modes',
}


def sensor_conditions(env):
    return dict(version=PROTOCOL,
                camera_render_modes={name: camera.get_render_mode().name
                                     for name, camera in env.cameras.items()},
                rgb_processing=RGB_SPEC)


def public_initial_conditions(env, observation, robot):
    conditions = original_public_initial_conditions(env, observation, robot)
    sensor = sensor_conditions(env)
    conditions['sensor_protocol'] = sensor
    for name, calibration in conditions['cameras'].items():
        calibration['render_mode'] = sensor['camera_render_modes'][name]
        calibration['raw_rgb_255_processing'] = RGB_SPEC['observation/raw_rgb_255'][name]
    return conditions


class AuthorRGBDSensors:
    def _start_alignment(self):
        super()._start_alignment()
        from pyrep.const import RenderMode
        camera = self.cameras['front']
        before = physics_time(), float(self.task.t)
        intrinsics, extrinsics = camera.get_intrinsic_matrix(), camera.get_matrix()
        camera.set_render_mode(RenderMode.OPENGL)
        if (physics_time(), float(self.task.t)) != before:
            raise SensorProtocolInvariantError('front renderer selection advanced physics/task clock')
        if not np.allclose(camera.get_intrinsic_matrix(), intrinsics, rtol=0, atol=1e-7):
            raise SensorProtocolInvariantError('front renderer selection changed intrinsics')
        if not np.allclose(camera.get_matrix(), extrinsics, rtol=0, atol=1e-7):
            raise SensorProtocolInvariantError('front renderer selection changed extrinsics')

    def observe(self):
        before = physics_time(), float(self.task.t)
        observation, robot = super().observe()
        raw = {}
        for name, camera in self.cameras.items():
            values = np.asarray(camera.capture_rgb(), dtype=np.float32)*255.
            if not np.isfinite(values).all():
                raise SensorProtocolInvariantError('nonfinite native RGB frame')
            raw[name] = np.clip(values.astype(np.uint8), 0, 255) if name == 'front' else values
        observation['observation/raw_rgb_255'] = raw
        observation['observation/sensor_protocol'] = sensor_conditions(self)
        if (physics_time(), float(self.task.t)) != before:
            raise SensorProtocolInvariantError('native RGB buffer access advanced physics/task clock')
        return observation, robot


class AlignedLocalSimulation(AuthorRGBDSensors, OriginalLocalSimulation):
    pass


class AlignedOfficialSimulation(AuthorRGBDSensors, OriginalOfficialSimulation):
    pass
