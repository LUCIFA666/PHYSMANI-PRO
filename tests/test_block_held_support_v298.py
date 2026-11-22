from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import numpy as np

from our_mani.benchmark_perception import RedBoxTracker, BoxPrior
from our_mani.contracts import Pose, ObjectState
from our_mani.multitask.block_held_support_v298 import (
    MeasuredTracker, SmoothBlockEnvironment, supported_box, held_collision)


def corners():
    return np.array([[x, y, z] for x in (-.025, .025)
        for y in (-.025, .025) for z in (-.025, .025)])


def top_cloud(z=.801363268):
    x, y = np.meshgrid(np.linspace(.075, .125, 12), np.linspace(-.025, .025, 12))
    return np.c_[x.ravel(), y.ravel(), np.full(x.size, z)]


def observation(timestamp):
    points = top_cloud()
    calibration = NS(backproject=lambda depth, mask: points.copy())
    rgb = np.zeros((1, len(points), 3), np.uint8)
    rgb[:, :, 0] = 255
    return dict(timestamp=timestamp, **{'observation/images': {'front': rgb},
        'observation/depths': {'front': np.ones((1, len(points)))},
        'observation/calibrations': {'front': calibration}})


PLANE = dict(height=.751999855, xy_min=[-.075, -.455], xy_max=[.575, .455])


class Shape:
    def __init__(self, handle):
        self.handle = handle
    def get_handle(self):
        return self.handle


class Target(Shape):
    def __init__(self):
        super().__init__(1)
        self.collidable = True
        self.pose = Pose([.1, 0, .775])
        self.pairs = {2}
    def set_collidable(self, value):
        self.collidable = value
    def check_collision(self, shape):
        return shape.handle in self.pairs


class Service:
    def __init__(self):
        self.target = Target()
        self.ghosts = [Shape(4), Shape(5)]
        self.queries = 0
        self.arm_other_collision = False
        self.env = NS(arm=NS(check_arm_collision=lambda:
            self.arm_other_collision or (self.target.collidable and 2 in self.target.pairs)))
    @contextmanager
    def estimated_target(self, pose):
        old = self.target.pose
        flag = self.target.collidable
        self.target.pose = pose
        try:
            yield
        finally:
            self.target.pose = old
            self.target.collidable = flag


class Tests(unittest.TestCase):
    def test_same_RGBD_keeps_original_tracker_geometry_velocity_and_history(self):
        original = RedBoxTracker()
        boundary = NS()
        measured = MeasuredTracker(boundary)
        for t in (0., .05, .1):
            a, ma = original.observe(observation(t))
            b, mb = measured.observe(observation(t))
            self.assertEqual(ma, mb)
            for attr in ('points_world', 'linear_velocity', 'angular_velocity'):
                np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))
            np.testing.assert_array_equal(a.pose.xyzw(), b.pose.xyzw())
            self.assertEqual(len(boundary.visual_measurement['points']), 144)
            for x, y in zip(original.history, measured.history):
                self.assertEqual(x[0], y[0])
                np.testing.assert_array_equal(x[1], y[1])

    def test_same_surface_support_correction_and_inconsistent_cloud_rejection(self):
        pose = Pose([.1, 0, .776363268])
        fitted, row = supported_box(pose, top_cloud(), corners(), PLANE, BoxPrior())
        self.assertTrue(row['accepted'])
        self.assertLess(row['z_correction_m'], .00064)
        self.assertGreater(fitted.apply(corners())[:, 2].min(), PLANE['height'])
        for points, candidate in [(top_cloud(1.1), pose),
                (top_cloud(.7), pose), (top_cloud(), Pose([.7, 0, .776363268]))]:
            _, bad = supported_box(candidate, points, corners(), PLANE, BoxPrior())
            self.assertFalse(bad['accepted'])

    def test_gripper_only_contact_never_exempts_forearm_static_or_arm_world(self):
        service = Service()
        pose = Pose([.1, 0, .8])
        old = service.target.pose
        self.assertFalse(held_collision(service, pose, [Shape(2), Shape(3)], {2}))
        self.assertIs(service.target.pose, old)
        self.assertTrue(service.target.collidable)
        for obstacle in (3, 4, 5):
            service.target.pairs.add(obstacle)
            self.assertTrue(held_collision(service, pose, [Shape(2), Shape(3)], {2}))
            service.target.pairs.remove(obstacle)
        service.arm_other_collision = True
        self.assertTrue(held_collision(service, pose, [Shape(2), Shape(3)], {2}))

    def test_collision_exception_restores_owned_target_pose_and_flags(self):
        service = Service()
        old = service.target.pose
        def fail(shape):
            raise RuntimeError('native collision unavailable')
        service.target.check_collision = fail
        with self.assertRaisesRegex(RuntimeError, 'unavailable'):
            held_collision(service, Pose([0, 0, 1]), [Shape(2), Shape(3)], {2})
        self.assertIs(service.target.pose, old)
        self.assertTrue(service.target.collidable)

    def test_first_contact_only_admission_persists_under_occlusion_then_clears(self):
        env = NS(_requested_opening=0., time=.3)
        boundary = SmoothBlockEnvironment(env)
        boundary.service = NS()
        boundary._install_map = lambda: None
        boundary.plane, boundary.vertices = PLANE, corners()
        robot = NS(timestamp=.3, contact=True, pose=Pose([.1, 0, .762]))
        body = ObjectState('target', .3, Pose([.1, 0, .776363268]), top_cloud(), np.zeros(3), np.zeros(3))
        boundary.contact_started_at = .3
        boundary.visual_measurement = dict(timestamp=.3, object_id='target', pose=body.pose,
            points=top_cloud(), prior=BoxPrior())
        boundary.supplied_decision(robot, body, dict(command='local_capture'), NS())
        self.assertTrue(boundary._held_valid())
        first = boundary.held_local
        env.time = .35
        robot = NS(timestamp=.35, contact=True, pose=Pose([.1, 0, .8]))
        boundary.visual_measurement = None
        boundary.supplied_decision(robot, None, dict(command='hold'), NS())
        self.assertIs(boundary.held_local, first)
        self.assertTrue(boundary._held_valid())
        # Opening/lost contact invalidates permission immediately, including
        # before another current RGBD measurement or controller step.
        env._requested_opening = 1.
        self.assertFalse(boundary._held_valid())
        env.observe = lambda: ({}, NS(timestamp=.4, contact=False))
        boundary.observe()
        self.assertIsNone(boundary.held_local)

    def test_stale_or_missing_first_contact_measurement_never_grants_permission(self):
        for timestamp in (None, .25):
            env = NS(_requested_opening=0., time=.3)
            b = SmoothBlockEnvironment(env)
            b._install_map = lambda: None
            b.contact_started_at = .3
            b.visual_measurement = None if timestamp is None else dict(timestamp=timestamp, object_id='target')
            robot = NS(timestamp=.3, contact=True)
            b.supplied_decision(robot, NS(object_id='target'), dict(command='hold'), NS())
            self.assertFalse(b._held_valid())
            self.assertIsNone(b.held_local)


if __name__ == '__main__':
    unittest.main()
