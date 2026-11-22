"""Causality and same-attempt label acceptance, without Native or fitting."""
from copy import deepcopy
import unittest
import numpy as np
from our_mani.local_mpc_p4_20261103.data import public_features, successful_contacts, CHECKPOINT_SHA256


def records():
    rows = []
    for i in range(5):
        t = .05*i
        obj = dict(object_id='target', timestamp=t, pose=dict(position=[.2, 0., .775]),
                   points_world=[[.175, -.025, .75], [.225, .025, .8], [.2, 0, .775]], visible=True, confidence=.8)
        r = dict(timestamp=t, mode='FREE', available=i==4, free_forecast_available=i==4,
                 Native_inputs_used=False, future_observations_used=False, observations_fabricated=False,
                 forecaster_is_injected_test_double=False, object_observation=obj)
        rows.append(dict(index=i, result=r, close_requested=False, rgbd=f'rgbd/{i:04d}.npz',
                         public_robot=dict(timestamp=t, tcp_position=[.2, 0, .8], tcp_quaternion=[0, 0, 0, 1],
                                           q=[0.]*7, dq=[0.]*7, finger_open_amounts=[1., 1.])))
    r['forecast'] = dict(object_ids=['target'], times=(t+np.arange(17)*.05).tolist(),
                         positions=np.tile([.2, 0, .775], (17, 1, 1)).tolist(),
                         rotations=np.tile(np.eye(3), (17, 1, 1, 1)).tolist(),
                         linear_velocities=np.zeros((17, 1, 3)).tolist(), angular_velocities=np.zeros((17, 1, 3)).tolist(),
                         metadata=dict(checkpoint_sha256=CHECKPOINT_SHA256, observed_until=t,
                                       history_frames={'target': 5}, contexts={'target': [1./8]*16}))
    return rows


class PublicFeaturesTests(unittest.TestCase):
    def test_shapes_units_separate_context_and_placeholder(self):
        rows = records(); features, meta = public_features(rows[-1], rows)
        self.assertEqual(features['future_tokens'].shape, (17, 16))
        self.assertEqual(features['robot'].shape, (25,))
        self.assertEqual(features['geometry'].shape, (4,))
        self.assertAlmostEqual(float(features['robot'][16]), .025, places=6)
        self.assertEqual(len(meta['real_history_rgbd']), 5)
        self.assertIn('not_task_classifier', meta['task_e_source'])
        self.assertFalse(meta['orientation_observed'])

    def test_future_real_history_or_interpolated_period_is_rejected(self):
        for value in (.25, .15):
            rows = records(); rows[1]['public_robot']['timestamp'] = value
            with self.assertRaises(ValueError): public_features(rows[-1], rows)
        rows = records(); rows[-1]['result']['forecast']['metadata']['observed_until'] = .25
        with self.assertRaises(ValueError): public_features(rows[-1], rows)

    def test_privileged_robot_and_label_channels_cannot_enter(self):
        rows = records(); rows[-1]['public_robot']['held'] = True
        with self.assertRaises(ValueError): public_features(rows[-1], rows)
        rows = records(); rows[-1]['label'] = [123]*3
        with self.assertRaises(ValueError): public_features(rows[-1], rows)
        rows = records(); rows[-1]['result']['Native_inputs_used'] = True
        with self.assertRaises(ValueError): public_features(rows[-1], rows)

    def test_nonfinite_forecast_or_context_and_stale_frame_are_rejected(self):
        for location in ('forecast', 'context', 'stale'):
            rows = records()
            if location == 'forecast': rows[-1]['result']['forecast']['positions'][1][0][0] = float('nan')
            elif location == 'context': rows[-1]['result']['forecast']['metadata']['contexts']['target'][0] = 0.
            else: rows[1]['result']['object_observation']['timestamp'] = 0.
            with self.assertRaises(ValueError): public_features(rows[-1], rows)


def initial():
    return dict(time=0., held=False, native_success=False, object_position=[0, 0, .775],
                object_quaternion=[0, 0, 0, 1], tcp_position=[0, 0, .76], tcp_quaternion=[0, 0, 0, 1])


class ContactLabelsTests(unittest.TestCase):
    def ticks(self):
        values = []
        for i in range(1, 21):
            s = initial(); s.update(time=i*.05, held=i>=2, native_success=i>=16)
            if i>=4: s.update(object_position=[0, 0, .825], tcp_position=[0, 0, .81])
            values.append(s)
        return values

    def test_actual_first_held_pose_and_tau_not_later_lift_or_advice(self):
        labels = successful_contacts(initial(), self.ticks())
        self.assertEqual(len(labels), 1)
        self.assertAlmostEqual(labels[0]['tau'], .1)
        np.testing.assert_allclose(labels[0]['object_tcp_position'], [0, 0, -.015], atol=1e-10)

    def test_close_request_without_held_or_hold_without_native_goal_not_positive(self):
        ticks = self.ticks()
        for tick in ticks: tick.update(held=False, requested_opening=0.)
        self.assertEqual(successful_contacts(initial(), ticks), [])
        ticks = self.ticks()
        for tick in ticks: tick['native_success'] = False
        self.assertEqual(successful_contacts(initial(), ticks), [])

    def test_distinct_attempts_cannot_share_goal_and_hold_evidence(self):
        ticks = self.ticks()
        for i, tick in enumerate(ticks):
            tick['native_success'] = i>=18
            if i==16: tick['held'] = False
        self.assertEqual(successful_contacts(initial(), ticks), [])

    def test_missing_tick_cannot_be_interpolated(self):
        ticks = self.ticks(); del ticks[4]
        with self.assertRaises(ValueError): successful_contacts(initial(), ticks)


if __name__ == '__main__': unittest.main()
