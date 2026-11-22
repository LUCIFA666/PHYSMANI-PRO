"""Causality and source separation for SE3 training windows."""
import copy
import unittest

import numpy as np
import torch

from our_mani.multitask.context_se3 import SE3ContextNetwork
from our_mani.multitask.track_dataset import (BodyTrackEpisode, examples, train_normalization,
                                             validate_motion_label_schema)


def episode(split='train', offset=0.):
    times = np.arange(12) * .05
    positions = np.stack((times, times * 0, times * 0 + .8), -1)[:, None] + offset
    arrays = dict(times=times, body_ids=np.array(['frame']), positions=positions,
        rotations=np.broadcast_to(np.eye(3), (12, 1, 3, 3)).copy(),
        linear_velocities=np.broadcast_to([1., 0., 0.], (12, 1, 3)).copy(),
        angular_velocities=np.zeros((12, 1, 3)), confidence=np.ones((12, 1)),
        valid=np.ones((12, 1), dtype=bool), modes=np.zeros((12, 1), dtype=int),
        rotation_observable=np.ones((12, 1), dtype=bool))
    for name in ('positions', 'rotations', 'linear_velocities', 'angular_velocities'):
        arrays['target_' + name] = arrays[name].copy()
    arrays['target_valid'] = arrays['valid'].copy()
    arrays['target_rotation_observable'] = arrays['rotation_observable'].copy()
    return BodyTrackEpisode(dict(split=split, task='synthetic_fixture', raw_episode_id='source0'), arrays)


class TrackDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_validation_extremes_cannot_fit_normalization(self):
        original, _ = train_normalization([episode()])
        mixed, coverage = train_normalization([episode(), episode('validation', offset=1000.)])
        self.assertEqual(original, mixed)
        self.assertEqual(len(coverage), 1)

    def test_future_targets_never_enter_history_and_mode_switch_stops_labels(self):
        model = SE3ContextNetwork()
        original = episode()
        before, _ = examples([original], model, steps=7)
        changed = copy.deepcopy(original)
        changed.arrays['target_positions'][7:] += 5.
        after, _ = examples([changed], model, steps=7)
        np.testing.assert_array_equal(before[0]['history'], after[0]['history'])
        np.testing.assert_array_equal(before[0]['state'], after[0]['state'])
        self.assertFalse(np.array_equal(before[0]['targets']['positions'], after[0]['targets']['positions']))
        changed.arrays['modes'][8:10] = 1
        switched, _ = examples([changed], model, steps=7)
        self.assertTrue(switched[0]['mask'][1:4].all())
        self.assertFalse(switched[0]['mask'][4:].any())
        unknown = copy.deepcopy(original)
        unknown.arrays['target_valid'][8] = False
        bounded, _ = examples([unknown], model, steps=7)
        self.assertTrue(bounded[0]['mask'][1:4].all())
        self.assertFalse(bounded[0]['mask'][4:].any())

    def test_replica_parent_groups_do_not_merge_physical_body_histories(self):
        one, two = episode(), episode(offset=.02)
        for i, value in enumerate((one, two)):
            value.entry['parent_raw_episode_id'] = 'source0'
            value.entry['raw_episode_id'] = f'source0__replica{i}'
        rows, coverage = examples([one, two], SE3ContextNetwork(), steps=7)
        self.assertEqual(len(coverage), 1)
        self.assertEqual(len({row['source_index'] for row in rows}), 1)
        self.assertEqual(len({row['body_group'] for row in rows}), 2)

    def test_derived_velocity_schema_cannot_mix_legacy_or_reintroduce_raw_lag(self):
        from our_mani.multitask.motion_labels import SCHEMA, backward_pose_supervision
        arrays=episode().arrays
        for name in ('native_pose_valid','mode_valid','dynamics_segment_valid'):
            arrays[name]=np.ones((12,1),bool)
        arrays['target_linear_velocities'][:]=[.2,0.,0.]
        derived,_=backward_pose_supervision(arrays)
        validate_motion_label_schema(derived,SCHEMA,SCHEMA)
        with self.assertRaisesRegex(ValueError,'schemas differ'):
            validate_motion_label_schema(derived,SCHEMA,None)
        with self.assertRaisesRegex(ValueError,'explicit manifest'):
            validate_motion_label_schema(derived,None,None)
        changed=copy.deepcopy(derived);changed['target_linear_velocities'][4]=[.2,0.,0.]
        with self.assertRaisesRegex(ValueError,'backward GT pose'):
            validate_motion_label_schema(changed,SCHEMA,SCHEMA)
        changed=copy.deepcopy(derived);changed['target_valid'][0]=True
        with self.assertRaisesRegex(ValueError,'boundary mask'):
            validate_motion_label_schema(changed,SCHEMA,SCHEMA)


if __name__ == '__main__':
    unittest.main()
