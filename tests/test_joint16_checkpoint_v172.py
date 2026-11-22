from copy import deepcopy
from pathlib import Path
import json
import unittest
from our_mani.multitask.context_se3 import MultiBodyContextDynamics
from our_mani.multitask.joint16_checkpoint_v172 import check_training_metadata_and_roster

ROOT = Path(__file__).resolve().parents[1]
B = ROOT/'artifacts/framework/multitask_r10'


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MultiBodyContextDynamics(B/'context_joint16_hold77_v157/dynamics.pt')
        cls.entries = json.loads((B/'context_joint16_hold77_v157/training_sources.json').read_text())['entries']
        cls.roster = json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']

    def test_actual_completed_checkpoint_and_original1600_roster_pass(self):
        proof = check_training_metadata_and_roster(self.model.metadata, self.entries, self.roster)
        self.assertEqual(proof['buzz_parents_with_conditional_rotation'], 200)
        self.assertEqual(len(proof['tasks']), 16)

    def test_original_old1400_checkpoint_is_rejected(self):
        model = MultiBodyContextDynamics(B/'context_se3_full_v6/dynamics.pt')
        with self.assertRaises(ValueError):
            check_training_metadata_and_roster(model.metadata, self.entries, self.roster)

    def test_missing_or_duplicate_parent_rejected_even_if_counts_are_claimed(self):
        for rows in (self.entries[:-1], self.entries[:-1]+[self.entries[0]]):
            with self.assertRaises(ValueError):
                check_training_metadata_and_roster(self.model.metadata, rows, self.roster)

    def test_incomplete_training_or_TEST_fit_rejected(self):
        for key, value in (('training_complete', False), ('completed_epochs', 29), ('TEST_used_for_fitting', True)):
            metadata = dict(self.model.metadata, **{key: value})
            with self.assertRaises(ValueError):
                check_training_metadata_and_roster(metadata, self.entries, self.roster)

    def test_old_orientation_none_buzz_channel_rejected(self):
        rows = deepcopy(self.entries)
        rows[0]['new_conditional_target_rotation_mask'] = False
        with self.assertRaises(ValueError):
            check_training_metadata_and_roster(self.model.metadata, rows, self.roster)


if __name__ == '__main__': unittest.main()
