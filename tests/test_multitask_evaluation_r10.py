"""Formal denominators cannot absorb demo or infrastructure records."""
import unittest

from our_mani.multitask.evaluation import (case_identity, native_completion_record, validate_completed_case,
                                          comparison_protocol_sha256)
from our_mani.multitask.protocol import MultitaskProtocolError


class EvaluationRecordsTests(unittest.TestCase):
    def test_val_release_pointer_does_not_create_a_config_hash_cycle(self):
        val=dict(dt=.05,context_checkpoint={'path':'context.pt','sha256':'a'*64},
                 gripper_timing_config={'position_m':.02},formal_comparison_ready=False)
        test=dict(val,formal_comparison_ready=True,
                  r10_validation_release={'path':'release.json','sha256':'b'*64})
        self.assertEqual(comparison_protocol_sha256(val),comparison_protocol_sha256(test))
        for changed in (dict(test,dt=.1),dict(test,context_checkpoint={'path':'context.pt','sha256':'c'*64}),
                        dict(test,gripper_timing_config={'position_m':.03}),dict(test,extra_policy_setting=True)):
            self.assertNotEqual(comparison_protocol_sha256(val),comparison_protocol_sha256(changed))

    def identity(self, **changes):
        values = dict(split='test', task='fixture', raw_episode_id='episode0', method='cv_geometry',
            source_row=dict(task='fixture', raw_episode_id='episode0'),
            config_sha256='a' * 64, inputs_sha256='b' * 64)
        values.update(changes)
        return case_identity(**values)

    def record(self, **changes):
        values = dict(identity=self.identity(), process_id=321, success=False, fatal=False, ticks=200,
            initial_conditions=dict(metric_only=True, policy_use_allowed=False))
        values.update(changes)
        ticks = values['ticks']
        values['alignment_receipt'] = dict(clock=dict(ticks=ticks, dt=.05, budget_seconds=10., simulation_seconds=ticks*.05),
            actual_physics_seconds=ticks*.05, actual_task_seconds=ticks*.05,
            native_success_latched=values['success'], native_terminal_failure_latched=values['fatal'])
        return native_completion_record(**values)

    def test_demo_replay_and_formal_records_have_distinct_identities(self):
        demo = self.identity(demo_only=True)
        record = self.record(identity=demo, success=True, ticks=12)
        self.assertFalse(record['formal_test_used'])
        with self.assertRaises(MultitaskProtocolError):
            validate_completed_case(record, self.identity())

    def test_censor_fatal_and_clock_corruption_rejected(self):
        for change in (dict(infrastructure_error='native unavailable'),
                       dict(official_task_success=True, native_terminal_failure=True),
                       dict(counted_clock=dict(ticks=200, dt=.05, seconds=9.)),
                       dict(TEST_used_for_fitting=True)):
            with self.subTest(change=change), self.assertRaises(MultitaskProtocolError):
                validate_completed_case(dict(self.record(), **change), self.identity())

    def test_full_budget_task_failure_is_valid_without_fake_terminal(self):
        result = validate_completed_case(self.record(), self.identity())
        self.assertTrue(result['passed'])
        self.assertFalse(result['official_task_success'])
        with self.assertRaises(MultitaskProtocolError):
            self.identity(replica=1)

    def test_actual_native_clock_cannot_be_replaced_by_synthetic_tick_count(self):
        record = self.record()
        record['alignment_receipt']['actual_physics_seconds'] = 999.
        with self.assertRaisesRegex(MultitaskProtocolError, 'actual native clock'):
            validate_completed_case(record, self.identity())

    def test_source_and_augmented_rollout_identities_remain_distinct(self):
        value = self.identity(split='train', raw_episode_id='episode0__policy_aug_r1',
            parent_raw_episode_id='episode0', replica=1, exploration_seed=17)
        self.assertEqual(value['source_raw_episode_id'], 'episode0')
        self.assertEqual(value['raw_episode_id'], 'episode0__policy_aug_r1')
        with self.assertRaises(MultitaskProtocolError):
            self.identity(split='validation', raw_episode_id='episode0__policy_aug_r1',
                parent_raw_episode_id='episode0', replica=1, exploration_seed=17)
        record = self.record()
        record['alignment_receipt']['clock']['simulation_seconds'] = float('nan')
        with self.assertRaises(MultitaskProtocolError):
            validate_completed_case(record, self.identity())
        record = self.record()
        record['task'] = 'another_task'
        with self.assertRaisesRegex(MultitaskProtocolError, 'top-level case identity'):
            validate_completed_case(record, self.identity())


if __name__ == '__main__':
    unittest.main()
