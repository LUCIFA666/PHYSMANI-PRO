"""No simulator is run by these TRAIN-only hook ownership/scope tests."""
from copy import deepcopy
from pathlib import Path
import unittest

from our_mani.multitask import train_teacher
from our_mani.multitask.train_teacher import FollowerConfig
from our_mani.multitask.train_teacher_requery import PlanningRequeryConfig, CountedPlanningRequeryFollower
from our_mani.multitask.train_teacher_requery_hook import scoped_counted_requery, file_sha, TASK


class RequeryHookTests(unittest.TestCase):
    def setUp(self):
        self.original = train_teacher.CommittedWaypointFollower
        self.case = dict(case_id='fixture', split='train', task=TASK,
                         behavior='TRAIN_teacher_committed_waypoint', teacher_used=True,
                         context_primary_view=False, interaction_eligible=True,
                         replica=1, behavior_replica=1, maximum_collection_ticks=200)
        self.sha = file_sha(Path(train_teacher.__file__))
        self.captures = []

    def scope(self, case=None, checksum=None):
        return scoped_counted_requery(self.case if case is None else case,
            expected_teacher_source_sha256=self.sha if checksum is None else checksum,
            configuration=PlanningRequeryConfig(), captures=self.captures)

    def construct(self):
        # Mirrors the collector's local import after the scope is entered.
        from our_mani.multitask.train_teacher import CommittedWaypointFollower
        return CommittedWaypointFollower(lambda: {}, lambda goal: None,
            FollowerConfig(.02, .1, .005, 3), initial_opening=1.)

    def test_collector_import_constructs_one_counted_follower_and_restores(self):
        with self.scope() as receipt:
            follower = self.construct()
            self.assertIsInstance(follower, CountedPlanningRequeryFollower)
            self.assertEqual(self.captures, [follower])
            with self.assertRaisesRegex(ValueError, 'one explicit follower'):
                self.construct()
        self.assertIs(train_teacher.CommittedWaypointFollower, self.original)
        self.assertTrue(receipt['complete'])
        self.assertTrue(receipt['original_factory_restored'])
        self.assertEqual(receipt['actual_requeries'], 0)

    def test_native_error_is_propagated_without_retry_and_factory_restored(self):
        with self.assertRaisesRegex(RuntimeError, 'native failure'):
            with self.scope() as receipt:
                self.construct()
                raise RuntimeError('native failure')
        self.assertIs(train_teacher.CommittedWaypointFollower, self.original)
        self.assertFalse(receipt['complete'])
        self.assertFalse(receipt['infrastructure_error_retry'])

    def test_other_split_task_hold_replica_or_budget_cannot_enter(self):
        for key, value in [('split', 'validation'), ('split', 'test'), ('task', 'other'),
                           ('context_primary_view', True), ('teacher_used', False),
                           ('replica', 0), ('replica', True), ('behavior_replica', 0),
                           ('behavior_replica', True), ('maximum_collection_ticks', 201),
                           ('maximum_collection_ticks', 200.)]:
            case = deepcopy(self.case); case[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                with self.scope(case):
                    self.fail('invalid scope entered')
            self.assertIs(train_teacher.CommittedWaypointFollower, self.original)

    def test_foreign_factory_or_wrong_source_is_rejected_before_patch(self):
        with self.assertRaisesRegex(ValueError, 'source/factory'):
            with self.scope(checksum='0' * 64):
                pass
        foreign = lambda: None
        train_teacher.CommittedWaypointFollower = foreign
        try:
            with self.assertRaisesRegex(ValueError, 'source/factory'):
                with self.scope():
                    pass
            self.assertIs(train_teacher.CommittedWaypointFollower, foreign)
        finally:
            train_teacher.CommittedWaypointFollower = self.original

    def test_reentry_is_rejected_without_replacing_outer_hook(self):
        with self.scope() as receipt:
            outer = train_teacher.CommittedWaypointFollower
            with self.assertRaisesRegex(ValueError, 'source/factory'):
                with scoped_counted_requery(self.case, expected_teacher_source_sha256=self.sha,
                        configuration=PlanningRequeryConfig(), captures=[]):
                    pass
            self.assertIs(train_teacher.CommittedWaypointFollower, outer)
            self.construct()
        self.assertTrue(receipt['complete'])

    def test_foreign_write_during_hook_is_detected_and_original_restored(self):
        with self.assertRaisesRegex(ValueError, 'owner/source changed'):
            with self.scope() as receipt:
                self.construct()
                train_teacher.CommittedWaypointFollower = lambda: None
        self.assertIs(train_teacher.CommittedWaypointFollower, self.original)
        self.assertFalse(receipt['complete'])
        self.assertFalse(receipt['hook_owner_unchanged'])

    def test_no_constructed_follower_is_not_a_complete_execution(self):
        with self.scope() as receipt:
            pass
        self.assertFalse(receipt['complete'])
        self.assertEqual(receipt['follower_instances'], 0)


if __name__ == '__main__':
    unittest.main()
