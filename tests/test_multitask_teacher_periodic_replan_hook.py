from copy import deepcopy
from pathlib import Path
import unittest

from our_mani.multitask import train_teacher
from our_mani.multitask.train_teacher import FollowerConfig
from our_mani.multitask.train_teacher_requery import PlanningRequeryConfig
from our_mani.multitask.train_teacher_periodic_replan import PeriodicReplanningConfig, PeriodicFeedbackReplanningFollower
from our_mani.multitask.train_teacher_periodic_replan_hook import scoped_periodic_replanning, file_sha


class PeriodicHookTests(unittest.TestCase):
    def case(self):
        return dict(case_id='fixed2', split='train', task='place_cups_on_rotating_frame_high_speed',
                    behavior='TRAIN_teacher_committed_waypoint', teacher_used=True,
                    context_primary_view=False, interaction_eligible=True,
                    replica=2, behavior_replica=2, maximum_collection_ticks=200)

    def hook(self, case=None, captures=None, source=None):
        return scoped_periodic_replanning(self.case() if case is None else case,
            captures=[] if captures is None else captures,
            expected_teacher_source_sha256=file_sha(Path(train_teacher.__file__)) if source is None else source,
            periodic_configuration=PeriodicReplanningConfig(), failure_configuration=PlanningRequeryConfig())

    def instantiate(self):
        return train_teacher.CommittedWaypointFollower(lambda: None, lambda goal: None,
                    FollowerConfig(.02, .1, .005, 10), initial_opening=1.)

    def test_exact_factory_scope_and_restore_with_truthful_zero_counters(self):
        original = train_teacher.CommittedWaypointFollower
        captures = []
        with self.hook(captures=captures) as receipt:
            follower = self.instantiate()
            self.assertIsInstance(follower, PeriodicFeedbackReplanningFollower)
            self.assertIs(captures[0], follower)
            self.assertIsNot(train_teacher.CommittedWaypointFollower, original)
        self.assertIs(train_teacher.CommittedWaypointFollower, original)
        self.assertTrue(receipt['complete'])
        self.assertTrue(receipt['original_factory_restored'])
        self.assertEqual(receipt['actual_periodic_requeries'], 0)
        self.assertEqual(receipt['actual_failed_planning_requeries'], 0)
        self.assertFalse(receipt['native_or_physics_steps_called_by_hook'])

    def test_wrong_task_split_replica_context_budget_or_source_reject(self):
        for name, value in (('task', 'other'), ('split', 'val'), ('replica', 1),
                            ('behavior_replica', 1), ('context_primary_view', True),
                            ('teacher_used', False), ('maximum_collection_ticks', 201)):
            case = self.case(); case[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                with self.hook(case=case):
                    pass
        with self.assertRaises(ValueError):
            with self.hook(source='f' * 64):
                pass

    def test_body_error_and_foreign_owner_restore_without_complete_witness(self):
        original = train_teacher.CommittedWaypointFollower
        with self.assertRaisesRegex(RuntimeError, 'fixed body'):
            with self.hook() as receipt:
                self.instantiate()
                raise RuntimeError('fixed body')
        self.assertIs(train_teacher.CommittedWaypointFollower, original)
        self.assertFalse(receipt['complete'])
        with self.assertRaisesRegex(ValueError, 'owner/source'):
            with self.hook() as receipt:
                self.instantiate()
                train_teacher.CommittedWaypointFollower = lambda: None
        self.assertIs(train_teacher.CommittedWaypointFollower, original)
        self.assertFalse(receipt['complete'])

    def test_nested_hook_or_second_instance_reject_and_restores(self):
        original = train_teacher.CommittedWaypointFollower
        with self.hook() as receipt:
            self.instantiate()
            with self.assertRaises(ValueError):
                with self.hook():
                    pass
            with self.assertRaises(ValueError):
                self.instantiate()
        self.assertIs(train_teacher.CommittedWaypointFollower, original)
        self.assertTrue(receipt['complete'])
        self.assertEqual(receipt['follower_instances'], 1)


if __name__ == '__main__':
    unittest.main()
