"""CPU-only real follower ledgers and separate augmentation boundary tests."""
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from our_mani.contracts import Pose
from our_mani.multitask import train_teacher
from our_mani.multitask.train_teacher import FollowerConfig, TeacherPathPlanningFailure
from our_mani.multitask.train_teacher_requery import PlanningRequeryConfig
from our_mani.multitask.train_teacher_requery_hook import scoped_counted_requery, file_sha, TASK
from our_mani.multitask.requery_acquisition import expected_cases, validate_counted_commands


class Robot:
    timestamp = 0.
    joints = np.zeros(7)
    pose = Pose(np.zeros(3))


class RequeryRawLedgerTests(unittest.TestCase):
    def fixture(self):
        case = dict(case_id='CPU_fixture', split='train', task=TASK,
                    behavior='TRAIN_teacher_committed_waypoint', teacher_used=True,
                    context_primary_view=False, interaction_eligible=True,
                    replica=1, behavior_replica=1, maximum_collection_ticks=200)
        def teacher():
            return dict(tcp_xyzw=Pose(np.ones(3)).xyzw(), opening=1., stage='wp3',
                        phase='transport', reference_body='spoke0', manipulated_body='cup0',
                        reference_role='selected_spoke', reference_kind='observed_se3',
                        anchor_id='cup:transport', manipulated_role='cup0')
        def planner(goal):
            raise TeacherPathPlanningFailure('CPU planning failure, no native code')
        captures = []
        with scoped_counted_requery(case, expected_teacher_source_sha256=file_sha(Path(train_teacher.__file__)),
                configuration=PlanningRequeryConfig(), captures=captures) as scope:
            from our_mani.multitask.train_teacher import CommittedWaypointFollower
            follower = CommittedWaypointFollower(teacher, planner, FollowerConfig(.02, .1, .005, 3), initial_opening=1.)
            robot = Robot(); lower, upper = np.full(7, -1.), np.full(7, 1.)
            follower.command(robot, lower, upper)
            for tick in range(1, 11):
                time = tick * .05
                follower.after_tick(dict(timestamp=time, gripper_actuation_done=False,
                    measured_tcp_xyzw=robot.pose.xyzw(), joint_limit_recovery_action=False))
                robot.timestamp = time
                if tick < 10:
                    follower.command(robot, lower, upper)
            follower.command(robot, lower, upper)
            follower.finish('fixed_tick_budget', .5)
        return dict(commands=deepcopy(follower.commands), phase_completion_events=[]), dict(scope)

    def test_real_counted_hook_ledger_has_one_fresh_query_and_preserved_failure(self):
        ledger, scope = self.fixture()
        proof = validate_counted_commands(ledger, scope, ticks=10)
        self.assertEqual(proof['actual_requeries'], 1)
        self.assertEqual(proof['failed_commands'], 2)
        self.assertEqual(proof['failed_commands_marked_completed'], 0)

    def test_wait_query_completion_and_configuration_forgeries_rejected(self):
        for corruption in ('wait9', 'early', 'completed', 'different_previous', 'positive_failure',
                           'scope_count', 'missing_configuration', 'duplicate_id', 'bool_id', 'unrestored'):
            ledger, scope = self.fixture(); before, after = ledger['commands']
            if corruption == 'wait9': before['planning_requery_counted_wait_ticks'] = 9
            elif corruption == 'early': after['generated_at'] = .45
            elif corruption == 'completed': before['command_completed_at'] = .5
            elif corruption == 'different_previous': after['fresh_query_after_failed_command_id'] = 1
            elif corruption == 'positive_failure': ledger['phase_completion_events'] = [dict(origin_command_id=0)]
            elif corruption == 'scope_count': scope['actual_requeries'] = 2
            elif corruption == 'missing_configuration': after.pop('TRAIN_planning_requery_configuration')
            elif corruption == 'duplicate_id': after['command_id'] = 0
            elif corruption == 'bool_id': after['command_id'] = True
            else: scope['original_factory_restored'] = False
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                validate_counted_commands(ledger, scope, ticks=10)

    def test_requery_count_cannot_exceed_actual_feedback_budget(self):
        ledger, scope = self.fixture()
        with self.assertRaisesRegex(ValueError, 'real waits'):
            validate_counted_commands(ledger, scope, ticks=9)


class FixedRosterTests(unittest.TestCase):
    def primary(self):
        rows = []
        for task in [TASK] + ['other_' + str(i) for i in range(13)]:
            for i in range(100):
                for teacher in (True, False):
                    behavior = 'TRAIN_teacher_committed_waypoint' if teacher else 'TRAIN_hold_motion_identification'
                    rows.append(dict(case_id=f'{task}:{i}:{teacher}', task=task, split='train',
                        raw_episode_id=f'episode{i}__{behavior}_r0', parent_raw_episode_id=f'episode{i}',
                        parent_source_group=f'train:{task}:episode{i}', behavior=behavior,
                        replica=0, behavior_replica=0, maximum_collection_ticks=200 if teacher else 60,
                        teacher_used=teacher, context_primary_view=not teacher, interaction_eligible=teacher,
                        source=dict(task=task, raw_episode_id=f'episode{i}', path=f'CPU_fixture/{i}'),
                        template_asset='CPU_fixture_geometry.json'))
        return dict(split='train', cases=rows)

    def test_all100_parents_get_distinct_new_id_and_original_source_and_fold(self):
        primary = self.primary(); original = deepcopy(primary); cases = expected_cases(primary)
        self.assertEqual(primary, original)
        self.assertEqual(len(cases), 100)
        self.assertEqual(len({r['case_id'] for r in cases}), 100)
        self.assertTrue(all(r['replica'] == 1 and r['context_primary_view'] is False for r in cases))
        for case in cases:
            before = next(r for r in primary['cases'] if r['task'] == TASK
                          and r['parent_raw_episode_id'] == case['parent_raw_episode_id'] and r['teacher_used'])
            self.assertEqual(case['parent_source_group'], before['parent_source_group'])
            self.assertEqual(case['source'], before['source'])
            self.assertEqual(case['template_asset'], before['template_asset'])

    def test_missing_parent_duplicate_parent_or_changed_primary_contract_rejected(self):
        for corruption in ('missing', 'duplicate', 'Context', 'budget'):
            primary = self.primary()
            if corruption == 'missing': primary['cases'][0]['task'] = 'unknown'
            elif corruption == 'duplicate': primary['cases'][2]['parent_source_group'] = primary['cases'][0]['parent_source_group']
            elif corruption == 'Context': primary['cases'][0]['context_primary_view'] = True
            else: primary['cases'][0]['maximum_collection_ticks'] = 201
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                expected_cases(primary)


if __name__ == '__main__':
    unittest.main()
