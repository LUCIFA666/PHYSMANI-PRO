"""CPU-only checks of native terminal semantics and privileged-data boundaries."""
from types import SimpleNamespace

import numpy as np
import unittest

from our_mani.multitask.specs import TASK_SPECS, get_task_spec
from our_mani.multitask.native_tasks import (
    NativeAuditError, NativeOutcomeLatch, metric_initial_snapshot, native_result,
    official_reset_plan, validate_archived_motion_state,
)



class Body:
    serial = 0

    def __init__(self, name):
        Body.serial += 1
        self.name, self.handle = name, Body.serial

    def get_name(self):
        return self.name

    def get_handle(self):
        return self.handle

    def get_pose(self):
        return [1., 2., 3., 0., 0., 0., 1.]

    def get_velocity(self):
        return [0., .2, 0.], [0., 0., .5]

    def get_objects_in_tree(self):
        class ExpertWaypoint(Body):
            def get_pose(self):
                raise AssertionError('audit must not read expert waypoint pose')
        return [ExpertWaypoint('waypoint_unavailable_to_policy')]

class TestNativeContracts(unittest.TestCase):

    def test_registry_distinguishes_manipulated_body_from_recorded_motion_label(self):
        assert len(TASK_SPECS) == 14
        assert len({spec.family for spec in TASK_SPECS.values()}) == 7
        bin_spec = get_task_spec('put_rubbish_in_moving_bin')
        assert bin_spec.primary_manipulated_roles == ('rubbish',)
        assert bin_spec.target_label.role == 'bin'
        ball_spec = get_task_spec('moving_basketball_in_hoop')
        assert ball_spec.target_label.dropped_roles == ('hoop',)
        assert get_task_spec('remove_cups_from_rotating_frame').target_label.identity_may_change_within_stage
        for spec in TASK_SPECS.values():
            public = spec.policy_contract()
            assert 'roles' not in public and 'dynamic_roots' not in public
            assert 'native_success_all' not in public and 'target_label' not in public

    def test_author_reset_exceptions_are_ordered_without_executing_any_task(self):
        button = official_reset_plan('push_moving_button_high_speed', 1)
        actions = [step.action for step in button.steps]
        assert actions.index('load_archived_motion_state_before_reset_to_demo') < actions.index('reset_to_demo')
        assert button.steps[0].action == 'set_variation' and button.steps[0].args == (1,)
        peg = official_reset_plan('insert_onto_rotating_peg', 2)
        init = next(step for step in peg.steps if step.action == 'init_episode')
        assert init.args == (2,) and dict(init.kwargs) == {'bool_random_place': False}
        with self.assertRaises(NativeAuditError):
            official_reset_plan('remove_cups_from_rotating_frame', 1)
        driver = {'x': [0., 0., .8], 'v': [0., .1, 0.], 'a': [0., 0., 0.]}
        with self.assertRaisesRegex(NativeAuditError, 'separate ball and hoop'):
            validate_archived_motion_state('moving_basketball_in_hoop', 0, {0: [driver]})
        assert validate_archived_motion_state('moving_basketball_in_hoop', 0,
                                              {0: [driver, driver]})['driver_count'] == 2

    def test_fatal_native_outcome_cannot_be_recovered_into_success(self):
        task = SimpleNamespace(success=lambda: (np.bool_(False), np.bool_(True)))
        failure = native_result(task)
        assert failure.terminal_failure and failure.terminate and not failure.success
        latch = NativeOutcomeLatch()
        assert latch.update(failure) == failure
        task.success = lambda: (True, True)
        assert latch.update(native_result(task)) == failure

    def test_metric_snapshot_is_complete_copied_and_not_an_observation_mapping(self):
        driver = {'x': [0., 0., .8], 'v': [0., .1, 0.], 'a': [0., 0., 0.], 't0': 0.}
        task = SimpleNamespace(ball=Body('ball'), hoop=Body('hoop'), t=.05, step_id=1,
                               var_index=0, stage='wp0', _has_been_picked=False,
                               target_state_list=[(driver, driver)],
                               var2target_state_list={0: [driver, driver]},
                               success=lambda: (False, False))
        snapshot = metric_initial_snapshot(task, 'moving_basketball_in_hoop',
                                           scene_object_resolver=Body, physics_time=lambda: 1.25)
        report = snapshot.to_report()
        assert report['metric_only'] and report['policy_use_allowed'] is False
        assert report['bodies']['hoop']['angular_velocity_rad_s'] == [0., 0., .5]
        assert len(report['active_motion_schedule']) == 2
        assert report['task_time_seconds'] == .05 and report['task_step_id'] == 1
        assert report['skipped_expert_waypoint_dummies'] == ['waypoint_unavailable_to_policy']
        assert report['full_native_physics_state_claimed'] is False
        report['active_motion_schedule'][0]['v'][0] = 999.
        assert task.target_state_list[0][0]['v'][0] == 0.
        assert snapshot.to_report()['active_motion_schedule'][0]['v'][0] == 0.
        with self.assertRaises(TypeError):
            {}.update(snapshot)
        clock = iter([1.25, 1.30])
        with self.assertRaisesRegex(NativeAuditError, 'physics clock'):
            metric_initial_snapshot(task, 'moving_basketball_in_hoop',
                                    scene_object_resolver=Body, physics_time=lambda: next(clock))

if __name__ == '__main__':
    unittest.main()
