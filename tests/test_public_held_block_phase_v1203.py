"""Synthetic public frames and original fake-v7 CPU tests; no Native task."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from our_mani.benchmark_perception import BoxPrior, box_surface
from our_mani.contracts import ObjectState, Pose, RobotState
from our_mani.multitask import public_held_block_phase_v1203 as new
from our_mani.multitask.block_held_support_v298 import supported_box
from our_mani.multitask.physmani_block_policy_v538 import BlockGeometryIntents
from tests import test_retained_preclose_block_integration_v1153 as old_fixture


CONFIG = dict(dt=.05, baseline_decision_ticks=10, warmup_ticks=1,
              local_max_speed=.5, local_max_acceleration=3.,
              forecast_seconds=1., preposition_height=1.)


def robot(when=0., *, contact=False, position=(.15, 0., .82)):
    return RobotState(when, Pose(position), np.zeros(3), np.zeros(3),
                      np.zeros(7), np.zeros(7), gripper_open=not contact, contact=contact)


def measured(when, *, position=(.15, 0., .775), points=None):
    pose = Pose(position)
    cloud = box_surface(pose, BoxPrior().dimensions, 96) if points is None else points
    return ObjectState('target', when, pose, cloud, np.zeros(3), np.zeros(3))


def collision():
    # Bypass only the Native/static-map constructor in these CPU fixtures.
    c = object.__new__(new.PublicHeldBlockCollision)
    c.held_local = None
    c.attempted = False
    c.events = []
    c.queries = 0
    c.last_contact_frame = None
    c.held_admission_events = []
    c.capture_epoch = 0
    return c


def frontend(raw=None):
    f = object.__new__(new.PublicHeldBlockFrontend)
    f.config = dict(CONFIG)
    f.task = 'pick_moving_target_on_the_table_high_speed'
    f.method = 'cv_geometry'
    f.model = None
    f.collision = collision()
    f.tracker = NS(prior=BoxPrior(), observe=Mock(return_value=(raw, {})))
    f.marker = NS(observe=Mock(return_value=(Pose([.3, 0., 1.]), {})))
    forecast = NS(times=np.array([0., 1.]), at=Mock(return_value=(Pose([.15, 0., .775]), None, None)))
    f.dynamics = NS(observe=Mock(), predict=Mock(return_value=forecast))
    f.robot = f.object = f.forecast = None
    f.goal_pose = Pose([.3, 0., 1.])
    f.close_confirmed = False
    f.public_capture_seen = False
    f.public_capture_lost = False
    f.frames = 0
    f.events = []
    f.last_time = None
    return f


def observe(f, when, *, contact=False, raw=None, position=(.15, 0., .82)):
    f.tracker.observe.return_value = (raw, {})
    f.observe({'timestamp': when}, robot(when, contact=contact, position=position))


def confirmed_close(p):
    # This callback is invoked only after successful apply by the unchanged
    #1114 executor.  Separate fake-v7 tests below exercise that order itself.
    p.confirm_opening(0., dict(phase='acquire', actually_dispatched=True),
                      dict(issued_at=0., distance_m=.01, angle_rad=0.), confirmed_after=.05)


class PublicFrameTests(unittest.TestCase):
    def test_held_quality_uses_original_exact_surface_without_support_or_Z_shift(self):
        raw = measured(.05)
        half = np.asarray(BoxPrior().dimensions)/2
        vertices = np.array([[x, y, z] for x in (-half[0], half[0])
                            for y in (-half[1], half[1]) for z in (-half[2], half[2])])
        plane = dict(height=.75, xy_min=np.array([-.1, -.1]), xy_max=np.array([.1, .1]))
        _, old = supported_box(raw.pose, raw.points_world, vertices, plane, BoxPrior())
        self.assertFalse(old['accepted'])
        accepted, proof = new.qualify_observed_inhand_box(raw.pose, raw.points_world, BoxPrior())
        self.assertIs(accepted, raw.pose)
        self.assertTrue(proof['accepted'])
        self.assertFalse(proof['support_plane_used'])
        self.assertFalse(proof['support_XY_footprint_used'])
        self.assertEqual(proof['z_correction_m'], 0.)

    def test_held_below_plane_points_do_not_require_table_support(self):
        raw = measured(.05, position=(.15, 0., .70))
        accepted, proof = new.qualify_observed_inhand_box(raw.pose, raw.points_world, BoxPrior())
        self.assertIs(accepted, raw.pose)
        self.assertTrue(proof['accepted'])
        self.assertEqual(proof['z_correction_m'], 0.)

    def test_surface_RMS_and_inlier_inequalities_match298_exactly(self):
        pose = Pose([0., 0., .8])
        good = pose.apply(np.tile([.025, 0., 0.], (13, 1)))
        bad = pose.apply(np.tile([.10, 0., 0.], (7, 1)))
        points = np.r_[good, bad]
        half = np.array([.025]*3)
        vertices = np.array([[x, y, z] for x in (-half[0], half[0])
                            for y in (-half[1], half[1]) for z in (-half[2], half[2])])
        plane = dict(height=.7, xy_min=np.array([-.5, -.5]), xy_max=np.array([.5, .5]))
        _, old = supported_box(pose, points, vertices, plane, BoxPrior())
        _, proof = new.qualify_observed_inhand_box(pose, points, BoxPrior())
        for key in ('accepted', 'surface_rms_m', 'inlier_fraction'):
            self.assertEqual(proof[key], old[key])
        self.assertTrue(proof['accepted'])
        self.assertEqual(proof['inlier_fraction'], .65)

    def test_invalid_small_nonfinite_RMS_or_inlier_cloud_stays_unknown(self):
        pose = Pose([0., 0., .8])
        good = pose.apply(np.tile([.025, 0., 0.], (12, 1)))
        cases = [np.zeros((12, 2)), good[:11], np.r_[good[:11], [[np.nan, 0., 0.]]],
                 pose.apply(np.tile([.04, 0., 0.], (12, 1))),
                 np.r_[good[:6], pose.apply(np.tile([.10, 0., 0.], (6, 1)))],
                 pose.apply(np.tile([.10, 0., 0.], (12, 1))), None]
        for points in cases:
            with self.subTest(shape=np.shape(points)):
                accepted, proof = new.qualify_observed_inhand_box(pose, points, BoxPrior())
                self.assertIsNone(accepted)
                self.assertFalse(proof['accepted'])
        accepted, proof = new.qualify_observed_inhand_box(pose, good, BoxPrior())
        self.assertIs(accepted, pose)
        self.assertEqual(proof['minimum_visible_points'], 12)
        self.assertEqual(proof['maximum_registration_rms_m'], .012)
        self.assertEqual(proof['minimum_inlier_fraction'], .65)
        with self.assertRaises(new.retained.ProtocolInvariantError):
            new.qualify_observed_inhand_box(pose, good, NS(dimensions=(.06, .05, .05)))

    def test_failed_current_quality_enters_original_lift_then_next_frame_admits_TF(self):
        f = frontend()
        p = new.PublicHeldBlockIntents(f)
        p.phase = 'acquire'
        confirmed_close(p)
        invalid = measured(.05, points=np.tile([.4, .4, 1.2], (12, 1)))
        observe(f, .05, contact=True, raw=invalid)
        self.assertIsNone(f.collision.held_local)
        first = p.select()
        self.assertEqual(p.phase, 'lift')
        np.testing.assert_array_equal(first[:3], f.robot.pose.position+[0., 0., .1])
        self.assertEqual(first[7], 0.)
        np.testing.assert_array_equal(p.lift_goal.rotation, f.robot.pose.rotation)
        self.assertIsNone(p.transport_goal)
        self.assertIsNone(f.forecast)
        lift_goal = p.lift_goal
        observe(f, .10, contact=True, raw=measured(.10), position=lift_goal.position)
        self.assertIsNotNone(f.collision.held_local)
        action = p.select()
        self.assertEqual(p.phase, 'transport')
        local = f.collision.held_local
        np.testing.assert_array_equal(action[:3], f.goal_pose.position-f.goal_pose.rotation@local.position)
        self.assertEqual(action[7], 0.)
        self.assertEqual([row['accepted'] for row in f.collision.held_admission_events], [False, True])
        self.assertEqual(f.tracker.observe.call_count, 2)
        f.dynamics.observe.assert_not_called()
        f.dynamics.predict.assert_not_called()

    def test_lift_arrival_without_TF_cannot_transport_or_release(self):
        f = frontend()
        p = new.PublicHeldBlockIntents(f)
        p.phase = 'acquire'
        confirmed_close(p)
        observe(f, .05, contact=True)
        action = p.select()
        self.assertEqual(action[7], 0.)
        frozen = p.lift_goal
        observe(f, .10, contact=True, position=frozen.position)
        self.assertIsNone(p.select())
        self.assertEqual(p.phase, 'lift')
        self.assertIs(p.lift_goal, frozen)
        self.assertIsNone(p.transport_goal)
        self.assertTrue(f.close_confirmed)

    def test_published_grasp_without_applied_own_close_never_advances_or_fits(self):
        f = frontend()
        p = new.PublicHeldBlockIntents(f)
        p.phase = 'acquire'
        observe(f, .05, contact=True, raw=measured(.05))
        with patch.object(new, 'qualify_observed_inhand_box', wraps=new.qualify_observed_inhand_box) as fit:
            self.assertIsNone(p.select())
            fit.assert_not_called()
        self.assertEqual(p.phase, 'acquire')
        self.assertIsNone(f.collision.held_local)
        self.assertEqual(f.collision.held_admission_events, [])

    def test_proposed_close_is_not_confirmation_and_delayed_grasp_is_preserved(self):
        f = frontend()
        p = new.PublicHeldBlockIntents(f)
        p.phase = 'acquire'
        observe(f, .0, raw=measured(.0))
        old_fixture.fresh_object(f, p, distance=.01)
        _, dispatched, _ = new.retained.parent.adapt(CONFIG, f, p)
        active = new.PublicHeldBlockWitness(dispatched)
        command, proof, proposed = active.qualify_opening(0., 1.)
        self.assertIsNotNone(proposed)
        self.assertFalse(f.close_confirmed)
        f.forecast = None
        f.robot = replace(f.robot, contact=True)
        self.assertIsNone(active.select())
        self.assertEqual(p.phase, 'acquire')
        active.confirm_opening(command, proof, proposed, confirmed_after=.05)
        observe(f, .05, contact=False, raw=measured(.05))
        self.assertTrue(f.close_confirmed)
        self.assertFalse(f.public_capture_lost)
        self.assertEqual(p.phase, 'acquire')
        observe(f, .10, contact=True, raw=measured(.10))
        self.assertIsNotNone(active.select())
        self.assertEqual(p.phase, 'lift')

    def test_same_counted_frame_never_refits_and_next_frame_can_retry(self):
        c = collision()
        r = robot(.05, contact=True)
        with patch.object(new, 'qualify_observed_inhand_box', wraps=new.qualify_observed_inhand_box) as fit:
            bad = measured(.05, points=np.tile([.4, .4, 1.2], (12, 1)))
            c.observe_contact(r, bad, BoxPrior(), True)
            c.observe_contact(r, measured(.05), BoxPrior(), True)
            self.assertEqual(fit.call_count, 1)
            self.assertIsNone(c.held_local)
            c.observe_contact(robot(.10, contact=True), measured(.10), BoxPrior(), True)
            self.assertEqual(fit.call_count, 2)
            self.assertIsNotNone(c.held_local)
        with self.assertRaises(new.retained.ProtocolInvariantError):
            c.observe_contact(robot(.0, contact=True), measured(.0), BoxPrior(), True)

    def test_original_pipeline_rejects_duplicate_frame_before_any_extra_observation(self):
        f = frontend()
        f.close_confirmed = True
        observe(f, .05, contact=True, raw=measured(.05))
        old = deepcopy(f.collision.held_admission_events)
        with self.assertRaises(ValueError):
            observe(f, .05, contact=True, raw=measured(.05))
        self.assertEqual(f.tracker.observe.call_count, 1)
        self.assertEqual(f.collision.held_admission_events, old)

    def test_stale_wrong_role_missing_cloud_do_not_make_TF_but_retry_is_allowed(self):
        c = collision()
        c.observe_contact(robot(.05, contact=True), measured(.0), BoxPrior(), True)
        c.observe_contact(robot(.10, contact=True), replace(measured(.10), object_id='other_public_role'), BoxPrior(), True)
        c.observe_contact(robot(.15, contact=True), None, BoxPrior(), True)
        self.assertIsNone(c.held_local)
        c.observe_contact(robot(.20, contact=True), measured(.20), BoxPrior(), True)
        self.assertIsNotNone(c.held_local)
        self.assertEqual([row['accepted'] for row in c.held_admission_events], [False, False, False, True])

    def test_contact_loss_clears_TF_and_does_not_resume_from_historical_close(self):
        f = frontend()
        p = new.PublicHeldBlockIntents(f)
        p.phase = 'acquire'
        confirmed_close(p)
        observe(f, .05, contact=True, raw=measured(.05))
        p.select()
        witness = deepcopy(p.close_witness)
        observe(f, .10, contact=False, raw=measured(.10))
        self.assertIsNone(f.collision.held_local)
        self.assertFalse(f.close_confirmed)
        self.assertTrue(f.public_capture_lost)
        self.assertIsNone(p.select())
        observe(f, .15, contact=True, raw=measured(.15))
        self.assertIsNone(f.collision.held_local)
        self.assertIsNone(p.select())
        self.assertEqual(p.close_witness, witness)

    def test_release_confirmation_then_explicit_reset_clear_capture_state(self):
        f = frontend()
        p = new.PublicHeldBlockIntents(f)
        p.phase = 'acquire'
        confirmed_close(p)
        observe(f, .05, contact=True, raw=measured(.05))
        p.select()
        p.confirm_opening(1., dict(release=True), None, confirmed_after=.10)
        self.assertEqual(p.phase, 'released')
        self.assertFalse(f.close_confirmed)
        self.assertIsNone(f.collision.held_local)
        self.assertIsNone(p.lift_goal)
        self.assertIsNone(p.transport_goal)
        observe(f, .10, contact=True, raw=measured(.10))
        self.assertIsNone(p.select())
        self.assertIsNone(f.collision.held_local)
        p.reset_public_capture()
        self.assertEqual(p.phase, 'approach')
        self.assertIsNone(p.close_witness)
        self.assertIsNone(p.selected)
        self.assertFalse(f.public_capture_lost)
        self.assertIsNone(f.collision.last_contact_frame)
        self.assertEqual(f.collision.capture_epoch, 1)

    def test_admitted_TF_is_not_refit_during_same_continuous_capture(self):
        c = collision()
        c.observe_contact(robot(.05, contact=True), measured(.05), BoxPrior(), True)
        local = c.held_local
        with patch.object(new, 'qualify_observed_inhand_box', side_effect=AssertionError('no refit of admitted capture')):
            c.observe_contact(robot(.10, contact=True), measured(.10), BoxPrior(), True)
        self.assertIs(c.held_local, local)

    def test_normal_valid_TF_lift_transport_actions_match_frozen538(self):
        old = frontend()
        current = frontend()
        for f in (old, current):
            f.robot = robot(.05, contact=True)
            f.frames = 2
            f.close_confirmed = True
            f.collision.held_local = Pose([0., 0., -.05])
        baseline = BlockGeometryIntents(old)
        candidate = new.PublicHeldBlockIntents(current)
        for p in (baseline, candidate):
            p.phase = 'acquire'
            p.close_witness = dict(issued_at=0., confirmed_after=.05)
        np.testing.assert_array_equal(candidate.select(), baseline.select())
        self.assertEqual(candidate.phase, baseline.phase)
        for f, p in ((old, baseline), (current, candidate)):
            f.robot = robot(.10, contact=True, position=p.lift_goal.position)
        np.testing.assert_array_equal(candidate.select(), baseline.select())
        self.assertEqual(candidate.phase, baseline.phase)

    def test_pregrasp_forecast_and_original_qualifier_are_inherited(self):
        self.assertIs(new.PublicHeldBlockIntents.qualify_opening, BlockGeometryIntents.qualify_opening)
        f = frontend()
        observe(f, .0, raw=measured(.0))
        self.assertEqual(f.dynamics.observe.call_count, 1)
        self.assertEqual(f.dynamics.predict.call_count, 1)
        f.tracker.observe.assert_called_once_with({'timestamp': .0})

    def test_active_receipt_labels_changed_public_phase_without_native_claim(self):
        f = frontend()
        p = new.PublicHeldBlockIntents(f)
        observe(f, .0, raw=measured(.0))
        _, dispatched, _ = new.retained.parent.adapt(CONFIG, f, p)
        receipt = new.PublicHeldBlockWitness(dispatched).receipt()
        self.assertFalse(receipt['public_held_block_phase_v1203']['native_success_claimed'])
        self.assertFalse(receipt['held_geometry']['original_held_support_geometry_reused'])
        self.assertFalse(receipt['retained_preclose_block_integration_v1153']['original_frontend_support_phase_and_native_actuator_unchanged'])
        self.assertTrue(receipt['retained_preclose_block_integration_v1153']['original_preclose_and_native_actuator_unchanged'])
        self.assertFalse(receipt['held_geometry']['held_target_vs_arm_and_static_obstacles_checked'])

    def test_explicit_factory_constructs_once_and_restores_same_executor_facade(self):
        env = NS(robot_model=object())
        f = frontend()
        p = new.PublicHeldBlockIntents(f)
        _, parent, executor = new.retained.parent.adapt(CONFIG, f, p)
        events = []
        with patch.object(new, 'PublicHeldBlockCollision', side_effect=lambda e: events.append('collision') or f.collision), \
             patch.object(new, 'PublicHeldBlockFrontend', side_effect=lambda *args: events.append('frontend') or f), \
             patch.object(new, 'PublicHeldBlockIntents', side_effect=lambda arg: events.append('policy') or p), \
             patch.object(new.retained.parent, 'adapt', return_value=(f, parent, executor)) as adapt:
            with self.assertRaisesRegex(RuntimeError, 'owned exit'):
                with new.integrated_block_components(Path('/fixture'), CONFIG, new.retained.parent.BLOCKS[0], 'cv_geometry', env) as result:
                    self.assertIs(result[0], f)
                    self.assertIs(result[1].source_policy, p)
                    self.assertIs(executor.skill_policy, result[1])
                    raise RuntimeError('owned exit')
            self.assertIs(executor.skill_policy, parent)
            self.assertEqual(events, ['collision', 'frontend', 'policy'])
            adapt.assert_called_once_with(CONFIG, f, p)
        with self.assertRaises(ValueError):
            new.construct(None, CONFIG, 'unknown_task', 'cv_geometry', env)

    def test_binding_pins_original_parents_and_declares_no_new_numeric_quality(self):
        root = Path(__file__).resolve().parents[1]
        with patch.object(new.retained, 'binding', return_value={'frozen1153': True}) as parent:
            receipt = new.binding(root, '/not_constructed_native_repo')
        parent.assert_called_once_with(root, '/not_constructed_native_repo')
        self.assertEqual(receipt['parent'], {'frozen1153': True})
        self.assertFalse(receipt['exact_box_quality']['numerical_thresholds_changed'])
        self.assertEqual(receipt['original_known_axis_box_dimensions_m'], [.05]*3)
        self.assertEqual(receipt['additional_native_getters_observations_predictions_or_success_calls'], 0)


class OriginalExecutorTests(unittest.TestCase):
    def setUp(self):
        self.fixture = old_fixture.Tests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def objects(self, env):
        original, _ = old_fixture.fixture.policy_for(env)
        f = frontend()
        f.config.update(original.config)
        f.robot = original.robot
        p = new.PublicHeldBlockIntents(f)
        p.phase = 'acquire'
        def update(observation, observed):
            f.robot = old_fixture.fixture.robot(observed.timestamp)
            old_fixture.fresh_object(f, p, distance=.01)
        f.observe = update
        update({}, NS(timestamp=env.time))
        old_fixture.choose(p, env)
        _, dispatched, executor = new.retained.parent.adapt(f.config, f, p)
        active = new.PublicHeldBlockWitness(dispatched)
        executor.skill_policy = active
        return f, p, active, executor

    def test_original_full_MRO_effects_getters_and_single_qualifier_match1153(self):
        baseline = old_fixture.Env()
        env = old_fixture.Env()
        _, _, _, _, old_executor = self.fixture.objects(baseline)
        f, p, active, executor = self.objects(env)
        with patch.object(p, 'qualify_opening', wraps=p.qualify_opening) as raw, \
             patch.object(p, 'confirm_opening', wraps=p.confirm_opening) as confirm:
            self.fixture.base.run_action(baseline, old_executor)
            self.fixture.base.run_action(env, executor)
        self.fixture.base.equal_native_effects(baseline, env)
        self.assertEqual(env.counts, baseline.counts)
        self.assertEqual(env.arm.calls, baseline.arm.calls)
        self.assertEqual((raw.call_count, confirm.call_count), (10, 10))
        self.assertTrue(f.close_confirmed)
        self.assertIsNotNone(p.close_witness)
        self.assertIsNone(executor.admit_joint_target)
        self.assertEqual(executor.collision_diagnostic_limit, 0)

    def test_failed_apply_cannot_confirm_or_enable_lift(self):
        env = old_fixture.Env(apply_error=new.retained.ProtocolInvariantError('original apply failed'))
        f, p, _, executor = self.objects(env)
        with patch.object(p, 'confirm_opening', wraps=p.confirm_opening) as confirm:
            with self.assertRaises(new.retained.ProtocolInvariantError):
                self.fixture.base.run_action(env, executor)
        confirm.assert_not_called()
        self.assertFalse(f.close_confirmed)
        self.assertIsNone(p.close_witness)

    def test_native_fake_release_effect_is_unchanged_but_capture_state_invalidates(self):
        env = old_fixture.Env()
        f, p, _, executor = self.objects(env)
        self.fixture.base.run_action(env, executor)
        self.assertTrue(f.close_confirmed)
        action = old_fixture.choose(p, env, opening=1., phase='transport')
        self.fixture.base.run_action(env, executor, action=action)
        self.assertEqual(env.ledger.release_requests, 1)
        self.assertEqual(env._requested_opening, 1.)
        self.assertEqual(p.phase, 'released')
        self.assertFalse(f.close_confirmed)
        self.assertIsNone(f.collision.held_local)


if __name__ == '__main__':
    unittest.main()
