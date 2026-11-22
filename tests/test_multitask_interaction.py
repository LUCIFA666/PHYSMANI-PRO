"""Synthetic CPU checks: observed rigid frames, constraints and outcome scope."""
from pathlib import Path
from dataclasses import replace
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import ObjectState, Pose, RobotState
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.reachability import TimedReachability
from our_mani.multitask.candidate_success import (
    ContinuationAttempts, FEATURE_NAMES, FEATURE_SCHEMA_SHA256, LABEL_SCOPE,
    CONTINUATION_POLICY, candidate_features, select_candidate,
)
from our_mani.multitask.interaction import (
    CandidateConfig, CandidateGenerator, InteractionArtifactError, InteractionController,
    InteractionFitConfig, InteractionLibrary, MPCConfig, PolicyState, SE3InteractionMPC,
    TRAINBodyTrack, TRAINInteractionSource, TRAINReferenceBinding, fit_interaction_library,
)


def synthetic_sources(kind='observed_se3'):
    sources = []
    times = np.arange(11)*.05
    for index, angle in enumerate((0., .7)):
        rotation = Rotation.from_euler('z', angle).as_matrix()
        base = np.array([index*.4, index*.2, 0.])
        body_poses, tcp = [], []
        for frame in range(len(times)):
            relative = np.array([0., 0., .1 if frame < 5 else .05])
            body_poses.append(Pose(base, rotation).xyzw())
            if kind == 'translation_only':
                tcp.append(Pose(base+relative).xyzw())
            else:
                tcp.append(Pose(base+rotation@relative, rotation).xyzw())
        track = TRAINBodyTrack('button', times, np.asarray(body_poses), ('external',)*len(times),
                               np.full(len(times), kind != 'translation_only'))
        binding = TRAINReferenceBinding('button', 'button', kind, 'one_TRAIN_anchor', 'button')
        sources.append(TRAINInteractionSource('push_moving_button', 'synthetic'+str(index),
            'train', 'synthetic', times, np.asarray(tcp), np.zeros(len(times)), np.asarray(tcp),
            ('wp0',)*5+('wp1',)*6, (binding,)*len(times), {'button': track}))
    return sources


def synthetic_library(kind='observed_se3'):
    return fit_interaction_library(synthetic_sources(kind),
        InteractionFitConfig(3, (0.,), 1., .9, 2, .01, .05, 1.))


class CartesianRobot:
    name = 'synthetic_cartesian_joint_fixture'
    lower, upper = np.full(7, -10.), np.full(7, 10.)
    velocity_limits, acceleration_limits = np.full(7, 10.), np.full(7, 100.)

    def inverse(self, pose, seed):
        # Pose rotations are immutable; SciPy 1.10 requires a writable buffer.
        return np.r_[pose.position, Rotation.from_matrix(pose.rotation.copy()).as_rotvec(), 0.]

    def forward(self, q):
        return Pose(q[:3], Rotation.from_rotvec(q[3:6]).as_matrix())

    def jacobian(self, q):
        return np.c_[np.eye(6), np.zeros(6)]

    def collision(self, q, future_offset):
        return False


def state(time=0., robot_z=.1, *, angle=0., observable=True):
    pose = Pose(np.zeros(3), Rotation.from_euler('z', angle).as_matrix())
    body = ObjectState('button', time, pose, np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]]),
                       np.zeros(3), np.array([0., 0., 1.]) if not observable else np.zeros(3))
    robot_pose = Pose(np.array([0., 0., robot_z]))
    robot = RobotState(time, robot_pose, np.zeros(3), np.zeros(3),
                       np.r_[robot_pose.position, np.zeros(4)], np.zeros(7))
    return PolicyState(robot, {'button': body}, {'bodies': {'button': dict(
        directly_visible=True, registration_valid=True, orientation_observable=observable,
        motion_mode='external',
        rotation_gauge=None if observable else 'sphere_gauge')}} , {'button': ('button',)})


def candidates_config():
    return CandidateConfig((.2, .4), .05, 1., 2., .1, .2, .1, .02, .02, 1.)


def mpc_config():
    return MPCConfig(.05, 3, 1., 10., 2., 20., 50., 10., .05, 100)


class TestObservedInteraction(unittest.TestCase):
    def test_value_fallback_consumes_one_query_then_all_phases_use_geometry(self):
        library = synthetic_library()
        class Critic:
            metadata = dict(continuation_sha256='test_controller',interaction_library_sha256=library.sha256,
                feature_schema_sha256=FEATURE_SCHEMA_SHA256,label_scope=LABEL_SCOPE,
                continuation_selection_policy=CONTINUATION_POLICY)
            calls = 0
            def predict(self,x,**kwargs):
                self.calls += 1
                return dict(probability=np.full(len(x),.7),supported=np.ones(len(x),bool),
                            ensemble_std=np.zeros(len(x)),ranking_enabled=False)
        critic = Critic()
        generator = CandidateGenerator(library,TimedReachability(CartesianRobot()),
            lambda **kw:dict(feasible=True,evidence={}),candidates_config(),allow_synthetic=True)
        controller = InteractionController(library,generator,SE3InteractionMPC(mpc_config()),
            predictor='cv',ranking='value',continuation_sha256='test_controller',
            success_model=critic,allow_synthetic=True)
        for tick in range(20):
            observed = state(tick*.05,robot_z=.1 if controller.phase_index == 0 else .05)
            forecast = ConstantVelocityDynamics().predict(list(observed.bodies.values()),1.,.05)
            controller.step(observed,forecast,10.-tick*.05)
            if controller.phase_index == 2:
                break
        self.assertEqual(controller.phase_index,2)
        self.assertEqual(critic.calls,1)
        records = controller.attempts.to_report()
        self.assertEqual(records[0]['predicted_probability'],.7)
        self.assertIsNone(records[1]['predicted_probability'])
        self.assertEqual(len(controller.selection_events),len(records))
        self.assertEqual(sum(e['value_query_this_selection'] for e in controller.selection_events),1)
        self.assertFalse(any(e['actual_override'] for e in controller.selection_events))
        self.assertEqual(controller.selection_events[0]['success_probability'],.7)
        self.assertTrue(all(e['selected_candidate_id']==e['candidate_ids'][e['selected_index']]
            and e['geometry_candidate_id']==e['candidate_ids'][e['geometry_index']] for e in controller.selection_events))

    def test_native_optional_trailing_release_can_be_absent_but_insert_cannot(self):
        sources = []
        stages = tuple(stage for stage in ('wp0','wp1-grasp','wp2','wp3','wp4') for _ in range(2))
        times = np.arange(len(stages))*.05
        poses = np.tile([0.,0.,0.,0.,0.,0.,1.],(len(times),1))
        binding = TRAINReferenceBinding('ring','ring','observed_se3','trained_anchor','ring')
        for index in range(2):
            track = TRAINBodyTrack('ring',times,poses,('external',)*len(times),np.ones(len(times),bool))
            sources.append(TRAINInteractionSource('insert_onto_rotating_peg',str(index),
                'train','synthetic',times,poses,np.zeros(len(times)),poses,stages,
                (binding,)*len(times),{'ring':track}))
        config = InteractionFitConfig(3,(0.,),1.,.9,2,.01,.05,1.)
        library = fit_interaction_library(sources,config)
        self.assertEqual(library.phase_order,('approach','acquire','lift','align','insert'))
        self.assertEqual([row['phase'] for row in library.metadata['omitted_optional_phases']],['release'])
        with self.assertRaises(InteractionArtifactError):
            fit_interaction_library([replace(source,stages=tuple('wp3' if stage == 'wp4' else stage
                                     for stage in source.stages)) for source in sources],config)

    def test_parent_grouped_augmentations_and_mode_boundaries(self):
        sources = []
        for original in synthetic_sources():
            for replica in range(2):
                source = replace(original, source_id=original.source_id+'rep'+str(replica),
                    raw_episode_id=original.source_id+'rep'+str(replica),
                    parent_raw_episode_id=original.source_id,
                    parent_source_group='train:push_moving_button:'+original.source_id)
                sources.append(source)
        library = fit_interaction_library(sources, InteractionFitConfig(3, (0.,), 1., .9, 2, .01, .05, 1.))
        self.assertEqual(len(library.metadata['TRAIN_source_ids']), 4)
        self.assertEqual(len(library.metadata['TRAIN_parent_source_groups']), 2)
        for diagnostic in library.metadata['source_grouped_lag_selection'].values():
            self.assertEqual(len(diagnostic['source_to_fold']), 2)
            self.assertEqual(len(diagnostic['fitted_rollouts']), 4)
        original = synthetic_sources()[0].body_tracks['button']
        switched = replace(original, motion_modes=('external',)*5+('attached',)*6)
        switched.at(switched.times[5], orientation_required=True, same_mode_as='attached')
        with self.assertRaises(InteractionArtifactError):
            switched.at((switched.times[4]+switched.times[5])/2,
                        orientation_required=True, same_mode_as='attached')
        switched_sources = [replace(source, body_tracks={'button': replace(
            source.body_tracks['button'], motion_modes=('external',)*8+('attached',)*3)})
            for source in synthetic_sources()]
        split_library = fit_interaction_library(switched_sources,
            InteractionFitConfig(3, (0.,), 1., .9, 2, .01, .05, 1.))
        press = split_library.for_phase('press')
        self.assertEqual({template.reference_mode for template in press}, {'external','attached'})
        self.assertTrue(all(template.duration_seconds <= .15+1e-8 for template in press))

    def test_world_anchor_uses_fitted_world_pose_without_a_goal_sensor(self):
        sources = []
        for source in synthetic_sources():
            references = tuple(replace(binding, reference_id=None, reference_role='public_world',
                reference_kind='world') for binding in source.references)
            sources.append(replace(source, references=references, body_tracks={}))
        library = fit_interaction_library(sources, InteractionFitConfig(3, (0.,), 1., .9, 2, .01, .05, 1.))
        observed = state()
        forecast = ConstantVelocityDynamics().predict(list(observed.bodies.values()), 1., .05)
        generator = CandidateGenerator(library, TimedReachability(CartesianRobot()),
            lambda **kwargs: dict(feasible=True, evidence={}), candidates_config(), allow_synthetic=True)
        candidates = generator.generate(observed, forecast, 0, 10., observed.robot.pose)
        self.assertTrue(candidates)
        self.assertTrue(all(candidate.reference_id == 'world:approach' for candidate in candidates))
        np.testing.assert_allclose(candidates[0].reference_pose.position, np.zeros(3))

    def test_TRAIN_relative_SE3_fit_is_invariant_and_missing_artifact_is_closed(self):
        library = synthetic_library()
        approach = library.for_phase('approach')[0]
        np.testing.assert_allclose(approach.at(.5)[0].position, [0., 0., .1], atol=1e-7)
        np.testing.assert_allclose(approach.at(.5)[0].rotation, np.eye(3), atol=1e-7)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InteractionArtifactError):
                InteractionLibrary.load(Path(directory)/'missing.json')
            path = Path(directory)/'synthetic.json'
            library.save(path)
            with self.assertRaises(InteractionArtifactError):
                InteractionLibrary.load(path)
        with self.assertRaises(InteractionArtifactError):
            TRAINInteractionSource('push_moving_button', 'illegal_val', 'validation',
                'TRAIN_demonstration', np.arange(2), np.zeros((2, 7)), np.zeros(2),
                np.zeros((2, 7)), ('wp0',)*2, (None,)*2, {})

    def test_candidate_budget_collision_and_unobservable_sphere_gauge(self):
        library = synthetic_library('translation_only')
        reach = TimedReachability(CartesianRobot())
        generator = CandidateGenerator(library, reach,
            lambda **kw: dict(feasible=kw['time'] < .3, reason='synthetic_dynamic_obstacle', evidence={}),
            candidates_config(), allow_synthetic=True)
        observed = state(angle=1.2, observable=False)
        forecast = ConstantVelocityDynamics().predict(list(observed.bodies.values()), 1., .05)
        candidates = generator.generate(observed, forecast, 0, 10., observed.robot.pose)
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].arrival_time, .2)
        np.testing.assert_allclose(candidates[0].pose.rotation, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(candidates[0].angular_velocity, 0., atol=1e-6)
        self.assertIn('dynamic_collision_synthetic_dynamic_obstacle', generator.last_diagnostics['rejections'])
        self.assertFalse(generator.generate(observed, forecast, 0, .1, observed.robot.pose))
        self.assertIn('insufficient_full_continuation_time_budget', generator.last_diagnostics['rejections'])
        feature = candidate_features(candidates[0], observed.robot, candidates[0].current_reference,
                                     10., predictor='cv')
        self.assertEqual(feature.shape, (len(FEATURE_NAMES),))

    def test_phase_success_is_pending_native_completion_and_abandonment_is_local(self):
        ledger = ContinuationAttempts(task_name='test', predictor='context',
                                      continuation_sha256='frozen_controller', library_sha256='trained_library')
        x = np.zeros(len(FEATURE_NAMES))
        ledger.begin('acquire', 'cup0@time1', 1., x, probability=.8, selected_supported=True)
        ledger.skill_completed('acquire', 2.)
        self.assertIsNone(ledger.to_report()[0]['official_task_success'])
        ledger.begin('insert', 'peg0@time3', 3., x)
        ledger.begin('insert', 'peg0@time4', 4., x)
        records = ledger.finish(True, 5., 'native_task_success')
        self.assertEqual([r['official_task_success'] for r in records], [True, False, True])
        self.assertTrue(records[0]['skill_success'])
        censored = ContinuationAttempts(task_name='test', predictor='cv',
                                        continuation_sha256='frozen', library_sha256='trained')
        censored.begin('press', 'button', 1., x)
        row = censored.finish(None, 2., 'infrastructure_interruption')[0]
        self.assertTrue(row['censored'])
        self.assertIsNone(row['official_task_success'])
        self.assertFalse(row['complete'])

    def test_probability_does_not_become_penalized_utility(self):
        class Candidate:
            def __init__(self, utility): self.geometry_utility = utility
        candidates = [Candidate(-.1), Candidate(-.3)]
        predictions = dict(probability=np.array([.7, .75]), supported=np.ones(2, bool),
                           ensemble_std=np.full(2, .1), ranking_enabled=True)
        result = select_candidate(candidates, predictions, minimum_gain=.02, disagreement_multiplier=1.)
        self.assertEqual(result.selected_index, 0)
        self.assertEqual(result.success_probability, .7)
        self.assertEqual(result.geometry_utility, -.1)
        self.assertFalse(result.actual_override)

    def test_SE3_MPC_respects_twist_limits_and_controller_has_no_native_outcome(self):
        library = synthetic_library()
        reach = TimedReachability(CartesianRobot())
        generator = CandidateGenerator(library, reach, lambda **kw: dict(feasible=True, evidence={}),
                                       candidates_config(), allow_synthetic=True)
        mpc = SE3InteractionMPC(mpc_config())
        observed = state()
        desired = [Pose(np.array([.01, 0., .1]), Rotation.from_euler('z', .05).as_matrix())]*3
        pose, velocity, angular, details = mpc.command(observed.robot, desired)
        self.assertLessEqual(np.linalg.norm(velocity), 1.+1e-5)
        self.assertLessEqual(np.linalg.norm(angular), 2.+1e-5)
        self.assertGreater(pose.position[0], 0.)
        self.assertFalse(details['learned_contact_dynamics'])
        controller = InteractionController(library, generator, mpc, predictor='cv', ranking='geometry',
                                           continuation_sha256='synthetic_test_only', allow_synthetic=True)
        for tick in range(20):
            time = tick*.05
            observed = state(time, robot_z=.1 if controller.phase_index == 0 else .05)
            forecast = ConstantVelocityDynamics().predict(list(observed.bodies.values()), 1., .05)
            controller.step(observed, forecast, 10.-time)
            if controller.phase_index == len(library.phase_order): break
        self.assertEqual(controller.phase_index, 2)
        self.assertTrue(controller.events)
        self.assertTrue(all(row['official_task_success'] is None for row in controller.attempts.to_report()))
        self.assertTrue(all(not event['official_task_success_claimed'] for event in controller.events))
        with self.assertRaises(ValueError):
            PolicyState(observed.robot, observed.bodies, {'metric_only': True}, observed.role_bindings)


if __name__ == '__main__':
    unittest.main()
