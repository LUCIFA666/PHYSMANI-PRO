"""CPU-only differential tests through the complete original v7/v6/v5 MRO."""
from contextlib import ExitStack
from collections import Counter
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import numpy as np

from our_mani.multitask import author_aligned_geometry_executor_v1114 as new
from our_mani.multitask import physmani_executor_v7 as v7
from our_mani.ablations import physmani_executor_v4 as v4

ACTION = np.array([0., 0., 1., 0., 0., 0., 1., 0.])
CONFIG = dict(dt=.05, baseline_decision_ticks=10)


class Arm:
    def __init__(self):
        self.q = np.zeros(7); self.target = np.zeros(7)
        self.velocity = np.zeros(7); self.calls = Counter(); self.targets = []

    def get_joint_count(self):
        self.calls['joint_count'] += 1
        return 7

    def get_joint_positions(self):
        self.calls['positions'] += 1
        return self.q.tolist()

    def get_joint_target_positions(self):
        self.calls['targets'] += 1
        return self.target.tolist()

    def get_joint_velocities(self):
        self.calls['velocities'] += 1
        return self.velocity.tolist()

    def set_joint_target_positions(self, q):
        self.calls['set_targets'] += 1
        self.target = np.asarray(q, float).copy()
        self.targets.append(self.target.tolist())


class Path:
    def __init__(self, arm, *, error=None, handle=None, done_after=3):
        self._arm = arm; self.calls = 0; self.error = error
        self._joint_position_action = None; self._rml_handle = handle
        self._path_points = np.zeros(14); self._path_done = False
        self.done_after = done_after

    def step(self):
        self.calls += 1
        if self.error:
            raise self.error
        self._joint_position_action = np.full(7, .7 + self.calls * .1)
        self._arm.set_joint_target_positions(self._joint_position_action)
        self._path_done = self.done_after is not None and self.calls >= self.done_after
        return self._path_done


class Env:
    def __init__(self, *, ticks=0, path_error=None, step_error=None, handle=None,
                 done_after=3, terminate_after=None, apply_error=None):
        self.arm = Arm()
        self.path = Path(self.arm, error=step_error, handle=handle, done_after=done_after)
        self.path_error = path_error; self.apply_error = apply_error
        self.terminate_after = terminate_after
        self.map_service = NS(plan_path=self.plan)
        self.scene = NS(check_target_in_workspace=lambda xyz: True)
        self.time = ticks * .05
        self.ledger = NS(ticks=ticks, remaining=(200-ticks)*.05, dt=.05,
                         grasp_requests=0, release_requests=0, before=lambda: None)
        self.robot_model = NS(lower=np.full(7, -.4), upper=np.full(7, .4),
                              velocity_limits=np.full(7, .1))
        self._native_terminal_latched = self._native_failure_latched = False
        self._native_success_latched = False; self._requested_opening = 1.
        self.protocol = dict(gripper_velocity=.2)
        self.counts = Counter(); self.phases = []; self.gripper_events = []; self.plan_calls = []
        self.gripper = NS(release=lambda: self.gripper_events.append(('release',)),
                          grasp=lambda obj: self.gripper_events.append(('grasp', obj)),
                          actuate=lambda opening, velocity: self.gripper_events.append(('actuate', opening, velocity)))
        self.task = NS(get_graspable_objects=self.graspables)

    def graspables(self):
        self.counts['graspable_objects'] += 1
        return ['public-independent-native-body']

    def plan(self, *args, **kwargs):
        self.plan_calls.append((args, kwargs))
        if self.path_error:
            raise self.path_error
        return self.path

    def _before_action(self):
        self.counts['before_action'] += 1

    def _tick(self, phase):
        if self.apply_error:
            raise self.apply_error
        before = self.arm.q.copy()
        self.arm.q = self.arm.target.copy()
        self.arm.velocity = (self.arm.q-before)/.05
        self.ledger.ticks += 1; self.ledger.remaining -= .05
        self.time = self.ledger.ticks*.05; self.phases.append(phase)
        if self.terminate_after is not None and len(self.phases) >= self.terminate_after:
            self._native_terminal_latched = self._native_failure_latched = True

    def observe(self):
        self.counts['observe'] += 1
        return {}, NS(timestamp=self.time)


class Skill:
    def __init__(self, env, *, defer=False):
        self.env = env; self.calls = []; self.defer = defer
        self.phase_index = 0; self.close_witness = None
        self.selected = dict(phase='acquire', manipulated_ids=['cup'], selected_at=env.time)
        self.frontend = NS(state=NS(robot=NS(timestamp=env.time)), actor=NS(requested_opening=1.))

    def qualify_opening(self, requested, current):
        self.frontend.state.robot.timestamp = self.env.time
        value = 1. if self.defer else requested
        self.calls.append(('qualify', self.env.time, requested, current))
        return value, dict(time=self.env.time, phase='acquire', desired_opening=requested,
            current_requested_opening=current, qualified_opening=value,
            current_position_error_m=None, current_orientation_error_rad=None,
            reason='legacy_defer' if self.defer else 'legacy', native_truth_read=False), None

    def confirm_opening(self, opening, proof, proposed, *, confirmed_after):
        self.calls.append(('confirm', confirmed_after, opening))
        if proposed is not None:
            self.close_witness = dict(proposed, confirmed_after=confirmed_after)
        self.frontend.actor.requested_opening = opening

    def receipt(self):
        return dict(own_close_witness=self.close_witness)


class Tests(unittest.TestCase):
    def setUp(self):
        constants = ModuleType('pyrep.const')
        constants.ConfigurationPathAlgorithms = NS(RRTConnect='original-native-RRTConnect')
        errors = ModuleType('pyrep.errors')
        errors.ConfigurationPathError = type('ConfigurationPathError', (Exception,), {})
        errors.IKError = type('IKError', (Exception,), {})
        backend = ModuleType('pyrep.backend'); backend.sim = NS(simRMLRemove=Mock())
        self.remove_handle = backend.sim.simRMLRemove
        scope = patch.dict(sys.modules, {'pyrep.const': constants, 'pyrep.errors': errors,
                                        'pyrep.backend': backend})
        scope.start(); self.addCleanup(scope.stop)

    def adapter(self, **kwargs):
        return new.AuthorAlignedGeometryExecutor(CONFIG,
            observe_public_frame=lambda *args: None, **kwargs)

    def run_action(self, env, adapter=None, action=ACTION):
        if adapter is None:
            adapter = object.__new__(v7.PhysManiAdapter)
            adapter.config = CONFIG; adapter.observe = lambda *args: None
        with ExitStack() as scopes:
            scopes.enter_context(patch.object(v7, 'physics_time', side_effect=lambda: env.time))
            scopes.enter_context(patch.object(v7, '_check_clock'))
            scopes.enter_context(patch.object(v7, 'modify_released_path', return_value={'same_extraction': True}))
            return adapter.execute(env, action)

    def equal_native_effects(self, baseline, observed):
        self.assertEqual(baseline.arm.calls, observed.arm.calls)
        self.assertEqual(baseline.arm.targets, observed.arm.targets)
        self.assertEqual(baseline.counts, observed.counts)
        self.assertEqual(baseline.gripper_events, observed.gripper_events)
        self.assertEqual(baseline.phases, observed.phases)
        self.assertEqual(baseline.path.calls, observed.path.calls)
        self.assertEqual(baseline.plan_calls, observed.plan_calls)
        self.assertEqual(baseline.ledger.ticks, observed.ledger.ticks)

    def test_full_mro_default_exact_native_calls_and_receipt_match_original_v7(self):
        baseline, actual = Env(), Env()
        expected = self.run_action(baseline)
        never = Mock(side_effect=AssertionError('public joint predicate forbidden by default'))
        skill = Skill(actual)
        executor = self.adapter(skill_policy=skill, admit_joint_target=never)
        result = self.run_action(actual, executor)
        self.equal_native_effects(baseline, actual)
        self.assertEqual(expected, result); never.assert_not_called()
        self.assertEqual(actual.phases, ['physmani_path']*9 + ['physmani_gripper'])
        self.assertEqual(len(skill.calls), 20)
        self.assertEqual([v[0] for v in skill.calls], ['qualify', 'confirm']*10)
        self.assertEqual([e for e in actual.gripper_events if e[0]=='grasp'], [('grasp', 'public-independent-native-body')])
        self.assertEqual(actual.plan_calls[0][1], dict(quaternion=[0.,0.,0.,1.], ignore_collisions=True,
            trials=100, max_configs=10, max_time_ms=10, trials_per_goal=5, algorithm='original-native-RRTConnect'))
        self.assertGreater(result['counterfactual_ours_reference_clip']['velocity_clip_ticks'], 0)
        self.assertEqual(result['common_joint_command_clipping']['velocity_clip_ticks'], 0)
        self.assertAlmostEqual(actual.arm.q[0], .4)

    def test_explicit_bounded_collision_false_is_recorded_without_veto_or_extra_path_step(self):
        baseline, actual = Env(done_after=None), Env(done_after=None)
        expected = self.run_action(baseline)
        callback = Mock(return_value=(False, {'pair':['wand','frame']}))
        executor = self.adapter(admit_joint_target=callback, collision_diagnostic_limit=2)
        result = self.run_action(actual, executor)
        self.equal_native_effects(baseline, actual); self.assertEqual(expected, result)
        self.assertEqual(callback.call_count, 2)
        receipt = executor.geometry_receipt()
        self.assertEqual(receipt['collision_diagnostic_calls'], 2)
        self.assertEqual(receipt['collision_diagnostic_skipped'], 7)
        self.assertTrue(all(e['would_reject'] for e in receipt['generated_target_admissions']))
        self.assertIsNone(result['arm_error'])

    def test_path_recovery_semantics_match_original_and_protected_errors_escape(self):
        for kind in ('planning','RML'):
            args = {'path_error':RuntimeError('planner rejected')} if kind=='planning' else {'step_error':ValueError('RML rejected')}
            baseline, actual = Env(**args), Env(**args)
            expected = self.run_action(baseline); result = self.run_action(actual, self.adapter())
            self.assertEqual(expected, result); self.equal_native_effects(baseline, actual)
            self.assertEqual(result['ticks'], 10)
        for message in ('CUDA out of memory', 'invalid object handle', 'simulator connection lost'):
            env = Env(step_error=RuntimeError(message), handle=42)
            original_plan = env.map_service.plan_path; original_apply = new.direct._apply_direct_target
            with self.assertRaises(new.InfrastructureExecutionError):
                self.run_action(env, self.adapter(skill_policy=Skill(env),
                    admit_joint_target=lambda q:(True,{}), collision_diagnostic_limit=1))
            self.assertIs(env.map_service.plan_path, original_plan)
            self.assertIs(new.direct._apply_direct_target, original_apply)
            self.assertEqual(env.ledger.ticks, 0)
        self.assertEqual(self.remove_handle.call_count, 3)

    def test_diagnostic_error_is_infrastructure_and_original_handle_cleanup_still_runs(self):
        env = Env(handle=17, done_after=None); callback=Mock(side_effect=ValueError('probe failed'))
        executor = self.adapter(admit_joint_target=callback, collision_diagnostic_limit=1)
        with self.assertRaises(new.InfrastructureExecutionError):
            self.run_action(env, executor)
        self.remove_handle.assert_called_once_with(17)
        self.assertEqual(env.ledger.ticks, 0)
        self.assertEqual(executor.geometry_receipt()['generated_target_admissions'][0]['diagnostic_error'], 'ValueError')

    def test_cleanup_error_is_infrastructure_never_a_task_outcome(self):
        env = Env(handle=18, done_after=None)
        self.remove_handle.side_effect = RuntimeError('native removal failed')
        with self.assertRaisesRegex(new.InfrastructureExecutionError, 'RML handle cleanup failed'):
            self.run_action(env, self.adapter())

    def test_partial_budget_and_terminal_stop_do_not_fabricate_final_close(self):
        for kwargs, ticks in (({'ticks':195}, 5), ({'terminate_after':3}, 3)):
            env = Env(**kwargs); source = Skill(env, defer=True)
            wrapper = new.AuthorRequestedOpeningWitness(source)
            result = self.run_action(env, self.adapter(skill_policy=wrapper))
            self.assertEqual(result['ticks'], ticks)
            self.assertEqual(env.ledger.grasp_requests, 0); self.assertIsNone(source.close_witness)
            self.assertTrue(all(e['proposed'] is None for e in wrapper.opening_events))
            self.assertEqual(env._requested_opening, 1.)

    def test_confirm_only_follows_successful_original_application_and_helpers_restore(self):
        env = Env(apply_error=new.ProtocolInvariantError('counted tick failed')); skill = Skill(env)
        before_direct = new.direct._apply_direct_target; before_command = v4._command_with_clipping_receipt
        with self.assertRaises(new.ProtocolInvariantError):
            self.run_action(env, self.adapter(skill_policy=skill))
        self.assertEqual([v[0] for v in skill.calls], ['qualify'])
        self.assertIs(new.direct._apply_direct_target, before_direct)
        self.assertIs(v4._command_with_clipping_receipt, before_command)

    def test_author_opening_overrides_legacy_defer_with_truthful_actual_edge_proof(self):
        env = Env(); source = Skill(env, defer=True); confirmed = []
        def hook(policy, opening, proof, proposed, *, confirmed_after):
            if proposed is not None:
                self.assertEqual(env.ledger.grasp_requests, 1)
                self.assertEqual(policy.close_witness['confirmed_after'], confirmed_after)
                confirmed.append((opening, proof, proposed, confirmed_after))
        wrapper = new.AuthorRequestedOpeningWitness(source, on_confirmed_opening=hook)
        self.run_action(env, self.adapter(skill_policy=wrapper))
        self.assertEqual(len(confirmed), 1)
        opening, proof, proposed, after = confirmed[0]
        self.assertEqual(opening, 0.); self.assertEqual(proof['qualified_opening'], 0.)
        self.assertEqual(proof['legacy_qualified_opening'], 1.)
        self.assertEqual(proof['legacy_qualification']['proof']['reason'], 'legacy_defer')
        self.assertFalse(proof['native_capture_claimed'])
        self.assertEqual(proposed['manipulated_ids'], ['cup'])
        self.assertAlmostEqual(after-proposed['issued_at'], .05)
        self.assertIsNone(proposed['distance_m'])
        self.assertEqual(env._requested_opening, 0.)
        self.assertEqual(len(source.calls), 20)

    def test_actual_command_witness_is_consumable_by_action_conditioned_public_observer(self):
        from our_mani.contracts import ObjectState, Pose, RobotState
        from our_mani.multitask.action_conditioned_mode_observer_v1115 import ActionConditionedPublicModeObserver
        identities = ('cup0', 'cup1', 'cup2')
        def states(timestamp):
            return [ObjectState(identity, timestamp, Pose([.02+.04*i,0.,0.]),
                    [[0.,0.,0.],[.01,0.,0.],[0.,.01,0.]], np.zeros(3), np.zeros(3))
                    for i,identity in enumerate(identities)]
        env = Env(); source = Skill(env, defer=True)
        source.selected['manipulated_ids'] = ['cup1']
        mode = ActionConditionedPublicModeObserver(identities, identities, .15)
        published = []
        def hook(policy, opening, proof, proposed, *, confirmed_after):
            if proposed is not None:
                published.append(mode.record_confirmed_close(selected=policy.selected, proof=proof,
                    proposed=proposed, confirmed_after=confirmed_after,
                    previous_states=states(proposed['issued_at']), confirmed_witness=policy.close_witness))
        self.run_action(env, self.adapter(skill_policy=new.AuthorRequestedOpeningWitness(source,
                                                 on_confirmed_opening=hook)))
        self.assertEqual(published, [True])
        robot = RobotState(env.time, Pose([0.,0.,0.]), np.zeros(3), np.zeros(3),
                           np.zeros(7), np.zeros(7), gripper_open=False, contact=True)
        modes, invalid = mode.observe(states(env.time-.05), robot)
        self.assertEqual(modes, {'cup0':'external', 'cup1':'attached', 'cup2':'external'})
        self.assertEqual(invalid, set())
        self.assertFalse(mode.receipt()['events'][-1]['native_grasp_identity_verified'])

    def test_no_neural_constructor_no_pose_mismatch_dispatch_and_invalid_config(self):
        with patch.object(new.OriginalExecutor, '__init__', side_effect=AssertionError('neural init forbidden')):
            executor = self.adapter(admit_pose_action=lambda a:(False, {'intent':'mismatch'}))
        env = Env()
        with self.assertRaises(new.ProtocolInvariantError): self.run_action(env, executor)
        self.assertEqual(env.ledger.ticks, 0); self.assertEqual(env.plan_calls, [])
        with self.assertRaises(new.ProtocolInvariantError): executor.predict(None)
        with self.assertRaises(ValueError): self.adapter(collision_diagnostic_limit=65)


if __name__ == '__main__':
    unittest.main()
