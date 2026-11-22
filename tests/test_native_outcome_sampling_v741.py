"""CPU-only outcome semantics and real counted-tick method integration."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from our_mani.aligned.protocol import TickLedger
from our_mani.multitask.native_tasks import NativeResult
from our_mani.multitask.native_outcome_sampling_v741 import (
    ActionOutcomeSampling, AuthorActionOutcomeSamplingMixin, NONTERMINAL,
)
from our_mani.multitask.protocol import (
    EpisodeTerminatedError, MultitaskInfrastructureError, MultitaskProtocolError,
)
from our_mani.multitask.simulation import MultitaskSimulation


FAILURE = NativeResult(False, True, True)
SUCCESS = NativeResult(True, True, False)


class SamplingStateTests(unittest.TestCase):
    def sampler(self, initial=NONTERMINAL):
        sampler = ActionOutcomeSampling()
        sampler.initialize(initial)
        return sampler

    def test_intra_action_failure_is_diagnostic_and_can_recover(self):
        sampler = self.sampler()
        sampler.begin_action(tick=0, label='execute')
        self.assertEqual(sampler.observe_tick(FAILURE, tick=1), NONTERMINAL)
        self.assertEqual(sampler.observe_tick(NONTERMINAL, tick=2), NONTERMINAL)
        self.assertEqual(sampler.end_action(NONTERMINAL, tick=2), NONTERMINAL)
        sampler.begin_action(tick=2, label='next')
        self.assertEqual(sampler.receipt()['intermediate_failure_samples'], 1)

    def test_current_action_end_failure_is_terminal_even_after_nonfatal_tick(self):
        sampler = self.sampler()
        sampler.begin_action(tick=0, label='hold', max_ticks=1)
        sampler.observe_tick(NONTERMINAL, tick=1)
        self.assertEqual(sampler.end_action(FAILURE, tick=1), FAILURE)
        with self.assertRaises(EpisodeTerminatedError):
            sampler.begin_action(tick=1, label='forbidden')

    def test_success_stops_immediately_and_cannot_be_overwritten_at_boundary(self):
        sampler = self.sampler()
        sampler.begin_action(tick=0, label='execute')
        sampler.observe_tick(FAILURE, tick=1)
        self.assertEqual(sampler.observe_tick(SUCCESS, tick=2), SUCCESS)
        with self.assertRaises(EpisodeTerminatedError):
            sampler.require_next_tick(3)
        self.assertEqual(sampler.end_action(FAILURE, tick=2), SUCCESS)
        with self.assertRaises(EpisodeTerminatedError):
            sampler.begin_action(tick=2, label='forbidden')

    def test_action_boundaries_are_explicit_and_not_a_global_tick_modulus(self):
        sampler = self.sampler()
        sampler.begin_action(tick=0, label='short_execute')
        for tick in (1, 2, 3):
            sampler.observe_tick(NONTERMINAL, tick=tick)
        sampler.end_action(NONTERMINAL, tick=3)
        sampler.begin_action(tick=3, label='single_public_hold', max_ticks=1)
        sampler.observe_tick(FAILURE, tick=4)
        self.assertEqual(sampler.end_action(FAILURE, tick=4), FAILURE)
        rows = sampler.receipt()['completed_actions']
        self.assertEqual([(r['start_tick'], r['end_tick']) for r in rows], [(0, 3), (3, 4)])

    def test_reset_failure_is_not_latched_and_reset_clears_previous_episode(self):
        sampler = self.sampler(FAILURE)
        self.assertEqual(sampler.outcome, NONTERMINAL)
        sampler.begin_action(tick=0, label='first')
        with self.assertRaises(MultitaskProtocolError):
            sampler.reset()
        sampler.observe_tick(SUCCESS, tick=1)
        sampler.end_action(SUCCESS, tick=1)
        sampler.reset()
        self.assertEqual(sampler.initialize(FAILURE), NONTERMINAL)
        self.assertEqual(sampler.receipt()['completed_actions'], [])
        self.assertFalse(sampler.aborted)

    def test_existing_reset_success_remains_terminal(self):
        sampler = self.sampler(SUCCESS)
        self.assertEqual(sampler.outcome, SUCCESS)
        with self.assertRaises(EpisodeTerminatedError):
            sampler.begin_action(tick=0, label='already_successful')

    def test_invalid_or_unbounded_action_sequences_fail_before_a_tick(self):
        sampler = self.sampler()
        for maximum in (0, 11, True):
            with self.subTest(maximum=maximum), self.assertRaises(MultitaskProtocolError):
                sampler.begin_action(tick=0, label='bad', max_ticks=maximum)
        sampler.begin_action(tick=0, label='one', max_ticks=1)
        with self.assertRaises(MultitaskProtocolError):
            sampler.begin_action(tick=0, label='nested')
        with self.assertRaises(MultitaskProtocolError):
            sampler.observe_tick(NONTERMINAL, tick=2)
        with self.assertRaises(MultitaskProtocolError):
            sampler.end_action(NONTERMINAL, tick=0)
        sampler.observe_tick(NONTERMINAL, tick=1)
        with self.assertRaises(MultitaskProtocolError):
            sampler.require_next_tick(2)

    def test_abort_is_not_clean_task_failure_or_retry_and_receipts_are_snapshots(self):
        sampler = self.sampler()
        sampler.begin_action(tick=0, label='execute')
        sampler.observe_tick(NONTERMINAL, tick=1)
        sampler.abort_action(tick=1, error='observer failure')
        receipt = sampler.receipt()
        self.assertTrue(receipt['aborted'])
        self.assertEqual(receipt['completed_actions'], [])
        self.assertFalse(receipt['aborted_actions'][0]['complete'])
        with self.assertRaises(MultitaskProtocolError):
            sampler.begin_action(tick=1, label='retry')
        sampler.reset()
        sampler.initialize(NONTERMINAL)
        sampler.begin_action(tick=0, label='new_episode')
        sampler.observe_tick(NONTERMINAL, tick=1)
        sampler.end_action(NONTERMINAL, tick=1)
        receipt = sampler.receipt()
        receipt['completed_actions'][0]['current_native_result']['success'] = True
        self.assertFalse(sampler.receipt()['completed_actions'][0]['current_native_result']['success'])


class CoreWithSampling(AuthorActionOutcomeSamplingMixin, MultitaskSimulation):
    """Use the real core tick/command methods without constructing a simulator."""


def core(current_result=lambda tick: (False, False)):
    env = object.__new__(CoreWithSampling)
    env.ledger = TickLedger(dt=.05, budget=10.)
    env._native_action_sampling = ActionOutcomeSampling()
    env._terminal_tick = None
    env._native_success_latched = env._native_failure_latched = env._native_terminal_latched = False
    env._aligned = True
    env.time = env._physics_origin = env._task_origin = env._fake_clock = 0.
    env.physics_time = lambda: env._fake_clock
    env.task = SimpleNamespace(t=0.)
    env.task.success = Mock(side_effect=lambda: current_result(env.ledger.ticks))
    env.scene_steps = 0

    def step():
        env.scene_steps += 1
        env._fake_clock += .05
        env.task.t += .05

    env.scene = SimpleNamespace(step=step)
    positions = np.zeros(7)
    env.arm = SimpleNamespace(get_joint_positions=lambda: positions.copy(),
        set_joint_target_positions=Mock())
    env.robot_model = SimpleNamespace(velocity_limits=np.ones(7), lower=np.full(7, -3.), upper=np.full(7, 3.))
    env._latch_native_result()
    return env


class MixinCountedTickTests(unittest.TestCase):
    def test_real_core_tick_ignores_transient_failure_and_ends_on_current_result(self):
        env = core(lambda tick: (False, tick == 1))
        with env.native_action_scope(label='execute'):
            env._tick('arm')
            self.assertFalse(env._native_terminal_latched)
            env._tick('arm')
        self.assertEqual(env.ledger.ticks, 2)
        self.assertFalse(env._native_terminal_latched)
        self.assertEqual(env.native_outcome, NONTERMINAL)
        self.assertEqual(env.task.success.call_count, 4)  # reset, two ticks, action end

    def test_action_end_requeries_current_task_without_advancing_any_clock(self):
        env = core()
        env.task.success.side_effect = [(False, False), (False, True)]
        with env.native_action_scope(label='public_hold', max_ticks=1):
            env._tick('hold')
            self.assertFalse(env._native_failure_latched)
            before = env.ledger.ticks, env._fake_clock, env.task.t, env.scene_steps
        self.assertEqual((env.ledger.ticks, env._fake_clock, env.task.t, env.scene_steps), before)
        self.assertTrue(env._native_failure_latched)
        self.assertEqual(env._terminal_tick, 1)

    def test_immediate_success_blocks_further_physics_and_survives_end_failure(self):
        env = core()
        env.task.success.side_effect = [(True, True), (False, True)]
        with env.native_action_scope(label='execute'):
            env._tick('arm')
            self.assertTrue(env._native_success_latched)
            with self.assertRaises(EpisodeTerminatedError):
                env._tick('arm')
        self.assertEqual(env.scene_steps, 1)
        self.assertEqual(env.native_outcome, SUCCESS)
        self.assertFalse(env._native_failure_latched)

    def test_commands_outside_scope_are_rejected_before_reference_or_physics_mutation(self):
        env = core()
        with self.assertRaises(MultitaskProtocolError):
            env.command_joints(np.ones(7))
        env.arm.set_joint_target_positions.assert_not_called()
        self.assertEqual(env.scene_steps, 0)
        with env.native_action_scope(label='execute', max_ticks=1):
            env.command_joints(np.ones(7))
        np.testing.assert_allclose(env.arm.set_joint_target_positions.call_args.args[0], np.full(7, .05))
        self.assertEqual(env.ledger.ticks, 1)

    def test_action_tick_allowance_rejects_extra_reference_before_writing_it(self):
        env = core()
        with env.native_action_scope(label='single_hold', max_ticks=1):
            env.command_joints(np.zeros(7))
            with self.assertRaises(MultitaskProtocolError):
                env.command_joints(np.ones(7))
        env.arm.set_joint_target_positions.assert_called_once()
        self.assertEqual(env.scene_steps, 1)

    def test_partial_final_budget_action_cannot_expand_the_200_tick_budget(self):
        env = core()
        for index in range(19):
            with env.native_action_scope(label='execute' + str(index)):
                for _ in range(10):
                    env._tick('arm')
        with env.native_action_scope(label='eight_ticks'):
            for _ in range(8):
                env._tick('arm')
        with env.native_action_scope(label='budget_partial') as boundary:
            self.assertEqual(boundary['max_ticks'], 2)
            env._tick('arm')
            env._tick('arm')
        self.assertEqual(env.ledger.ticks, 200)
        self.assertAlmostEqual(env._fake_clock, 10.)
        self.assertEqual(env.native_action_sampling_receipt()['completed_actions'][-1]['ticks'], 2)
        with self.assertRaises(MultitaskProtocolError):
            with env.native_action_scope(label='over_budget'):
                self.fail('exhausted scope cannot enter')
        self.assertEqual(env.scene_steps, 200)

    def test_body_exceptions_propagate_unchanged_and_do_not_finish_or_retry_action(self):
        env = core()
        error = RuntimeError('public geometry callback failed')
        try:
            with env.native_action_scope(label='execute'):
                env._tick('arm')
                raise error
        except RuntimeError as caught:
            self.assertIs(caught, error)
        else:
            self.fail('action exception was swallowed')
        receipt = env.native_action_sampling_receipt()
        self.assertIsNone(receipt['active_action'])
        self.assertTrue(receipt['aborted'])
        self.assertEqual(receipt['completed_actions'], [])
        self.assertEqual(env.task.success.call_count, 2)  # No normal end query.
        with self.assertRaises(MultitaskProtocolError):
            env._tick('forbidden_retry')
        self.assertEqual(env.scene_steps, 1)

    def test_end_condition_exception_is_infrastructure_with_original_cause(self):
        env = core()
        error = ValueError('native detector unavailable')
        env.task.success.side_effect = [(False, False), error]
        with self.assertRaises(MultitaskInfrastructureError) as caught:
            with env.native_action_scope(label='hold', max_ticks=1):
                env._tick('hold')
        self.assertIs(caught.exception.__cause__, error)
        self.assertTrue(env.native_action_sampling_receipt()['aborted'])
        self.assertEqual(env.native_action_sampling_receipt()['completed_actions'], [])
        self.assertEqual(env.scene_steps, 1)

    def test_malformed_tick_outcome_and_zero_tick_return_are_not_clean_cases(self):
        env = core()
        env.task.success.return_value = None
        env.task.success.side_effect = lambda: ('not a bool', False)
        with self.assertRaises(MultitaskInfrastructureError):
            with env.native_action_scope(label='bad_native'):
                env._tick('arm')
        self.assertTrue(env.native_action_sampling_receipt()['aborted'])
        env = core()
        with self.assertRaises(MultitaskProtocolError):
            with env.native_action_scope(label='zero_tick'):
                pass
        self.assertEqual(env.scene_steps, 0)
        self.assertTrue(env.native_action_sampling_receipt()['aborted'])

    def test_reset_hook_cooperates_with_base_and_refuses_mid_action_reset(self):
        class ResetBase:
            def _start_alignment(self):
                self.ledger = TickLedger()
                self._terminal_tick = None
                self._latch_native_result()
                return 'base_reset_complete'

            def alignment_receipt(self):
                return {'base_receipt': True}

        class ResetEnv(AuthorActionOutcomeSamplingMixin, ResetBase):
            pass

        env = ResetEnv()
        env.task = SimpleNamespace(success=Mock(return_value=(False, True)))
        self.assertEqual(env._start_alignment(), 'base_reset_complete')
        self.assertFalse(env._native_terminal_latched)
        self.assertTrue(env.alignment_receipt()['base_receipt'])
        self.assertTrue(env.alignment_receipt()['native_outcome_sampling']['reset_failure_is_diagnostic_only'])
        env._native_action_sampling.begin_action(tick=0, label='active')
        with self.assertRaises(MultitaskProtocolError):
            env._start_alignment()
        env._native_action_sampling.abort_action(tick=0, error='reset test cleanup')
        env.task.success.return_value = (True, True)
        env._start_alignment()
        self.assertTrue(env._native_success_latched)
        self.assertEqual(env._terminal_tick, 0)
        self.assertFalse(env.native_action_sampling_receipt()['aborted'])


if __name__ == '__main__':
    unittest.main()
