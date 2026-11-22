"""Explicit author-action outcome sampling; no simulator or policy mutation.

The author arm loop observes success at each physical tick, but the outer
TaskEnvironment.step reads the current failure result after the whole action.
An intermediate failure is diagnostic only. Action boundaries are explicit;
neither elapsed time nor a global tick modulus defines them.

Integration is opt-in: derive a new simulation class with the mixin first in
its MRO, then wrap each execute or individual hold in native_action_scope.
This module does not patch any frozen class, factory, command or condition.
It targets the core simulation, not the block adapter whose receipt forbids
extra task.success calls.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import math

from .native_tasks import NativeResult, native_result
from .protocol import (
    EpisodeTerminatedError, MultitaskInfrastructureError, MultitaskProtocolError,
)


NONTERMINAL = NativeResult(False, False, False)


def _result(value):
    if not isinstance(value, NativeResult):
        raise TypeError('a native task result is required, never policy diagnostics')
    if any(type(getattr(value, key)) is not bool for key in ('success', 'terminate', 'terminal_failure')):
        raise TypeError('native outcome fields must be booleans')
    if value.terminal_failure != (value.terminate and not value.success):
        raise ValueError('inconsistent native terminal-failure result')
    return value


class ActionOutcomeSampling:
    """Pure episode state machine with explicit, bounded action boundaries."""

    def __init__(self):
        self.active = None
        self.reset()

    def reset(self):
        if self.active is not None:
            raise MultitaskProtocolError('cannot reset inside a native action')
        self.outcome = NONTERMINAL
        self.initial_result = self.last_raw = None
        self.last_tick = 0
        self.actions = []
        self.aborted_actions = []
        self.aborted = False
        self.intermediate_failure_samples = 0

    def initialize(self, result, *, tick=0):
        if type(tick) is not int or tick != 0 or self.last_tick != 0 or self.actions or self.active is not None or self.aborted:
            raise MultitaskProtocolError('initial native sample belongs only to reset tick zero')
        if self.initial_result is not None:
            raise MultitaskProtocolError('native reset result was already sampled')
        result = _result(result)
        self.initial_result = self.last_raw = result
        # Preserve existing immediate reset-success semantics. Reset failure
        # does not replace the author's action-end failure observation.
        self.outcome = NativeResult(True, True, False) if result.success else NONTERMINAL
        return self.outcome

    def begin_action(self, *, tick, label, max_ticks=10):
        if self.aborted:
            raise MultitaskProtocolError('an aborted episode requires reset, not a retry')
        if self.outcome.terminate:
            raise EpisodeTerminatedError('a terminal episode cannot start another action')
        if self.active is not None:
            raise MultitaskProtocolError('native action scopes cannot nest')
        if self.initial_result is None or type(tick) is not int or tick != self.last_tick:
            raise MultitaskProtocolError('action start must match the current counted native tick')
        if type(max_ticks) is not int or not 1 <= max_ticks <= 10:
            raise MultitaskProtocolError('an author action permits one to ten counted ticks')
        if not isinstance(label, str) or not label:
            raise MultitaskProtocolError('an explicit native action label is required')
        self.active = dict(action_index=len(self.actions), label=label, start_tick=tick, max_ticks=max_ticks)
        return dict(self.active)

    def require_active(self):
        if self.aborted:
            raise MultitaskProtocolError('an aborted episode cannot advance')
        if self.outcome.terminate:
            raise EpisodeTerminatedError('a terminal native result cannot advance')
        if self.active is None:
            raise MultitaskProtocolError('physical commands require an explicit native action scope')

    def require_next_tick(self, tick):
        self.require_active()
        if type(tick) is not int or tick != self.last_tick + 1:
            raise MultitaskProtocolError('native outcome samples must follow each counted physical tick')
        if tick - self.active['start_tick'] > self.active['max_ticks']:
            raise MultitaskProtocolError('native action exceeded its explicit tick allowance')

    def observe_tick(self, result, *, tick):
        self.require_next_tick(tick)
        result = _result(result)
        self.last_tick, self.last_raw = tick, result
        self.intermediate_failure_samples += int(result.terminal_failure)
        if result.success:
            self.outcome = NativeResult(True, True, False)
        # Failure is deliberately not OR'ed into the sampled episode outcome.
        return self.outcome

    def end_action(self, result, *, tick):
        if self.aborted or self.active is None:
            raise MultitaskProtocolError('only an active native action can finish')
        result = _result(result)
        if type(tick) is not int or tick != self.last_tick or tick <= self.active['start_tick']:
            raise MultitaskProtocolError('an action must finish after its actual counted ticks')
        self.last_raw = result
        if not self.outcome.success:
            self.outcome = NativeResult(True, True, False) if result.success else result
        row = dict(self.active, end_tick=tick, ticks=tick - self.active['start_tick'],
            current_native_result=asdict(result), sampled_outcome=asdict(self.outcome), complete=True)
        self.actions.append(row)
        self.active = None
        return self.outcome

    def abort_action(self, *, tick, error):
        # An exception is not a completed action or a clean task failure.
        if self.active is not None:
            self.aborted_actions.append(dict(self.active, observed_tick=tick, error=str(error), complete=False))
        self.active = None
        self.aborted = True

    def receipt(self):
        return dict(schema='r10_explicit_author_action_outcome_sampling_v741',
            success_sampled_each_tick=True, failure_sampled_at_explicit_action_end=True,
            global_tick_modulus_used=False, reset_failure_is_diagnostic_only=True,
            reset_success_semantics_preserved=True,
            initial_native_result=None if self.initial_result is None else asdict(self.initial_result),
            sampled_outcome=asdict(self.outcome), last_tick=self.last_tick,
            active_action=None if self.active is None else dict(self.active),
            aborted=self.aborted, completed_actions=deepcopy(self.actions),
            aborted_actions=deepcopy(self.aborted_actions),
            intermediate_failure_samples=self.intermediate_failure_samples)


class AuthorActionOutcomeSamplingMixin:
    """Opt-in core-simulation subclass adapter; physics methods stay inherited."""

    def _start_alignment(self):
        previous = getattr(self, '_native_action_sampling', None)
        if previous is not None and previous.active is not None:
            raise MultitaskProtocolError('cannot reset alignment inside a native action')
        self._native_action_sampling = ActionOutcomeSampling()
        return super()._start_alignment()

    def _read_current_native_result(self):
        try:
            return native_result(self.task)
        except Exception as exc:
            raise MultitaskInfrastructureError('invalid native task outcome') from exc

    def _sync_sampled_native_result(self, value):
        self._outcome = self._native_action_sampling
        self._native_success_latched = value.success
        self._native_failure_latched = value.terminal_failure
        self._native_terminal_latched = value.terminate
        if value.terminate and self._terminal_tick is None:
            self._terminal_tick = self.ledger.ticks
        return value

    def _latch_native_result(self):
        sampler = self._native_action_sampling
        current = self._read_current_native_result()
        value = (sampler.initialize(current, tick=self.ledger.ticks) if sampler.active is None
                 else sampler.observe_tick(current, tick=self.ledger.ticks))
        return self._sync_sampled_native_result(value)

    def _before_action(self):
        # Check the action allowance before inherited code can write a joint
        # or gripper reference, as well as checking again before physics.
        self._native_action_sampling.require_next_tick(self.ledger.ticks + 1)
        return super()._before_action()

    def _tick(self, phase):
        self._native_action_sampling.require_next_tick(self.ledger.ticks + 1)
        return super()._tick(phase)

    @contextmanager
    def native_action_scope(self, *, label, max_ticks=10):
        sampler = self._native_action_sampling
        # This bound never expands the existing ledger budget. A final partial
        # decision is an action boundary at its actual last physical tick.
        if type(max_ticks) is not int or not 1 <= max_ticks <= 10:
            raise MultitaskProtocolError('an author action permits one to ten counted ticks')
        remaining = int(math.floor((self.ledger.remaining + 1e-8) / self.ledger.dt))
        allowance = min(max_ticks, remaining)
        token = sampler.begin_action(tick=self.ledger.ticks, label=label, max_ticks=allowance)
        try:
            self._before_action()
            yield token
            # Like TaskEnvironment.step: re-read current task.success after
            # the action, without a physical step or direct condition access.
            current = self._read_current_native_result()
            value = sampler.end_action(current, tick=self.ledger.ticks)
            self._sync_sampled_native_result(value)
        except BaseException as exc:
            sampler.abort_action(tick=self.ledger.ticks, error=type(exc).__name__)
            raise

    def native_action_sampling_receipt(self):
        return self._native_action_sampling.receipt()

    def alignment_receipt(self):
        result = super().alignment_receipt()
        result['native_outcome_sampling'] = self.native_action_sampling_receipt()
        return result
