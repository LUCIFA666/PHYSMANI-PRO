"""Unwired cost-ranked reuse of already-scored CEM rollouts.

No optimizer, model, executor or simulator is constructed here. A future caller
must offer the original 88/88/80 batches before they are discarded. Retaining
eight alternatives leaves the 256 rollout budget unchanged, but can add up to
64 original static collision queries and has no real-time guarantee. Querying
the first action is not certification of its H8 trajectory or contact dynamics.
"""
from copy import deepcopy
import hashlib
import json
import time
import numpy as np

from .joint_goals import PoEJointGoalHelper

VERSION = 'unwired_existing_CEM_top8_original_action_qualification_20261103_v1'
ITERATION_SIZES = (88, 88, 80)
CAPACITY = 8
HORIZON = 8


def _plain(value):
    if isinstance(value, np.ndarray):
        return dict(array=value.tolist(), dtype=str(value.dtype), shape=list(value.shape))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError('Only explicit numeric/public planning state is supported')


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    return int(value)


class RankedRolloutPool:
    """Keep stable global top8 from the unchanged three model-scoring batches.

    ``states`` is the original [N,9,...] rollout dictionary, ``actions`` the
    original [N,8,...] dictionary, and ``parts`` each original [N] cost term.
    Rank zero exactly follows old argmin/strictly-better-across-iterations tie
    order. The original model feasibility is preserved via finite scores; no
    rejected/infinite-cost model row is made eligible by a collision query.
    """
    def __init__(self, state, capacity=CAPACITY):
        if _integer(capacity, 'capacity') != CAPACITY:
            raise ValueError('This research version fixes topK=8 before outcomes')
        if (not isinstance(state, dict) or isinstance(state.get('time'), (bool, np.bool_))
                or not np.isscalar(state.get('time')) or not np.isfinite(state.get('time', np.nan))):
            raise ValueError('Explicit current planning state with finite time required')
        q = np.asarray(state.get('q'), float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('Current actual q7 required')
        self._state = deepcopy(state)
        self.state_sha256 = _digest(state)
        self.capacity = capacity
        self._entries = []
        self._batches = []
        self._candidate_sha = None
        self._sealed_sha = None

    def offer_batch(self, iteration, scores, modes, candidates, parts, states, actions, dt):
        iteration = _integer(iteration, 'iteration')
        if iteration != len(self._batches) or iteration >= len(ITERATION_SIZES):
            raise ValueError('Offer exactly original iteration order 0,1,2 once')
        if dt != .1 or isinstance(dt, (bool, np.bool_)):
            raise ValueError('Original 100ms/H8 CEM timing required')
        scores = np.asarray(scores, float)
        modes = np.asarray(modes)
        n = ITERATION_SIZES[iteration]
        if (scores.shape != (n,) or np.isnan(scores).any() or np.isneginf(scores).any()
                or modes.shape != (n,) or modes.dtype.kind not in 'iu'
                or np.any((modes < 0) | (modes >= 8))):
            raise ValueError('Original finite-or-positive-infinity cost and K8 mode arrays required')
        if not np.array_equal(modes, np.repeat(np.arange(8), n // 8)):
            raise ValueError('Original candidate-major row order required')
        if (len(candidates) != 8 or len({row['id'] for row in candidates}) != 8
                or not all(np.isfinite(row['deadline']) for row in candidates)):
            raise ValueError('Same K8 distinct absolute-deadline candidates required')
        candidate_sha = _digest(candidates)
        if self._candidate_sha is not None and self._candidate_sha != candidate_sha:
            raise ValueError('Candidate identity/deadlines changed between CEM batches')
        if not states or not actions or not parts:
            raise ValueError('Full original rollout/cost dictionaries required')
        for value in states.values():
            if np.asarray(value).shape[:2] != (n, HORIZON + 1):
                raise ValueError('Full H8 predicted-state ledger required')
        for value in actions.values():
            if np.asarray(value).shape[:2] != (n, HORIZON):
                raise ValueError('Full H8 action ledger required')
        if any(np.asarray(value).shape != (n,) for value in parts.values()):
            raise ValueError('One original cost-term value per scored row required')
        if 'joint_goal' not in actions or np.asarray(actions['joint_goal']).shape != (n, 8, 7):
            raise ValueError('Original full modeled joint-goal sequences required')
        additions = []
        finite = np.flatnonzero(np.isfinite(scores))
        order = finite[np.argsort(scores[finite], kind='stable')[:self.capacity]]
        for row in order:
            candidate = candidates[int(modes[row])]
            sequence = []
            for k in range(HORIZON):
                action = {key: np.asarray(value)[row, k].copy()
                    if np.asarray(value)[row, k].shape else np.asarray(value)[row, k].item()
                    for key, value in actions.items()}
                action['dt'] = dt
                sequence.append(action)
            predicted = [deepcopy(self._state)]
            for k in range(1, HORIZON + 1):
                item = {key: np.asarray(value)[row, k].copy()
                    if np.asarray(value)[row, k].shape else np.asarray(value)[row, k].item()
                    for key, value in states.items()}
                if 'lift_goal_position' in self._state:
                    item['lift_goal_position'] = deepcopy(self._state['lift_goal_position'])
                predicted.append(item)
            additions.append(dict(cost=float(scores[row]), iteration=iteration,
                row_index=int(row), candidate_index=int(modes[row]),
                candidate_id=candidate['id'], deadline=float(candidate['deadline']),
                candidate=deepcopy(candidate), sequence=sequence, predicted_states=predicted,
                cost_terms={key: float(np.asarray(value)[row]) for key, value in parts.items()},
                feedback_sha256=self.state_sha256))
        entries = self._entries + additions
        entries.sort(key=lambda item: (item['cost'], item['iteration'], item['row_index']))
        self._entries = entries[:self.capacity]
        self._candidate_sha = candidate_sha
        self._batches.append(dict(iteration=iteration, scored=n, finite=int(len(finite))))
        if len(self._batches) == 3:
            self._sealed_sha = _digest(self._entries)

    def ranked(self):
        if len(self._batches) != 3 or sum(row['scored'] for row in self._batches) != 256:
            raise ValueError('All original 256 scoring rows must be offered before qualification')
        if _digest(self._entries) != self._sealed_sha:
            raise ValueError('Retained modeled rollouts changed after scoring')
        return deepcopy(self._entries)


def select_first_qualified(pool, helper, backend, state, max_checks=CAPACITY, mode=None):
    """Query existing actions in cost order; never execute/re-solve an action.

    A successful row is a *whole selection bundle*: its candidate, H8 sequence,
    cost terms and predictions must replace the old best together, before the
    original phase/warm builder runs. Replacing only q while retaining rank0
    phase/candidate/warm is incorrect. This function does not commit any of them.
    """
    if type(pool) is not RankedRolloutPool or type(helper) is not PoEJointGoalHelper:
        raise TypeError('Exact pool and original PoEJointGoalHelper required')
    if getattr(helper.qualify_action, '__func__', None) is not PoEJointGoalHelper.qualify_action:
        raise TypeError('Original endpoint/seven-path qualification method required')
    if getattr(helper.solve_action_target, '__func__', None) is not PoEJointGoalHelper.solve_action_target:
        raise TypeError('Original exact-q/FK/bounds action solver required')
    max_checks = _integer(max_checks, 'max_checks')
    if max_checks != CAPACITY:
        raise ValueError('This research version fixes at most eight checks')
    if _digest(state) != pool.state_sha256:
        raise ValueError('Qualification must use the identical current feedback state')
    if not callable(getattr(backend, 'clock', None)):
        raise TypeError('Explicit current no-step clock service required')
    now = float(state['time'])
    def same_clock():
        value = backend.clock()
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or abs(float(value) - now) > 1e-8:
            raise ValueError('Collision query changed or did not match current physics clock')
    same_clock()
    attempts = []
    selected = None
    started = time.perf_counter()
    for rank, item in enumerate(pool.ranked()[:max_checks]):
        if _digest(state) != pool.state_sha256:
            raise ValueError('Feedback changed during lazy qualification')
        action = deepcopy(item['sequence'][0])
        opening = action.get('opening')
        if opening not in (0., 1.) or isinstance(opening, (bool, np.bool_)):
            raise ValueError('Original modeled binary opening required')
        reason = None
        if mode == 'REOPENING_UNKNOWN_CONTACT' and opening != 1.:
            reason = 'reopen_model_proposed_close'
        if mode == 'ACTIVE_VISUAL_VERIFICATION_PROBE' and opening != 0.:
            reason = 'probe_model_proposed_open'
        if reason:
            goal, qualification = None, dict(collision_probes=[], physical_steps=0)
        else:
            goal, reason, qualification = helper.qualify_action(backend, state, action,
                require_predicted_joint_goal=True)
        same_clock()
        if _digest(state) != pool.state_sha256:
            raise ValueError('Qualification mutated current feedback')
        attempts.append(dict(rank=rank, iteration=item['iteration'], row_index=item['row_index'],
            candidate_id=item['candidate_id'], cost=item['cost'], reason=reason,
            qualification=deepcopy(qualification)))
        if reason is None:
            if goal is None or not np.array_equal(np.asarray(goal), np.asarray(action['joint_goal'])):
                raise ValueError('Original qualifier must return the exact modeled q7')
            selected = deepcopy(item)
            selected['action'] = action
            selected['qualified_joint_goal'] = np.asarray(goal).copy()
            selected['selected_rank'] = rank
            break
    return dict(version=VERSION, status='QUALIFIED_FIRST_ACTION' if selected is not None else 'HOLD_REQUIRED',
        selection_bundle=selected, attempts=attempts, model_sequences_scored=256,
        retained_topK=CAPACITY, Native_collision_queries=sum(len(row['qualification'].get('collision_probes', [])) for row in attempts),
        physical_steps=0, executor_calls=0, new_model_rollouts=0,
        original_qualifier_used=True, whole_H8_collision_certified=False,
        phase_or_warm_committed=False, failure_hold_rule='actual_q_keep_requested_gripper_clear_warm',
        wall_seconds=time.perf_counter() - started, real_time_guaranteed=False,
        integrated=False, formal_qualification=False)
