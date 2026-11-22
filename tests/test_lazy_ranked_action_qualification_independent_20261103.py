"""Independent synthetic contracts: real PoE qualifier, no Native or models."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from our_mani.local_mpc_control_20261103.joint_goals import (
    JointGoalConfig, PoEJointGoalHelper,
)
from our_mani.local_mpc_control_20261103.lazy_ranked_action_qualification_prototype_20261103 import (
    ITERATION_SIZES, RankedRolloutPool, select_first_qualified,
)
from our_mani.local_mpc_p2_20261103.action_adapter import matrix_quaternion
from our_mani.local_mpc_p2_20261103.kinematics import PoEKinematics, SCHEMA


class NoNativeBackend:
    def __init__(self, collision_calls=(), always_collide=False):
        self.now = 1.25
        self.calls = []
        self.collision_calls = set(collision_calls)
        self.always_collide = always_collide
        self.on_query = None

    def clock(self):
        return self.now

    def probe_joint_collision(self, q):
        self.calls.append(np.asarray(q).copy())
        if self.on_query is not None:
            self.on_query()
        return self.always_collide or len(self.calls) in self.collision_calls

    def solve_ik(self, *_args, **_kwargs):
        raise AssertionError('No Native IK or new solve')

    def step(self, *_args, **_kwargs):
        raise AssertionError('No physics step')


def fixture():
    matrix = np.eye(4)
    matrix[:3, 3] = [.2, 0., 1.]
    snapshot = dict(schema=SCHEMA, q_ref=np.zeros(7),
        space_screws_vw=np.tile([0., 0., 0., 0., 0., 1.], (7, 1)),
        tcp_reference_matrix=matrix, twist_order='v_omega')
    chain = PoEKinematics(snapshot)
    helper = PoEJointGoalHelper(chain,
        SimpleNamespace(lower=np.full(7, -2.5), upper=np.full(7, 2.5)),
        JointGoalConfig(lower=(-2., -2., .76), upper=(2., 2., 1.65)))
    state = dict(time=1.25, q=np.zeros(7), dq=np.zeros(7),
        tcp_position=matrix[:3, 3].copy(), tcp_quaternion=np.array([0., 0., 0., 1.]),
        tcp_velocity=np.zeros(3), object_position=np.array([.2, 0., .94]),
        object_quaternion=np.array([0., 0., 0., 1.]),
        object_linear_velocity=np.zeros(3), object_angular_velocity=np.zeros(3),
        opening=1., requested_opening=1., held=False,
        phase_state=dict(phase='align', dwell=.1, updated_at=1.25))
    candidates = [dict(id=f'candidate-{i}', deadline=1.8 + .1*i,
        object_tcp_position=np.array([0., 0., .06]),
        object_tcp_quaternion=np.array([0., 0., 0., 1.]),
        allowed_relative_speed=.08) for i in range(8)]
    return helper, state, candidates


def batch(helper, state, iteration):
    n = ITERATION_SIZES[iteration]
    offset = sum(ITERATION_SIZES[:iteration])
    q = np.zeros((n, 8, 7))
    q[:, :, 0] = .001 * (offset + np.arange(n)[:, None] + 1)
    q[:, :, 6] = np.arange(8)[None] * .00003
    transforms = helper.kinematics(q)
    actions = dict(joint_goal=q.copy(), tcp_position=transforms[:, :, :3, 3].copy(),
        tcp_quaternion=np.asarray([[matrix_quaternion(m[:3, :3]) for m in row]
            for row in transforms]),
        opening=np.ones((n, 8)), joint_goal_feasible=np.ones((n, 8), bool),
        tcp_velocity_reference=np.zeros((n, 8, 3)),
        angular_velocity_reference=np.zeros((n, 8, 3)))
    states = dict(q=np.concatenate((np.broadcast_to(state['q'], (n, 1, 7)), q), axis=1),
        time=np.broadcast_to(state['time'] + .1*np.arange(9), (n, 9)).copy(),
        opening=np.ones((n, 9)), held=np.zeros((n, 9), bool))
    scores = 10. + offset + np.arange(n, dtype=float)
    modes = np.repeat(np.arange(8), n//8)
    parts = dict(position=scores*.6, orientation=scores*.4)
    return scores, modes, parts, states, actions


def full_pool(helper, state, candidates, edits=None):
    pool = RankedRolloutPool(state)
    originals = []
    for i in range(3):
        scores, modes, parts, states, actions = batch(helper, state, i)
        if edits is not None:
            edits(i, scores, modes, parts, states, actions)
        pool.offer_batch(i, scores, modes, candidates, parts, states, actions, .1)
        originals.append((scores, modes, parts, states, actions))
    return pool, originals


class LazyRankedQualificationIndependentTests(unittest.TestCase):
    def test_rank_zero_matches_original_global_argmin_and_stable_ties(self):
        helper, state, candidates = fixture()
        def edits(i, scores, *_):
            if i == 0: scores[3] = 2.
            if i == 1: scores[2] = 1.
            if i == 2: scores[0] = 1.
        pool, originals = full_pool(helper, state, candidates, edits)
        old = None
        for i, (scores, *_rest) in enumerate(originals):
            row = int(np.argmin(scores))
            if old is None or scores[row] < old[0]:
                old = (float(scores[row]), i, row)
        ranked = pool.ranked()
        self.assertEqual((ranked[0]['cost'], ranked[0]['iteration'], ranked[0]['row_index']), old)
        self.assertEqual([(v['iteration'], v['row_index']) for v in ranked[:2]], [(1, 2), (2, 0)])
        self.assertEqual(len(ranked), 8)
        self.assertEqual(sum(v['scored'] for v in pool._batches), 256)

    def test_rejected_best_selects_second_whole_bundle_and_exact_C0(self):
        helper, state, candidates = fixture()
        pool, originals = full_pool(helper, state, candidates)
        backend = NoNativeBackend(collision_calls={1})
        original = deepcopy(state)
        result = select_first_qualified(pool, helper, backend, state)
        self.assertEqual(result['status'], 'QUALIFIED_FIRST_ACTION')
        selected = result['selection_bundle']
        self.assertEqual(selected['selected_rank'], 1)
        self.assertEqual(selected['row_index'], 1)
        self.assertEqual(selected['candidate_id'], candidates[0]['id'])
        np.testing.assert_array_equal(selected['candidate']['object_tcp_position'],
            candidates[0]['object_tcp_position'])
        np.testing.assert_array_equal(selected['qualified_joint_goal'], originals[0][4]['joint_goal'][1, 0])
        np.testing.assert_array_equal(selected['action']['joint_goal'], selected['sequence'][0]['joint_goal'])
        np.testing.assert_array_equal(selected['sequence'][-1]['joint_goal'], originals[0][4]['joint_goal'][1, -1])
        self.assertEqual(len(selected['predicted_states']), 9)
        self.assertEqual(selected['deadline'], candidates[0]['deadline'])
        self.assertEqual(selected['cost_terms']['position'], originals[0][2]['position'][1])
        self.assertEqual(len(backend.calls), 9)
        self.assertEqual(result['attempts'][1]['qualification']['DLS_solves'], 0)
        self.assertEqual(result['executor_calls'], 0)
        self.assertEqual(result['new_model_rollouts'], 0)
        self.assertFalse(result['phase_or_warm_committed'])
        np.testing.assert_array_equal(state['q'], original['q'])
        self.assertEqual(state['phase_state'], original['phase_state'])

    def test_original_endpoint_plus_all_seven_path_samples_are_preserved(self):
        helper, state, candidates = fixture()
        pool, _ = full_pool(helper, state, candidates)
        backend = NoNativeBackend()
        result = select_first_qualified(pool, helper, backend, state)
        probes = result['attempts'][0]['qualification']['collision_probes']
        alphas = [1., .125, .25, .375, .5, .625, .75, .875]
        self.assertEqual([p['alpha'] for p in probes], alphas)
        q = result['selection_bundle']['qualified_joint_goal']
        for alpha, actual in zip(alphas, backend.calls):
            np.testing.assert_array_equal(actual, q if alpha == 1. else (1-alpha)*state['q'] + alpha*q)
        self.assertEqual(result['Native_collision_queries'], 8)
        self.assertFalse(result['whole_H8_collision_certified'])

    def test_all_rejected_or_all_model_infinite_returns_hold_without_commit(self):
        helper, state, candidates = fixture()
        pool, _ = full_pool(helper, state, candidates)
        backend = NoNativeBackend(always_collide=True)
        result = select_first_qualified(pool, helper, backend, state)
        self.assertEqual(result['status'], 'HOLD_REQUIRED')
        self.assertIsNone(result['selection_bundle'])
        self.assertEqual(len(result['attempts']), 8)
        self.assertEqual(len(backend.calls), 8)
        self.assertFalse(result['phase_or_warm_committed'])
        def infinite(_i, scores, *_): scores[:] = np.inf
        pool, _ = full_pool(helper, state, candidates, infinite)
        backend = NoNativeBackend()
        result = select_first_qualified(pool, helper, backend, state)
        self.assertEqual(result['status'], 'HOLD_REQUIRED')
        self.assertEqual(result['attempts'], [])
        self.assertEqual(backend.calls, [])

    def test_feedback_mutation_and_clock_change_are_not_accepted(self):
        helper, state, candidates = fixture()
        pool, _ = full_pool(helper, state, candidates)
        changed = deepcopy(state); changed['q'][0] += .01
        with self.assertRaises(ValueError):
            select_first_qualified(pool, helper, NoNativeBackend(), changed)
        backend = NoNativeBackend()
        backend.on_query = lambda: state['phase_state'].update(dwell=.2)
        with self.assertRaises(ValueError):
            select_first_qualified(pool, helper, backend, state)
        helper, state, candidates = fixture()
        pool, _ = full_pool(helper, state, candidates)
        backend = NoNativeBackend()
        backend.on_query = lambda: setattr(backend, 'now', 1.30)
        with self.assertRaises(ValueError):
            select_first_qualified(pool, helper, backend, state)

    def test_reopen_and_probe_modes_keep_original_binary_opening_gate(self):
        helper, state, candidates = fixture()
        def closed_best(i, _scores, _modes, _parts, _states, actions):
            if i == 0: actions['opening'][0] = 0.
        pool, _ = full_pool(helper, state, candidates, closed_best)
        result = select_first_qualified(pool, helper, NoNativeBackend(), state,
            mode='REOPENING_UNKNOWN_CONTACT')
        self.assertEqual(result['attempts'][0]['reason'], 'reopen_model_proposed_close')
        self.assertEqual(result['attempts'][0]['qualification']['collision_probes'], [])
        self.assertEqual(result['selection_bundle']['selected_rank'], 1)
        result = select_first_qualified(pool, helper, NoNativeBackend(), state,
            mode='ACTIVE_VISUAL_VERIFICATION_PROBE')
        self.assertEqual(result['selection_bundle']['selected_rank'], 0)
        self.assertEqual(result['selection_bundle']['action']['opening'], 0.)

    def test_full_budget_and_original_helper_identity_are_required(self):
        helper, state, candidates = fixture()
        pool = RankedRolloutPool(state)
        with self.assertRaises(ValueError): pool.ranked()
        scores, modes, parts, states, actions = batch(helper, state, 0)
        pool.offer_batch(0, scores, modes, candidates, parts, states, actions, .1)
        with self.assertRaises(ValueError): pool.offer_batch(0, scores, modes, candidates, parts, states, actions, .1)
        with self.assertRaises(ValueError): select_first_qualified(pool, helper, NoNativeBackend(), state)
        pool, _ = full_pool(helper, state, candidates)
        helper.qualify_action = lambda *_a, **_k: (np.zeros(7), None, {})
        with self.assertRaises(TypeError): select_first_qualified(pool, helper, NoNativeBackend(), state)
        with self.assertRaises(ValueError): select_first_qualified(pool, fixture()[0], NoNativeBackend(), state, max_checks=1)


if __name__ == '__main__':
    unittest.main()
