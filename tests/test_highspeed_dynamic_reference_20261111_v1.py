"""Reachable warm/expiry contracts plus exact pending-close/top8 preservation."""
import ast
from copy import deepcopy
import inspect
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

from our_mani.contracts import Forecast
from our_mani.cube_native_contact_feedback_live_20261104_v1.planner import NativePendingForecastPlannerV1
from our_mani.local_mpc_control_20261103.lazy_free_joint_goal_cem_v72 import _LazyJointGoalCEMPlannerV72
from our_mani.local_mpc_control_20261103.forecast_control import BoundForecastExecutionModel
from our_mani.highspeed_control_20261111_v1.dynamic_cem import (
    DynamicPendingForecastPlanner, _DynamicLazyPlanner, DYNAMIC_BASIS,
)
from tests.test_local_mpc_control_20261103_joint_goal_cem import fixture as base_fixture
from tests import test_cube_dynamic_reference_independent_20261104_v1 as old_tests


def reachable_forecast(state):
    t = np.arange(17)*.05
    state['object_linear_velocity'] = np.array([.0001, 0., 0.])
    state['object_angular_velocity'] = np.zeros(3)
    return Forecast(('target',), state['time']+t,
        state['object_position'][None,None]+t[:,None,None]*state['object_linear_velocity'][None,None],
        np.tile(np.eye(3),(17,1,1,1)), np.tile(state['object_linear_velocity'],(17,1,1)), np.zeros((17,1,3)))


def reachable_fixture(now=0., held=False):
    base, state, candidates = base_fixture()
    state['time'] = now
    state['phase_state']['updated_at'] = now-.1
    state['held'] = held
    for c in candidates: c['deadline'] += now
    forecast = reachable_forecast(state)
    binding = BoundForecastExecutionModel(base.execution_model, forecast, anchor_time=now)
    p = old_tests.module.DynamicReferenceForecastJointGoalCEMPlannerV1(execution_model=binding, goal_helper=base.goal_helper)
    return p, state, candidates, forecast


class Backend:
    def __init__(self, now): self.now, self.queries = now, 0
    def clock(self): return self.now
    def probe_joint_collision(self, q): self.queries += 1; return False


def live_fixture(state=None, candidates=None, model=None, helper=None):
    if state is None:
        base, state, candidates = base_fixture()
        model, helper = base.execution_model, base.goal_helper
    state = deepcopy(state)
    state.update(Native_contact_feedback_used=True, Native_held_used=True,
        Native_object_state_used=False, Native_success_used=False, grasp_motion_mode='FREE_GRU')
    state['actuator_state']['gripper_target'] = state['requested_opening']
    forecast = reachable_forecast(state)
    binding = BoundForecastExecutionModel(model, forecast, anchor_time=state['time'],
        public_bridge_verified=True, held_source='public_estimate')
    backend = Backend(state['time'])
    p = DynamicPendingForecastPlanner(execution_model=binding, goal_helper=helper, backend=backend)
    return p, state, candidates, backend


class DynamicContracts(unittest.TestCase):
    def test_prior_warm_and_expiry_tests_on_reachable_nonstationary_fixture(self):
        # The old fixture loses grasp IK at late H8 endpoints. These two tests
        # need a complete feasible first plan; production thresholds stay exact.
        with patch.object(old_tests, 'fixture', side_effect=reachable_fixture), patch.object(old_tests, 'forecast_for', side_effect=reachable_forecast):
            old_tests.DynamicReferenceIndependent.test_same_frame_and_shift_one_absolute_warm_not_forecast_or_deadline_shift(self)
            old_tests.DynamicReferenceIndependent.test_expired_warm_dropped_and_cannot_revive_old_deadline(self)

    def test_native_pending_rollout_changes_only_four_dynamic_table_indices(self):
        def node(fn): return ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
        count = [0]
        class Normalize(ast.NodeTransformer):
            def visit_Subscript(self, n):
                self.generic_visit(n)
                if isinstance(n.value,ast.Subscript) and isinstance(n.value.value,ast.Attribute) and n.value.value.attr=='_nominals':
                    if isinstance(n.slice,ast.Tuple) and len(n.slice.elts)==3 and isinstance(n.slice.elts[-1],ast.Name) and n.slice.elts[-1].id=='k':
                        n.slice.elts=n.slice.elts[:-1];count[0]+=1
                return n
        self.assertEqual(ast.dump(node(NativePendingForecastPlannerV1._rollout_batch)), ast.dump(Normalize().visit(node(_DynamicLazyPlanner._rollout_batch))))
        self.assertEqual(count[0],4)

    def test_ranked_collision_selection_and_cem_change_only_warm_rebase_index(self):
        def node(fn): return ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
        class Normalize(ast.NodeTransformer):
            def visit_Subscript(self,n):
                self.generic_visit(n)
                if isinstance(n.value,ast.Subscript) and isinstance(n.value.value,ast.Attribute) and n.value.value.attr=='_nominals':
                    if isinstance(n.slice,ast.Tuple) and len(n.slice.elts)==3:n.slice.elts=n.slice.elts[:2]
                return n
        self.assertEqual(ast.dump(node(_LazyJointGoalCEMPlannerV72.plan)),ast.dump(Normalize().visit(node(_DynamicLazyPlanner.plan))))

    def test_public_dynamic_plan_keeps_original_collision_queries_and_absolute_warm(self):
        p,s,c,b = live_fixture()
        r = p.plan(s,c)
        self.assertIsNotNone(r['action'])
        self.assertGreater(b.queries,0)
        self.assertEqual(r['budget']['sequences'],256)
        self.assertEqual(r['warm_start']['nominal_basis_version'],DYNAMIC_BASIS)
        next_state=deepcopy(r['predicted_states'][1])
        next_state['actuator_state']={}
        next_state['phase_state']=deepcopy(r['phase_state'])
        p2,s2,c2,b2=live_fixture(next_state,c,p.execution_model.execution_model,p.goal_helper)
        r2=p2.plan(s2,c2,warm_start=r['warm_start'])
        self.assertEqual(r2['diagnostics']['warm_start_evaluations_counted'],3)
        self.assertEqual(r2['warm_start']['nominal_basis_version'],DYNAMIC_BASIS)
        self.assertEqual(r['warm_start']['deadline'],next(v['deadline'] for v in c if v['id']==r['selected_candidate_id']))

    def test_private_or_unknown_warm_and_mutated_feedback_rejected(self):
        p,s,c,b=live_fixture()
        with self.assertRaises(ValueError): p.plan(dict(s,Native_object_state_used=True),c)
        p,s,c,b=live_fixture()
        r=p.plan(s,c)
        p2,s2,c2,b2=live_fixture()
        with self.assertRaises(ValueError):p2.plan(s2,c2,warm_start=dict(r['warm_start'],nominal_basis_version='static'))


if __name__=='__main__':unittest.main()
