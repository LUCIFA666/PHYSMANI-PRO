"""Exact old-control AST contracts for the independent V68 additions."""
import ast
from copy import deepcopy
import inspect
import textwrap
import unittest

from our_mani.local_mpc_control_20261103 import joint_goal_cem as old
from our_mani.local_mpc_control_20261103 import public_candidate_local_hold_v66 as hold
from our_mani.local_mpc_control_20261103 import public_candidate_scope12_v67 as v67
from our_mani.local_mpc_control_20261103 import response_seed_hold_planner_v68 as new
from our_mani.local_mpc_control_20261103 import public_candidate_response_seed_v68 as v68


def tree(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]


def dump(value):
    return ast.dump(value, include_attributes=False)


class ResponseSeedSourceContracts(unittest.TestCase):
    def test_rollout_cost_dynamics_constraints_exact_except_explicit_fresh_injection(self):
        def normalize(function):
            node = tree(function)
            node.body = [item for item in node.body if not (
                isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('warm_mask', 'fresh_mask') for t in item.targets))
                and not (isinstance(item, ast.If) and any(isinstance(n, ast.Name) and n.id == 'fixed_sequence' for n in ast.walk(item.test)))]
            class Strip(ast.NodeTransformer):
                def visit_If(self, item):
                    if any(isinstance(n, ast.Name) and n.id in ('warm_mask', 'fresh_mask') for n in ast.walk(item.test)):
                        return None
                    return self.generic_visit(item)
            return Strip().visit(node)
        self.assertEqual(dump(normalize(old.JointGoalCEMPlanner._rollout_batch)),
                         dump(normalize(new.ResponseSeedHoldPlannerV68._rollout_batch)))

    def test_dispatch_changes_only_hold_planner_class(self):
        before, after = tree(hold._V66V4Dispatch.decide), tree(v68._V68V4Dispatch.decide)
        class Rename(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == 'ResponseSeedHoldPlannerV68':node.id = 'LocalHoldPlannerV66'
                return node
        self.assertEqual(dump(before), dump(Rename().visit(after)))

    def test_runner_exact_dispatch_and_monitor_math_match_v67(self):
        before, after = tree(v67.run_public_candidate_scope12_v67), tree(v68.run_public_candidate_response_seed_v68)
        names = {'run_public_candidate_response_seed_v68': 'run_public_candidate_scope12_v67',
                 '_validate_v68_runner': '_validate_v67_runner',
                 'ObservedPublicCandidateResponseSeedV68Backend': 'ObservedPublicCandidateScope12V67Backend',
                 'CandidateResponseSeedV68Coordinator': 'CandidateScope12V67Coordinator'}
        strings = {'public_candidate_response_seed_v68_provenance': 'public_candidate_scope12_v67_provenance',
                   'public_candidate_position_v68_decisions': 'public_candidate_position_v67_decisions',
                   'P4_candidate_transformer_v68_': 'P4_candidate_transformer_v67_',
                   'P4_public_candidate_response_seed_v68_development': 'P4_public_candidate_scope12_v67_development'}
        class Normalize(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                node.name = names.get(node.name, node.name);return self.generic_visit(node)
            def visit_Name(self, node):
                node.id = names.get(node.id, node.id);return node
            def visit_Constant(self, node):
                if isinstance(node.value, str):
                    for a, b in strings.items():node.value = node.value.replace(a, b)
                return node
            def visit_Expr(self, node):
                if isinstance(node.value, ast.Call) and any(k.arg == 'response_seed' for k in node.value.keywords):return None
                return self.generic_visit(node)
        self.assertEqual(dump(before), dump(Normalize().visit(after)))

    def test_sampling_scoring_and_update_remain_inside_three_iteration_loop(self):
        node = tree(new.ResponseSeedHoldPlannerV68._plan_cold_with_fresh_seed)
        loop = next(n for n in node.body if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple))
        calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertEqual(sum(n.func.attr == '_rollout_batch' for n in calls), 1)
        self.assertTrue(any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Subscript)
            and isinstance(t.value, ast.Name) and t.value.id == 'samples' for t in n.targets) for n in ast.walk(loop)))
        self.assertTrue(any(isinstance(n, ast.If) and dump(n.test).find("value=0") >= 0
                            and any(isinstance(x, ast.Name) and x.id == 'iteration' for x in ast.walk(n.test)) for n in loop.body))
        # The replacement is local index1, preserving ordinary mean/zero index0.
        row_assignment = next(n for n in ast.walk(loop) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and t.slice.value == 'row_index' for t in n.targets))
        self.assertIsInstance(row_assignment.value, ast.BinOp)
        self.assertEqual(row_assignment.value.right.value, 1)

    def test_source_map_explicit_three_new_files_and_original_model_study(self):
        sources = v68.response_seed_producer_source_map()
        for name in ('response_seed_generator_v68.py', 'response_seed_hold_planner_v68.py',
                     'public_candidate_response_seed_v68.py', 'hold_response_feedforward_prototype_20261103.py'):
            self.assertEqual(sum(path.endswith('/' + name) for path in sources), 1)
        self.assertIn('CandidateScope12V67Coordinator', [c.__name__ for c in v68.CandidateResponseSeedV68Coordinator.__mro__])
        self.assertLess(v68.CandidateResponseSeedV68Coordinator.__mro__.index(v68._V68V4Dispatch),
                        v68.CandidateResponseSeedV68Coordinator.__mro__.index(hold._V66V4Dispatch))


if __name__ == '__main__':
    unittest.main()
