"""Independent source/pure contracts; no model weights, forward or Native launch."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from our_mani.cube_v72_live_20261104_v1 import backend, contracts, candidate_runtime, producer
from our_mani.local_mpc_control_20261103 import public_candidate_stable_dwell_v72 as old
from our_mani.local_mpc_20261102.native import TruthNativeBackend

ROOT = Path(__file__).resolve().parents[1]


def tree(fn):
    return ast.parse(textwrap.dedent(inspect.getsource(fn)))


class NormalizeRunner(ast.NodeTransformer):
    names = {'run_cube_v72_live':'run_public_candidate_stable_dwell_v72',
        'ObservedCubeV72Backend':'ObservedPublicCandidateStableDwellV72Backend',
        'CubeV72Coordinator':'CandidateStableDwellV72Coordinator'}
    def visit_Name(self, node):
        node.id = self.names.get(node.id,node.id);return node
    def visit_FunctionDef(self,node):
        node.name=self.names.get(node.name,node.name);return self.generic_visit(node)
    def visit_ImportFrom(self,node):
        if node.module and node.module.startswith('local_mpc_control_20261103.'):
            node.module=node.module.split('.',1)[1];node.level=1
        return node


class CubeLiveIndependent(unittest.TestCase):
    def test_original_observer_coordinator_and_mechanics_methods(self):
        for name in ('sample','tick'):
            self.assertIs(getattr(producer.ObservedCubeV72Backend,name),
                          getattr(old.ObservedPublicCandidateStableDwellV72Backend,name))
        self.assertIs(producer.CubeV72Coordinator.decide,old.CandidateStableDwellV72Coordinator.decide)
        for name in ('solve_ik','probe_joint_collision','set_joint_target_positions',
                     'actuate_gripper','tick','close'):
            self.assertIs(getattr(backend.CubeTaskNativeBackendV1,name),getattr(TruthNativeBackend,name))

    def test_runner_entire_ast_only_declared_names_import_path(self):
        actual=NormalizeRunner().visit(tree(producer.run_cube_v72_live))
        reference=tree(old.run_public_candidate_stable_dwell_v72)
        self.assertEqual(ast.dump(actual,include_attributes=False),ast.dump(reference,include_attributes=False))

    def test_actual_tasks_generated_reset_no_motion_or_pose_assignment(self):
        self.assertEqual(backend.TASKS,('pick_moving_target_on_the_table',
                                      'pick_moving_target_on_the_table_high_speed'))
        source=tree(backend.CubeTaskNativeBackendV1)
        stores=[n for n in ast.walk(source) if isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store)]
        forbidden={'target_state_list','target_block','t','time','velocity','motion'}
        self.assertFalse([ast.unparse(n) for n in stores if n.attr in forbidden])
        calls=[n for n in ast.walk(source) if isinstance(n,ast.Call)]
        initial=[n for n in calls if ast.unparse(n.func)=='AlignedLocalSimulation.__init__']
        self.assertEqual(len(initial),1)
        self.assertEqual(ast.unparse(initial[0].args[1]),'task_name')
        reset=tree(backend.CubeTaskNativeBackendV1.reset_scenario)
        self.assertEqual(sum(isinstance(n,ast.Call) and ast.unparse(n.func)=='self.raw.task_environment.reset'
                             for n in ast.walk(reset)),1)
        with self.assertRaises(ValueError):
            backend.CubeTaskNativeBackendV1('unsupported','missing',seed=1,execute_native=False)

    def test_fixed_all_eight_profile_case_pairs_and_control_reference(self):
        plan=contracts.validate_plan(ROOT/'config/framework/cube_v72_live_20261104_v1/plan.json')
        self.assertEqual(len(plan['cases']),4)
        self.assertEqual(len(plan['profiles_order'])*len(plan['cases']),8)
        self.assertEqual([c['seed'] for c in plan['cases']],[2611041100,2611041101,2611041110,2611041111])
        reference=contracts.read(contracts.REFERENCE)
        allowed={'schema','task','seed','variation','simulation_seconds','dataset_use','formal_evaluation',
            'calibration_train_episode','candidate_profile','candidate_transformer_checkpoint',
            'candidate_transformer_checkpoint_sha256','candidate_model_freeze_manifest','native_generated_motion',
            'Native_labels_used_by_policy','official_VAL_or_TEST_source','P4_collection_metadata',
            'native_task_source_sha256','native_task_scene_sha256'}
        for case in plan['cases']:
            configs=[contracts.case_config(plan,case,p) for p in plan['profiles_order']]
            for config in configs:
                self.assertEqual({k:v for k,v in config.items() if k not in allowed},
                                 {k:v for k,v in reference.items() if k not in allowed})
                self.assertEqual(config['simulation_seconds'],8.)
                self.assertEqual(config['task'],case['task'])
            differing={k for k in configs[0] if configs[0][k]!=configs[1][k]}
            self.assertEqual(differing,{'candidate_profile','candidate_transformer_checkpoint',
                                      'candidate_transformer_checkpoint_sha256'})
        self.assertEqual(plan['candidate_profiles']['cube_candidate_v1']['sha256'],
                         '81391a2d26f0a601f76364306ee9b924edddc7f3dd38276000de96859e34553b')

    def test_negative_checkpoint_and_provider_admission_before_load(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'not_checkpoint.pt';p.write_bytes(b'not weights')
            with patch.object(candidate_runtime.torch,'load',side_effect=AssertionError('must not load')):
                with self.assertRaises(ValueError):
                    candidate_runtime.FrozenCubeCandidateInferenceV1(p,expected_sha256='0'*64)
                with self.assertRaises(ValueError):
                    candidate_runtime.candidate_loader(p,'0'*64,'world_tcp')
        with self.assertRaises(TypeError):
            candidate_runtime.CubeLearnedCandidatesV1(object(),object())

    def test_ledger_and_symmetry_math_is_original(self):
        for name in ('update','_validate_current','events','candidates'):
            self.assertIs(getattr(candidate_runtime.CubeLearnedCandidatesV1,name),
                          getattr(candidate_runtime.LearnedCandidates,name))
        self.assertIs(candidate_runtime.CubeSymmetricLearnedCandidatesV1.update,
                      candidate_runtime.SymmetricLearnedCandidates.update)
        ctor=inspect.getsource(candidate_runtime.FrozenCubeCandidateInferenceV1.__init__)
        for field in ('training_updates','task_classifier_trained','train_parents','selection_parents','model_spec'):
            self.assertIn(field,ctor)
        self.assertNotIn('WorldTCP',inspect.getsource(producer.run_cube_v72_live))


if __name__=='__main__':unittest.main()
