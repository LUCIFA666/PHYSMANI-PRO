"""Fixed scalar profile contracts; synthetic only, never real fresh scoring."""
from pathlib import Path
from unittest.mock import patch
import ast
import inspect
import unittest
import numpy as np
from our_mani.near_goal_depth_refiner_20261103_v6 import evaluation as e, fresh_evaluation as f, qualification as q
from our_mani.near_goal_depth_refiner_20261103_v5 import qualification as oldq
from our_mani.near_goal_fresh4_calibration_20261103 import evaluation as oldf
from scripts import qualify_near_goal_depth_refiner_20261103_v6 as cli
ROOT=Path(__file__).resolve().parents[1]


def fixture(precision=.995,coverage=.95):
    rows=[dict(scene_id=p+'/'+s,eligible=s in ('shot02','shot03','shot04','shot05','shot06'))for p in f.PARTITION_IDS for s in f.SHOT_IDS]
    eligibility=dict(protocol=f.BASELINE_ELIGIBILITY_PROTOCOL,scope='fresh4_calibration',records=rows,
        frozen_before_NN_predictions=True,GT_not_forwarded=True,required_eligible=20)
    eligibility['content_sha256']=f._canonical_sha(eligibility)
    report=dict(threshold=.995,pixels=dict(predicted_positive_count=1000,conservative_precision_lowerbound=precision,
        known_precision=1.,U=1),positions=dict(samples=32,oracle_geometry_eligible=20,oracle_eligible_coverage=coverage,
        qualified_visible_centre_error={'p95_m':.002},owner_unobserved_falseavailable=0,
        negatives={'sphere_only':{'samples':8,'falseavailable':0},'cube_far':{'samples':4,'falseavailable':0}}),
        records=[dict(scene_id=r['scene_id'],oracle_geometry_eligible=r['eligible'])for r in rows],
        baseline_eligibility_protocol=f.BASELINE_ELIGIBILITY_PROTOCOL,geometry_version=f.GEOMETRY_VERSION,
        fresh_eligibility_sha256=eligibility['content_sha256'])
    return eligibility,report


class FixedProfileTests(unittest.TestCase):
    def test_evaluation_only_changes_two_declared_constants(self):
        a=ast.parse((ROOT/'our_mani/near_goal_depth_refiner_20261103_v4/evaluation.py').read_text())
        b=ast.parse(inspect.getsource(e))
        for tree in (a,b):
            for node in tree.body:
                if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id in {'THRESHOLD_GRID','QUALITY_PROTOCOL'}for t in node.targets):
                    node.value=ast.Constant(value='declared_profile_constant')
        self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))
        self.assertEqual(e.THRESHOLD_GRID,(.995,))

    def test_fresh_boolean_numerical_gates_are_unchanged(self):
        def passed(module):
            tree=ast.parse(inspect.getsource(module.choose_fresh_calibrated_threshold))
            node=next(n for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='passed'for t in n.targets))
            return ast.dump(node.value,include_attributes=False)
        self.assertEqual(passed(f),passed(oldf))

    def test_scalar_pass_fail_and_no_fresh_threshold_search(self):
        eligibility,report=fixture()
        result=f.choose_fresh_calibrated_threshold([report],eligibility)
        self.assertTrue(result['calibration_passed']);self.assertEqual(result['selected_threshold'],.995)
        self.assertFalse(result['threshold_selected_from_fresh']);self.assertFalse(result['selection_stage_raw_gate_pass'])
        for precision,coverage in ((.994999,.95),(.995,.89999)):
            eligibility,report=fixture(precision,coverage)
            failed=f.choose_fresh_calibrated_threshold([report],eligibility)
            self.assertFalse(failed['calibration_passed']);self.assertIsNone(failed['selected_threshold'])
        eligibility,report=fixture();report['threshold']=.999
        with self.assertRaises(ValueError):f.choose_fresh_calibrated_threshold([report],eligibility)

    def test_raw_greater_equal_and_other_threshold_rejected(self):
        p=np.empty((4,2,64,64),np.float32);p[:,1]=np.float32(.995);p[:,0]=1-p[:,1]
        crops={'valid_crop':np.ones((4,64,64),bool)}
        with patch.object(e,'measure_current_cube',return_value={}):
            self.assertTrue(e.probability_to_observation({},crops,p,.995)['raw_Cube_mask'].all())
            p[:,1]=np.nextafter(np.float32(.995),np.float32(0));p[:,0]=1-p[:,1]
            self.assertFalse(e.probability_to_observation({},crops,p,.995)['raw_Cube_mask'].any())
            with self.assertRaises(ValueError):e.probability_to_observation({},crops,p,.99)

    def test_fit_proof_is_same_function_and_distinct_from_profile(self):
        self.assertIs(q.validate_selected_checkpoint,oldq.validate_selected_checkpoint)
        self.assertEqual(q.FIT_PROTOCOL_SHA256,'472c14d9b10ecbd06a67f9bdd545543ec235f906ea571e322c5c660775b94710')
        self.assertEqual(q.PROTOCOL_SHA256,'cfb3785d6888e2f73c3c972c4f0780664d4af93c5f48f15330bd707e39e18725')
        profile=q.check_profile_bindings()
        self.assertEqual(profile['fixed_threshold'],.995);self.assertFalse(profile['selection_basis']['selection_raw_gate_pass'])

    def test_default_cli_never_consumes(self):
        with patch.object(cli,'check_qualification_inputs',return_value={'complete':True}),\
             patch.object(cli,'qualify_selected_checkpoint',side_effect=AssertionError('no consumption'))as run:
            self.assertEqual(cli.main(['--selected','unused','--expected-sha256','a'*64,'--fresh-manifest','unused']),0)
            run.assert_not_called()


if __name__=='__main__':unittest.main()
