"""Focused production-hook CPU review; no Native, training or core writes."""
import ast
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from our_mani.local_mpc_control_20261103 import public_grasp_candidates_v5 as v5
from our_mani.local_mpc_control_20261103 import public_grasp_v4 as v4
from our_mani.local_mpc_control_20261103.active_visual_attachment import ActiveVisualAttachment
from our_mani.local_mpc_control_20261103.probe_nominal_cache_v53 import CachedNominalVerificationProbePlanner
from our_mani.local_mpc_control_20261103.public_grasp_probe import VerificationProbePlanner
from our_mani.local_mpc_control_20261103.response_factory import build_planner
from our_mani.local_mpc_p4_20261103.runtime import FrozenPilotInference,PublicRowCache
from our_mani.local_mpc_20261102.records import freeze_sources,sha
from scripts.replay_probe_nominal_counterfactual_20261103 import candidate_ledger,numpy_spec
from scripts.replay_probe_cached_nominal_counterfactual_20261103 import make_limits

ROOT=Path(__file__).resolve().parents[1]
CASE=ROOT/'runtime/framework/local_mpc_20261102/public_v41_probe12_canaries_v1/uniform_012'


class SyntheticCollisionOnly:
    """CPU code-path fixture, never a Native collision certificate."""
    def __init__(self):self.calls=[]
    def probe_joint_collision(self,q):self.calls.append(np.asarray(q).copy());return False


class ProductionHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows=[json.loads(x) for x in (CASE/'public_grasp_v41_decisions.jsonl').read_text().splitlines()]
        cls.rows=[r for r in rows if r['decision']['mode']=='ACTIVE_VISUAL_VERIFICATION_PROBE'][:2]
        cls.core,cls.helper,_=build_planner(ROOT,make_limits(cls.rows[0]),controller='coupled',action_space='joint',
            joint_goal_config={'lower':(-.1,-.65,.755)},terminal_config={'enabled':True},reference_frequency=6.)
        cls.inference=FrozenPilotInference(ROOT/'artifacts/framework/local_mpc_20261102/P4_candidate_training_v1/final.pt',
            expected_sha256='61e5827b4838649b8e477048bea56fcace18697f906fb610a3f963bba2437f60')

    def coordinator(self,**kwargs):
        return v5.CandidateGraspV5Coordinator(self.core,self.helper,SimpleNamespace(),ActiveVisualAttachment(),
            inference=self.inference,cache=PublicRowCache(),**kwargs)

    def test_default_and_false_delegate_to_original_components(self):
        for kwargs in ({},{'cache_fixed_probe_nominal_ik':False}):
            coordinator=self.coordinator(**kwargs)
            self.assertIsNone(coordinator.probe_nominal_cache)
            self.assertIsNone(coordinator.probe_cache_sources)
            sentinel=(object(),object(),object())
            with patch.object(v4.ActiveVisualGraspCoordinator,'_verification_components',return_value=sentinel) as original:
                result=coordinator._verification_components('backend','state','pool','model','spec')
                self.assertIs(result,sentinel);original.assert_called_once_with('backend','state','pool','model','spec')

    def test_two_recorded_real_feedback_frames_use_one_cache_and_fresh_typed_qualification(self):
        coordinator=self.coordinator(cache_fixed_probe_nominal_ik=True);cache=coordinator.probe_nominal_cache
        keys=[];planners=[];feedback=[];goals=[];state_copies=[]
        for row in self.rows:
            state=deepcopy(row['decision']['state']);spec=numpy_spec(row['decision']['probe'])
            planning=candidate_ledger(row)
            # FREE expiry is intentionally irrelevant to admitted probe scope.
            for c in planning:c['proposal_available']=False;c['parallel_gripper_symmetry_choice']=0
            state_copies.append(deepcopy(state));oracle=SyntheticCollisionOnly()
            candidates,receipt,planner=coordinator._verification_components(oracle,state,planning,self.core.execution_model,spec)
            self.assertIs(type(planner),CachedNominalVerificationProbePlanner)
            self.assertIs(planner.nominal_cache,cache);self.assertEqual(len(oracle.calls),8)
            self.assertTrue(all(c['native_pose_qualified'] for c in candidates))
            result=planner.plan(state,candidates)
            self.assertEqual(result['budget']['sequences'],256);self.assertEqual(result['budget']['horizon'],8)
            self.assertEqual(result['budget']['iteration_sequences'],[88,88,80])
            self.assertEqual(result['action']['opening'],0.)
            self.assertTrue(all(not s['held'] for s in result['predicted_states']))
            self.assertEqual(spec['deadline'],row['decision']['probe']['deadline'])
            self.assertEqual(spec['config']['lift_m'],.012)
            self.assertFalse(receipt['old_fresh_DLS_certificate_reused'])
            # The production runner's unchanged selected gate still performs
            # fresh point +7path probes and returns exactly the selected q.
            before=len(oracle.calls)
            goal,reason,selected=self.helper.qualify_action(oracle,state,result['action'],require_predicted_joint_goal=True)
            self.assertIsNone(reason);self.assertEqual(len(oracle.calls)-before,8)
            np.testing.assert_array_equal(goal,result['action']['joint_goal'])
            self.assertEqual(selected['DLS_solves'],0)
            keys.append(receipt['cache_key']);feedback.append(receipt['feedback_sha256']);planners.append(planner)
            goals.append(cache.lookup(receipt['cache_key'])['joint_goal'])
            self.assertEqual(state['actuator_state'],state_copies[-1]['actuator_state'])
        self.assertIsNot(planners[0],planners[1]);self.assertIs(coordinator.probe_nominal_cache,cache)
        self.assertEqual(cache.solve_count,1);self.assertEqual(cache.reuse_count,1)
        self.assertEqual(keys[0],keys[1]);self.assertNotEqual(feedback[0],feedback[1])
        np.testing.assert_array_equal(goals[0],goals[1])
        with self.assertRaisesRegex(ValueError,'fresh actual state'):
            planners[0].plan(self.rows[1]['decision']['state'],candidate_ledger(self.rows[1]))

    def test_declared_source_hashes_are_actual_and_source_inventory_gates_the_flag(self):
        coordinator=self.coordinator(cache_fixed_probe_nominal_ik=True)
        self.assertEqual(len(coordinator.probe_cache_sources),8)
        for name,expected in coordinator.probe_cache_sources.items():
            self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(),expected)
        config=json.loads((ROOT/'config/framework/P4_student_closedloop_devVAL_v52_v1/00_val_static.json').read_text())
        with TemporaryDirectory(prefix='v53_CPU_source_inventory_',dir=ROOT/'config/framework') as directory:
            path=Path(directory)/'config.json'
            config['cache_fixed_probe_nominal_ik']=False;path.write_text(json.dumps(config));old=freeze_sources(path,'p1')
            config['cache_fixed_probe_nominal_ik']=True;path.write_text(json.dumps(config));new=freeze_sources(path,'p1')
        module='our_mani/local_mpc_control_20261103/probe_nominal_cache_v53.py'
        self.assertNotIn(module,old);self.assertEqual(new[module],sha(ROOT/module))
        self.assertEqual(set(new)-set(old),{module})
        self.assertGreaterEqual(len(new),92)

    def test_only_verification_constructor_extracted_from_prior_v52_decide_monitor_unchanged(self):
        snapshot=ROOT/'runtime/framework/local_mpc_20261102/P4_student_closedloop_devVAL_v52_v1/00_val_static/source_snapshot'
        original=ast.parse((snapshot/'our_mani/local_mpc_control_20261103/public_grasp_v4.py').read_text())
        oldclass=next(node for node in original.body if isinstance(node,ast.ClassDef) and node.name=='ActiveVisualGraspCoordinator')
        old=next(node for node in oldclass.body if isinstance(node,ast.FunctionDef) and node.name=='decide')
        current=ast.parse(inspect.getsource(v4.ActiveVisualGraspCoordinator.decide).lstrip()).body[0]
        # Dedent is required for method source after stripping the first line.
        import textwrap
        current=ast.parse(textwrap.dedent(inspect.getsource(v4.ActiveVisualGraspCoordinator.decide))).body[0]
        class Normalize(ast.NodeTransformer):
            def visit_If(self,node):
                if ast.dump(node.test)==ast.dump(ast.parse("mode=='ACTIVE_VISUAL_VERIFICATION_PROBE'",mode='eval').body):
                    node.body=[ast.Pass()]
                return self.generic_visit(node)
        self.assertEqual(ast.dump(Normalize().visit(old)),ast.dump(Normalize().visit(current)))
        active='our_mani/local_mpc_control_20261103/active_visual_attachment.py'
        self.assertEqual(sha(snapshot/active),sha(ROOT/active))
        bridge=next(node for node in original.body if isinstance(node,ast.ClassDef) and node.name=='ActiveVisualPublicBridge')
        observe=next(node for node in bridge.body if isinstance(node,ast.FunctionDef) and node.name=='observe')
        current_observe=ast.parse(textwrap.dedent(inspect.getsource(v4.ActiveVisualPublicBridge.observe))).body[0]
        self.assertEqual(ast.dump(observe),ast.dump(current_observe))


if __name__=='__main__':unittest.main()
