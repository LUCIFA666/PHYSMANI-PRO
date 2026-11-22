"""New-source contracts; no renderer, physics, model forward or TEST data."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

from our_mani.local_mpc_control_20261103 import public_candidate_position_v64 as old
from our_mani.local_mpc_control_20261103 import public_candidate_reverification_v65 as new
from our_mani.local_mpc_p4_20261103.runtime import _walk_public
from tests.test_public_reverification_epoch_prototype_20261103 import frame


class V65SourceContracts(unittest.TestCase):
    def test_runner_has_exact_original_control_math_dispatch_and_budget(self):
        text=inspect.getsource(new.run_public_candidate_reverification_v65)
        replacements={
            'run_public_candidate_reverification_v65':'run_public_candidate_position_v64',
            '_validate_v65_runner':'_validate_v64_runner',
            'ObservedPublicCandidateReverificationV65Backend':'ObservedPublicCandidatePositionV64Backend',
            'CandidateReverificationV65Coordinator':'CandidatePositionV64Coordinator',
            'public_candidate_reverification_v65_provenance.json':'public_candidate_position_v64_provenance.json',
            'public_candidate_position_v65_decisions.jsonl':'public_candidate_position_v64_decisions.jsonl',
            'P4_candidate_transformer_v65_':'P4_candidate_transformer_v64_',
            'P4_public_candidate_reverification_v65_development':'P4_public_candidate_position_v64_development',
        }
        for a,b in sorted(replacements.items(),key=lambda pair:-len(pair[0])):text=text.replace(a,b)
        self.assertEqual(ast.dump(ast.parse(text),include_attributes=False),
            ast.dump(ast.parse(inspect.getsource(old.run_public_candidate_position_v64)),include_attributes=False))

    def test_source_closure_retains_original_and_adds_exact_epoch_dependencies(self):
        sources=new.position_producer_source_map()
        self.assertTrue(set(old.position_producer_source_map()).issubset(sources))
        for name,sha in new.EXTRA_SOURCES.items():self.assertEqual(sources[str((new.ROOT/name).resolve())],sha)
        self.assertIn(str(Path(new.__file__).resolve()),sources)

    def test_outer_inner_robot_identity_is_value_checked_without_copying_sensor_identity(self):
        _,inner=frame(8);outer=deepcopy(inner)
        self.assertIsNot(inner,outer);self.assertTrue(new._same_robot_values(inner,outer))
        outer.q[0]+=.0001
        self.assertFalse(new._same_robot_values(inner,outer))
        self.assertFalse(new._same_robot_values(inner,SimpleNamespace(**vars(inner))))

    def test_public_summary_is_small_and_initial_epoch_has_no_attachment_claim(self):
        bridge=object.__new__(new.PublicCandidateReverificationBridgeV65)
        bridge.delegate=SimpleNamespace(active=SimpleNamespace(attempt_id=1))
        bridge.active=bridge.delegate.active
        bridge.witness=bridge.epoch_owner=None;bridge.used_attempt_ids=set();bridge._inner_robot=None
        summary=bridge.public_summary();_walk_public(summary,typed=True)
        self.assertEqual(summary['verification_epoch'],0)
        self.assertFalse(summary['current_reverified_position_maintenance_passed'])
        self.assertFalse(summary['old_failed_owner_reset'])
        self.assertTrue(all(not isinstance(x,(dict,list,tuple,np.ndarray))for x in summary.values()))

    def test_observer_source_has_one_staged_target_one_bridge_consume_and_no_actuator_call(self):
        tree=ast.parse(inspect.getsource(new).strip())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='PublicCandidateReverificationBridgeV65')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='observe')
        calls=[ast.unparse(n.func)for n in ast.walk(method)if isinstance(n,ast.Call)]
        self.assertEqual(calls.count('d._track_once'),1)
        self.assertEqual(calls.count('self.bridge.observe'),1)
        self.assertNotIn('self.delegate.observe',calls)
        self.assertFalse(any('execute' in call or 'active_test_proposal' in call for call in calls))


if __name__=='__main__':unittest.main()
