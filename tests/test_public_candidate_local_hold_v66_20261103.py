"""V66 source/control dispatch checks, no model forward or Native."""
import ast
import inspect
from pathlib import Path
import textwrap
import unittest
from our_mani.local_mpc_control_20261103 import public_candidate_local_hold_v66 as subject
from our_mani.local_mpc_control_20261103 import public_candidate_reverification_v65 as base
from our_mani.local_mpc_control_20261103.public_grasp_v4 import ActiveVisualGraspCoordinator
from our_mani.local_mpc_control_20261103.public_grasp_v41 import ActiveVisualGraspV41Coordinator


def tree(fn):return ast.parse(textwrap.dedent(inspect.getsource(fn)))
def dump(value):return ast.dump(value,include_attributes=False)


class V66ControlSourceTests(unittest.TestCase):
    def test_runner_exact_v65_physics_qualification_dispatch_budget(self):
        source=inspect.getsource(subject.run_public_candidate_local_hold_v66)
        changes={'run_public_candidate_local_hold_v66':'run_public_candidate_reverification_v65',
            '_validate_v66_runner':'_validate_v65_runner',
            'ObservedPublicCandidateLocalHoldV66Backend':'ObservedPublicCandidateReverificationV65Backend',
            'CandidateLocalHoldV66Coordinator':'CandidateReverificationV65Coordinator',
            'public_candidate_local_hold_v66_provenance.json':'public_candidate_reverification_v65_provenance.json',
            'public_candidate_position_v66_decisions.jsonl':'public_candidate_position_v65_decisions.jsonl',
            'P4_candidate_transformer_v66_':'P4_candidate_transformer_v65_',
            'P4_public_candidate_local_hold_v66_development':'P4_public_candidate_reverification_v65_development'}
        for a,b in sorted(changes.items(),key=lambda pair:-len(pair[0])):source=source.replace(a,b)
        self.assertEqual(dump(ast.parse(source)),dump(tree(base.run_public_candidate_reverification_v65)))

    def test_v4_dispatch_only_hold_planner_and_new_qualification_changed(self):
        current=tree(subject._V66V4Dispatch.decide);function=current.body[0]
        branch_count=0;remove=[]
        for node in ast.walk(function):
            if isinstance(node,ast.If)and ast.unparse(node.test)=="mode == 'ACTIVE_VISUAL_VERIFICATION_PROBE'":
                hold=node.orelse[0]
                self.assertIsInstance(hold,ast.If)
                self.assertEqual(ast.unparse(hold.test),"mode == 'ATTACHED_PUBLIC'")
                self.assertIn('LocalHoldPlannerV66',ast.unparse(hold))
                node.orelse=hold.orelse;branch_count+=1
        for index,node in enumerate(function.body):
            if isinstance(node,ast.If)and ast.unparse(node.test)=="mode == 'ATTACHED_PUBLIC'":
                self.assertEqual(ast.unparse(node.body[0]),'qualification = deepcopy(planner.fresh_nominal_qualification)')
                remove.append(index)
        self.assertEqual(branch_count,1);self.assertEqual(len(remove),1)
        function.body.pop(remove[0])
        self.assertEqual(dump(current),dump(tree(ActiveVisualGraspCoordinator.decide)))

    def test_real_mro_reaches_dispatch_after_v41_without_monkeypatch(self):
        sequence=subject.CandidateLocalHoldV66Coordinator.__mro__
        self.assertIn(base.CandidateReverificationV65Coordinator,sequence)
        self.assertIs(sequence[sequence.index(ActiveVisualGraspV41Coordinator)+1],subject._V66V4Dispatch)
        self.assertIs(sequence[sequence.index(subject._V66V4Dispatch)+1],ActiveVisualGraspCoordinator)

    def test_observer_keeps_same_exact_v65_measurement_and_reverification(self):
        self.assertIs(subject.ObservedPublicCandidateLocalHoldV66Backend.sample,base.ObservedPublicCandidateReverificationV65Backend.sample)
        self.assertIs(subject.ObservedPublicCandidateLocalHoldV66Backend.__init__,base.ObservedPublicCandidateReverificationV65Backend.__init__)
        self.assertIs(subject.ObservedPublicCandidateLocalHoldV66Backend.receipt,base.ObservedPublicCandidateReverificationV65Backend.receipt)

    def test_source_map_retains_all_v65_plus_exact_two_prototypes_and_protocols(self):
        sources=subject.position_producer_source_map();old=base.position_producer_source_map()
        self.assertTrue(set(old.items()).issubset(sources.items()))
        for name,value in subject.EXTRA_SOURCES.items():self.assertEqual(sources[str((subject.ROOT/name).resolve())],value)
        self.assertIn(str(Path(subject.__file__).resolve()),sources)
        self.assertEqual(len(sources),len(old)+5)


if __name__=='__main__':unittest.main()
