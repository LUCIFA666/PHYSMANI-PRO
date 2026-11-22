"""Pending-cert V67 static composition checks; no qualified object is faked."""
import ast
import inspect
from pathlib import Path
import textwrap
import unittest
from our_mani.local_mpc_control_20261103 import public_candidate_scope12_v67 as subject
from our_mani.local_mpc_control_20261103 import public_candidate_scope12_measurement_v67 as measurement
from our_mani.local_mpc_control_20261103 import public_candidate_scope12_reverification_v67 as reverify
from our_mani.local_mpc_control_20261103 import public_candidate_position_v64 as oldmeasure
from our_mani.local_mpc_control_20261103 import public_candidate_reverification_v65 as oldreverify
from our_mani.local_mpc_control_20261103 import public_candidate_local_hold_v66 as hold
from our_mani.local_mpc_control_20261103.public_grasp_v41 import ActiveVisualGraspV41Coordinator


def tree(fn):return ast.parse(textwrap.dedent(inspect.getsource(fn)))
def dump(value):return ast.dump(value,include_attributes=False)


class NormalizeMeasurement(ast.NodeTransformer):
    def visit_Constant(self,node):
        if node.value=='v67':node.value='v64'
        return node
    def visit_Call(self,node):
        self.generic_visit(node)
        node.keywords=[k for k in node.keywords if k.arg!='candidate_scope12_position_v67']
        return node


class V67CompositionTests(unittest.TestCase):
    def test_runner_original_v66_dispatch_budget_physical_judge_calls(self):
        code=inspect.getsource(subject.run_public_candidate_scope12_v67)
        changes={'run_public_candidate_scope12_v67':'run_public_candidate_local_hold_v66',
            '_validate_v67_runner':'_validate_v66_runner',
            'ObservedPublicCandidateScope12V67Backend':'ObservedPublicCandidateLocalHoldV66Backend',
            'CandidateScope12V67Coordinator':'CandidateLocalHoldV66Coordinator',
            'public_candidate_scope12_v67_provenance.json':'public_candidate_local_hold_v66_provenance.json',
            'public_candidate_position_v67_decisions.jsonl':'public_candidate_position_v66_decisions.jsonl',
            'P4_candidate_transformer_v67_':'P4_candidate_transformer_v66_',
            'P4_public_candidate_scope12_v67_development':'P4_public_candidate_local_hold_v66_development'}
        for a,b in sorted(changes.items(),key=lambda pair:-len(pair[0])):code=code.replace(a,b)
        self.assertEqual(dump(ast.parse(code)),dump(tree(hold.run_public_candidate_local_hold_v66)))

    def test_mro_reuses_same_v66_hold_implementation(self):
        mro=subject.CandidateScope12V67Coordinator.__mro__
        self.assertIs(mro[mro.index(ActiveVisualGraspV41Coordinator)+1],hold._V66V4Dispatch)
        self.assertIs(subject.LocalHoldPlannerV66,hold.LocalHoldPlannerV66)
        self.assertIn(reverify.CandidateScope12ReverificationV67Coordinator,mro)

    def test_measurement_primitive_and_maintenance_observation_math_unchanged(self):
        pairs=[(measurement.Scope12GoalAwareActiveVisualPublicBridgeV67.observe,oldmeasure.PositionGoalAwareActiveVisualPublicBridgeV64.observe),
            (measurement.PublicCandidateScope12MeasurementBridgeV67.observe,oldmeasure.PublicCandidatePositionBridgeV64.observe),
            (measurement.PublicCandidateScope12MeasurementBridgeV67._track_once,oldmeasure.PublicCandidatePositionBridgeV64._track_once)]
        for new,old in pairs:self.assertEqual(dump(NormalizeMeasurement().visit(tree(new))),dump(tree(old)))

    def test_new_typed_reverification_preserves_every_original_frame_transition(self):
        self.assertEqual(dump(NormalizeMeasurement().visit(tree(reverify.PublicCandidateScope12ReverificationBridgeV67.observe))),
            dump(tree(oldreverify.PublicCandidateReverificationBridgeV65.observe)))
        self.assertEqual(dump(tree(reverify.ObservedPublicCandidateScope12ReverificationV67Backend.sample)),
            dump(tree(oldreverify.ObservedPublicCandidateReverificationV65Backend.sample)))
        self.assertIs(subject.ObservedPublicCandidateScope12V67Backend.sample,reverify.ObservedPublicCandidateScope12ReverificationV67Backend.sample)

    def test_complete_source_map_retains_old_hold_and_new_exact_types(self):
        sources=subject.position_producer_source_map()
        self.assertTrue(set(hold.position_producer_source_map().items()).issubset(sources.items()))
        for name in ('depth_scope12_goal_perception_v67.py','goal_aware_scope12_binding_v67.py',
                'public_candidate_scope12_measurement_v67.py','public_candidate_scope12_reverification_v67.py','public_candidate_scope12_v67.py'):
            self.assertIn(str(Path(subject.__file__).with_name(name).resolve()),sources)

    def test_pending_args_or_old_controller_flags_cannot_admit(self):
        with self.assertRaises((TypeError,ValueError)):
            reverify.PublicCandidateScope12ReverificationBridgeV67(refiner_checkpoint=None,refiner_checkpoint_sha256=None)
        with self.assertRaisesRegex(ValueError,'V67'):
            subject._validate_v67_runner(None,dict(candidate_position_measurement_v64=True,
                public_reverification_epoch_v65=True,hold_micro_orientation_local_nominal_v66=True))


if __name__=='__main__':unittest.main()
