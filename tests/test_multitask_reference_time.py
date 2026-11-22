"""Issued versus rolling references and original TRAIN label preservation."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import unittest

import numpy as np

from our_mani.contracts import Pose
from our_mani.multitask.artifact_fitting import interaction_source
from our_mani.multitask.interaction import (_completion_reference, _selected_reference_at,
    PhaseTemplate, InteractionArtifactError, InteractionLibrary, CandidateGenerator,
    InteractionController, SE3InteractionMPC)
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.reachability import TimedReachability
from our_mani.multitask.reference_time import load_rule, CURRENT, COMMAND_ISSUE
from scripts.report_multitask_phase_support_r10 import verify_canonical_case
from tests.test_multitask_interaction import synthetic_library, state, CartesianRobot, candidates_config, mpc_config

ROOT = Path(__file__).resolve().parents[1]
RULE = dict(path='artifacts/framework/multitask_r10/basketball_reference_time_rule_v1_setup/reference_time_rule.json',
    sha256='4d0f8b45dd3958940d0fba0b3aa9d0f489c88cdcadc40b70195c027cbdce277d')


class ReferenceTimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule = load_rule(ROOT, RULE)
        plan = json.loads((ROOT/'artifacts/framework/multitask_r10/TRAIN_full_collection_plan_v11.json').read_text())
        expected = next(c for c in plan['cases'] if c['case_id'] == '170fe0e96faada34')
        cls.case, _ = verify_canonical_case(ROOT,
            ROOT/'artifacts/framework/multitask_r10/TRAIN_full_collection_v11/cases/train/moving_basketball_in_hoop/170fe0e96faada34/case.json',
            expected, plan['plan_sha256'])

    def test_historical_rule_does_not_rehash_changed_live_fitter_as_old_source(self):
        self.assertGreater(len(self.rule.verified_files_sha256), 1000)
        paths = self.rule.verified_files_sha256
        self.assertNotIn(str(ROOT/'our_mani/multitask/interaction.py'), paths)
        self.assertTrue(any(path.endswith('/our_mani/multitask/interaction.py') for path in paths))

    def test_actual_raw_terminal_labels_actions_masks_and_ledger_remain_exact(self):
        case = copy.deepcopy(self.case)
        original, _ = interaction_source(case)
        before = {key: value.copy() for key, value in case['arrays'].items()}
        ledger = copy.deepcopy(case['command_receipt'])
        derived, audit = interaction_source(case, reference_time_rule=self.rule)
        self.assertTrue(audit['reference_time_derivation']['original_raw_labels_and_masks_unchanged'])
        self.assertEqual(case['command_receipt'], ledger)
        for key in before:
            np.testing.assert_array_equal(before[key], case['arrays'][key])
        self.assertEqual(len(original.completion_events), len(derived.completion_events))
        for raw, event in zip(original.completion_events, derived.completion_events):
            np.testing.assert_array_equal(raw.terminal_reference_xyzw, event.terminal_reference_xyzw)
            np.testing.assert_array_equal(raw.terminal_tcp_xyzw, event.terminal_tcp_xyzw)
            if event.phase == 'approach':
                self.assertEqual(event.binding.reference_time_basis, COMMAND_ISSUE)
                self.assertGreater(np.linalg.norm(event.issued_reference_xyzw[:3]-event.terminal_reference_xyzw[:3]), .1)
                np.testing.assert_array_equal(_completion_reference(derived,event,0.).xyzw(),event.issued_reference_xyzw)
            else:
                self.assertEqual(event.binding.reference_time_basis, CURRENT)
                self.assertIsNone(event.issued_reference_xyzw)

    def test_missing_or_wrong_issued_pose_is_rejected_after_original_validation(self):
        for change in ('missing','wrong','bad_quaternion'):
            with self.subTest(change=change):
                case = copy.deepcopy(self.case)
                row = case['command_receipt']['commands'][0]
                if change == 'missing':
                    del row['issued_reference_native_pose_xyzw']
                elif change == 'wrong':
                    row['issued_reference_native_pose_xyzw'][0] += .2
                else:
                    row['issued_reference_native_pose_xyzw'][3:] = [0.,0.,0.,0.]
                with self.assertRaisesRegex(ValueError,'issued reference'):
                    interaction_source(case,reference_time_rule=self.rule)

    def test_invalid_original_terminal_truth_cannot_be_replaced_by_good_issue_pose(self):
        case = copy.deepcopy(self.case)
        case['command_receipt']['phase_completion_events'][0]['current_reference_native_pose_xyzw'][0] += 1.
        with self.assertRaisesRegex(ValueError,'terminal body labels'):
            interaction_source(case,reference_time_rule=self.rule)

    def test_issue_event_requires_provenance_and_disallows_a_forward_lag(self):
        source,_ = interaction_source(copy.deepcopy(self.case),reference_time_rule=self.rule)
        event = source.completion_events[0]
        with self.assertRaisesRegex(InteractionArtifactError,'issued reference pose'):
            replace(event,issued_reference_xyzw=None)
        with self.assertRaisesRegex(InteractionArtifactError,'ledger provenance'):
            replace(event,reference_time_provenance=None)
        with self.assertRaisesRegex(InteractionArtifactError,'no forward motion lag'):
            _completion_reference(source,event,.05)

    def test_selected_issue_anchor_is_fixed_while_current_reference_keeps_rolling(self):
        template = replace(synthetic_library('translation_only').templates[0],reference_time_basis=COMMAND_ISSUE,
            reference_time_rule_sha256=RULE['sha256'])
        goal = SimpleNamespace(template=template,reference_id='button',reference_pose=Pose(np.array([2.,0.,0.])))
        class Drifting:
            calls = 0
            def at(self,body,time):
                self.calls += 1
                return Pose(np.array([time*10.,0.,0.])),np.array([10.,0.,0.]),np.zeros(3)
        forecast=Drifting()
        for time in (.2,.5,10.):
            pose,velocity,angular=_selected_reference_at(goal,forecast,time,Pose(np.zeros(3)))
            np.testing.assert_array_equal(pose.position,[2.,0.,0.])
            np.testing.assert_array_equal(velocity,np.zeros(3))
        self.assertEqual(forecast.calls,0)
        goal.template=replace(template,reference_time_basis=CURRENT,reference_time_rule_sha256=None)
        first=_selected_reference_at(goal,forecast,.2,Pose(np.zeros(3)))[0]
        second=_selected_reference_at(goal,forecast,.5,Pose(np.zeros(3)))[0]
        self.assertGreater(np.linalg.norm(first.position-second.position),2.)
        self.assertEqual(forecast.calls,2)

    def test_template_cannot_silently_claim_issue_semantics_without_rule_binding(self):
        template=synthetic_library('translation_only').templates[0]
        with self.assertRaisesRegex(ValueError,'bound TRAIN derivation'):
            replace(template,reference_time_basis=COMMAND_ISSUE)
        with self.assertRaisesRegex(ValueError,'unknown explicit'):
            replace(template,reference_time_basis='arrival_or_maybe_issue')
        payload=copy.deepcopy(synthetic_library('translation_only').payload)
        payload['templates'][0].update(reference_time_basis=COMMAND_ISSUE,reference_time_rule_sha256=RULE['sha256'])
        with self.assertRaisesRegex(InteractionArtifactError,'explicit reference-time source/rule binding'):
            InteractionLibrary(payload,allow_synthetic=True)

    def test_real_generator_predicts_arrival_then_controller_never_requeries_that_anchor(self):
        # Writable copy is needed by the older native-environment SciPy; it is
        # specific to this CPU robot fixture, not a production source edit.
        class CPUCartesianRobot(CartesianRobot):
            def inverse(self,pose,seed):
                from scipy.spatial.transform import Rotation
                return np.r_[pose.position,Rotation.from_matrix(pose.rotation.copy()).as_rotvec(),0.]
        payload=copy.deepcopy(synthetic_library('translation_only').payload)
        payload['templates'][0].update(reference_time_basis=COMMAND_ISSUE,reference_time_rule_sha256=RULE['sha256'])
        payload['metadata'].update(reference_time_schema='r10_explicit_reference_time_basis_v1',
            reference_time_helper_sha256=hashlib.sha256((ROOT/'our_mani/multitask/reference_time.py').read_bytes()).hexdigest(),
            reference_time_rule_sha256_values=[RULE['sha256']])
        library=InteractionLibrary(payload,allow_synthetic=True)
        generator=CandidateGenerator(library,TimedReachability(CPUCartesianRobot()),
            lambda **kwargs:dict(feasible=True,evidence={}),candidates_config(),allow_synthetic=True)
        initial=state(0.)
        initial=replace(initial,bodies={'button':replace(initial.bodies['button'],linear_velocity=np.array([.3,0.,0.]))})
        forecast=ConstantVelocityDynamics().predict(list(initial.bodies.values()),1.,.05)
        candidates=generator.generate(initial,forecast,0,10.,initial.robot.pose)
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertAlmostEqual(candidate.reference_pose.position[0],.3*candidate.arrival_time)
            np.testing.assert_array_equal(candidate.reference_velocity,np.zeros(3))
        controller=InteractionController(library,generator,SE3InteractionMPC(mpc_config()),
            predictor='cv',ranking='geometry',continuation_sha256='CPU-reference-time-test',allow_synthetic=True)
        controller.step(initial,forecast,10.)
        selected=controller.goal.reference_pose.position.copy()
        later=state(.05)
        later=replace(later,bodies={'button':replace(later.bodies['button'],pose=Pose(np.array([1.,0.,0.])))})
        class NoRollingLookup:
            metadata={}
            def at(self,*args):
                raise AssertionError('command_issue controller queried a moving forecast after selection')
        action=controller.step(later,NoRollingLookup(),9.95)
        self.assertEqual(action['command'],'joint_trajectory')
        np.testing.assert_array_equal(controller.goal.reference_pose.position,selected)


if __name__ == '__main__':
    unittest.main()
