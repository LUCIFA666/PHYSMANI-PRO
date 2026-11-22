"""Real CPU production fitter/consumer with synthetic raw-source fixtures."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from our_mani.contracts import Pose
from our_mani.multitask import command_goal_reference as goal,release_preflight as release
from our_mani.multitask.interaction import fit_interaction_library,InteractionFittingPending
from tests.test_multitask_completed_skill_fitting import completed_source,fitting_config


def source_with_goal(index):
    source=completed_source(index);event=source.completion_events[0]
    pose=source.body_tracks[event.binding.reference_id].at(event.attempt_start_time,orientation_required=True).xyzw()
    reference=goal.TRAINCommandGoalReference(source.task_name,source.source_id,source.grouping_source,
        event.origin_command_id,event.attempt_start_time,event.binding,event.reference_mode,goal.ISSUE,pose,pose,
        dict(terminal_reference_xyzw=event.terminal_reference_xyzw.tolist()),dict(schema=goal.SCHEMA,TRAIN_only=True,
            original_events_masks_and_command_ids_preserved=True,intent_label_is_actual_future_dynamics_observation=False,
            runtime_native_driver_or_teacher_parameter_access_allowed=False,source_declared_forecast_seconds=0.,
            privileged_TRAIN_native_driver_label_used=False))
    evidence=goal.TRAINCommandGoalEvidence(source.task_name,source.source_id,source.grouping_source,'a'*64,
        (reference,),(),{}, {},'b'*64,'c'*64,dict(actual_raw_source_verified=True))
    return replace(source,command_goal_reference_evidence=evidence)


class CommandGoalProductionFittingTests(unittest.TestCase):
    def setUp(self):
        # Only the typed task-scope allowlist is extended for this existing
        # synthetic button fixture. Production fitting and consumer math run
        # unchanged; no native permission loader/actual raw proof is claimed.
        scope=patch.dict(goal.TASK_PHASES,{'push_moving_button':{'approach':goal.ISSUE}})
        scope.start();self.addCleanup(scope.stop)

    def test_actual_production_kernel_and_formal_consumer_agree_on_mixed_partition_basis(self):
        sources=[source_with_goal(i) for i in range(8)]
        config=fitting_config(minimum_sources=8,minimum_completed_parent_sources=8,development_only=False)
        library=fit_interaction_library(sources,config)
        self.assertFalse(library.metadata['public_completion_verifier_calibrated'])
        release._command_goal_library_metadata(library.metadata,sources,SimpleNamespace(sha256='a'*64))
        for template in library.templates:
            key,parents,attempts=release._completion_support(sources,template)
            self.assertEqual(len(parents),8);self.assertEqual(len(attempts),8)
            release._command_goal_geometry(sources,template)
            if template.phase=='approach':
                self.assertEqual(template.reference_time_basis,goal.INTENT_BASIS)
                self.assertIsNotNone(template.completion_geometry_xyzw)
            else:self.assertEqual(template.reference_time_basis,'current')

    def test_tampered_canonical_geometry_is_rejected_from_actual_commands(self):
        sources=[source_with_goal(i) for i in range(8)]
        library=fit_interaction_library(sources,fitting_config(minimum_sources=8,minimum_completed_parent_sources=8,development_only=False))
        template=library.for_phase('approach')[0];bad=template.completion_geometry_xyzw.copy();bad[0]+=.1
        with self.assertRaisesRegex(ValueError,'canonical geometry differs'):
            release._command_goal_geometry(sources,replace(template,completion_geometry_xyzw=bad))

    def test_invalid_declared_intent_is_pending_without_legacy_completion_fallback(self):
        sources=[source_with_goal(i) for i in range(8)];first=sources[0];evidence=first.command_goal_reference_evidence
        sources[0]=replace(first,command_goal_reference_evidence=replace(evidence,references=(),
            pending_commands=(dict(command_id=0,reason='invalid raw issue mask'),)))
        with self.assertRaisesRegex(InteractionFittingPending,'invalid explicit completion'):
            fit_interaction_library(sources,fitting_config(minimum_sources=8,minimum_completed_parent_sources=8,development_only=False))

    def test_parent_replica_does_not_pass_eight_parent_requirement(self):
        sources=[source_with_goal(i) for i in range(2)]
        with self.assertRaises(InteractionFittingPending):
            fit_interaction_library(sources,fitting_config(minimum_sources=8,minimum_completed_parent_sources=8,development_only=False))


if __name__=='__main__':unittest.main()
