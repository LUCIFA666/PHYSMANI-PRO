"""CPU boundary tests; synthetic fixtures never qualify native performance."""
from dataclasses import replace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import Pose
from our_mani.multitask import command_goal_reference as goal
from our_mani.multitask.interaction import TRAINReferenceBinding,fit_interaction_library
from tests.test_multitask_completed_skill_fitting import completed_source,fitting_config


def reference(kind=goal.ISSUE):
    task='beat_the_rotating_buzz';pose=Pose([.1,.2,.3],Rotation.from_rotvec([.2,-.3,.4]).as_matrix()).xyzw()
    intended=pose if kind==goal.ISSUE else Pose([.2,.1,.3],Rotation.from_rotvec([.2,-.3,.6]).as_matrix()).xyzw()
    proof=dict(schema=goal.SCHEMA,TRAIN_only=True,original_events_masks_and_command_ids_preserved=True,
        intent_label_is_actual_future_dynamics_observation=False,runtime_native_driver_or_teacher_parameter_access_allowed=False,
        source_declared_forecast_seconds=0. if kind==goal.ISSUE else .5,
        privileged_TRAIN_native_driver_label_used=kind==goal.FORECAST)
    return goal.TRAINCommandGoalReference(task,'fixture0','train:'+task+':episode0',0,0.,
        TRAINReferenceBinding('wand','wand','observed_se3','approach:wand','wand'),'external',kind,
        intended,pose,dict(original_terminal_reference_xyzw=[4,5,6,0,0,0,1]),proof)


def empty_evidence(source):
    return goal.TRAINCommandGoalEvidence(source.task_name,source.source_id,source.grouping_source,'a'*64,
        (),(),{}, {},'b'*64,'c'*64,dict(actual_raw_source_verified=True))


class CommandGoalReferenceTests(unittest.TestCase):
    def test_observed_SE3_intent_retains_orientation_and_actual_metric_separately(self):
        value=reference();self.assertGreater(Rotation.from_quat(value.intent_reference_xyzw[3:]).magnitude(),.1)
        self.assertEqual(value.original_completion_metric['original_terminal_reference_xyzw'][:3],[4,5,6])
        self.assertFalse(value.intent_reference_xyzw.flags.writeable)
        with self.assertRaises(ValueError):value.intent_reference_xyzw[0]=99

    def test_issue_pose_must_not_change_to_arrival_metric_or_world_goal(self):
        value=reference();bad=value.intent_reference_xyzw.copy();bad[0]+=.1
        with self.assertRaisesRegex(ValueError,'preserve actual original'):
            replace(value,intent_reference_xyzw=bad)

    def test_forecast_intent_cannot_be_called_actual_future_observation(self):
        value=reference(goal.FORECAST)
        with self.assertRaisesRegex(ValueError,'separation'):
            replace(value,provenance=dict(value.provenance,intent_label_is_actual_future_dynamics_observation=True))
        with self.assertRaisesRegex(ValueError,'source0.5'):
            replace(value,provenance=dict(value.provenance,source_declared_forecast_seconds=.8))

    def test_runtime_accepts_only_causal_existing_predictor_pose_and_freezes_it(self):
        p=Pose([1,2,3]);value=goal.freeze_runtime_reference(candidate_forecast_pose=p,predictor='context',
            observed_until=.9,selected_at=1.,forecast_time=1.3)
        pose,linear,angular=goal.fixed_runtime_reference(value)
        np.testing.assert_array_equal(pose.position,[1,2,3]);np.testing.assert_array_equal(linear,np.zeros(3))
        np.testing.assert_array_equal(angular,np.zeros(3))
        for bad in (reference(),dict(yaw_speed=10)):
            with self.assertRaisesRegex(ValueError,'cannot enter runtime'):
                goal.freeze_runtime_reference(candidate_forecast_pose=bad,predictor='cv',observed_until=0,selected_at=1,forecast_time=2)

    def test_runtime_future_observations_or_native_predictor_rejected(self):
        for changes in (dict(observed_until=1.1),dict(forecast_time=.9),dict(predictor='native_teacher')):
            kw=dict(candidate_forecast_pose=Pose([0,0,0]),predictor='cv',observed_until=.9,selected_at=1.,forecast_time=1.3)
            kw.update(changes)
            with self.assertRaisesRegex(ValueError,'causal predictor'):goal.freeze_runtime_reference(**kw)

    def test_complete100_default_full_curve_diagnostic_reproduces_actual_kernel(self):
        sources=[completed_source(i) for i in range(100)]
        config=fitting_config(minimum_sources=8,minimum_completed_parent_sources=8,development_only=False,
            lag_grid_seconds=(0.,.05),tolerance_quantile=.95,duration_quantile=.95)
        expected=fit_interaction_library(sources,config)
        result=goal.fit_diagnostic(sources,[empty_evidence(s) for s in sources],config)
        self.assertTrue(result['all_required_phases_and_observed_partitions_passed'])
        for key,row in result['phase_fits'].items():
            baseline=expected.metadata['completed_skill_fitting']['phase_fits'][key]
            for name in ('selected_lag','position_tolerance_m','orientation_tolerance_rad','duration_seconds','duration_budget_seconds'):
                self.assertAlmostEqual(row[name],baseline[name],places=12)
            curve=expected.for_phase(key.split('|')[0])[0]
            np.testing.assert_allclose(row['fitted_curve']['positions'],curve.positions,rtol=0,atol=1e-12)
            self.assertEqual(row['source_to_fold'],baseline['source_to_fold'])

    def test_incomplete_or_parent_replica_fixture_cannot_claim_original100(self):
        sources=[completed_source(i) for i in range(100)];config=fitting_config(
            minimum_sources=8,minimum_completed_parent_sources=8,development_only=False)
        replica=replace(sources[0],source_id='synthetic_replica_of_parent0',augmentation_id='synthetic_replica')
        for values in (sources[:99],sources[:99]+[replica]):
            with self.assertRaisesRegex(ValueError,'all100 original'):
                goal.fit_diagnostic(values,[empty_evidence(s) for s in values],config)

    def test_draft_cannot_be_loaded_for_runtime_or_formal_release(self):
        with self.assertRaisesRegex(ValueError,'not production-authorized'):
            goal.load_rule('.',{},purpose='production')


if __name__=='__main__':unittest.main()
