"""Completion evidence, right censoring and terminal error scope; CPU only."""
from dataclasses import replace
import unittest

import numpy as np

from our_mani.multitask.artifact_fitting import interaction_source
from our_mani.multitask.interaction import (InteractionArtifactError,InteractionFittingPending,
    InteractionFitConfig,fit_interaction_library)
from tests.test_multitask_artifact_fitting import teacher_case


def completed_source(index):
    source,_=interaction_source(teacher_case(index))
    return replace(source,source_kind='synthetic')


def fitting_config(**changes):
    values=dict(progress_knots=3,lag_grid_seconds=(0.,),duration_quantile=.9,tolerance_quantile=.9,
        minimum_sources=2,minimum_position_tolerance_m=.01,minimum_orientation_tolerance_rad=.05,
        minimum_lag_sample_fraction=.5,minimum_completed_parent_sources=2,
        maximum_position_tolerance_m=.03,maximum_orientation_tolerance_rad=.15,development_only=True)
    values.update(changes)
    return InteractionFitConfig(**values)


class CompletedSkillFittingTests(unittest.TestCase):
    def test_large_transient_command_error_does_not_inflate_terminal_tolerance(self):
        sources=[]
        for index in range(2):
            source=completed_source(index);raw=source.expert_tcp_xyzw.copy();raw[:,0]+=.4
            events=tuple(replace(event,committed_world_waypoint_xyzw=event.committed_world_waypoint_xyzw+
                np.array([.4,0.,0.,0.,0.,0.,0.])) for event in source.completion_events)
            sources.append(replace(source,expert_tcp_xyzw=raw,completion_events=events))
        library=fit_interaction_library(sources,fitting_config())
        self.assertTrue(all(t.position_tolerance_m==.01 for t in library.templates))
        for row in library.metadata['source_grouped_lag_selection'].values():
            errors=row['candidates'][0]['terminal_crossfit_errors']
            self.assertTrue(all(e['committed_world_waypoint_position_error_m']>.39 for e in errors))
            self.assertTrue(all(e['current_reference_endpoint_position_error_m']<1e-8 for e in errors))

    def test_incomplete_all_open_attempt_does_not_teach_duration_or_opening(self):
        sources=[completed_source(0),completed_source(1)]
        base=fit_interaction_library(sources,fitting_config())
        failed=completed_source(2)
        poses=failed.measured_tcp_xyzw.copy();poses[:,0]=9.
        failed=replace(failed,completion_events=(),measured_tcp_xyzw=poses,curve_tcp_xyzw=poses,
            expert_opening=np.zeros(len(failed.times)),
            command_ledger=tuple(dict(row,cutoff_at=10.,command_incomplete_at_cutoff=True) for row in failed.command_ledger))
        fitted=fit_interaction_library(sources+[failed],fitting_config())
        self.assertEqual(fitted.metadata['BC_excluded_source_ids'],[failed.source_id])
        for before,after in zip(base.templates,fitted.templates):
            self.assertEqual(before.duration_seconds,after.duration_seconds)
            self.assertEqual(before.duration_budget_seconds,after.duration_budget_seconds)
            np.testing.assert_array_equal(before.positions,after.positions)
            np.testing.assert_array_equal(before.opening,after.opening)
        excluded=[row for row in fitted.metadata['completed_skill_fitting']['curation'] if not row['included_in_BC']]
        self.assertEqual(len(excluded),2)
        self.assertTrue(all(row['duration_is_right_censored_lower_bound'] for row in excluded))
        self.assertTrue(all(row['raw_rows_preserved'] for row in excluded))

    def test_current_reference_terminal_dispersion_exceeding_cap_remains_pending(self):
        first=completed_source(0);second=completed_source(1)
        poses=second.measured_tcp_xyzw.copy();poses[:,0]+=.1
        events=tuple(replace(event,terminal_tcp_xyzw=event.terminal_tcp_xyzw+
            np.array([.1,0.,0.,0.,0.,0.,0.])) for event in second.completion_events)
        second=replace(second,measured_tcp_xyzw=poses,curve_tcp_xyzw=poses,completion_events=events)
        with self.assertRaises(InteractionFittingPending) as caught:
            fit_interaction_library([first,second],fitting_config())
        failed=list(caught.exception.diagnostics['phase_fits'].values())
        self.assertTrue(all(row['position_tolerance_m']>.09 for row in failed))
        self.assertTrue(all(row['passed'] is False for row in failed))
        self.assertTrue(all('no clipping' in row['reason'] for row in failed))

    def test_command_done_alone_never_becomes_phase_completion(self):
        source=completed_source(0)
        with self.assertRaisesRegex(InteractionArtifactError,'adjacent forward'):
            replace(source,completion_events=(replace(source.completion_events[0],to_phase='native_success'),))
        with self.assertRaisesRegex(InteractionArtifactError,'executed TRAIN'):
            replace(source.completion_events[-1],executed_tick_count=0)

    def test_successful_acquire_endpoint_is_not_an_external_mode_bridge(self):
        sources=[]
        for index in range(2):
            source=completed_source(index);track=source.body_tracks['button']
            track=replace(track,motion_modes=('external',)*5+('attached',)*6)
            events=(source.completion_events[0],replace(source.completion_events[1],reference_mode='attached'))
            sources.append(replace(source,body_tracks={'button':track},completion_events=events))
        library=fit_interaction_library(sources,fitting_config(lag_grid_seconds=(0.,.05)))
        approach=library.for_phase('approach')[0]
        self.assertEqual(approach.reference_mode,'external')
        self.assertEqual(approach.lag_seconds,0.)
        for diagnostic in library.metadata['source_grouped_lag_selection'].values():
            self.assertFalse(diagnostic['candidates'][1]['valid'])
        self.assertTrue(all(row['terminal_reference_is_external_dynamics_observation'] is False
            for row in library.metadata['completed_skill_fitting']['curation']))

    def test_replicas_do_not_manufacture_completed_parent_support(self):
        sources=[]
        for index in range(2):
            for replica in range(4):
                source=completed_source(index)
                sources.append(replace(source,source_id=source.source_id+'copy'+str(replica),
                    raw_episode_id=source.raw_episode_id+'copy'+str(replica)))
        with self.assertRaises(InteractionFittingPending) as caught:
            fit_interaction_library(sources,fitting_config(minimum_completed_parent_sources=8,development_only=False))
        self.assertTrue(all(row['completed_parent_count']==2
            for row in caught.exception.diagnostics['phase_fits'].values()))
        with self.assertRaisesRegex(InteractionArtifactError,'eight completed TRAIN parents'):
            fit_interaction_library(sources,fitting_config(development_only=False))

    def test_unknown_terminal_reference_is_reported_without_fabricating_a_label(self):
        case=teacher_case();case['command_receipt']['phase_completion_events'][0]['current_reference_native_pose_valid']=False
        source,audit=interaction_source(case)
        self.assertEqual(len(source.completion_events),1)
        self.assertEqual(len(audit['excluded_phase_completion_events']),1)
        # The event is a completed skill with unusable labels, not a synthetic
        # negative and not an unobserved duration to use in a fitted quantile.
        other=completed_source(1)
        with self.assertRaises(InteractionFittingPending) as caught:
            fit_interaction_library([replace(source,source_kind='synthetic'),other],fitting_config())
        row=next(row for row in caught.exception.diagnostics['curation'] if row['source_id']==source.source_id and row['command_id']==0)
        self.assertTrue(row['skill_completion_observed'])
        self.assertFalse(row['duration_is_right_censored_lower_bound'])
        self.assertFalse(row['included_in_BC'])

    def test_terminal_full_orientation_mask_cannot_be_promoted_by_metric_snapshot(self):
        case=teacher_case();case['arrays']['target_rotation_observable'][5,0]=False
        source,audit=interaction_source(case)
        self.assertEqual([e.phase for e in source.completion_events],['press'])
        self.assertIn('orientation label unavailable',audit['excluded_phase_completion_events'][0]['reason'])
        self.assertFalse(case['arrays']['target_rotation_observable'][5,0])


if __name__=='__main__':unittest.main()
