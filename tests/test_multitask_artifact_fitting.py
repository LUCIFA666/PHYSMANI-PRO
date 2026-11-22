"""TRAIN-only label masks and source-fold separation; no native simulator."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from our_mani.multitask.artifact_fitting import (colour_parent_crossfit,fit_artifacts,
    fit_layout,interaction_source,load_train_cases)
from our_mani.multitask.evaluation import file_sha
from our_mani.multitask.interaction import TRAINBodyTrack,InteractionArtifactError
from our_mani.multitask.public_roles import ColourFitConfig,TRAINColourSample


def teacher_case(index=0):
    n,b=11,3;task='push_moving_button';parent='episode'+str(index);raw=parent+'__native_aug_r0'
    times=np.arange(n)*.05;poses=np.tile([0.,0.,.1,0.,0.,0.,1.],(n,1))
    arrays=dict(times=times,body_ids=np.asarray(['button','plate','wrap']),
        target_positions=np.zeros((n,b,3)),target_rotations=np.tile(np.eye(3),(n,b,1,1)),
        native_pose_valid=np.ones((n,b),bool),mode_valid=np.ones((n,b),bool),
        target_valid=np.ones((n,b),bool),target_rotation_observable=np.ones((n,b),bool),
        modes=np.zeros((n,b),np.int8),teacher_action_valid=np.ones(n,bool),
        teacher_tcp_xyzw=poses.copy(),measured_tcp_xyzw=poses.copy(),teacher_opening=np.ones(n),
        teacher_applied_opening=np.ones(n),teacher_desired_opening=np.zeros(n),
        teacher_stage=np.asarray(['wp0']*5+['wp1']*6),teacher_phase=np.asarray(['approach']*5+['press']*6),
        teacher_command_id=np.asarray([0]*5+[1]*6),
        teacher_reference_body=np.asarray(['button']*n),teacher_reference_role=np.asarray(['button']*n),
        teacher_reference_kind=np.asarray(['observed_se3']*n),teacher_anchor_id=np.asarray(['actual_anchor']*n),
        teacher_manipulated_role=np.asarray(['button']*n))
    arrays['target_valid'][5:,1]=False # public press excludes plate dynamics labels
    entry=dict(task=task,split='train',raw_episode_id=raw,parent_raw_episode_id=parent,
        parent_source_group=f'train:{task}:{parent}',source_id=task+':'+raw,augmentation_id=raw,
        source_kind='TRAIN_native_generated_augmentation',cache_sha256='a'*64,
        interaction_opening_array='teacher_applied_opening',interaction_tcp_array='measured_tcp_xyzw',
        interaction_eligible=True,teacher_used=True)
    commands=[];events=[]
    for cid,phase,begin,end,to_phase in ((0,'approach',0.,.25,'press'),(1,'press',.25,.5,'native_success')):
        command=dict(command_id=cid,generated_at=begin,raw_phase=phase,executed_tick_count=5,
            reference_body='button',reference_role='button',reference_kind='observed_se3',anchor_id='actual_anchor',
            manipulated_role='button',command_completed_at=end)
        commands.append(command)
        events.append(dict(version='r10_TRAIN_phase_completion_event_v1',phase=phase,from_phase=phase,to_phase=to_phase,
            evidence='completed_command_then_ordered_teacher_phase_transition' if cid==0 else
                'TRAIN_native_success_during_executed_final_required_phase',origin_command_id=cid,eligible_command_ids=[cid],
            attempt_start_time=begin,attempt_end_time=end,origin_reference_body='button',origin_reference_role='button',
            origin_reference_kind='observed_se3',origin_anchor_id='actual_anchor',origin_manipulated_role='button',
            origin_reference_motion_mode='external',origin_measured_tcp_xyzw=poses[0].tolist(),
            current_measured_tcp_xyzw=poses[0].tolist(),current_reference_native_pose_xyzw=[0.,0.,0.,0.,0.,0.,1.],
            current_reference_native_pose_valid=True,committed_world_waypoint_xyzw=poses[0].tolist(),
            label_namespace='TRAIN_metric_only',origin_executed_tick_count=5,origin_command_completed=True))
    return dict(entry=entry,arrays=arrays,command_receipt_verified=True,
        command_receipt=dict(version='r10_TRAIN_committed_waypoint_commands_v1',commands=commands,phase_completion_events=events))


class ArtifactFittingTests(unittest.TestCase):
    def test_unreferenced_controlled_plate_does_not_invalidate_button_teacher(self):
        case=teacher_case();before=case['arrays']['target_valid'].copy()
        source,audit=interaction_source(case)
        self.assertEqual(set(source.body_tracks),{'button'})
        self.assertTrue(all(binding is not None for binding in source.references))
        np.testing.assert_array_equal(before,case['arrays']['target_valid'])
        self.assertFalse(audit['dynamics_masks_modified'])
        self.assertFalse(audit['controlled_interaction_labels_are_exogenous_dynamics'])
        self.assertTrue(source.expert_opening.all())
        self.assertFalse(audit['raw_desired_opening_used_as_applied'])

    def test_legacy_raw_or_mismatched_actual_opening_is_not_silently_fitted(self):
        case=teacher_case();case['entry'].pop('interaction_opening_array')
        with self.assertRaisesRegex(ValueError,'actual applied opening'): interaction_source(case)
        case=teacher_case();case['arrays']['teacher_opening'][0]=0.
        with self.assertRaisesRegex(ValueError,'compatibility alias'): interaction_source(case)

    def test_real_reference_invalid_mode_is_excluded_and_lag_cannot_bridge_it(self):
        case=teacher_case();case['arrays']['mode_valid'][5,0]=False
        source,audit=interaction_source(case)
        self.assertIsNone(source.references[5])
        self.assertEqual(audit['excluded_reference_commands'],1)
        with self.assertRaisesRegex(InteractionArtifactError,'invalid offline reference'):
            source.body_tracks['button'].at(.3,orientation_required=True,same_mode_as='external',not_before=.2)

    def test_offline_controlled_reference_label_is_explicit_not_dynamics_fill(self):
        case=teacher_case();case['arrays']['teacher_reference_body'][5:]='plate'
        source,audit=interaction_source(case)
        self.assertEqual(audit['controlled_reference_teacher_labels_used'],6)
        self.assertTrue(source.body_tracks['plate'].pose_valid.all())
        self.assertFalse(case['arrays']['target_valid'][5:,1].any())

    def test_mode_roundtrip_and_clock_gap_are_never_interpolated_as_one_lag(self):
        poses=np.tile([0.,0.,0.,0.,0.,0.,1.],(3,1))
        track=TRAINBodyTrack('physical_body',np.array([0.,.05,.1]),poses,
            ('external','attached','external'),np.ones(3,bool))
        with self.assertRaisesRegex(InteractionArtifactError,'mode boundary'):
            track.at(.1,orientation_required=True,same_mode_as='external',not_before=0.)
        gap=TRAINBodyTrack('physical_body',np.array([0.,.05,.2]),poses,
            ('external',)*3,np.ones(3,bool))
        with self.assertRaisesRegex(InteractionArtifactError,'clock gap'):
            gap.at(.15,orientation_required=True)
        unknown_rotation=TRAINBodyTrack('physical_body',np.array([0.,.05,.1]),poses,
            ('external',)*3,np.array([True,False,True]))
        with self.assertRaisesRegex(InteractionArtifactError,'observable/supervised orientation'):
            unknown_rotation.at(.1,orientation_required=True,same_mode_as='external',not_before=0.)

    def colour_samples(self,count=4):
        result=[]
        for i in range(count):
            for role,name,rgb in (('peg0','red',[220,20,20]),('peg1','blue',[20,20,220])):
                result.append(TRAINColourSample('insert_onto_rotating_peg',
                    'train:insert_onto_rotating_peg:episode'+str(i),role,(role+'_surface',),name,
                    np.tile(rgb,(20,1))))
        return result

    def test_colour_parent_folds_include_all_replicas_and_need_disjoint_support(self):
        config=ColourFitConfig(2,.9,.05,.01,10,.05)
        samples=self.colour_samples();samples.extend(self.colour_samples()[:2]) # extra replica of parent0
        report=colour_parent_crossfit(samples,config,minimum_accuracy=.9)
        self.assertTrue(report['passed'])
        for fold in report['folds']:
            self.assertFalse(set(fold['fit_parent_sources'])&set(fold['held_parent_sources']))
        pending=colour_parent_crossfit(self.colour_samples(2),config,minimum_accuracy=.9)
        self.assertFalse(pending['valid']);self.assertFalse(pending['passed'])

    def test_rigid_relation_requires_every_parent_not_a_favourable_case(self):
        cases=[teacher_case(0),teacher_case(1)]
        config=dict(minimum_parent_sources=2,minimum_samples_per_rollout=5,
            position_tolerance_m=.002,orientation_tolerance_rad=.02)
        accepted=fit_layout(cases,'push_moving_button',config)
        self.assertEqual(len(accepted['accepted']),1)
        cases[1]['arrays']['target_positions'][:,2,0]=.1
        rejected=fit_layout(cases,'push_moving_button',config)
        self.assertFalse(rejected['accepted']);self.assertEqual(len(rejected['rejected']),1)

    def write_cache(self,root,case):
        folder=root/case['entry']['raw_episode_id'];folder.mkdir()
        cache=folder/'body_tracks.npz';np.savez_compressed(cache,**case['arrays'])
        entry=dict(case['entry'],cache=str(cache.relative_to(root)),cache_sha256=file_sha(cache))
        entry['frames']=len(case['arrays']['times']);ticks=entry['frames']-1
        record=dict(entry=entry,status='completed',split='train',case_isolation='fresh_simulator_process_per_episode',
            case_id=entry['raw_episode_id'],plan_sha256='a'*64,
            process_id=42,alignment=dict(clock=dict(ticks=ticks,dt=.05,budget_seconds=10.,simulation_seconds=ticks*.05),
                actual_physics_seconds=ticks*.05,actual_task_seconds=ticks*.05),
            teacher_actions_used=True,policy_history_contains_native_GT=False,
            files_sha256={entry['cache']:entry['cache_sha256']})
        commands=folder/'TRAIN_teacher_commands.json';commands.write_text(json.dumps(case['command_receipt']))
        entry.update(phase_completion_events_receipt=str(commands.relative_to(root)),
            teacher_command_receipt=str(commands.relative_to(root)))
        record['files_sha256'][str(commands.relative_to(root))]=file_sha(commands)
        colour=folder/'TRAIN_color_annotations.json'
        colour.write_text(json.dumps(dict(version='r10_TRAIN_role_color_supervision_v1',split='train',
            task=entry['task'],source_group=entry['parent_source_group'],role_color_labels={'button':dict(
                visible=True,rgb_pixels_255=[[220,20,20]]*20,component_names=['button_surface'],
                provenance='current_rgbd',source_group=entry['parent_source_group'],teacher_color_annotation=dict(
                    color_name='red',source_split='train',source_group=entry['parent_source_group'],
                    label_is_runtime_policy_observation=False))})))
        record['files_sha256'][str(colour.relative_to(root))]=file_sha(colour)
        launcher=root/'launch.sh';launcher.write_text('# CPU fixture only, never launched\n')
        child=folder/'raw_child.json';child.write_text(json.dumps(record))
        record['parent_process_receipt']=dict(child_case_path=str(child.relative_to(root)),
            child_case_sha256=file_sha(child),launcher=str(launcher.relative_to(root)),
            launcher_sha256=file_sha(launcher),process_id=42)
        path=folder/'case.json';path.write_text(json.dumps(record))
        return entry,dict(path=str(path.relative_to(root)),sha256=file_sha(path),case_id=entry['raw_episode_id']),cache

    def test_collector_receipt_hashes_and_train_split_cannot_be_bypassed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);entry,receipt,cache=self.write_cache(root,teacher_case())
            path=root/'manifest.json';data=dict(version='r10_causal_body_tracks_v1',complete=True,TEST_used=False,
                plan_sha256='a'*64,
                expected=1,completed=1,missing_case_ids=[],entries=[entry],case_receipts=[receipt]);path.write_text(json.dumps(data))
            _,cases=load_train_cases(root,path);self.assertEqual(len(cases),1)
            cache.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'): load_train_cases(root,path)
            data['entries'][0]['split']='validation';path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'TRAIN teacher/hold'): load_train_cases(root,path)

    def test_partial_native_coverage_remains_pending_and_single_colour_is_not_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);entries=[];receipts=[]
            for i in range(2):
                entry,receipt,_=self.write_cache(root,teacher_case(i));entries.append(entry);receipts.append(receipt)
            manifest=root/'collector.json';manifest.write_text(json.dumps(dict(version='r10_causal_body_tracks_v1',
                plan_sha256='a'*64,
                complete=True,TEST_used=False,expected=len(entries),completed=len(entries),missing_case_ids=[],
                entries=entries,case_receipts=receipts)))
            config=root/'fit.json';config.write_text(json.dumps(dict(fit_split='train',VAL_used=False,TEST_used=False,
                interaction=dict(progress_knots=3,lag_grid_seconds=[0.],duration_quantile=1.,tolerance_quantile=.9,
                    minimum_sources=2,minimum_position_tolerance_m=.01,minimum_orientation_tolerance_rad=.05,
                    minimum_lag_sample_fraction=1.,minimum_completed_parent_sources=2,
                    maximum_position_tolerance_m=.03,maximum_orientation_tolerance_rad=.15,development_only=True),
                colour=dict(minimum_sources_per_colour=2,within_colour_quantile=.9,minimum_acceptance_radius=.05,
                    minimum_margin=.01,minimum_visible_pixels=10,maximum_observation_age=.05),
                rigid_layout=dict(minimum_parent_sources=2,minimum_samples_per_rollout=5,
                    position_tolerance_m=.002,orientation_tolerance_rad=.02),colour_parent_cv_minimum_accuracy=.9)))
            # This mock isolates the *reporting* separation from the independently
            # strict real-1400 prior verifier; no fixture is called real TRAIN data.
            prior=dict(original_parent_groups=[entry['parent_source_group'] for entry in entries],
                original_TRAIN_source_count=1400,fixture_only=True)
            with patch('our_mani.multitask.artifact_fitting.original_prior_binding',return_value=prior):
                report=fit_artifacts(root,manifest,'unused','unused',config,'output')
            self.assertFalse(report['formal_comparison_ready'])
            self.assertFalse(report['formal_freeze_authorized'])
            self.assertEqual(report['native_TRAIN_parent_sources'],2)
            task=report['per_task']['push_moving_button']
            self.assertEqual(task['interaction_status'],'TRAIN_fitted_pending_formal_release')
            self.assertFalse(task['all_original_TRAIN_parents_acquired'])
            self.assertFalse(report['colour_parent_crossfit']['passed'])
            self.assertFalse((root/'output'/'colour.json').exists())


if __name__=='__main__': unittest.main()
