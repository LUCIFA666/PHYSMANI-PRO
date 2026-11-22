"""Synthetic offline evidence only: no native, GPU, teacher call, or fitting."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scripts import analyze_multitask_smoke_r10 as analysis
from scripts.collect_multitask_training_r10 import context_window_eligibility,fit_rigid_layout
from our_mani.multitask.specs import get_task_spec


def arrays_for(task='moving_basketball_in_hoop',frames=40,press_at=None):
    spec=get_task_spec(task)
    ids=[role.name for role in spec.roles if role.name not in spec.audit_only_goal_roles and role.kind == 'body']
    t,b=frames,len(ids);times=np.arange(t)*.05
    positions=np.zeros((t,b,3));positions[:,:,0]=times[:,None]*.2+np.arange(b)[None,:]
    world_omega=np.array([0.,.3,.4])
    initial=Rotation.from_rotvec([.8,0.,0.]).as_matrix()
    rotations=Rotation.from_rotvec(times[:,None]*world_omega).as_matrix() @ initial
    rotations=np.repeat(rotations[:,None],b,axis=1)
    valid=np.ones((t,b),bool);action=np.arange(t)<t-1
    commands=np.full(t,'move',dtype='<U20');commands[-1]='no_action_terminal'
    arrays=dict(times=times,body_ids=np.asarray(ids),positions=positions.copy(),rotations=rotations.copy(),
        linear_velocities=np.zeros((t,b,3)),angular_velocities=np.zeros((t,b,3)),confidence=np.ones((t,b)),
        valid=valid.copy(),rotation_observable=np.zeros((t,b),bool),modes=np.zeros((t,b),np.int8),
        mode_valid=valid.copy(),native_pose_valid=valid.copy(),dynamics_segment_valid=valid.copy(),
        target_positions=positions.copy(),target_rotations=rotations.copy(),
        target_linear_velocities=np.zeros((t,b,3)),target_angular_velocities=np.zeros((t,b,3)),
        target_valid=valid.copy(),target_rotation_observable=valid.copy(),teacher_action_valid=action,
        teacher_stage=np.where(action,'wp0',''),teacher_phase=np.where(action,'approach',''),
        teacher_reference_body=np.where(action,ids[0],''),teacher_reference_role=np.where(action,ids[0],''),
        teacher_reference_kind=np.where(action,'translation_only' if ids[0] == 'ball' else 'observed_se3',''),
        teacher_manipulated_body=np.where(action,ids[0],''),teacher_manipulated_role=np.where(action,ids[0],''),
        public_command_kind=commands)
    if press_at is not None:
        j=ids.index('plate')
        arrays['public_command_kind'][press_at]='press'
        arrays['teacher_stage']=np.where(action,np.where(np.arange(t)>=press_at,'wp1','wp0'),'')
        arrays['teacher_phase']=np.where(action,np.where(np.arange(t)>=press_at,'press','approach'),'')
        for name in ('valid','target_valid','dynamics_segment_valid'):
            arrays[name][press_at:,j]=False
    diagnostics=dict(version='r10_current_observation_registration_diagnostics_v1',frames=[dict(
        timestamp=float(times[i]),public_command_kind=str(arrays['public_command_kind'][i]),
        bodies={role:dict(directly_visible=True,registration_valid=True,orientation_observable=False,
            orientation_observability='none',rotation_ambiguous=False,registration_rms_m=.005) for role in ids},
        rigid_groups=[],mode_identity_ambiguous=[]) for i in range(t)])
    return arrays,diagnostics


class OfflineSmokeAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.thresholds=dict(linear_motion_m_per_s=.001,angular_motion_rad_per_s=.01,
            reported_zero_m_per_s=1e-6,reported_zero_rad_per_s=1e-6)

    def test_world_twist_and_reported_zero_are_detected_without_label_rewrite(self):
        arrays,_=arrays_for()
        original={key:value.copy() for key,value in arrays.items()}
        result=analysis.native_twist_audit(arrays,0,self.thresholds)
        self.assertAlmostEqual(result['raw_native']['linear_m_per_s']['vector_rms_error'],.2)
        self.assertAlmostEqual(result['raw_native']['angular_rad_per_s']['vector_rms_error'],.5)
        self.assertEqual(result['raw_native']['linear_m_per_s']['pose_motion_but_reported_zero_pairs'],39)
        self.assertEqual(result['segmented']['linear_m_per_s']['pairs'],38)
        for key in arrays:
            np.testing.assert_array_equal(arrays[key],original[key])
        arrays['target_linear_velocities'][:,:,0]=.2
        arrays['target_angular_velocities'][:]=[0.,.3,.4]
        matched=analysis.native_twist_audit(arrays,0,self.thresholds)
        self.assertLess(matched['raw_native']['angular_rad_per_s']['vector_rms_error'],1e-12)

    def test_contiguous_visibility_breaks_at_gap_mode_change_and_unknown(self):
        times=np.arange(12)*.05;times[8:]+=.05
        modes=np.zeros(12);modes[5:]=2
        mask=np.ones(12,bool);mask[3]=False
        result=analysis.continuity(mask,times,modes)
        self.assertEqual(result['run_lengths_frames'],[3,1,3,4])
        self.assertEqual(result['contiguous_five_frame_endpoints'],0)

    def fixture(self,folder,task='moving_basketball_in_hoop',frames=40,press_at=None):
        plan=analysis.read(analysis.BASE/'TRAIN_smoke_collection_plan_v3.json')
        case=next(case for case in plan['cases'] if case['task'] == task)
        arrays,diagnostics=arrays_for(task,frames,press_at)
        out=Path(folder)/'cases'/'train'/task/case['case_id'];out.mkdir(parents=True)
        track=out/'body_tracks.npz';np.savez_compressed(track,**arrays)
        diag=out/'tracker_diagnostics.json';diag.write_text(json.dumps(diagnostics))
        layout=out/'TRAIN_rigid_layout.json';layout.write_text(json.dumps(fit_rigid_layout(
            arrays,task=task,source_group=case['parent_source_group'])))
        files={str(path.relative_to(analysis.ROOT)):analysis.sha(path) for path in (track,diag,layout)}
        eligibility=context_window_eligibility(arrays)
        entry=dict(split='train',task=task,raw_episode_id=case['raw_episode_id'],
            parent_raw_episode_id=case['parent_raw_episode_id'],parent_source_group=case['parent_source_group'],
            cache=str(track.relative_to(analysis.ROOT)),cache_sha256=analysis.sha(track),
            observation_source='causal_RGBD_and_proprioception',target_source='TRAIN_VAL_native_supervision',
            frames=frames,context_window_eligibility=eligibility,
            context_training_eligible=eligibility['usable_context_windows']>0)
        seconds=float(arrays['times'][-1])
        record=dict(version='r10_single_case_collection_receipt_v1',status='completed',case_id=case['case_id'],
            plan_sha256=plan['plan_sha256'],task=task,split='train',raw_episode_id=case['raw_episode_id'],
            parent_source_group=case['parent_source_group'],case_isolation='fresh_simulator_process_per_episode',
            process_id=424242,policy_history_contains_native_GT=False,teacher_actions_used=True,
            teacher_paths_requested=False,files_sha256=files,entry=entry,CPU_synthetic_fixture=True,
            alignment=dict(clock=dict(ticks=frames-1,dt=.05,budget_seconds=10.,simulation_seconds=seconds),
                actual_physics_seconds=seconds,actual_task_seconds=seconds,native_terminal_latched=True,
                native_success_latched=False,native_terminal_failure_latched=False))
        path=out/'case.json';path.write_text(json.dumps(record))
        return plan,case,path,arrays

    def test_partial_verified_case_full_orientation_not_promoted_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory(prefix='cpu_smoke_analysis_',dir=analysis.BASE) as folder:
            plan,case,path,arrays=self.fixture(folder)
            result=analysis.analyze(analysis.BASE/'TRAIN_smoke_collection_plan_v3.json',folder)
            self.assertEqual(result['verified_cases'],1);self.assertEqual(result['missing_cases'],13)
            self.assertFalse(result['complete'])
            report=result['cases'][0]
            self.assertEqual(report['bodies']['ball']['full_orientation_history']['frames'],0)
            self.assertEqual(report['bodies']['ball']['registration_rms_m']['count'],40)
            self.assertEqual(report['teacher']['required_phases_missing'],['acquire','lift','deposit'])
            record=json.loads(path.read_text());cache=analysis.absolute(record['entry']['cache'])
            with cache.open('ab') as stream:stream.write(b'tamper')
            rejected=analysis.analyze(analysis.BASE/'TRAIN_smoke_collection_plan_v3.json',folder)
            self.assertEqual(rejected['verified_cases'],0);self.assertEqual(rejected['invalid_cases'],1)
            self.assertFalse(rejected['diagnostics_valid'])

    def test_immediate_terminal_preserved_without_success_filter_or_context_windows(self):
        with tempfile.TemporaryDirectory(prefix='cpu_smoke_analysis_',dir=analysis.BASE) as folder:
            self.fixture(folder,frames=1)
            report=analysis.analyze(analysis.BASE/'TRAIN_smoke_collection_plan_v3.json',folder)
            case=report['cases'][0]
            self.assertEqual(case['frames'],1);self.assertTrue(case['immediate_terminal'])
            self.assertEqual(case['teacher']['teacher_action_rows'],0)
            self.assertEqual(case['context_window_eligibility']['usable_context_windows'],0)
            self.assertEqual(len(report['short_terminal_cases']),1)
            self.assertFalse(case['short_terminal_is_infrastructure_failure'])

    def test_plate_press_mask_keeps_raw_native_GT_and_excludes_segmented_dynamics(self):
        with tempfile.TemporaryDirectory(prefix='cpu_smoke_analysis_',dir=analysis.BASE) as folder:
            plan,case,path,arrays=self.fixture(folder,task='push_moving_button',frames=20,press_at=8)
            report=analysis.audit_case(path,plan,case,self.thresholds)
            mask=report['public_press_mask']
            self.assertTrue(mask['passed']);self.assertEqual(mask['rows_after_first_press'],12)
            self.assertEqual(mask['raw_native_pose_valid_after_press'],12)
            plate=report['bodies']['plate']
            self.assertEqual(plate['native_twist']['raw_native_pairs'],19)
            self.assertEqual(plate['native_twist']['segmented_pairs'],7)
            self.assertEqual(plate['public_non_exogenous_masked_rows'],12)
            full_roles=[row for row in report['teacher']['role_availability_by_phase'] if row['requires_full_reference_rotation']]
            self.assertTrue(full_roles)
            self.assertTrue(all(row['required_reference_available_rows'] == 0 for row in full_roles))
            arrays['valid'][10,1]=True
            with self.assertRaisesRegex(analysis.SmokeAuditError,'public segment'):
                analysis.validate_arrays(arrays,analysis.read(path.parent/'tracker_diagnostics.json'),get_task_spec(case['task']))

    def test_occluded_body_cannot_count_as_fresh_history(self):
        arrays,diagnostics=arrays_for()
        diagnostics['frames'][3]['bodies']['ball']['directly_visible']=False
        with self.assertRaisesRegex(analysis.SmokeAuditError,'falsely marks'):
            analysis.validate_arrays(arrays,diagnostics,get_task_spec('moving_basketball_in_hoop'))

    def test_parent_raw_child_equality_is_required(self):
        with tempfile.TemporaryDirectory(prefix='cpu_smoke_analysis_',dir=analysis.BASE) as folder:
            plan,case,path,_=self.fixture(folder)
            record=json.loads(path.read_text());raw=path.parent/'raw_child.json';raw.write_text(json.dumps(record))
            launcher=analysis.ROOT/'scripts/run_local_sim.sh'
            record['parent_process_receipt']=dict(child_case_path=str(raw.relative_to(analysis.ROOT)),
                child_case_sha256=analysis.sha(raw),launcher=str(launcher.relative_to(analysis.ROOT)),
                launcher_sha256=analysis.sha(launcher),command=['bash',str(launcher),'--case-id',case['case_id']])
            path.write_text(json.dumps(record));analysis.audit_case(path,plan,case,self.thresholds)
            record['entry']['frames']=39;path.write_text(json.dumps(record))
            with self.assertRaisesRegex(analysis.SmokeAuditError,'exact preserved raw child'):
                analysis.audit_case(path,plan,case,self.thresholds)


if __name__ == '__main__':unittest.main()
