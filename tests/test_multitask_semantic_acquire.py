"""Pure CPU raw-evidence boundaries; no native, GPU or patched fitter types."""
from copy import deepcopy
import unittest
import numpy as np

from our_mani.multitask import semantic_acquire as semantic
from our_mani.multitask.interaction import (
    TRAINPhaseCompletionEvent, TRAINReferenceBinding, InteractionArtifactError)


def fixture(parent='episode0'):
    task=semantic.TASKS[0];n=9;t=np.arange(n)*.05
    entry=dict(task=task,split='train',teacher_used=True,interaction_eligible=True,
        source_kind='TRAIN_native_generated_augmentation',parent_raw_episode_id=parent,
        parent_source_group='train:'+task+':'+parent,source_id=task+':'+parent+'__r0',augmentation_id=parent+'__r0')
    arrays={name:np.ones((n,1),bool) for name in semantic.MASKS}
    arrays.update(times=t,body_ids=np.array(['ball']),modes=np.array([[0],[0],[0],[1],[1],[1],[1],[1],[1]],np.int8),
        robot_contact=np.array([False]*3+[True]*6),teacher_action_valid=np.array([True]*8+[False]),
        teacher_command_id=np.array([1]*4+[2]*4+[-1]),teacher_phase=np.array(['acquire']*8+['lift']),
        measured_tcp_xyzw=np.tile([0.,0.,0.,0.,0.,0.,1.],(n,1)))
    arrays['target_valid'][3,0]=arrays['target_twist_valid'][3,0]=False
    commands=[]
    for cid,begin,end in [(1,0.,.2),(2,.2,.4)]:
        commands.append(dict(command_id=cid,generated_at=begin,raw_phase='acquire',
            reference_body='ball',manipulated_body='ball',reference_role='ball',manipulated_role='ball',
            reference_kind='translation_only',executed_tick_count=4,command_completed_at=end))
    event=dict(phase='acquire',from_phase='acquire',to_phase='lift',origin_command_id=2,
        eligible_command_ids=[2],evidence='completed_command_then_ordered_teacher_phase_transition',
        label_namespace='TRAIN_metric_only',attempt_start_time=.2,attempt_end_time=.4,
        origin_command_completed=True,origin_executed_tick_count=4,
        current_measured_tcp_xyzw=[0.,0.,0.,0.,0.,0.,1.])
    ledger=dict(version='r10_TRAIN_committed_waypoint_commands_v1',commands=commands,
        phase_completion_events=[event])
    bindings={name:dict(path=name+'.json',sha256='a'*64) for name in ('canonical_receipt','cache','commands')}
    return entry,arrays,ledger,bindings


class SemanticAcquireTests(unittest.TestCase):
    def derive(self,values):
        entry,arrays,ledger,bindings=values
        return semantic.derive_acquire(entry,arrays,ledger,raw_bindings=bindings)

    def test_additive_whole_phase_two_real_mode_segments_keep_boundary_masks_and_raw_commands(self):
        values=fixture();before=deepcopy(values);result=self.derive(values)
        self.assertTrue(result['physical_acquire_evidence'])
        self.assertEqual(result['phase_command_ids'],[1,2])
        self.assertEqual(result['mode_segments'][0]['frame_indices'],(0,1,2))
        self.assertEqual(result['mode_segments'][1]['frame_indices'],(3,4,5,6,7))
        self.assertEqual(result['mode_segments'][1]['raw_command_ids'],(1,2))
        self.assertFalse(result['boundary']['target_valid']);self.assertFalse(result['boundary']['target_twist_valid'])
        self.assertEqual(result['original_acquire_completion_events'],before[2]['phase_completion_events'])
        self.assertEqual(result['raw_acquire_commands'],before[2]['commands'])
        for name in semantic.MASKS:np.testing.assert_array_equal(values[1][name],before[1][name])
        self.assertFalse(result['production_consumer_enabled'])

    def test_command_completed_without_forward_phase_remains_censored(self):
        values=fixture();values[2]['phase_completion_events']=[];result=self.derive(values)
        self.assertEqual(result['status'],'censored_no_original_acquire_completion')
        self.assertFalse(result['physical_acquire_evidence']);self.assertEqual(result['mode_segments'],[])
        self.assertTrue(all('command_completed_at' in x for x in result['raw_acquire_commands']))

    def test_teacher_stage_transition_without_public_grasp_cannot_label_physical_acquire(self):
        values=fixture();values[1]['robot_contact'][:]=False;result=self.derive(values)
        self.assertFalse(result['physical_acquire_evidence']);self.assertEqual(result['mode_segments'],[])
        self.assertEqual(len(result['original_acquire_completion_events']),1)

    def test_invalid_association_or_mode_change_preserved_no_cross_boundary_curve(self):
        for invalid in ('invalid_mode','invalid_pose','released','unknown_mode','teacher_gap'):
            values=fixture()
            if invalid=='invalid_mode':values[1]['mode_valid'][3,0]=False
            elif invalid=='invalid_pose':values[1]['native_pose_valid'][4,0]=False
            elif invalid=='released':values[1]['modes'][5,0]=2
            elif invalid=='unknown_mode':values[1]['modes'][5,0]=3
            else:values[1]['teacher_action_valid'][4]=False
            with self.subTest(invalid=invalid):
                result=self.derive(values);self.assertFalse(result['physical_acquire_evidence'])
                self.assertFalse(result['mode_segments']);self.assertTrue(result['original_mode_runs'])

    def test_source_scope_parent_and_unknown_body_fail_closed(self):
        for change in ('VAL','teacher','parent','reference','body'):
            values=fixture()
            if change=='VAL':values[0]['split']='validation'
            elif change=='teacher':values[0]['teacher_used']=False
            elif change=='parent':values[0]['parent_source_group']='train:other:source'
            elif change=='reference':values[2]['commands'][0]['reference_body']='unknown'
            else:values[1]['body_ids']=np.array(['other'])
            with self.subTest(change=change),self.assertRaises(ValueError):self.derive(values)

    def test_duplicate_noncontiguous_or_reordered_command_and_bad_terminal_fail_closed(self):
        for change in ('duplicate','phase_reentry','clock','terminal','raw_event_id','namespace'):
            values=fixture()
            if change=='duplicate':values[2]['commands'][1]['command_id']=1
            elif change=='phase_reentry':
                values[2]['commands'][1]['command_id']=3
                values[1]['teacher_command_id'][4:8]=3
                values[2]['phase_completion_events'][0].update(origin_command_id=3,eligible_command_ids=[3])
                values[2]['commands'].insert(1,dict(values[2]['commands'][0],command_id=2,
                    generated_at=.1,raw_phase='lift'))
            elif change=='clock':values[2]['commands'][1]['generated_at']=-1.
            elif change=='terminal':values[2]['phase_completion_events'][0]['attempt_end_time']=.425
            elif change=='raw_event_id':values[2]['phase_completion_events'][0]['eligible_command_ids']=[1,2]
            else:values[2]['phase_completion_events'][0]['label_namespace']='runtime_policy'
            with self.subTest(change=change),self.assertRaises(ValueError):self.derive(values)

    def test_clock_gap_missing_mask_wrong_binding_and_terminal_tcp_fail_closed(self):
        for change in ('gap','mask','bindings','TCP'):
            values=fixture()
            if change=='gap':values[1]['times'][4:]+=.1
            elif change=='mask':values[1].pop('target_twist_valid')
            elif change=='bindings':values[3]['unbound_runtime']=dict(path='x',sha256='b'*64)
            else:values[2]['phase_completion_events'][0]['current_measured_tcp_xyzw'][0]=.1
            with self.subTest(change=change),self.assertRaises(ValueError):self.derive(values)

    def test_support_groups_replicas_by_original_parent_and_preserves_existing_gate(self):
        rows=[self.derive(fixture('episode'+str(i))) for i in range(8)]
        duplicate=deepcopy(rows[0]);duplicate['source_id']+='__r1';duplicate['augmentation_id']+='__r1'
        result=semantic.summarize(rows+[duplicate])[semantic.TASKS[0]]
        self.assertEqual(result['acquisitions'],9);self.assertEqual(result['original_parents'],8)
        self.assertEqual(len(result['external_parent_sources']),8);self.assertEqual(len(result['attached_parent_sources']),8)
        self.assertEqual(result['existing_minimum_completed_parents'],8)
        self.assertEqual((result['existing_position_cap_m'],result['existing_orientation_cap_rad']),(.03,.15))
        self.assertFalse(result['formal_fitter_acceptance_proven'])

    def test_actual_legacy_completion_type_refuses_honest_multicommand_segment(self):
        binding=TRAINReferenceBinding('ball','ball','translation_only','acquire:ball','ball')
        pose=np.array([0.,0.,0.,0.,0.,0.,1.])
        with self.assertRaisesRegex(InteractionArtifactError,'explicit executed TRAIN phase completion evidence'):
            TRAINPhaseCompletionEvent(phase='acquire',to_phase='lift',
                evidence='completed_command_then_ordered_teacher_phase_transition',origin_command_id=1,
                eligible_command_ids=(1,2),attempt_start_time=0.,attempt_end_time=.4,binding=binding,
                reference_mode='attached',origin_measured_tcp_xyzw=pose,terminal_tcp_xyzw=pose,
                terminal_reference_xyzw=pose,committed_world_waypoint_xyzw=pose,
                executed_tick_count=8,command_completed=True,label_namespace='TRAIN_metric_only')
        self.assertFalse(semantic.legacy_fitter_interface()['compatible'])

    def test_segment_contract_rejects_fake_frame_continuity_or_duplicate_raw_commands(self):
        for frames,ids in [((0,2),(1,)),((0,1),(1,1))]:
            with self.assertRaises(ValueError):
                semantic.AcquireModeSegment('external',frames,ids,0.,.15,3,
                    'public_external_to_attached_boundary',2)


if __name__=='__main__':unittest.main()
