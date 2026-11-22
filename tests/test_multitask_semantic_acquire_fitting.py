"""Real optional fitter branch with synthetic CPU evidence; never native SR."""
from dataclasses import replace
from pathlib import Path
import hashlib
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.multitask import semantic_acquire as semantic
from our_mani.multitask.interaction import (
    TRAINSemanticAcquireEvidence, TRAINSemanticAcquireSegment, TRAINInteractionSource,
    TRAINReferenceBinding, TRAINPhaseCompletionEvent, TRAINBodyTrack,
    InteractionArtifactError, InteractionFittingPending, fit_interaction_library)
from tests.test_multitask_semantic_acquire import fixture
from tests.test_multitask_completed_skill_fitting import fitting_config


def source(index, *, physical=True, offset=0., angle=0., parent=None):
    entry,a,q,bindings=fixture(parent or 'episode'+str(index));n=len(a['times'])
    binding=TRAINReferenceBinding('ball','ball','translation_only','acquire:ball','ball')
    poses=a['measured_tcp_xyzw'].copy();poses[:,0]=offset
    poses[:,3:]=Rotation.from_rotvec([0.,0.,angle]).as_quat()
    body=np.tile([0.,0.,0.,0.,0.,0.,1.],(n,1))
    track=TRAINBodyTrack('ball',a['times'],body,('external',)*3+('attached',)*6,
        np.ones(n,bool),pose_valid=np.ones(n,bool))
    event=TRAINPhaseCompletionEvent('acquire','lift','completed_command_then_ordered_teacher_phase_transition',
        2,(2,),.2,.4,binding,'attached',poses[4],poses[8],body[8],poses[4],4,True,'TRAIN_metric_only')
    derived=semantic.derive_acquire(entry,a,q,raw_bindings=bindings)
    segments=[]
    if physical:
        for s in derived['mode_segments']:
            segments.append(TRAINSemanticAcquireSegment(
                f"acquire:{s['mode']}:{s['frame_indices'][0]}-{s['frame_indices'][-1]}:terminal{s['terminal_frame']}",
                'acquire',binding,s['mode'],s['frame_indices'],s['raw_command_ids'],s['terminal_frame'],
                s['start_time'],s['end_time'],poses[s['frame_indices'][0]],poses[s['terminal_frame']],
                body[s['terminal_frame']],s['terminal_kind']))
    sid=entry['source_id']+':fixture'+str(index)
    evidence=TRAINSemanticAcquireEvidence(entry['task'],sid,entry['parent_source_group'],bindings,'a'*64,
        hashlib.sha256(Path(semantic.__file__).read_bytes()).hexdigest(),
        derived['status'] if physical else 'pending_missing_public_grasp_or_ball_association',tuple(segments),'b'*64)
    return TRAINInteractionSource(task_name=entry['task'],source_id=sid,split='train',source_kind='synthetic',
        times=a['times'],expert_tcp_xyzw=poses,expert_opening=np.zeros(n),measured_tcp_xyzw=poses,
        stages=('wp1',)*8+('wp2',),references=(binding,)*n,body_tracks={'ball':track},
        parent_source_group=entry['parent_source_group'],raw_episode_id=entry['parent_raw_episode_id']+'__r0',
        parent_raw_episode_id=entry['parent_raw_episode_id'],augmentation_id='fixture'+str(index),
        command_ids=a['teacher_command_id'],completion_events=(event,),curve_tcp_xyzw=poses,
        curve_label_kind='measured_executed_TCP',command_ledger=tuple(q['commands']),
        semantic_acquire_evidence=evidence)


def config():
    return fitting_config(minimum_sources=8,minimum_completed_parent_sources=8,development_only=False,
        lag_grid_seconds=(0.,.05),progress_knots=3)


def diagnostic(sources):
    # These fixtures intentionally contain only acquire.  The real fitter
    # therefore remains pending other required phases, while its actual
    # per-mode support/crossfit/cap/duration checks run without mocks.
    try:fit_interaction_library(sources,config())
    except InteractionFittingPending as error:return error.diagnostics
    raise AssertionError('acquire-only synthetic fixtures cannot release a full task library')


class SemanticAcquireFittingTests(unittest.TestCase):
    def test_eight_parents_fit_both_modes_keep_raw_ids_and_do_not_double_count_legacy_curve(self):
        values=[source(i) for i in range(8)];r=diagnostic(values)
        fits=r['phase_fits'];self.assertEqual(len(fits),2)
        self.assertTrue(all(x['passed'] and x['completed_parent_count']==8 for x in fits.values()))
        self.assertEqual(sorted(x['fitted_samples'] for x in fits.values()),[32,48])
        np.testing.assert_allclose(sorted(x['duration_seconds'] for x in fits.values()),[.15,.25],rtol=0,atol=1e-15)
        legacy=[x for x in r['curation'] if x.get('reason')=='represented_by_verified_semantic_acquire_evidence']
        segments=[x for x in r['curation'] if x.get('segment_id')]
        self.assertEqual(len(legacy),16);self.assertFalse(any(x['included_in_BC'] for x in legacy))
        self.assertEqual(len(segments),16);self.assertTrue(all(x['included_in_BC'] for x in segments))
        self.assertTrue(any(x['raw_command_ids']==[1,2] for x in segments))
        self.assertTrue(all(x['attempt_id']==x['source_id']+':semantic:'+x['segment_id'] for x in segments))
        self.assertTrue(all(x.completion_events[0].eligible_command_ids==(2,) for x in values))

    def test_seven_physical_parents_plus_phase_advance_without_grasp_stays_pending(self):
        values=[source(i) for i in range(7)]+[source(7,physical=False)];r=diagnostic(values)
        self.assertTrue(all(not x['passed'] and x['completed_parent_count']==7 for x in r['phase_fits'].values()))
        suppressed=[x for x in r['curation'] if x['source_id']==values[-1].source_id]
        self.assertFalse(any(x['included_in_BC'] for x in suppressed))
        self.assertTrue(any(x['original_phase_completion_event_present'] for x in suppressed))
        self.assertTrue(any(not x['duration_is_right_censored_lower_bound'] for x in suppressed))

    def test_replicas_share_parent_and_do_not_manufacture_eighth_support(self):
        values=[source(i) for i in range(7)]+[source(7,parent='episode0')];r=diagnostic(values)
        self.assertTrue(all(x['completed_parent_count']==7 and not x['passed'] for x in r['phase_fits'].values()))

    def test_semantic_endpoint_caps_remain_pending_without_clipping(self):
        for failure in ('position','angle'):
            values=[source(i,offset=(-.1 if i<4 else .1) if failure=='position' else 0.,
                angle=(-.2 if i<4 else .2) if failure=='angle' else 0.) for i in range(8)]
            with self.subTest(failure=failure):
                fits=diagnostic(values)['phase_fits']
                self.assertTrue(all(not x['passed'] and 'no clipping' in x['reason'] for x in fits.values()))
                self.assertTrue(all(x['maximum_position_tolerance_m']==.03
                    and x['maximum_orientation_tolerance_rad']==.15 for x in fits.values()))

    def test_positive_lag_never_bridges_external_contact_boundary(self):
        fits=diagnostic([source(i) for i in range(8)])['phase_fits']
        external=next(x for x in fits.values() if x['reference_mode']=='external')
        self.assertEqual(external['selected_lag'],0.)
        self.assertFalse(external['candidates'][1]['valid'])

    def test_mixed_rule_or_legacy_supervision_rejected(self):
        values=[source(i) for i in range(8)]
        for mixed in (replace(values[-1],semantic_acquire_evidence=None),
                replace(values[-1],semantic_acquire_evidence=replace(values[-1].semantic_acquire_evidence,
                    derivation_rule_sha256='c'*64))):
            with self.assertRaisesRegex(InteractionArtifactError,'mix'):
                fit_interaction_library(values[:-1]+[mixed],config())

    def test_typed_frame_command_reference_and_terminal_tampering_rejected(self):
        value=source(0);e=value.semantic_acquire_evidence;segment=e.segments[1]
        for change in ('command','TCP','phase'):
            with self.subTest(change=change),self.assertRaises(InteractionArtifactError):
                if change=='command':bad=replace(segment,raw_command_ids=(1,2,99))
                elif change=='TCP':bad=replace(segment,terminal_tcp_xyzw=segment.terminal_tcp_xyzw+
                    np.array([.1,0.,0.,0.,0.,0.,0.]))
                else:bad=replace(segment,binding=replace(segment.binding,anchor_id='unbound'))
                replace(value,semantic_acquire_evidence=replace(e,segments=(e.segments[0],bad)))


if __name__=='__main__':unittest.main()
