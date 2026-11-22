"""Completed skill support, censored commands and canonical byte boundaries."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from our_mani.multitask.interaction import TRAINReferenceBinding,TRAINPhaseCompletionEvent,TRAINInteractionSource
from scripts.report_multitask_phase_support_r10 import (ROOT,sha,classify_attempts,
    summarize_attempts,verify_canonical_case)


IDENTITY=np.array([0.,0.,0.,0.,0.,0.,1.])


def source_fixture(parent='episode1',suffix='a',anchor='fixture:approach'):
    binding=TRAINReferenceBinding('world',None,'world',anchor,'button')
    press=TRAINReferenceBinding('world',None,'world','fixture:press','button')
    event=TRAINPhaseCompletionEvent(phase='approach',to_phase='press',
        evidence='completed_command_then_ordered_teacher_phase_transition',origin_command_id=0,eligible_command_ids=(0,),
        attempt_start_time=0.,attempt_end_time=.1,binding=binding,reference_mode='not_applicable',
        origin_measured_tcp_xyzw=IDENTITY,terminal_tcp_xyzw=IDENTITY,terminal_reference_xyzw=IDENTITY,
        committed_world_waypoint_xyzw=IDENTITY,executed_tick_count=2,command_completed=True,label_namespace='TRAIN_metric_only')
    poses=np.tile(IDENTITY,(4,1))
    return TRAINInteractionSource(task_name='push_moving_button',source_id=parent+':'+suffix,split='train',
        source_kind='TRAIN_native_generated_augmentation',times=np.array([0.,.05,.1,.15]),
        expert_tcp_xyzw=poses,expert_opening=np.ones(4),measured_tcp_xyzw=poses,
        stages=('wp0','wp0','wp1','wp1'),references=(binding,binding,press,None),body_tracks={},
        parent_source_group='train:push_moving_button:'+parent,augmentation_id=suffix,
        raw_episode_id=parent+'__'+suffix,parent_raw_episode_id=parent,command_ids=np.array([0,0,1,-1]),
        completion_events=(event,),curve_tcp_xyzw=poses,curve_label_kind='measured_executed_TCP',
        command_ledger=(dict(command_id=0,raw_phase='approach',generated_at=0.,executed_tick_count=2,
            last_executed_at=.1,command_completed_at=.1),
            dict(command_id=1,raw_phase='press',generated_at=.1,executed_tick_count=1,last_executed_at=.15,
                command_completed_at=.15)))


class PhaseSupportSemanticsTests(unittest.TestCase):
    def test_failed_task_keeps_earlier_completed_skill_and_waypoint_only_is_censored(self):
        rows=classify_attempts(source_fixture(),task_success=False)
        summary=summarize_attempts(rows,['approach','press'])
        self.assertEqual(summary['approach']['distinct_completed_TRAIN_parents'],1)
        self.assertEqual(summary['approach']['completed_skill_attempts_from_eventually_failed_tasks'],1)
        self.assertEqual(summary['press']['distinct_completed_TRAIN_parents'],0)
        self.assertEqual(summary['press']['right_censored_incomplete_attempts'],1)
        self.assertEqual(summary['press']['waypoint_command_done_without_usable_completed_skill'],1)

    def test_repeated_rollouts_do_not_multiply_distinct_original_parent_support(self):
        rows=classify_attempts(source_fixture(suffix='a'),task_success=True)
        rows+=classify_attempts(source_fixture(suffix='b'),task_success=True)
        summary=summarize_attempts(rows,['approach'])['approach']
        self.assertEqual(summary['completed_skill_attempts'],2)
        self.assertEqual(summary['distinct_completed_TRAIN_parents'],1)

    def test_invalid_completed_label_is_not_mislabeled_as_censored_or_success(self):
        source=replace(source_fixture(),excluded_completion_events=(dict(command_id=1,phase='press',reason='terminal full orientation unavailable'),))
        rows=classify_attempts(source,task_success=False);press=summarize_attempts(rows,['press'])['press']
        self.assertEqual(press['completed_but_label_or_curve_excluded_attempts'],1)
        self.assertEqual(press['right_censored_incomplete_attempts'],0)
        self.assertEqual(press['distinct_completed_TRAIN_parents'],0)

    def test_phase_aggregate_eight_does_not_hide_four_parent_reference_partitions(self):
        rows=[]
        for i in range(8):
            source=source_fixture(parent='episode'+str(i),anchor='fixture:A' if i<4 else 'fixture:B')
            rows.extend(classify_attempts(source,task_success=True))
        phase=summarize_attempts(rows,['approach'])['approach']
        self.assertTrue(phase['aggregate_phase_parent_support_met'])
        self.assertEqual(len(phase['reference_partitions']),2)
        self.assertTrue(all(row['distinct_completed_TRAIN_parents']==4 and not row['necessary_parent_support_met']
            for row in phase['reference_partitions'].values()))

    def test_completed_event_without_actual_usable_curve_never_counts_support(self):
        source=source_fixture();source=replace(source,references=(None,None,source.references[2],None))
        rows=classify_attempts(source,task_success=True)
        phase=summarize_attempts(rows,['approach'])['approach']
        self.assertEqual(phase['distinct_completed_TRAIN_parents'],0)
        self.assertEqual(phase['completed_but_label_or_curve_excluded_attempts'],1)
        self.assertEqual(phase['right_censored_incomplete_attempts'],0)

    def test_no_observations_is_zero_pending_support_not_synthetic_default(self):
        phase=summarize_attempts([],['approach'])['approach']
        self.assertEqual(phase['distinct_completed_TRAIN_parents'],0)
        self.assertFalse(phase['aggregate_phase_parent_support_met'])
        self.assertFalse(phase['formal_fitting_acceptance_proven'])


class CanonicalPhaseReceiptBoundaryTests(unittest.TestCase):
    def test_exact_raw_child_and_track_hash_are_required_before_audit(self):
        with tempfile.TemporaryDirectory(prefix='phase-support-fixture-',dir=ROOT/'runtime') as directory:
            root=Path(directory);cache=root/'body_tracks.npz';source=root/'original.npz';commands=root/'commands.json'
            launcher=root/'launcher.sh';launcher.write_text('fixture only\n')
            np.savez(cache,times=np.array([0.,.05]));np.savez(source,fixture=np.array([1.]))
            commands.write_text(json.dumps(dict(version='r10_TRAIN_committed_waypoint_commands_v1',commands=[],phase_completion_events=[])))
            expected=dict(case_id='fixturecase',task='push_moving_button',split='train',raw_episode_id='episode1__teacher',
                parent_raw_episode_id='episode1',parent_source_group='train:push_moving_button:episode1',
                behavior='TRAIN_teacher_committed_waypoint',teacher_used=True,interaction_eligible=True,maximum_collection_ticks=200)
            entry=dict(expected,cache='body_tracks.npz',cache_sha256=sha(cache),frames=2,
                source_demonstration=dict(path='original.npz',sha256=sha(source)),
                phase_completion_events_receipt='commands.json',teacher_command_receipt='commands.json')
            child=dict(status='completed',split='train',case_id='fixturecase',plan_sha256='f'*64,
                task=expected['task'],raw_episode_id=expected['raw_episode_id'],parent_source_group=expected['parent_source_group'],
                case_isolation='fresh_simulator_process_per_episode',teacher_actions_used=True,policy_history_contains_native_GT=False,
                process_id=123,entry=entry,files_sha256={'body_tracks.npz':sha(cache),'commands.json':sha(commands)},
                alignment=dict(clock=dict(ticks=1,dt=.05,budget_seconds=10.,simulation_seconds=.05),
                    actual_physics_seconds=.05,actual_task_seconds=.05))
            raw=root/'child.json';raw.write_text(json.dumps(child))
            record={**child,'parent_process_receipt':dict(child_case_path='child.json',child_case_sha256=sha(raw),
                launcher='launcher.sh',launcher_sha256=sha(launcher),process_id=456)}
            canonical=root/'case.json';canonical.write_text(json.dumps(record))
            case,_=verify_canonical_case(root,canonical,expected,'f'*64)
            self.assertTrue(case['command_receipt_verified'])
            np.savez(cache,times=np.array([0.,.051]))
            with self.assertRaisesRegex(ValueError,'body-track bytes'):
                verify_canonical_case(root,canonical,expected,'f'*64)


if __name__=='__main__':unittest.main()
