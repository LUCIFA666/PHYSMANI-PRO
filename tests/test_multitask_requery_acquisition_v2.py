"""CPU-only v2 audit of actual follower feedback arrays and raw clocks."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from our_mani.contracts import Pose
from our_mani.multitask import train_teacher
from our_mani.multitask import requery_acquisition as audit
from our_mani.multitask.requery_acquisition import validate_counted_commands, RAW_AUDIT_VERSION
from our_mani.multitask.train_teacher import FollowerConfig, TeacherPathPlanningFailure
from our_mani.multitask.train_teacher_requery import PlanningRequeryConfig
from our_mani.multitask.train_teacher_requery_hook import scoped_counted_requery, file_sha, TASK


def actual_CPU_follower_rows(*, ticks=12, successful_second=False):
    """Use real follower callbacks; imitate unchanged collector row timing.

    Motor feedback here is an explicit CPU fixture, never native evidence.
    """
    case=dict(case_id='CPU_v2',split='train',task=TASK,behavior='TRAIN_teacher_committed_waypoint',
              teacher_used=True,context_primary_view=False,interaction_eligible=True,
              replica=1,behavior_replica=1,maximum_collection_ticks=200)
    def teacher():
        return dict(tcp_xyzw=Pose(np.ones(3)).xyzw(),opening=1.,stage='wp3',phase='transport',
            reference_body='spoke0',manipulated_body='cup0',reference_role='selected_spoke',
            reference_kind='observed_se3',anchor_id='cup:transport',manipulated_role='cup0')
    planned=[]
    class CPUPath:
        def step(self): return True
        def get_executed_joint_position_action(self): return np.zeros(7)
        def close(self): pass
    def planner(goal):
        planned.append(goal)
        if successful_second and len(planned)==2:return CPUPath()
        raise TeacherPathPlanningFailure('CPU fixed planner failure')
    rows=[];captures=[]
    with scoped_counted_requery(case,expected_teacher_source_sha256=file_sha(Path(train_teacher.__file__)),
            configuration=PlanningRequeryConfig(),captures=captures) as scope:
        follower=train_teacher.CommittedWaypointFollower(teacher,planner,FollowerConfig(.02,.1,.005,3),initial_opening=1.)
        robot=SimpleNamespace(timestamp=0.,joints=np.zeros(7),pose=Pose(np.ones(3)))
        for tick in range(ticks):
            robot.timestamp=tick*.05
            result=follower.command(robot,np.full(7,-1.),np.full(7,1.))
            feedback=dict(timestamp=(tick+1)*.05,gripper_actuation_done=True,
                measured_tcp_xyzw=robot.pose.xyzw(),joint_limit_recovery_action=result['joint_limit_recovery_action'])
            completed=follower.after_tick(feedback)
            rows.append(dict(times=tick*.05,teacher_command_id=result['command_id'],
                teacher_command_is_new=result['command_is_new'],teacher_command_generated_at=result['command_generated_at'],
                teacher_command_age_seconds=result['command_age_seconds'],actuator_feedback_timestamp=feedback['timestamp'],
                teacher_execution_stage=result['execution_state'],teacher_execution_valid=result['teacher_execution_valid'],
                teacher_action_valid=bool(result['label']['stage']) and result['teacher_execution_valid'],
                teacher_command_completed=completed,teacher_joint_limit_recovery=result['joint_limit_recovery_action']))
        rows.append(dict(times=ticks*.05,teacher_command_id=-1,teacher_command_is_new=False,
            teacher_command_generated_at=0.,teacher_command_age_seconds=0.,actuator_feedback_timestamp=ticks*.05,
            teacher_execution_stage='no_teacher',teacher_execution_valid=False,teacher_action_valid=False,
            teacher_command_completed=False,teacher_joint_limit_recovery=False))
        follower.finish('fixed_tick_budget',ticks*.05)
    arrays={key:np.asarray([row[key] for row in rows]) for key in rows[0]}
    return dict(commands=deepcopy(follower.commands),phase_completion_events=[]),dict(scope),arrays


class ActualFeedbackAuditV2Tests(unittest.TestCase):
    def verify(self,ledger,scope,arrays,ticks=12):
        return validate_counted_commands(ledger,scope,ticks=ticks,arrays=arrays)

    def test_real_CPU_follower_rows_reconstruct_exact_ten_feedbacks_and_partial_failure(self):
        ledger,scope,arrays=actual_CPU_follower_rows()
        proof=self.verify(ledger,scope,arrays)
        self.assertEqual(proof['raw_audit_version'],RAW_AUDIT_VERSION)
        self.assertIs(proof['actual_feedback_arrays_verified'],True)
        self.assertEqual(proof['actual_requeries'],1)
        wait=proof['reconstructed_requery_waits'][0]
        self.assertEqual(wait['wait_action_frames'],list(range(10)))
        self.assertEqual(wait['next_issue_frame'],10)
        self.assertEqual(wait['feedback_frame_times'],[i*.05 for i in range(1,11)])
        self.assertEqual(ledger['commands'][-1]['planning_requery_counted_wait_ticks'],2)
        self.assertNotIn('command_completed_at',ledger['commands'][-1])

    def test_successful_command_then_normal_query_does_not_invent_failure_marker(self):
        ledger,scope,arrays=actual_CPU_follower_rows(successful_second=True)
        self.assertEqual(len(ledger['commands']),3)
        self.assertIn('command_completed_at',ledger['commands'][1])
        self.assertNotIn('fresh_query_after_failed_command_id',ledger['commands'][2])
        self.assertEqual(self.verify(ledger,scope,arrays)['actual_requeries'],1)
        row=ledger['commands'][2];row.update(fresh_query_after_failed_command_id=1,
            same_episode_after_counted_wait=True,actual_planning_requery_index=2)
        scope['actual_requeries']=2
        with self.assertRaisesRegex(ValueError,'nonfailed'):
            self.verify(ledger,scope,arrays)

    def test_missing_or_partial_requery_markers_cannot_lower_scope_count(self):
        fields=('fresh_query_after_failed_command_id','same_episode_after_counted_wait','actual_planning_requery_index')
        for removed in (fields,*[(key,) for key in fields]):
            ledger,scope,arrays=actual_CPU_follower_rows()
            for key in removed:ledger['commands'][1].pop(key)
            scope['actual_requeries']=0
            with self.subTest(removed=removed),self.assertRaisesRegex(ValueError,'all fresh-query markers'):
                self.verify(ledger,scope,arrays)

    def test_command_clock_shift_nan_bool_off_grid_negative_or_reversed_rejected(self):
        for mode in ('shift100','NaN','boolean','offgrid','negative','reversed'):
            ledger,scope,arrays=actual_CPU_follower_rows()
            if mode=='shift100':
                for row in ledger['commands']:
                    for key in ('generated_at','planning_requery_wait_started_at','planning_requery_wait_completed_at',
                                'planning_requery_last_actual_feedback_at','cutoff_at'):
                        if key in row:row[key]+=100.
            elif mode=='NaN':ledger['commands'][1]['generated_at']=float('nan')
            elif mode=='boolean':ledger['commands'][0]['generated_at']=False
            elif mode=='offgrid':ledger['commands'][1]['generated_at']=.501
            elif mode=='negative':ledger['commands'][0]['generated_at']=-.05
            else:ledger['commands'][1]['generated_at']=0.
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.verify(ledger,scope,arrays)

    def test_feedback_replay_skip_unissued_row_generation_and_issue_mask_rejected(self):
        for mode in ('replay','skip','unknownID','wronggen','missingnew','extraneousnew','terminalaction','prematureID','wrongage'):
            ledger,scope,arrays=actual_CPU_follower_rows()
            if mode=='replay':arrays['actuator_feedback_timestamp'][4]=.2
            elif mode=='skip':arrays['actuator_feedback_timestamp'][4]=.3
            elif mode=='unknownID':arrays['teacher_command_id'][3]=8
            elif mode=='wronggen':arrays['teacher_command_generated_at'][4]=.05
            elif mode=='missingnew':arrays['teacher_command_is_new'][10]=False
            elif mode=='extraneousnew':arrays['teacher_command_is_new'][4]=True
            elif mode=='terminalaction':arrays['teacher_command_id'][-1]=1
            elif mode=='prematureID':arrays['teacher_command_id'][9]=1
            else:arrays['teacher_command_age_seconds'][4]=.05
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.verify(ledger,scope,arrays)

    def test_failed_wait_cannot_be_promoted_and_real_feedback_count_cannot_be_omitted(self):
        for mode in ('action','execute','complete','truncatedwait','wronglast','missingarray'):
            ledger,scope,arrays=actual_CPU_follower_rows()
            if mode=='action':arrays['teacher_action_valid'][2]=True
            elif mode=='execute':arrays['teacher_execution_valid'][2]=True
            elif mode=='complete':arrays['teacher_command_completed'][2]=True
            elif mode=='truncatedwait':
                ledger['commands'][-1]['planning_requery_counted_wait_ticks']=1
                ledger['commands'][-1]['planning_requery_last_actual_feedback_at']=.55
            elif mode=='wronglast':ledger['commands'][0]['planning_requery_last_actual_feedback_at']=.45
            else:arrays.pop('actuator_feedback_timestamp')
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.verify(ledger,scope,arrays)

    def test_full_wait_at_budget_end_is_not_an_unissued_requery(self):
        ledger,scope,arrays=actual_CPU_follower_rows(ticks=10)
        proof=self.verify(ledger,scope,arrays,ticks=10)
        self.assertEqual(proof['actual_requeries'],0)
        self.assertEqual(proof['reconstructed_requery_waits'],[])
        self.assertEqual(len(ledger['commands']),1)

    def test_zero_tick_terminal_does_not_create_commands_or_feedback(self):
        ledger,scope,arrays=actual_CPU_follower_rows(ticks=0)
        proof=self.verify(ledger,scope,arrays,ticks=0)
        self.assertEqual(proof['actual_requeries'],0)
        self.assertEqual(proof['actual_feedback_rows'],0)

    def test_arrival_completion_cutoff_and_reference_time_order_is_strict(self):
        for mode in ('arrival','last_execution','reference','cutoff'):
            ledger,scope,arrays=actual_CPU_follower_rows(successful_second=True)
            if mode=='arrival':ledger['commands'][1]['measured_arrival_at']=.6
            elif mode=='last_execution':ledger['commands'][1]['last_executed_at']=.6
            elif mode=='reference':ledger['commands'][1]['command_terminal_reference_metric_time']=.6
            else:ledger['commands'][0]['cutoff_at']=.45
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.verify(ledger,scope,arrays)


class TypedManifestV2Tests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parents[1]
        self.temp=tempfile.TemporaryDirectory(prefix='requery_manifest_CPU_',dir=self.root/'artifacts/framework/multitask_r10')
        self.addCleanup(self.temp.cleanup);self.directory=Path(self.temp.name)
        self.proposal=self.directory/'proposal.json';self.proposal.write_text(json.dumps(dict(original_plan=dict(path='CPU_primary.json',sha256='CPU_original'))))
        self.registry=dict(proposal=audit.lineage.pointer(self.root,self.proposal),execution=dict(plan=dict(path='CPU_actual.json',sha256='CPU_actual')))
        self.projection=dict(version=audit.MANIFEST_VERSION,execution_registration=dict(path='CPU_registry',sha256='CPU_registry'),
            actual_plan=self.registry['execution']['plan'],plan_sha256='CPU_plan_internal',
            primary_K1_roster_unmodified=True,primary_Context_view_unmodified=True,complete=True)
        self.path=self.directory/'manifest.json';self.path.write_text(json.dumps(self.projection))
        self.proof=dict(complete=True,acquisition_projection=self.projection,verified_files_sha256={})
        self.context=dict(registry=self.registry,plan=dict(plan_sha256='CPU_plan_internal'),
            primary=dict(plan_sha256='CPU_original_internal'),files={})

    def test_exact_projection_preserves_actual_plan_and_original_primary_identity(self):
        with patch.object(audit,'audit_collection',return_value=deepcopy(self.proof)) as collection,patch.object(audit,'verify_registration',return_value=self.context):
            proof=audit.audit_manifest(self.root,self.path)
        collection.assert_called_once_with(self.root,self.projection['execution_registration'],require_complete=True)
        self.assertEqual(proof['base_plan_sha256'],'CPU_original_internal')
        self.assertEqual(proof['primary_plan'],dict(path='CPU_primary.json',sha256='CPU_original'))
        self.assertEqual(proof['acquisition_projection']['actual_plan'],self.projection['actual_plan'])
        self.assertEqual(proof['verified_files_sha256'][str(self.path)],audit.lineage.file_sha(self.path))

    def test_wrong_schema_changed_projection_or_false_actual_plan_rejected(self):
        for mode in ('schema','projection','actualplan'):
            value=deepcopy(self.projection);proof=deepcopy(self.proof)
            if mode=='schema':value['version']='r10_other_v1'
            elif mode=='projection':value['complete']=False
            else:value['actual_plan']=dict(path='CPU_other.json',sha256='other');proof['acquisition_projection']=value
            self.path.write_text(json.dumps(value))
            with patch.object(audit,'audit_collection',return_value=proof),patch.object(audit,'verify_registration',return_value=self.context),self.subTest(mode=mode),self.assertRaises(ValueError):
                audit.audit_manifest(self.root,self.path)

    def test_incomplete_actual_collection_cannot_satisfy_required_manifest(self):
        def incomplete(root,pointer,*,require_complete):
            if require_complete:raise ValueError('all100 committed new teacher sources required')
            return deepcopy(self.proof)
        with patch.object(audit,'audit_collection',side_effect=incomplete),self.assertRaisesRegex(ValueError,'all100 committed'):
            audit.audit_manifest(self.root,self.path)


if __name__=='__main__':unittest.main()
