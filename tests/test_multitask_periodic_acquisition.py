"""Real CPU follower callbacks and strict saved-row audits; no native evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.multitask import periodic_acquisition as audit,train_teacher
from our_mani.multitask.train_teacher import FollowerConfig,TeacherPathPlanningFailure
from our_mani.multitask.train_teacher_periodic_replan import PeriodicReplanningConfig
from our_mani.multitask.train_teacher_periodic_replan_hook import scoped_periodic_replanning,file_sha
from our_mani.multitask.train_teacher_requery import PlanningRequeryConfig
ROOT=Path(__file__).resolve().parents[1]


def CPU_rows(*,ticks=12,fail=False,done=False,gripper_done_at=None,recovery_ticks=(),phase='transport'):
    case=dict(case_id='CPU_periodic',split='train',task=audit.TASK,behavior=audit.BEHAVIOR,
        teacher_used=True,context_primary_view=False,interaction_eligible=True,
        replica=2,behavior_replica=2,maximum_collection_ticks=200)
    def teacher():return dict(tcp_xyzw=Pose(np.ones(3)).xyzw(),opening=0.,stage='wp3',phase=phase,
        reference_body='spoke0',manipulated_body='cup0',reference_role='selected_spoke',
        reference_kind='observed_se3',anchor_id='CPU:transport',manipulated_role='cup0')
    class CPUPath:
        def step(self):return done
        def get_executed_joint_position_action(self):return np.zeros(7)
        def close(self):pass
    def planner(goal):
        if fail:raise TeacherPathPlanningFailure('CPU preserved failure')
        return CPUPath()
    rows=[];captures=[]
    with scoped_periodic_replanning(case,expected_teacher_source_sha256=file_sha(train_teacher.__file__),
        periodic_configuration=PeriodicReplanningConfig(),failure_configuration=PlanningRequeryConfig(),captures=captures) as scope:
        follower=train_teacher.CommittedWaypointFollower(teacher,planner,FollowerConfig(.02,.1,.005,3),initial_opening=1.)
        robot=SimpleNamespace(timestamp=0.,joints=np.zeros(7),pose=Pose(np.ones(3)))
        for tick in range(ticks):
            robot.timestamp=tick*.05;robot.joints=np.full(7,1.001) if tick in recovery_ticks else np.zeros(7)
            result=follower.command(robot,np.full(7,-1.),np.full(7,1.))
            feedback=dict(timestamp=(tick+1)*.05,gripper_actuation_done=gripper_done_at is not None and tick>=gripper_done_at,
                measured_tcp_xyzw=robot.pose.xyzw(),joint_limit_recovery_action=result['joint_limit_recovery_action'])
            completed=follower.after_tick(feedback)
            rows.append(dict(times=tick*.05,teacher_command_id=result['command_id'],teacher_command_is_new=result['command_is_new'],
                teacher_command_generated_at=result['command_generated_at'],teacher_command_age_seconds=result['command_age_seconds'],
                actuator_feedback_timestamp=feedback['timestamp'],teacher_execution_stage=result['execution_state'],
                teacher_execution_valid=result['teacher_execution_valid'],teacher_action_valid=result['teacher_execution_valid'],
                teacher_command_completed=completed,teacher_joint_limit_recovery=result['joint_limit_recovery_action']))
        rows.append(dict(times=ticks*.05,teacher_command_id=-1,teacher_command_is_new=False,teacher_command_generated_at=0.,
            teacher_command_age_seconds=0.,actuator_feedback_timestamp=ticks*.05,teacher_execution_stage='no_teacher',
            teacher_execution_valid=False,teacher_action_valid=False,teacher_command_completed=False,teacher_joint_limit_recovery=False))
        follower.finish('fixed_tick_budget',ticks*.05)
    arrays={key:np.asarray([row[key] for row in rows]) for key in rows[0]}
    return dict(commands=deepcopy(follower.commands),phase_completion_events=[]),dict(scope),arrays


class PeriodicActualFeedbackTests(unittest.TestCase):
    def verify(self,ledger,scope,arrays,*,ticks=12,**kwargs):
        return audit.validate_counted_commands(ledger,scope,ticks=ticks,arrays=arrays,**kwargs)

    def test_ten_actual_rows_cut_path_and_new_query_preserves_incomplete_command(self):
        ledger,scope,arrays=CPU_rows();proof=self.verify(ledger,scope,arrays)
        self.assertEqual(proof['actual_periodic_requeries'],1);self.assertEqual(proof['actual_failed_planning_requeries'],0)
        self.assertEqual(proof['reconstructed_periodic_interruptions'][0]['actual_action_frames'],list(range(10)))
        self.assertNotIn('command_completed_at',ledger['commands'][0]);self.assertTrue(proof['actual_periodic_feedback_arrays_verified'])

    def test_final_budget_pending_interrupt_is_not_a_query_or_completed_skill(self):
        ledger,scope,arrays=CPU_rows(ticks=10);proof=self.verify(ledger,scope,arrays,ticks=10)
        self.assertEqual(proof['actual_periodic_requeries'],0);self.assertEqual(proof['pending_periodic_interruptions'],1)
        self.assertEqual(ledger['commands'][0]['cutoff_reason'],'fixed_tick_budget')

    def test_failed_wait_is_a_separate_preserved_counter(self):
        ledger,scope,arrays=CPU_rows(fail=True);proof=self.verify(ledger,scope,arrays)
        self.assertEqual(proof['actual_periodic_requeries'],0);self.assertEqual(proof['actual_failed_planning_requeries'],1)
        self.assertEqual(proof['failed_commands'],2);self.assertEqual(proof['reconstructed_requery_waits'][0]['actual_wait_ticks'],10)

    def test_actual_gripper_actuation_can_defer_without_fabricating_periodic_completion(self):
        ledger,scope,arrays=CPU_rows(done=True);proof=self.verify(ledger,scope,arrays)
        self.assertEqual(proof['actual_periodic_requeries'],0);self.assertEqual(proof['interrupted_commands'],0)
        self.assertEqual(ledger['commands'][0]['periodic_replan_deferred_reason'],'actual_gripper_actuation_or_release_settling')

    def test_pre_and_post_cut_recovery_delays_require_real_recovery_rows(self):
        for recovery in ((9,),(10,11)):
            ledger,scope,arrays=CPU_rows(ticks=14,recovery_ticks=recovery)
            proof=self.verify(ledger,scope,arrays,ticks=14);self.assertEqual(proof['actual_periodic_requeries'],1)
            changed=deepcopy(arrays);changed['teacher_joint_limit_recovery'][recovery[0]]=False
            with self.subTest(recovery=recovery),self.assertRaises(ValueError):self.verify(ledger,scope,changed,ticks=14)

    def test_early_fake_completed_wrong_counter_and_missing_marker_reject(self):
        for mutation in ('cut9','complete','count','failed_alias','missing_marker','pending','path_unclosed','post_counter','fake_defer'):
            ledger,scope,arrays=CPU_rows();old,new=ledger['commands'];nested=old['periodic_interruption']
            if mutation=='cut9':nested['actual_feedback_ticks']=9;nested['last_actual_feedback_at']=.45
            elif mutation=='complete':old['command_completed_at']=.5
            elif mutation=='count':scope['actual_periodic_requeries']=0
            elif mutation=='failed_alias':new['fresh_query_after_failed_command_id']=0
            elif mutation=='missing_marker':new.pop('same_episode_after_periodic_actual_feedback')
            elif mutation=='pending':scope['pending_periodic_interruptions']=1
            elif mutation=='path_unclosed':nested['actual_path_closed']=False
            elif mutation=='post_counter':old['post_query_actual_feedback_ticks']=9
            else:nested['actual_feedback_ticks']=11;nested['last_actual_feedback_at']=.55
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.verify(ledger,scope,arrays)

    def test_motor_row_tampering_is_not_accepted_as_periodic_feedback(self):
        for mutation in ('replay','skip','ID','new','age','complete','valid','terminal','missing'):
            ledger,scope,arrays=CPU_rows()
            if mutation=='replay':arrays['actuator_feedback_timestamp'][4]=.2
            elif mutation=='skip':arrays['actuator_feedback_timestamp'][4]=.3
            elif mutation=='ID':arrays['teacher_command_id'][9]=1
            elif mutation=='new':arrays['teacher_command_is_new'][10]=False
            elif mutation=='age':arrays['teacher_command_age_seconds'][4]=.05
            elif mutation=='complete':arrays['teacher_command_completed'][9]=True
            elif mutation=='valid':arrays['teacher_execution_valid'][3]=False
            elif mutation=='terminal':arrays['teacher_command_id'][-1]=1
            else:arrays.pop('actuator_feedback_timestamp')
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.verify(ledger,scope,arrays)

    def test_interrupted_ordered_event_rejects_but_original_native_terminal_outcome_is_independent(self):
        ledger,scope,arrays=CPU_rows(ticks=10,phase='hang')
        ledger['phase_completion_events']=[dict(origin_command_id=0,evidence='completed_command_then_ordered_teacher_phase_transition',phase='transport',to_phase='align',attempt_end_time=.5,origin_command_completed=False)]
        with self.assertRaises(ValueError):self.verify(ledger,scope,arrays,ticks=10)
        event=ledger['phase_completion_events'][0];event.update(evidence='TRAIN_native_success_during_executed_final_required_phase',phase='hang',to_phase='native_success')
        with self.assertRaises(ValueError):self.verify(ledger,scope,arrays,ticks=10)
        arrays.update(body_ids=np.asarray(['spoke0']),measured_tcp_xyzw=np.tile(Pose(np.ones(3)).xyzw(),(11,1)),
            target_positions=np.zeros((11,1,3)),target_rotations=np.tile(np.eye(3),(11,1,1,1)),native_pose_valid=np.ones((11,1),dtype=bool))
        event.update(origin_reference_body='spoke0',origin_reference_kind='observed_se3',eligible_command_ids=[0],
            current_measured_tcp_xyzw=Pose(np.ones(3)).xyzw().tolist(),current_reference_native_pose_xyzw=Pose(np.zeros(3)).xyzw().tolist(),
            current_reference_native_pose_valid=True)
        self.verify(ledger,scope,arrays,ticks=10,native_outcome=dict(success=True))
        self.assertNotIn('command_completed_at',ledger['commands'][0])
        event['current_measured_tcp_xyzw'][0]+=.01
        with self.assertRaises(ValueError):self.verify(ledger,scope,arrays,ticks=10,native_outcome=dict(success=True))


class PeriodicRosterTests(unittest.TestCase):
    def test_actual_all100_original_parents_get_new_replica2_ids_seeds_and_unchanged_geometry(self):
        p=ROOT/'artifacts/framework/multitask_r10/TRAIN_full_collection_plan_v11.json';primary=audit.lineage.read(p);old=deepcopy(primary)
        cases=audit.expected_cases(primary);self.assertEqual(primary,old);self.assertEqual(len(cases),100)
        self.assertTrue(all(c['replica']==c['behavior_replica']==2 for c in cases))
        self.assertFalse({c['case_id'] for c in cases}&{c['case_id'] for c in primary['cases']})
        for row in cases:
            before=next(c for c in primary['cases'] if c['parent_source_group']==row['parent_source_group'] and c['teacher_used'])
            self.assertEqual(row['template_asset'],before['template_asset']);self.assertEqual(row['source'],before['source'])
            self.assertNotEqual(audit.lineage.expected_reset(row,primary['protocol'])['seed'],audit.lineage.expected_reset(before,primary['protocol'])['seed'])

    def test_validation_missing_duplicate_or_context_parent_rejects(self):
        p=ROOT/'artifacts/framework/multitask_r10/TRAIN_full_collection_plan_v11.json'
        for mutation in ('split','missing','duplicate','Context'):
            primary=audit.lineage.read(p);selected=next(c for c in primary['cases'] if c['task']==audit.TASK and c['teacher_used'])
            if mutation=='split':primary['split']='validation'
            elif mutation=='missing':primary['cases'].remove(selected)
            elif mutation=='duplicate':next(c for c in primary['cases'] if c['task']==audit.TASK and c['teacher_used'] and c is not selected).update(parent_source_group=selected['parent_source_group'])
            else:selected['context_primary_view']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):audit.expected_cases(primary)


if __name__=='__main__':unittest.main()
