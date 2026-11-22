"""Small synthetic contracts, no trained weights, model NN, or Native launch."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from our_mani.contracts import Forecast,ObjectState,Pose
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from our_mani.cube_corrected_live_20261104_v1.candidate_runtime import pack_public_row
from our_mani.cube_native_contact_feedback_live_20261104_v1.candidate_runtime import NativeContactPublicCacheV1
from our_mani.cube_native_contact_feedback_live_20261104_v1.backend import NativeContactCubeBackendV1
from our_mani.cube_native_contact_feedback_live_20261104_v1.coordinator import NativeContactCoordinatorV1
from our_mani.cube_native_contact_feedback_live_20261104_v1.planner import NativePendingForecastPlannerV1
from our_mani.local_mpc_control_20261103.forecast_control import BoundForecastExecutionModel
from tests.test_local_mpc_forecast_control_20261103 import bound_fixture
from tests.test_lazy_free_planner_v71_independent_20261103 import QueryOnlyBackend
SHA='1'*64


def frame(i):
    now=i*.05;p=np.array([.3,now*.1,.8]);obj=ObjectState('target',now,Pose(p),np.array([p,p+[.01,0,0],p+[0,.01,0]]),np.array([0,.1,0]),np.zeros(3))
    robot=PublicRobotSensors(now,[.3,0,1.],[0,0,0,1],np.zeros(7),np.zeros(7),[1,1])
    f=Forecast(('target',),now+np.arange(17)*.05,np.tile(p,(17,1,1)),np.tile(np.eye(3),(17,1,1,1)),np.zeros((17,1,3)),np.zeros((17,1,3)),
        metadata=dict(checkpoint_sha256=SHA,observed_until=now,history_frames={'target':i+1},contexts={'target':np.full(16,1/8).tolist()}))if i>=4 else None
    result=dict(timestamp=now,mode='FREE',available=i>=4,free_forecast_available=i>=4,controlled_mode_available=False,
        Native_inputs_used=False,future_observations_used=False,observations_fabricated=False,forecaster_is_injected_test_double=False,
        frozen_checkpoint_sha256=SHA,object_observation=obj,forecast=f)
    return pack_public_row(result,robot,i>=5,i,f'rgbd/{i:04d}.npz',expected_forecast_sha256=SHA)


class NativeContactContracts(unittest.TestCase):
    def test_real_close_intent_retained_five_to_six_public_history_not_cleared(self):
        cache=NativeContactPublicCacheV1(expected_forecast_sha256=SHA)
        for i in range(6):cache.observe(frame(i))
        self.assertEqual(len(cache._rows),6);self.assertTrue(cache.latest['close_requested'])
        features,provenance=cache.features(current_time=.25)
        self.assertEqual(features['future_tokens'].shape,(17,16));self.assertEqual(features['robot'].shape,(25,))
        self.assertEqual(provenance['real_history_timestamps'],[i*.05 for i in range(6)])
        self.assertTrue(provenance['actual_close_intent_retained'])
        self.assertFalse(provenance['Native_pose_or_contact_in_numeric_NN_inputs'])

    def test_membership_reader_uses_no_native_object_pose_and_task_success_latches(self):
        raw=NativeContactCubeBackendV1.__new__(NativeContactCubeBackendV1)
        target=object();raw.task=SimpleNamespace(target_block=target)
        raw.gripper=SimpleNamespace(get_grasped_objects=lambda:[target]);raw.clock=lambda:.05
        raw.native_task_success_latched=False;raw.native_success_first_time=None
        raw._last_native_result=dict(success=False,terminate=False)
        self.assertTrue(raw.read_contact_feedback()['held'])
        self.assertFalse(raw._latch_original_task_success())
        raw._last_native_result['success']=True;raw._latch_original_task_success()
        raw._last_native_result['success']=False;raw._latch_original_task_success()
        self.assertTrue(raw.native_task_success_latched);self.assertEqual(raw.native_success_first_time,.05)
        self.assertFalse(raw.native_task_judge_receipt()['visual_probe_hold_or_goal_is_success_gate'])

    def test_first_actual_success_tick_truncates_original_two_tick_executor(self):
        from our_mani.local_mpc_20261102.executor import ShortHorizonExecutor,JointLimits,ExecutionConfig
        from our_mani.local_mpc_control_20261103.completed_gripper import CompletedGripperBackend
        raw=NativeContactCubeBackendV1.__new__(NativeContactCubeBackendV1)
        raw._closed=False;raw._fault=None;raw.budget_ticks=160
        raw.raw=SimpleNamespace(ledger=SimpleNamespace(ticks=0));raw.now=0.;raw.requested_opening=1.
        raw.native_task_success_latched=False;raw.native_success_first_time=None
        raw._last_native_result=dict(success=False,terminate=False,terminal_failure=False)
        raw.q=np.zeros(7);raw.clock=lambda:raw.now
        raw.get_joint_positions=lambda:raw.q.copy();raw.get_joint_velocities=lambda:np.zeros(7)
        raw.set_joint_target_positions=lambda q:setattr(raw,'q',np.asarray(q).copy())
        raw.actuate_gripper=lambda opening:True
        def actual_tick(phase):
            raw.now+=.05;raw.raw.ledger.ticks+=1
            raw._last_native_result['success']=True;raw._latch_original_task_success()
        raw.tick=actual_tick
        backend=CompletedGripperBackend(raw)
        executor=ShortHorizonExecutor(backend,JointLimits([-2]*7,[2]*7,[3]*7),ExecutionConfig())
        receipt=executor.execute_joint_target(np.zeros(7),opening=1.,phase='synthetic_success')
        self.assertEqual(receipt.executed_ticks,1);self.assertEqual(raw.raw.ledger.ticks,1)
        self.assertEqual(raw.native_success_first_time,.05);self.assertTrue(backend.terminal())

    def test_native_held_overrides_missing_visual_and_carries_closed(self):
        robot=PublicRobotSensors(.2,[.3,0,1],[0,0,0,1],np.zeros(7),np.zeros(7),[.1,.1])
        pose=Pose([.3,0,.98]);backend=SimpleNamespace(latest_contact=dict(timestamp=.2,held=True),
            latest={'mode':'UNRESOLVED','available':False},latest_carry=dict(estimated_pose=pose,linear_velocity=np.zeros(3),angular_velocity=np.zeros(3)),
            goal_pose=Pose([.4,0,1.]),robot_history=SimpleNamespace(robot_twist=(np.zeros(3),np.zeros(3),True)))
        c=NativeContactCoordinatorV1.__new__(NativeContactCoordinatorV1);c.backend=backend;c.core=SimpleNamespace(config=None,execution_model=None,planner=SimpleNamespace(constraint_cost=None),terminal_config=None);c.helper=None
        c.mode='FREE_GRU';c.mode_events=[];c._last_held=False;c.phase=c.warm=None;c.candidates=[{}]*8;c.carry_stage=None
        class Planner:
            def __init__(self,*a,**kw):self.fresh_nominal_qualification={}
            def plan(self,state,candidates,**kw):
                assert state['held']is True and state['Native_held_used']is True and state['Native_object_state_used']is False
                return dict(action=dict(joint_goal=np.zeros(7),opening=0.),selected_candidate_id='already_selected',diagnostics={})
        with patch('our_mani.cube_native_contact_feedback_live_20261104_v1.coordinator.NativeContactHoldPlannerV1',Planner):
            answer=c.decide(robot,dict(gripper_target=0.,gripper_done=True))
        self.assertEqual(answer['mode'],'NATIVE_LIFT');self.assertEqual(answer['action']['opening'],0.)
        self.assertFalse(answer['visual_attachment_gate_used'])

    def test_actual_completed_close_nonmembership_can_reopen_without_visual_proof(self):
        c=NativeContactCoordinatorV1.__new__(NativeContactCoordinatorV1)
        c.backend=SimpleNamespace(latest_contact=dict(timestamp=.2,held=False),latest={'mode':'UNRESOLVED'},goal_pose=Pose([0,0,1]))
        c.mode='NATIVE_PENDING_CLOSE_FOLLOW';c.mode_events=[];c._last_held=False;c.phase={'phase':'close'};c.warm={};c.carry_stage=None
        robot=PublicRobotSensors(.2,[0,0,1],[0,0,0,1],np.zeros(7),np.zeros(7),[0,0])
        answer=c.decide(robot,dict(gripper_target=0.,gripper_done=True))
        self.assertEqual(answer['action']['opening'],1.);self.assertTrue(answer['direct_membership_miss_open'])
        self.assertFalse(answer['visual_proof_failure_used'])

    def test_actual_native_pending_full_numeric256_plan_keeps_closed_not_old_visual_reopen(self):
        base,state,candidates,forecast,_=bound_fixture()
        state.update(grasp_motion_mode='FREE_GRU',Native_contact_feedback_used=True,Native_held_used=True,Native_object_state_used=False,Native_success_used=False)
        state['requested_opening']=0.;state['opening']=.5
        state['actuator_state']['gripper_target']=0.;state['actuator_state']['gripper_done']=False
        state['phase_state']=dict(phase='close',dwell=0.,lost_hold=0.,updated_at=-.1,close_entered_at=-.1,reopening=False)
        result=dict(available=True,free_forecast_available=True,mode='FREE',timestamp=0.,forecast=forecast,
            Native_inputs_used=False,future_observations_used=False,observations_fabricated=False,rotation_observable=False)
        model=BoundForecastExecutionModel.from_public_bridge(base.execution_model,result);backend=QueryOnlyBackend()
        planner=NativePendingForecastPlannerV1(execution_model=model,goal_helper=base.goal_helper,terminal_config=base.terminal_config,backend=backend)
        answer=planner.plan(state,candidates)
        self.assertIsNotNone(answer['action']);self.assertEqual(answer['action']['opening'],0.)
        self.assertEqual(answer['budget']['sequences'],256);self.assertEqual(len(backend.queries),8)
        self.assertTrue(state['Native_held_used']);self.assertTrue(answer['diagnostics']['Native_contact_pending_protocol']['actual_unfinished_close_kept_closed'])

if __name__=='__main__':unittest.main()
