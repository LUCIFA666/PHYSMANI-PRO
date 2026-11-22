"""Independent synthetic integration through the original qualifier/executor."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np

from our_mani.contracts import ObjectState,Pose
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from our_mani.local_mpc_control_20261103.active_visual_attachment import ActiveVisualAttachment
from our_mani.local_mpc_control_20261103.completed_gripper import CompletedGripperBackend
from our_mani.local_mpc_20261102.executor import ShortHorizonExecutor,JointLimits,ExecutionConfig
from our_mani.cube_bounded_reobserve_live_20261104_v1.supervision import (
    ReobservePublicLifecycle,clear_free_state_on_stale,guard_new_close,execute_selected_or_reobserve)


def frame(time,*,visible=True,ready=False,q=0.,opening=1.):
    robot=PublicRobotSensors(time,[0.,0.,1.],[0.,0.,0.,1.],np.full(7,q),np.zeros(7),[opening,opening])
    obj=ObjectState('target',time,Pose([.2,0.,.8]),np.zeros((3,3)),np.zeros(3),np.zeros(3))if visible else None
    result=dict(timestamp=time,Native_inputs_used=False,future_observations_used=False,observations_fabricated=False,
        object_observation=obj,observed_target_available=visible,free_forecast_available=ready and visible,
        mode='FREE'if visible and ready else'STALE')
    return result,robot


class IndependentBoundedIntegration(unittest.TestCase):
    def mechanics(self,*,collision=False,closed=False,owner=None):
        life=ReobservePublicLifecycle();owner=owner or ActiveVisualAttachment()
        life.observe(*frame(0.,ready=True,opening=0. if closed else 1.),active=owner,
            effective_attachment=None,close_requested=closed)
        life.observe(*frame(.05,visible=False,q=.1,opening=0. if closed else 1.),active=owner,
            effective_attachment=None,close_requested=closed)
        class Mechanical:
            def __init__(self):
                self.q=np.full(7,.1);self.dq=np.zeros(7);self.target=self.q.copy();self.t=.05
                self.ticks=0;self.probes=[];self.grips=[];self.requested_opening=0. if closed else 1.
                self.reobserve_lifecycle=life;self.tick_phases=[]
            def clock(self):return self.t
            def get_joint_positions(self):return self.q.copy()
            def get_joint_velocities(self):return self.dq.copy()
            def joint_limits(self):return dict(lower=np.full(7,-2.),upper=np.full(7,2.),velocity=np.full(7,3.))
            def probe_joint_collision(self,q):self.probes.append(np.asarray(q).copy());return collision
            def set_joint_target_positions(self,q):self.target=np.asarray(q)
            def actuate_gripper(self,opening):self.grips.append(opening);self.requested_opening=float(opening);return True
            def remaining_ticks(self):return 20-self.ticks
            def terminal(self):return False
            def tick(self,phase):
                self.q=self.target.copy();self.t+=.05;self.ticks+=1;self.tick_phases.append(phase)
                # Same observation path is advanced once by each physical tick.
                life.observe(*frame(self.t,visible=False,q=float(self.q[0]),opening=self.requested_opening),
                    active=owner,effective_attachment=None,close_requested=self.requested_opening<.5)
        raw=Mechanical();backend=CompletedGripperBackend(raw)
        executor=ShortHorizonExecutor(backend,JointLimits([-2.]*7,[2.]*7,[3.]*7),ExecutionConfig(),
            initial_gripper_opening=raw.requested_opening)
        return raw,backend,executor,life,owner

    def test_success_is_original_eight_queries_and_two_ticks_not_retreat_then_hold(self):
        raw,backend,executor,life,owner=self.mechanics()
        original=dict(mode='UNAVAILABLE',action=None,state=None)
        receipt,action,recovery=execute_selected_or_reobserve(backend,executor,original,raw.q.copy(),1.,dict(source='hold'))
        self.assertEqual(raw.ticks,2);self.assertEqual(len(raw.probes),8)
        self.assertTrue(recovery['outcome']['executed'])
        self.assertEqual(receipt['executed_ticks'],2)
        self.assertEqual(executor.last_receipt.command_id,1)
        self.assertEqual(raw.grips,[1.])
        np.testing.assert_allclose(receipt['target_joints'],np.full(7,.07))
        self.assertEqual(action['source'],'bounded_public_reobserve_joint_not_MPC_selected_action')
        self.assertIsNone(action['opening_argument'])
        self.assertTrue(all(p=='STALE_safe_public_reobservation'for p in raw.tick_phases))
        self.assertEqual(life.supervisor._cycles,1)
        self.assertEqual(life.fresh_frames,0)
        self.assertEqual(owner.attempt_id,0);self.assertIsNone(owner.confirmation)

    def test_collision_rejection_executes_one_original_hold_and_no_retreat_tick(self):
        raw,backend,executor,life,_=self.mechanics(collision=True)
        receipt,action,recovery=execute_selected_or_reobserve(backend,executor,
            dict(mode='UNAVAILABLE',action=None),raw.q.copy(),1.,dict(source='actual_q_hold'))
        self.assertFalse(recovery['outcome']['executed']);self.assertEqual(len(raw.probes),1)
        self.assertEqual(raw.ticks,2);self.assertEqual(receipt.executed_ticks,2)
        self.assertEqual(executor.last_receipt.command_id,1)
        self.assertEqual(action['source'],'actual_q_hold')
        np.testing.assert_allclose(receipt.target_joints,np.full(7,.1))
        self.assertTrue(all(p=='P4_candidate_transformer_v72_UNAVAILABLE'for p in raw.tick_phases))
        self.assertEqual(life.supervisor._cycles,1)

    def test_closed_and_any_previous_attempt_never_retreat_or_reopen(self):
        previous=ActiveVisualAttachment();previous.attempt_id=1;previous.attempt_started=.01
        previous.attempt_active=True;previous.failed=True
        previous.spec={'relative_r0':[0.,0.,.1],'deadline':.8,'anchor':'immutable-test'}
        for closed,owner in ((True,None),(False,previous)):
            raw,backend,executor,life,actual_owner=self.mechanics(closed=closed,owner=owner)
            before=deepcopy(vars(actual_owner))
            receipt,_,recovery=execute_selected_or_reobserve(backend,executor,
                dict(mode='UNAVAILABLE',action=None),raw.q.copy(),raw.requested_opening,dict(source='hold'))
            self.assertIsNone(recovery);self.assertEqual(raw.probes,[]);self.assertEqual(raw.ticks,2)
            self.assertEqual(receipt.gripper_target,0. if closed else 1.)
            self.assertEqual(vars(actual_owner),before)
            self.assertFalse(life.latest['preclose_reobserve_eligible'])

    def test_five_actual_frames_gate_close_and_real_gap_restarts_count(self):
        life=ReobservePublicLifecycle();owner=ActiveVisualAttachment()
        life.observe(*frame(0.,ready=True),active=owner,effective_attachment=None,close_requested=False)
        life.observe(*frame(.05,visible=False),active=owner,effective_attachment=None,close_requested=False)
        coordinator=SimpleNamespace(warm='old-warm',phase={'dwell':.1})
        close=dict(mode='FREE_GRU',action=dict(opening=0.,joint_goal=np.zeros(7)))
        for i in range(2,7):
            result,robot=frame(i*.05,ready=True)
            object_before=result['object_observation']
            saved=deepcopy(result)
            life.observe(result,robot,active=owner,effective_attachment=None,close_requested=False)
            guarded=guard_new_close(close,coordinator,life,dict(gripper_target=1.))
            if i<6:self.assertTrue(guarded['new_close_suppressed']);self.assertIsNone(guarded['action'])
            else:self.assertIs(guarded,close)
            self.assertIs(result['object_observation'],object_before)
            self.assertTrue(np.array_equal(result['object_observation'].pose.position,saved['object_observation'].pose.position))
        life.observe(*frame(.45,ready=True),active=owner,effective_attachment=None,close_requested=False)
        self.assertEqual(life.fresh_frames,1)
        self.assertTrue(guard_new_close(close,coordinator,life,dict(gripper_target=1.))['new_close_suppressed'])
        with self.assertRaises(ValueError):life.observe(*frame(.45,ready=True),active=owner,effective_attachment=None,close_requested=False)

    def test_stale_clears_only_free_memory_once_and_preserves_attempt_proof(self):
        life=ReobservePublicLifecycle();owner=ActiveVisualAttachment()
        coordinator=SimpleNamespace(warm={'sequence':'old'},phase={'dwell':.1},active=owner)
        life.observe(*frame(0.,ready=True),active=owner,effective_attachment=None,close_requested=False)
        life.observe(*frame(.05,visible=False),active=owner,effective_attachment=None,close_requested=False)
        before=deepcopy(vars(owner))
        report=clear_free_state_on_stale(coordinator,life)
        self.assertTrue(report['free_warm_and_dwell_cleared']);self.assertIsNone(coordinator.warm);self.assertIsNone(coordinator.phase)
        self.assertEqual(vars(owner),before)
        self.assertFalse(clear_free_state_on_stale(coordinator,life)['free_warm_and_dwell_cleared'])
        # Existing owner identity/proof is never a resettable free phase.
        owner.attempt_id=1;owner.attempt_started=.05;owner.failed=True;owner.spec={'r0':[0,0,.2],'deadline':1.}
        life.observe(*frame(.1,ready=False),active=owner,effective_attachment=None,close_requested=True)
        life.stale_entry_serial+=1;coordinator.warm='retained';coordinator.phase={'held':'private-test-memory'}
        before=deepcopy(vars(owner));report=clear_free_state_on_stale(coordinator,life)
        self.assertFalse(report['free_warm_and_dwell_cleared']);self.assertEqual(coordinator.warm,'retained')
        self.assertEqual(vars(owner),before)

    def test_new_open_to_close_also_requires_frames_after_prior_attempt_but_existing_close_is_preserved(self):
        life=ReobservePublicLifecycle();life.no_attempt=False;life.ever_confirmed=True
        life.latest={'fresh_five_and_original_forecast_ready':False}
        coordinator=SimpleNamespace(warm='old',phase={'dwell':.1})
        decision=dict(mode='ACTIVE_VISUAL_VERIFICATION_PROBE',action=dict(opening=0.,joint_goal=np.zeros(7)))
        self.assertIs(guard_new_close(decision,coordinator,life,dict(gripper_target=0.)),decision)
        blocked=guard_new_close(decision,coordinator,life,dict(gripper_target=1.))
        self.assertTrue(blocked['new_close_suppressed'])


if __name__=='__main__':unittest.main()
