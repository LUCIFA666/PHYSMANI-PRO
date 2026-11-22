"""Public-only bounded proposals; synthetic robot/target, no Native or NN."""
import unittest
import numpy as np
from our_mani.contracts import ObjectState,Pose
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from our_mani.cube_stale_reobserve_20261104_v1 import StaleReobserveSupervisorV1,execute_reobserve_cycle
from our_mani.local_mpc_20261102.executor import ShortHorizonExecutor,JointLimits,ExecutionConfig
from our_mani.local_mpc_control_20261103.completed_gripper import CompletedGripperBackend


def frame(t,*,visible=True,ready=True,q=0.,position=0.):
    robot=PublicRobotSensors(t,[0,0,1],[0,0,0,1],np.full(7,q),np.zeros(7),[1,1])
    obj=ObjectState('target',t,Pose([position,0,.8]),np.zeros((3,3)),np.zeros(3),np.zeros(3))if visible else None
    result=dict(timestamp=t,Native_inputs_used=False,future_observations_used=False,observations_fabricated=False,
        object_observation=obj,observed_target_available=visible,free_forecast_available=ready and visible)
    return result,robot


class ReobserveContracts(unittest.TestCase):
    def _mechanical(self,collision=False):
        class B:
            def __init__(self):self.q=np.full(7,.1);self.dq=np.zeros(7);self.target=self.q.copy();self.t=.05;self.ticks=0;self.probes=[];self.grips=[];self.requested_opening=1.
            def clock(self):return self.t
            def get_joint_positions(self):return self.q.copy()
            def get_joint_velocities(self):return self.dq.copy()
            def joint_limits(self):return dict(lower=np.full(7,-2.),upper=np.full(7,2.),velocity=np.full(7,3.))
            def probe_joint_collision(self,q):self.probes.append(np.asarray(q).copy());return collision
            def set_joint_target_positions(self,q):self.target=np.asarray(q)
            def actuate_gripper(self,x):self.grips.append(x);return True
            def remaining_ticks(self):return 10-self.ticks
            def terminal(self):return False
            def tick(self,phase):self.q=self.target.copy();self.t+=.05;self.ticks+=1
        raw=B();backend=CompletedGripperBackend(raw)
        executor=ShortHorizonExecutor(backend,JointLimits([-2]*7,[2]*7,[3]*7),ExecutionConfig())
        s=StaleReobserveSupervisorV1();s.observe(*frame(0.),close_requested=False)
        s.observe(*frame(.05,visible=False,q=.1),close_requested=False)
        return raw,backend,executor,s,s.propose()

    def test_missing_never_becomes_measured_centre_or_close(self):
        s=StaleReobserveSupervisorV1()
        s.observe(*frame(0.,position=0.),close_requested=False)
        s.observe(*frame(.05,q=.01,position=.01),close_requested=False)
        info=s.observe(*frame(.1,visible=False,q=.1),close_requested=False)
        p=s.propose()
        self.assertTrue(info['forbid_new_close']);self.assertTrue(p['forbid_new_close'])
        self.assertEqual(p['kind'],'REOBSERVE_JOINT');self.assertIsNone(p['opening_command'])
        np.testing.assert_allclose(p['joint_goal'],[.07]*7)
        self.assertFalse(p['object_roi_hint']['measurement'])
        np.testing.assert_allclose(p['object_roi_hint']['position_world'],[.02,0,.8])
        self.assertFalse(p['object_roi_hint']['replaces_public_sphere_center'])

    def test_closed_priorattempt_or_confirmed_never_retargets_or_resets_owner(self):
        for kwargs in (dict(close_requested=True),dict(close_requested=False,attempt_id='attempt1'),dict(close_requested=False,previously_confirmed=True)):
            s=StaleReobserveSupervisorV1();s.observe(*frame(0.),close_requested=False)
            s.observe(*frame(.05,visible=False,q=.1),**kwargs);p=s.propose()
            self.assertEqual(p['kind'],'KEEP_ORIGINAL_HOLD');self.assertIsNone(p['opening_command'])
            self.assertIsNone(p['object_roi_hint']);self.assertFalse(p['original_proof_owner_reset'])

    def test_three_attempt_bound_and_prediction_expiry_do_not_auto_retry(self):
        s=StaleReobserveSupervisorV1();s.observe(*frame(0.),close_requested=False)
        for i in range(1,6):
            s.observe(*frame(i*.05,visible=False,q=.1),close_requested=False)
            p=s.propose()
            if i<=3:self.assertEqual(p['kind'],'REOBSERVE_JOINT')
            else:self.assertEqual(p['kind'],'KEEP_ORIGINAL_HOLD')
            if i>=4:self.assertIsNone(p['object_roi_hint'])
            s.commit(p['proposal_id'],executed_ticks=0)
        self.assertEqual(s._cycles,3)

    def test_real_frame_and_forecast_readiness_required_no_stale_commit(self):
        s=StaleReobserveSupervisorV1();s.observe(*frame(0.,ready=False),close_requested=False)
        p=s.propose();self.assertTrue(p['forbid_new_close']);s.commit(p['proposal_id'],executed_ticks=0)
        with self.assertRaises(ValueError):s.commit(p['proposal_id'],executed_ticks=0)
        with self.assertRaises(ValueError):s.observe(*frame(0.),close_requested=False)
        s.observe(*frame(.05,visible=False,q=.1),close_requested=False)
        p=s.propose();s.commit(p['proposal_id'],executed_ticks=0)
        s.observe(*frame(.1,ready=False,q=.1),close_requested=False)
        p=s.propose();self.assertTrue(p['forbid_new_close']);s.commit(p['proposal_id'],executed_ticks=0)
        s.observe(*frame(.15,ready=True,q=.1),close_requested=False)
        self.assertFalse(s.propose()['forbid_new_close'])

    def test_optional_hook_runs_original_8_checks_then_original_two_ticks_KEEP(self):
        raw,b,e,s,p=self._mechanical()
        r=execute_reobserve_cycle(s,b,e,p)
        self.assertTrue(r['executed']);self.assertEqual(len(raw.probes),8)
        self.assertEqual(raw.ticks,2);self.assertEqual(raw.grips,[1.])
        self.assertEqual(r['executor']['target_joints'],tuple(p['joint_goal']))
        self.assertEqual(r['commit']['actual_executed_ticks'],2)
        self.assertFalse(r['opening_command_changed'])

    def test_optional_hook_collision_rejects_with_zero_steps_and_no_grip(self):
        raw,b,e,s,p=self._mechanical(collision=True)
        r=execute_reobserve_cycle(s,b,e,p)
        self.assertFalse(r['executed']);self.assertEqual(len(raw.probes),1)
        self.assertEqual(raw.ticks,0);self.assertEqual(raw.grips,[])
        self.assertEqual(r['commit']['actual_executed_ticks'],0)
        self.assertEqual(s._cycles,1)


if __name__=='__main__':unittest.main()
