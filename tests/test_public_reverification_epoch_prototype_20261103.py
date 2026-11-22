from copy import deepcopy
import unittest
import numpy as np
from our_mani.contracts import ObjectState,Pose
from our_mani.local_mpc_control_20261103.active_visual_attachment import ActiveVisualAttachment,ActiveVisualConfig
from our_mani.local_mpc_control_20261103.postconfirmed_maintenance_candidate_20261103 import PostConfirmedPositionMaintenance
from our_mani.local_mpc_control_20261103.public_reverification_epoch_prototype_20261103 import PublicReverificationEpochPrototype
from tests.test_local_mpc_active_visual_attachment_phase_scope_20261103 import sensor_frame,actual_h


def result(obj,robot,attachment=None):
    a=attachment or dict(timestamp=robot.timestamp,protocol='active_vertical_visual_comotion_v4',attempt_id=1,attempt_active=True,mode='UNRESOLVED')
    return dict(timestamp=robot.timestamp,object_observation=obj,attachment_estimate=a,
        original_probe_estimator_diagnostic=a,mode=a['mode'],available=a['mode']=='ATTACHED_PUBLIC',
        controlled_mode_available=a['mode']=='ATTACHED_PUBLIC',Native_inputs_used=False,
        future_observations_used=False,observations_fabricated=False)


def fixture():
    baseline,_=sensor_frame(0)
    active=ActiveVisualAttachment(ActiveVisualConfig(enforce_probe_anchor_orientation_after_confirmation=False))
    obj,robot=sensor_frame(1);active.observe(obj,robot,close_requested=True);active.begin_probe(obj,robot,actual_h(robot))
    for i in range(2,7):
        obj,robot=sensor_frame(i,height=.012*(i-1)/5)
        a=active.observe(obj,robot,close_requested=True)
    assert a['mode']=='ATTACHED_PUBLIC'
    old=PostConfirmedPositionMaintenance();old.start(result(obj,robot,a),robot,close_requested=True)
    _,robot=sensor_frame(7,height=.012);a=active.observe(None,robot,close_requested=True)
    old.observe(result(None,robot,a),robot,close_requested=True);assert old.failed
    return PublicReverificationEpochPrototype(old,baseline),old,baseline


def frame(i,height=.012,shift=(0.,0.,0.)):
    obj,robot=sensor_frame(i,height=height,relative_shift=shift)
    return result(obj,robot),robot


def begin(owner,shift=(0.,0.,0.)):
    r,robot=frame(8,shift=shift);owner.observe(r,robot,close_requested=True)
    return owner.begin_witness(r,robot,actual_h(robot)),r,robot


class ReverificationPrototypeTests(unittest.TestCase):
    def test_fresh_witness_requires_explicit_commit_preserves_old_failure_and_baselines(self):
        owner,old,baseline=fixture();original=old.r0.copy();spec,_,_=begin(owner)
        self.assertAlmostEqual(spec['target_position'][2]-spec['anchor_position'][2],.012)
        for i in range(9,14):
            r,robot=frame(i,.012+.012*(i-8)/5);current=owner.observe(r,robot,close_requested=True)
        self.assertTrue(current['current_fresh_witness_passed']);self.assertEqual(owner.verification_epoch,0)
        committed=owner.commit_verified_epoch(r,robot);self.assertEqual(committed['verification_epoch'],1)
        self.assertEqual(committed['evidence'].timestamp,robot.timestamp);self.assertTrue(old.failed)
        np.testing.assert_array_equal(old.r0,original)
        np.testing.assert_array_equal(owner.original_preclose_baseline.pose.position,baseline.pose.position)
        with self.assertRaises(ValueError):owner.commit_verified_epoch(r,robot)

    def test_static_reappearance_and_done_do_not_claim_attachment(self):
        owner,old,_=fixture();spec,r,robot=begin(owner)
        for i in range(9,14):r,robot=frame(i);current=owner.observe(r,robot,close_requested=True)
        self.assertFalse(current['current_fresh_witness_passed']);self.assertTrue(old.failed)
        with self.assertRaises(ValueError):owner.commit_verified_epoch(r,robot)
        self.assertEqual(owner.active_test_proposal()['executed'],False)
        self.assertEqual(owner.active_test_proposal()['probe_spec']['deadline'],spec['deadline'])

    def test_no_proxy_tcp_only_motion_fails_full_fixed_relative_window(self):
        owner,old,_=fixture();_,r0,_=begin(owner);anchor=r0['object_observation'].pose.position.copy()
        for i in range(9,14):
            r,robot=frame(i,.012+.012*(i-8)/5);obj=r['object_observation']
            r['object_observation']=ObjectState('target',robot.timestamp,Pose(anchor),obj.points_world,np.zeros(3),np.zeros(3),1.)
            current=owner.observe(r,robot,close_requested=True)
        self.assertTrue(current['witness_failed']);self.assertFalse(current['current_fresh_witness_passed'])
        self.assertTrue(old.failed);self.assertEqual(owner.verification_epoch,0)

    def test_gap_open_stale_attempt_model_only_and_GT_never_reset(self):
        for fault in ('gap','open','wrongattempt','missing','model_only'):
            with self.subTest(fault=fault):
                owner,old,_=fixture();begin(owner);r,robot=frame(10 if fault=='gap' else 9,.015)
                if fault=='wrongattempt':r['attachment_estimate']['attempt_id']=2
                if fault in ('missing','model_only'):
                    r['object_observation']=None
                    if fault=='model_only':r['controlled']={'estimated_world_pose':Pose([0,0,999])}
                current=owner.observe(r,robot,close_requested=fault!='open')
                self.assertTrue(current['witness_failed']);self.assertTrue(old.failed)
        owner,old,_=fixture();r,robot=frame(8);r['GT_model_hint']=False
        with self.assertRaises(ValueError):owner.observe(r,robot,close_requested=True)

    def test_no_rolling_reference_or_safety_priority_bypass(self):
        owner,old,_=fixture();begin(owner,shift=(.009,0,0));r,robot=frame(9,.015,shift=(.011,0,0))
        current=owner.observe(r,robot,close_requested=True)
        self.assertEqual(current['reason'],'fixed_original_r0_over10mm_no_anchor_laundering')
        owner,old,_=fixture();begin(owner)
        for i in range(9,13):r,robot=frame(i,.012+.009*(i-8)/4);owner.observe(r,robot,close_requested=True)
        r,robot=frame(13,.028);current=owner.observe(r,robot,close_requested=True)
        self.assertTrue(current['witness_failed']);self.assertFalse(current['current_fresh_witness_passed'])

    def test_source_and_old_failure_immutable_no_stale_h_or_sameframe_mutation(self):
        owner,old,_=fixture();r,robot=frame(8);owner.observe(r,robot,close_requested=True)
        h=actual_h(robot);h['reference_clock']=float('nan')
        with self.assertRaises(ValueError):owner.begin_witness(r,robot,h)
        body=r['object_observation'];r['object_observation']=ObjectState('target',body.timestamp,
            Pose(body.pose.position+[.001,0,0]),body.points_world,body.linear_velocity,body.angular_velocity,body.confidence)
        with self.assertRaises(ValueError):owner.begin_witness(r,robot,actual_h(robot))
        owner,old,_=fixture();old.failed=False;r,robot=frame(8)
        with self.assertRaises(ValueError):owner.observe(r,robot,close_requested=True)
        owner,old,_=fixture();begin(owner);owner.spec['deadline']+=.1
        with self.assertRaises(ValueError):owner.active_test_proposal()

    def test_previous_good_witness_cannot_authorize_later_insufficient_current_motion(self):
        owner,old,_=fixture();begin(owner)
        for i in range(9,14):
            r,robot=frame(i,.012+.012*(i-8)/5);current=owner.observe(r,robot,close_requested=True)
        self.assertTrue(current['current_fresh_witness_passed'])
        r,robot=frame(14,.018);current=owner.observe(r,robot,close_requested=True)
        self.assertFalse(current['current_fresh_witness_passed'])
        self.assertEqual(current['current_whole_trace_metrics']['real_probe_frames'],7)
        with self.assertRaises(ValueError):owner.commit_verified_epoch(r,robot)
        self.assertEqual(owner.verification_epoch,0);self.assertTrue(old.failed)


if __name__=='__main__':unittest.main()
