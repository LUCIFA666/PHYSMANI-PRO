import json
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import ObjectState,Pose
from our_mani.benchmark_perception import RedBoxTracker
from our_mani.local_mpc_control_20261103.public_attachment import PublicAttachmentEstimator,PublicTargetFrameCache
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors,PublicMotionBridge
from tests.test_local_mpc_public_motion_bridge_20261103 import observation,StaticPublicGoal,MockFrozenForecaster


def public_sample(t,*,opening=.6,tcp=None,target=None,rotation=None,confidence=1.):
    # Synthetic public-sensor contract trace; no Native contact/held label.
    tcp=np.array([.2+.08*t,-.15,.8]) if tcp is None else np.asarray(tcp)
    R=np.eye(3) if rotation is None else rotation
    position=tcp+R@np.array([0,0,-.04]) if target is None else np.asarray(target)
    q=Rotation.from_matrix(R).as_quat()
    robot=PublicRobotSensors(float(t),tcp,q,np.zeros(7),np.zeros(7),[opening,opening])
    points=position+np.array([[0,0,0],[.01,0,0],[0,.01,0]])
    obj=ObjectState('target',float(t),Pose(position),points,np.zeros(3),np.zeros(3),confidence)
    return obj,robot


def close_trace(estimator,*,stationary=False,empty=False,rotate=False,independent=False):
    rows=[]
    for i in range(10):
        t=i*.05;opening=max(0. if empty else .6,1.-(.25 if empty else .1)*i)
        tcp=np.array([.2,-.15,.8]) if stationary else None
        R=Rotation.from_euler('z',.4*t).as_matrix() if rotate else None
        target=[.2,-.15,.76] if independent else None
        obj,robot=public_sample(t,opening=opening,tcp=tcp,target=target,rotation=R)
        # t=0 is the real pre-close open-intent frame. Closure starts at t=.05.
        rows.append(estimator.observe(obj,robot,close_requested=i>0))
    return rows


class PublicAttachmentTests(unittest.TestCase):
    def test_observed_relative_motion_and_stalled_nonempty_fingers_needed(self):
        estimator=PublicAttachmentEstimator();rows=close_trace(estimator)
        self.assertTrue(all(row['mode']!='ATTACHED_PUBLIC' for row in rows[:9]))
        result=rows[-1];self.assertEqual(result['mode'],'ATTACHED_PUBLIC')
        self.assertEqual(result['evidence'].event,'attachment_confirmed')
        self.assertFalse(result['evidence'].orientation_observable)
        self.assertTrue(all(result['diagnostic']['gates'].values()))
        self.assertFalse(result['actions_emitted']);self.assertFalse(result['force_or_frictional_grasp_proof'])
        for row in rows:json.dumps({k:v for k,v in row.items() if k!='evidence'},allow_nan=False)

    def test_stationary_near_hand_and_closed_intent_do_not_confirm(self):
        result=close_trace(PublicAttachmentEstimator(),stationary=True)[-1]
        self.assertEqual(result['mode'],'UNRESOLVED')
        self.assertFalse(result['diagnostic']['gates']['actual_TCP_motion_witness'])
        self.assertFalse(result['diagnostic']['gates']['visual_object_motion_witness'])

    def test_fully_closed_empty_endpoint_is_not_obstruction_evidence(self):
        result=close_trace(PublicAttachmentEstimator(),empty=True)[-1]
        self.assertEqual(result['mode'],'UNRESOLVED')
        self.assertFalse(result['diagnostic']['gates']['nonempty_grasp_opening'])
        self.assertTrue(result['diagnostic']['gates']['stalled_opening'])

    def test_actual_fingers_still_closing_are_not_stalled_obstruction(self):
        estimator=PublicAttachmentEstimator()
        for i in range(10):
            obj,robot=public_sample(i*.05,opening=.95-.015*i)
            result=estimator.observe(obj,robot,close_requested=i>0)
        self.assertEqual(result['mode'],'UNRESOLVED')
        self.assertFalse(result['diagnostic']['gates']['stalled_opening'])

    def test_two_finger_asymmetry_is_not_hidden_by_scalar_aperture_mean(self):
        estimator=PublicAttachmentEstimator()
        for i in range(10):
            obj,robot=public_sample(i*.05,opening=1.)
            first=max(.6,1.-.1*i);second=max(.8,1.-.05*i)
            robot=PublicRobotSensors(robot.timestamp,robot.tcp_position,robot.tcp_quaternion,robot.q,robot.dq,[first,second])
            result=estimator.observe(obj,robot,close_requested=i>0)
        self.assertEqual(result['mode'],'UNRESOLVED')
        self.assertFalse(result['diagnostic']['gates']['fingers_consistent'])

    def test_stationary_world_object_is_not_following_moving_TCP(self):
        result=close_trace(PublicAttachmentEstimator(),independent=True)[-1]
        self.assertEqual(result['mode'],'UNRESOLVED')
        self.assertFalse(result['diagnostic']['gates']['visual_object_motion_witness'])

    def test_unobserved_rotation_cannot_certify_rotating_attachment(self):
        result=close_trace(PublicAttachmentEstimator(),rotate=True)[-1]
        self.assertEqual(result['mode'],'UNRESOLVED')
        self.assertFalse(result['diagnostic']['gates']['orientation_consistency'])
        self.assertIsNone(result['diagnostic']['relative_rotation_deviation_rad'])

    def test_after_confirmation_stationary_visible_hold_may_maintain_but_not_drift(self):
        estimator=PublicAttachmentEstimator();close_trace(estimator)
        anchor=np.array([.2+.08*.45,-.15,.8])
        for i in range(6):
            obj,robot=public_sample(.5+i*.05,tcp=anchor)
            result=estimator.observe(obj,robot,close_requested=True)
        self.assertEqual(result['mode'],'ATTACHED_PUBLIC')
        obj,robot=public_sample(.8,tcp=anchor,target=anchor+np.array([.03,0,-.04]))
        result=estimator.observe(obj,robot,close_requested=True)
        self.assertEqual(result['mode'],'UNRESOLVED')

    def test_visual_loss_or_100ms_gap_never_carries_attachment_forward(self):
        estimator=PublicAttachmentEstimator();close_trace(estimator)
        _,robot=public_sample(.5)
        missing=estimator.observe(None,robot,close_requested=True)
        self.assertEqual(missing['mode'],'STALE');self.assertNotEqual(missing['evidence'].event,'attachment_confirmed')
        obj,robot=public_sample(.55);result=estimator.observe(obj,robot,close_requested=True)
        self.assertEqual(result['mode'],'UNRESOLVED');self.assertEqual(result['consecutive_valid_frames'],1)
        obj,robot=public_sample(.65);result=estimator.observe(obj,robot,close_requested=True)
        self.assertEqual(result['mode'],'STALE');self.assertEqual(result['consecutive_valid_frames'],0)

    def test_open_request_alone_not_release_but_visible_separation_reestablishes_FREE(self):
        estimator=PublicAttachmentEstimator();close_trace(estimator)
        for i in range(4):
            obj,robot=public_sample(.5+i*.05,opening=1.)
            result=estimator.observe(obj,robot,close_requested=False)
        self.assertEqual(result['mode'],'UNRESOLVED')
        for i in range(4):
            t=.7+i*.05;tcp=np.array([.4+.08*t,-.15,.8]);obj,robot=public_sample(t,opening=1.,tcp=tcp,target=[.2,-.15,.76])
            result=estimator.observe(obj,robot,close_requested=False)
        self.assertEqual(result['mode'],'FREE');self.assertEqual(result['evidence'].event,'release_confirmed')
        self.assertFalse(result['attempt_active'])

    def test_opening_with_unchanged_attached_offset_does_not_fake_release(self):
        estimator=PublicAttachmentEstimator()
        for i in range(10):
            t=i*.05;tcp=np.array([.2+.08*t,-.15,.8])
            obj,robot=public_sample(t,opening=max(.6,1.-.1*i),tcp=tcp,target=tcp+[0,0,-.08])
            result=estimator.observe(obj,robot,close_requested=i>0)
        self.assertEqual(result['mode'],'ATTACHED_PUBLIC')
        for i in range(4):
            t=.5+i*.05;tcp=np.array([.2+.08*t,-.15,.8]);obj,robot=public_sample(t,opening=1.,tcp=tcp,target=tcp+[0,0,-.08])
            result=estimator.observe(obj,robot,close_requested=False)
        self.assertEqual(result['mode'],'UNRESOLVED')
        self.assertFalse(result['diagnostic']['release_relative_change_observed'])

    def test_Native_raw_dicts_and_robot_held_channel_rejected(self):
        estimator=PublicAttachmentEstimator();obj,robot=public_sample(0,opening=1.)
        with self.assertRaises(ValueError):estimator.observe({'object_position':[0,0,0],'held':True},robot,close_requested=True)
        estimator.reset()
        raw={key:getattr(robot,key) for key in ('timestamp','tcp_position','tcp_quaternion','q','dq','finger_open_amounts')};raw['native_success']=True
        with self.assertRaises(ValueError):estimator.observe(obj,raw,close_requested=True)
        estimator.reset();obj,_=public_sample(.05)
        with self.assertRaises(ValueError):estimator.observe(obj,robot,close_requested=True)

    def test_frame_cache_preserves_current_release_frame_and_refuses_duplicate_or_GT_diagnostics(self):
        cache=PublicTargetFrameCache();obj,_=public_sample(0.)
        cache.stage_public_target(obj,timestamp=0.);cache.reset()
        same,_=cache.observe(observation(0.));self.assertIs(same,obj)
        with self.assertRaises(ValueError):cache.observe(observation(0.))
        cache.reset()
        with self.assertRaises(ValueError):cache.stage_public_target(obj,timestamp=0.)
        with self.assertRaises(ValueError):cache.stage_public_target(None,timestamp=.05,diagnostic={'Native_held':True})
        cache.stage_public_target(None,timestamp=.05)
        with self.assertRaises(ValueError):cache.observe(observation(.05))

    def test_mode_reset_keeps_measured_pose_and_timestamp_but_restarts_public_velocity_history(self):
        tracker=RedBoxTracker();tracker.history.append((0.,np.zeros(3)));tracker.last_velocity[:]=.1
        cache=PublicTargetFrameCache(public_tracker=tracker);obj,_=public_sample(.05)
        obj=ObjectState(obj.object_id,obj.timestamp,obj.pose,obj.points_world,np.ones(3),np.zeros(3),obj.confidence)
        cache.stage_public_target(obj,timestamp=.05);cache.reset()
        current,diagnostic=cache.observe(observation(.05))
        np.testing.assert_array_equal(current.pose.position,obj.pose.position)
        self.assertEqual(current.timestamp,obj.timestamp);np.testing.assert_array_equal(current.linear_velocity,np.zeros(3))
        self.assertEqual(len(tracker.history),0);np.testing.assert_array_equal(tracker.last_velocity,np.zeros(3))
        self.assertTrue(diagnostic['public_velocity_history_reset'])

    def test_one_real_frame_once_estimator_then_bridge_same_timestamp(self):
        estimator=PublicAttachmentEstimator();cache=PublicTargetFrameCache();forecaster=MockFrozenForecaster()
        bridge=PublicMotionBridge(tracker=cache,goal_marker=StaticPublicGoal(),forecaster=forecaster)
        for i in range(10):
            t=i*.05;obj,robot=public_sample(t,opening=max(.6,1.-.1*i))
            cache.stage_public_target(obj,timestamp=t)
            attachment=estimator.observe(obj,robot,close_requested=i>0)
            result=bridge.observe(observation(t),robot,close_requested=i>0,evidence=attachment['evidence'])
        self.assertTrue(result['controlled_mode_available']);self.assertFalse(result['free_forecast_available'])
        self.assertEqual(forecaster.predict_calls,0)
        for i in range(4):
            t=.5+i*.05;obj,robot=public_sample(t,opening=1.,tcp=[.45,-.15,.8],target=[.2,-.15,.76])
            cache.stage_public_target(obj,timestamp=t)
            attachment=estimator.observe(obj,robot,close_requested=False)
            result=bridge.observe(observation(t),robot,evidence=attachment['evidence'])
        self.assertEqual(result['mode'],'FREE');self.assertEqual(result['history_frames'],1)
        for i in range(4):
            t=.7+i*.05;obj,robot=public_sample(t,opening=1.,tcp=[.45,-.15,.8],target=[.2,-.15,.76])
            cache.stage_public_target(obj,timestamp=t)
            attachment=estimator.observe(obj,robot,close_requested=False)
            result=bridge.observe(observation(t),robot,evidence=attachment['evidence'])
        self.assertTrue(result['free_forecast_available']);self.assertEqual(result['history_frames'],5)
        self.assertEqual(forecaster.predict_calls,1)

    def test_first_tick_closure_uses_real_preclose_fingers_then_common_motion_and_stall(self):
        estimator=PublicAttachmentEstimator();obj,robot=public_sample(0.,opening=1.)
        estimator.observe(obj,robot,close_requested=False)
        results=[]
        for i in range(1,7):
            obj,robot=public_sample(i*.05,opening=.8)
            results.append(estimator.observe(obj,robot,close_requested=True))
        self.assertTrue(all(row['mode']=='UNRESOLVED' for row in results[:-1]))
        result=results[-1]
        self.assertEqual(result['mode'],'ATTACHED_PUBLIC')
        self.assertTrue(all(result['diagnostic']['gates'].values()))
        self.assertEqual(result['closure_baseline']['timestamp'],0.)
        self.assertEqual(result['closure_baseline']['finger_open_amounts'],[1.,1.])
        np.testing.assert_allclose(results[0]['diagnostic']['observed_closure'],[.2,.2],atol=1e-15)
        self.assertFalse(result['native_held_or_success_used'])

    def test_no_preclose_frame_never_fabricates_open_fingers_or_closure(self):
        estimator=PublicAttachmentEstimator()
        for i in range(6):
            obj,robot=public_sample(i*.05,opening=.8)
            result=estimator.observe(obj,robot,close_requested=True)
        self.assertEqual(result['mode'],'UNRESOLVED')
        self.assertTrue(result['diagnostic']['gates']['real_consecutive_feedback_window'])
        self.assertFalse(result['diagnostic']['gates']['closure_actually_observed'])
        self.assertIsNone(result['diagnostic']['observed_closure'])
        self.assertFalse(result['closure_baseline']['available'])
        self.assertIsNone(result['closure_baseline']['finger_open_amounts'])

    def test_gap_at_first_close_does_not_reuse_stale_preclose_frame(self):
        for gap in (.1,.06):
            with self.subTest(gap=gap):
                estimator=PublicAttachmentEstimator();obj,robot=public_sample(0.,opening=1.)
                estimator.observe(obj,robot,close_requested=False)
                obj,robot=public_sample(gap,opening=.8)
                first=estimator.observe(obj,robot,close_requested=True)
                self.assertEqual(first['mode'],'STALE')
                for i in range(1,7):
                    obj,robot=public_sample(gap+i*.05,opening=.8)
                    result=estimator.observe(obj,robot,close_requested=True)
                self.assertEqual(result['mode'],'UNRESOLVED')
                self.assertFalse(result['closure_baseline']['available'])
                self.assertIsNone(result['diagnostic']['observed_closure'])

    def test_new_fresh_open_intent_frame_after_gap_can_seed_later_real_close(self):
        estimator=PublicAttachmentEstimator();obj,robot=public_sample(0.,opening=1.)
        estimator.observe(obj,robot,close_requested=False)
        obj,robot=public_sample(.1,opening=.95)
        self.assertEqual(estimator.observe(obj,robot,close_requested=False)['mode'],'STALE')
        for i in range(1,7):
            obj,robot=public_sample(.1+i*.05,opening=.8)
            result=estimator.observe(obj,robot,close_requested=True)
        self.assertEqual(result['mode'],'ATTACHED_PUBLIC')
        self.assertEqual(result['closure_baseline']['timestamp'],.1)
        self.assertEqual(result['closure_baseline']['finger_open_amounts'],[.95,.95])

    def test_gap_during_attempt_invalidates_old_closure_origin(self):
        estimator=PublicAttachmentEstimator();obj,robot=public_sample(0.,opening=1.)
        estimator.observe(obj,robot,close_requested=False)
        obj,robot=public_sample(.05,opening=.8)
        self.assertTrue(estimator.observe(obj,robot,close_requested=True)['closure_baseline']['available'])
        obj,robot=public_sample(.15,opening=.8)
        result=estimator.observe(obj,robot,close_requested=True)
        self.assertEqual(result['mode'],'STALE')
        self.assertFalse(result['closure_baseline']['available'])
        self.assertEqual(result['closure_baseline']['reason'],'missing_or_non_50ms_feedback_invalidates_closure_baseline')


if __name__=='__main__':unittest.main()
