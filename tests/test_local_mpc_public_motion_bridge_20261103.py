from collections import deque
import hashlib
import unittest

import numpy as np

from our_mani.contracts import Forecast,ObjectState,Pose
from our_mani.perception import CameraCalibration
from our_mani.local_mpc_control_20261103.public_motion_bridge import (
    PublicMotionBridge,PublicRobotSensors,PublicMotionEvidence,CHECKPOINT,CHECKPOINT_SHA256)


def observation(t):
    cameras=('front','8','16','36')
    return {'timestamp':float(t),'observation/images':{c:np.zeros((2,2,3),dtype=np.uint8) for c in cameras},
            'observation/depths':{c:np.ones((2,2),dtype=np.float32) for c in cameras},
            'observation/calibrations':{c:CameraCalibration(np.eye(3),Pose(np.zeros(3))) for c in cameras}}


def robot(t,opening=1.):
    return PublicRobotSensors(float(t),[.5,0.,.8],[0.,0.,0.,1.],[0.]*7,[0.]*7,[opening,opening])


class SyntheticPublicTracker:
    """Synthetic public-position contract input, never a performance fixture."""
    def __init__(self,low_times=()):
        self.low_times=tuple(low_times);self.observed=[];self.history=deque();self.last_velocity=np.zeros(3)
    def reset(self):self.history.clear();self.last_velocity=np.zeros(3)
    def observe(self,frame):
        t=frame['timestamp'];self.observed.append(t)
        p=np.array([.25+.02*t,-.15,.775]);points=p+np.array([[0,0,0],[.01,0,0],[0,.01,0]])
        confidence=.1 if any(abs(t-low)<1e-8 for low in self.low_times) else 1.
        obj=ObjectState('target',t,Pose(p),points,np.array([.02,0,0]),np.zeros(3),confidence)
        self.history.append(t)
        return obj,{'synthetic_contract_only':True,'world_identity_prior':True}


class StaticPublicGoal:
    def __init__(self):self.calls=0
    def observe(self,frame):self.calls+=1;return Pose([.2,.1,1.]),{'synthetic_static_goal_contract':True}


class MockFrozenForecaster:
    def __init__(self):self.observed=[];self.predict_calls=0;self.checkpoint_sha256='mock_not_real_frozen_weights'
    def reset(self):self.observed=[]
    def observe(self,objects,modes,orientation_observability):
        assert modes=={'target':'external'} and orientation_observability=={'target':False}
        self.observed.append(objects[0].timestamp)
    def predict(self,objects,horizon,dt):
        self.predict_calls+=1;assert horizon==.8 and dt==.05
        obj=objects[0];n=17;positions=np.broadcast_to(obj.pose.position,(n,1,3)).copy()
        positions[:,0,0]+=np.arange(n) # obvious index sentinel for off-by-one tests
        rotations=np.broadcast_to(np.eye(3),(n,1,3,3)).copy()
        velocities=np.broadcast_to(np.array([20.,0.,0.]),(n,1,3)).copy()
        return Forecast(('target',),obj.timestamp+np.arange(n)*dt,positions,rotations,velocities,np.zeros((n,1,3)))


class PublicMotionBridgeTests(unittest.TestCase):
    def create(self,**kwargs):
        tracker=kwargs.pop('tracker',SyntheticPublicTracker());goal=StaticPublicGoal();forecaster=MockFrozenForecaster()
        return PublicMotionBridge(tracker=tracker,goal_marker=goal,forecaster=forecaster,**kwargs),tracker,goal,forecaster
    def warm(self,bridge,start=0.):
        return [bridge.observe(observation(start+i*.05),robot(start+i*.05)) for i in range(5)]

    def test_five_real_frames_and_absolute_H8_sampling_include_current_index0(self):
        bridge,tracker,goal,forecaster=self.create();rows=self.warm(bridge)
        self.assertTrue(all(not row['available'] for row in rows[:4]));result=rows[-1]
        self.assertTrue(result['free_forecast_available']);self.assertEqual(result['history_frames'],5)
        self.assertEqual(len(result['forecast'].times),17)
        np.testing.assert_allclose(result['endpoint_times'],.2+np.arange(1,9)*.1,atol=1e-12)
        np.testing.assert_allclose(result['positions'][:,0],.25+.02*.2+np.arange(2,17,2),atol=1e-12)
        self.assertFalse(result['rotation_observable']);self.assertEqual(goal.calls,1)
        self.assertEqual(tracker.observed,[0.,.05,.1,.15000000000000002,.2])

    def test_no_Native_keys_or_held_flag_can_enter_observation_robot_or_evidence(self):
        bridge,*_=self.create();bad=observation(0);bad['Native_object_pose']=[0,0,0]
        with self.assertRaises(ValueError):bridge.observe(bad,robot(0))
        sensors=dict(timestamp=0.,tcp_position=[0,0,1],tcp_quaternion=[0,0,0,1],q=[0]*7,dq=[0]*7,finger_open_amounts=[1,1],held=True)
        with self.assertRaises(ValueError):bridge.observe(observation(0),sensors)
        with self.assertRaises(ValueError):PublicMotionEvidence('attachment_confirmed',0.,('Native_held',),Pose([0,0,0]))
        with self.assertRaises(ValueError):bridge.observe(observation(0),robot(.05))

    def test_100ms_gap_clears_history_and_never_creates_midpoint_frames(self):
        bridge,tracker,_,forecaster=self.create()
        bridge.observe(observation(0),robot(0));bridge.observe(observation(.05),robot(.05))
        result=bridge.observe(observation(.15),robot(.15))
        self.assertEqual(result['mode'],'STALE');self.assertEqual(result['history_frames'],0)
        self.assertFalse(result['available']);self.assertEqual(tracker.observed,[0.,.05])
        rows=self.warm(bridge,.2)
        self.assertTrue(rows[-1]['available']);self.assertEqual(len(forecaster.observed),5)
        with self.assertRaises(ValueError):bridge.observe(observation(.4),robot(.4))

    def test_non_50ms_or_extra_camera_channels_are_not_admitted(self):
        bridge,*_=self.create();bridge.observe(observation(0),robot(0))
        result=bridge.observe(observation(.03),robot(.03))
        self.assertFalse(result['available']);self.assertIn('50ms',result['reason'])
        bad=observation(.08);bad['observation/images']['Native_object_mask']=np.zeros((2,2))
        with self.assertRaises(ValueError):bridge.observe(bad,robot(.08))

    def test_low_confidence_clears_history_but_static_goal_cache_survives(self):
        bridge,_,goal,_=self.create(tracker=SyntheticPublicTracker(low_times=(.15,)))
        for t in (0.,.05,.1):bridge.observe(observation(t),robot(t))
        bad=bridge.observe(observation(.15),robot(.15))
        self.assertEqual(bad['mode'],'STALE');self.assertEqual(bad['history_frames'],0)
        self.assertTrue(bad['goal_cached']);self.assertEqual(goal.calls,1)
        rows=self.warm(bridge,.2);self.assertTrue(rows[-1]['available']);self.assertEqual(goal.calls,1)

    def test_close_request_is_unresolved_and_public_release_rewarms_not_permanently_blocked(self):
        bridge,_,_,forecaster=self.create();self.warm(bridge)
        result=bridge.observe(observation(.25),robot(.25),close_requested=True)
        self.assertEqual(result['mode'],'UNRESOLVED');self.assertFalse(result['available'])
        self.assertEqual(forecaster.predict_calls,1)
        result=bridge.observe(observation(.3),robot(.3));self.assertEqual(result['mode'],'UNRESOLVED')
        release=PublicMotionEvidence('release_confirmed',.35,('rgbd_object_release','public_gripper_opening'))
        result=bridge.observe(observation(.35),robot(.35),evidence=release)
        self.assertEqual(result['history_frames'],1);self.assertFalse(result['available'])
        for t in (.4,.45,.5,.55):result=bridge.observe(observation(t),robot(t))
        self.assertTrue(result['available']);self.assertEqual(forecaster.predict_calls,2)

    def test_repeated_free_public_evidence_does_not_reset_real_history(self):
        bridge,*_=self.create()
        for i in range(5):
            t=i*.05;evidence=PublicMotionEvidence('free_motion_confirmed',t,('rgbd_free_motion_consistency',))
            result=bridge.observe(observation(t),robot(t),evidence=evidence)
        self.assertTrue(result['free_forecast_available']);self.assertEqual(result['history_frames'],5)

    def test_public_attachment_returns_only_controlled_mode_and_expires_without_fresh_evidence(self):
        bridge,_,_,forecaster=self.create();self.warm(bridge)
        evidence=PublicMotionEvidence('attachment_confirmed',.25,('rgbd_relative_pose_tracking','public_robot_pose'),Pose([0,0,.05]))
        result=bridge.observe(observation(.25),robot(.25,.1),evidence=evidence,close_requested=True)
        self.assertEqual(result['mode'],'ATTACHED_PUBLIC');self.assertTrue(result['controlled_mode_available'])
        self.assertFalse(result['free_forecast_available']);self.assertIsNone(result['forecast'])
        np.testing.assert_allclose(result['controlled']['estimated_world_pose'].position,[.5,0,.85])
        self.assertEqual(forecaster.predict_calls,1)
        self.assertTrue(bridge.observe(observation(.3),robot(.3,.1))['controlled_mode_available'])
        stale=bridge.observe(observation(.35),robot(.35,.1));self.assertFalse(stale['available'])
        self.assertIn('attachment_relative_pose_stale',stale['reason'])

    def test_repeated_stale_checks_do_not_erase_unresolved_grasp_mode(self):
        bridge,*_=self.create();self.warm(bridge)
        bridge.observe(observation(.25),robot(.25),close_requested=True)
        self.assertFalse(bridge.latest(.4)['available']);self.assertFalse(bridge.latest(.45)['available'])
        bridge.observe(observation(.5),robot(.5));result=bridge.observe(observation(.55),robot(.55))
        self.assertEqual(result['mode'],'UNRESOLVED');self.assertFalse(result['available'])
        with self.assertRaises(ValueError):bridge.latest(float('nan'))

    def test_real_frozen_v671_load_with_synthetic_causal_positions_is_contract_only(self):
        before=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest();goal=StaticPublicGoal()
        bridge=PublicMotionBridge(tracker=SyntheticPublicTracker(),goal_marker=goal)
        result=self.warm(bridge)[-1]
        self.assertTrue(result['available']);self.assertFalse(result['forecaster_is_injected_test_double'])
        self.assertEqual(result['positions'].shape,(8,3));self.assertTrue(np.isfinite(result['positions']).all())
        self.assertFalse(result['rotation_observable']);self.assertEqual(result['forecast'].metadata['history_frames']['target'],5)
        self.assertTrue(all(not p.requires_grad for p in bridge.forecaster.model.parameters()))
        self.assertEqual(before,CHECKPOINT_SHA256);self.assertEqual(before,hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest())
        self.assertTrue(result['forecast'].metadata['full_orientation_observable']['target'] is False)


if __name__=='__main__':unittest.main()
