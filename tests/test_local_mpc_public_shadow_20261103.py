"""Public observer causality and clock checks, CPU fixtures only."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.local_mpc_control_20261103.public_shadow import PublicShadowBackend


class Backend:
    def __init__(self):self.t=0.;self.ticks=0;self.captures=0
    def clock(self):return self.t
    def read_state(self):
        return dict(time=self.t,tcp_position=[.3,0,1.],tcp_quaternion=[0,0,0,1.],q=[0]*7,dq=[0]*7,
            finger_open_amounts=[1.,1.],requested_opening=1.,object_position=[.2,.1,.775],
            object_quaternion=[0,0,0,1.],object_linear_velocity=[0,.02,0],object_angular_velocity=[0,0,0],
            held=False,native_success=False,secret_native_handle=314159)
    def observe_rgbd(self):
        self.captures+=1
        return {'timestamp':self.t,'observation/images':{c:np.zeros((2,2,3),np.uint8) for c in ('front','8','16','36')},
            'observation/depths':{c:np.ones((2,2),np.float32) for c in ('front','8','16','36')},
            'observation/calibrations':{c:CameraCalibration(np.eye(3),Pose([0,0,0])) for c in ('front','8','16','36')},
            'secret_native_mask':True}
    def tick(self,phase):self.ticks+=1;self.t=self.ticks*.05;return self.read_state()


class BridgeSpy:
    def __init__(self):self.times=[]
    def observe(self,observation,robot,*,close_requested):
        assert set(observation)=={'timestamp','observation/images','observation/depths','observation/calibrations'}
        assert set(vars(robot))=={'timestamp','tcp_position','tcp_quaternion','q','dq','finger_open_amounts'}
        self.times.append(observation['timestamp'])
        return dict(timestamp=robot.timestamp,available=False,fixture_only=True)


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.raw=Backend();self.bridge=BridgeSpy()
        self.observer=PublicShadowBackend(self.raw,Path(self.tmp.name)/'public',bridge=self.bridge,save_frames=False)
    def test_exactly_one_frame_after_each_actual_tick_no_interpolated_frames(self):
        self.observer.sample();self.observer.tick('control');self.observer.tick('control')
        self.assertEqual(self.raw.ticks,2);self.assertEqual(self.raw.captures,3)
        self.assertEqual(self.bridge.times,[0.,.05,.1]);self.assertEqual(self.raw.clock(),.1)
    def test_repeated_collection_reuses_frame_without_second_forecast(self):
        self.observer.sample();a=self.observer.observe_rgbd();b=self.observer.observe_rgbd()
        self.assertIs(a,b);self.assertEqual(self.raw.captures,1);self.assertEqual(len(self.bridge.times),1)
        with self.assertRaisesRegex(ValueError,'unique'):self.observer.sample()
    def test_privileged_labels_separate_and_never_enter_public_result(self):
        self.observer.sample();base=Path(self.tmp.name)/'public'
        result=json.loads((base/'public_outputs.jsonl').read_text())
        self.assertNotIn('held',result);self.assertNotIn('native_success',result)
        self.assertNotIn('secret_native',json.dumps(result))
        labels=json.loads((base/'evaluation_labels.jsonl').read_text())
        self.assertIn('held',labels);self.assertIn('object_position',labels)
        self.assertFalse(self.observer.receipt()['controller_uses_public_forecast'])
    def test_incorrect_camera_clock_fails_instead_of_resampling(self):
        original=self.raw.observe_rgbd
        def bad():r=original();r['timestamp']=.05;return r
        self.raw.observe_rgbd=bad
        with self.assertRaisesRegex(ValueError,'timestamp'):self.observer.sample()
        self.assertEqual(self.bridge.times,[]);self.assertEqual(self.raw.ticks,0)


if __name__=='__main__':unittest.main()
