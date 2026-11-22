"""Pure CPU tests of true-clock recording and public/private frame semantics."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from our_mani.contracts import ObjectState,Pose
from our_mani.cube_motion_capture_recovery_20261104_v1.backend import CubeMotionObservationBackendV1
from our_mani.cube_motion_capture_recovery_20261104_v1.collector import current_public_target
from our_mani.cube_v72_live_20261104_v1.backend import CubeTaskNativeBackendV1
from our_mani.generic_reference_mpc_20261104_v1.train_labels import TRAINLabelRecorder,TRAINRecordingExecutor
from our_mani.local_mpc_20261102.executor import JointLimits,ExecutionConfig
from our_mani.local_mpc_control_20261103.completed_gripper import CompletedGripperBackend


class MotionCaptureTests(unittest.TestCase):
    def test_actual_clock_not_counted_frame_time(self):
        b=object.__new__(CubeMotionObservationBackendV1)
        b._physics_origin=100.;b._physics_clock=lambda:100.25000003
        b.raw=SimpleNamespace(ledger=SimpleNamespace(time=123.,ticks=5))
        self.assertAlmostEqual(b.clock(),.25000003,places=12)
        self.assertNotEqual(b.clock(),5*.05)

    def test_current_measurement_before_forecast_and_no_stale_pose(self):
        obj=ObjectState('target',.2,Pose([.1,.2,.3],np.eye(3)),np.ones((3,3)),np.zeros(3),np.zeros(3),.8,True)
        result=dict(available=False,observed_target_available=True,object_observation=obj,
            depth_refiner_trace=dict(measurement_available=True,accepted_source='FAR_R100_DEVELOPMENT_CONSISTENCY'))
        target=current_public_target(result)
        self.assertTrue(target['available']);self.assertEqual(target['position_world'],[.1,.2,.3])
        self.assertFalse(target['future_prediction_used_as_current_measurement'])
        result['observed_target_available']=False
        target=current_public_target(result)
        self.assertFalse(target['available']);self.assertIsNone(target['position_world'])
        self.assertIsNone(target['confidence'])
        self.assertNotIn('held',target)

    def test_same_original_RGBD_capture_once_with_true_timestamp_and_actual_clips(self):
        b=object.__new__(CubeMotionObservationBackendV1)
        b._physics_origin=100.;b._task_origin=.5;b._physics_clock=lambda:100.25
        b._capture_cache=b._capture_cache_clock=None;b._check_live_clock=lambda:None
        b._clock_tuple=lambda:(100.25,.75,5,5)
        b.get_joint_positions=lambda:np.zeros(7);b.get_joint_velocities=lambda:np.ones(7)*.02
        tip=SimpleNamespace(get_pose=lambda:[0,0,1,0,0,0,1]);b.arm=SimpleNamespace(get_tip=lambda:tip)
        b.gripper=SimpleNamespace(get_joint_positions=lambda:[.04,.04],get_joint_velocities=lambda:[0,0],get_open_amount=lambda:[1,1])
        cam=SimpleNamespace(get_matrix=lambda:np.eye(4),get_intrinsic_matrix=lambda:np.eye(3),
            get_near_clipping_plane=lambda:.01,get_far_clipping_plane=lambda:4.,get_render_mode=lambda:SimpleNamespace(name='OPENGL'))
        names=('front','8','16','36');b.raw=SimpleNamespace(cameras={n:cam for n in names})
        original=dict(timestamp=999.,**{'observation/images':{n:np.full((256,256,3),127,np.uint8)for n in names},
            'observation/depths':{n:np.ones((256,256),np.float32)for n in names}})
        with patch.object(CubeTaskNativeBackendV1,'observe_rgbd',return_value=original)as capture:
            packet=b.read_snapshot();obs=b.observe_rgbd();again=b.read_snapshot()
            self.assertEqual(capture.call_count,1)
        self.assertIs(packet,again);self.assertEqual(packet['timestamp'],.25);self.assertEqual(obs['timestamp'],.25)
        self.assertEqual(packet['physics_timestamp'],100.25)
        self.assertTrue(np.all(packet['rgb']==127));np.testing.assert_array_equal(packet['camera_clips'],np.tile([.01,4.],(4,1)))
        self.assertNotIn('object_position',packet)

    def test_original_executor_records_public_before_private_after_each_real_commit(self):
        class B:
            dataset_use='train'
            def __init__(self):self._ticks=0;self.p=10.;self.task=20.;self.q=np.zeros(7);self.dq=np.zeros(7);self.goal=self.q.copy();self.issued=None;self.transition=None;self.order=[]
            def clock(self):return self.p-10.
            def get_joint_positions(self):return self.q.copy()
            def get_joint_velocities(self):return self.dq.copy()
            def set_joint_target_positions(self,q):self.goal=np.asarray(q)
            def terminal(self):return False
            def remaining_ticks(self):return 10-self._ticks
            def actuate_gripper(self,x):return True
            def _bind_issued_training_reference(self,r):self.issued=deepcopy(dict(r,physics_tick_before=self._ticks,actual_clock_before=self.clock()))
            def tick(self,phase):
                old=self.p;tick=self._ticks;self.p+=.050000012;self.task+=.05;self._ticks+=1
                self.q=self.goal.copy();self.transition=dict(tick_before=tick,tick_after=self._ticks,physics_time_before=old,physics_time_after=self.p)
                self.order.append(('public',self._ticks))
            def _training_label_snapshot(self):
                self.order.append(('private',self._ticks))
                return dict(physics_tick=self._ticks,actual_relative_physics_timestamp=self.clock(),
                    physics_time_origin=10.,physics_time_absolute=self.p,task_time_absolute=self.task,
                    last_physical_tick=self.transition,issued_reference=self.issued)
        b=B();rows=[];rec=TRAINLabelRecorder(b,rows.append)
        e=TRAINRecordingExecutor(CompletedGripperBackend(b),JointLimits([-2]*7,[2]*7,[3]*7),ExecutionConfig(),recorder=rec)
        b.order.append(('public',0));rec.record_initial(e.state())
        e.execute_joint_target(np.zeros(7),opening=1.)
        self.assertEqual(b.order,[('public',0),('private',0),('public',1),('private',1),('public',2),('private',2)])
        self.assertEqual(len(rows),3)
        self.assertNotEqual(rows[-1]['actual_relative_physics_timestamp'],.1)
        self.assertAlmostEqual(rows[-1]['observed_dt'],.050000012,places=12)
        np.testing.assert_array_equal(rows[-1]['issued_reference']['q_ref'],e.state()['reference_positions'])


if __name__=='__main__':unittest.main()
