"""Synthetic true-clock/private-label boundary tests; no Native/models/data."""
from copy import deepcopy
import json
import unittest
import numpy as np
from our_mani.generic_reference_mpc_20261104_v1.train_labels import TRAINLabelRecorder, TRAINRecordingExecutor, run_hold_teacher
from our_mani.generic_reference_mpc_20261104_v1.native_backend import make_executor
from our_mani.local_mpc_20261102.executor import ShortHorizonExecutor


class Robot:
    dataset_use='train';requested_opening=1.
    def __init__(self):
        self.q=np.zeros(7);self.dq=np.zeros(7);self._ticks=0;self.origin=10.;self.absolute=10.
        self._last_tick_observation=None;self._issued_training_reference=None
    def clock(self):return self.absolute-self.origin
    def joint_limits(self):return dict(lower=np.full(7,-3),upper=np.full(7,3),velocity=np.full(7,2.))
    def get_joint_positions(self):return self.q.copy()
    def get_joint_velocities(self):return self.dq.copy()
    def set_joint_target_positions(self,q):self.target=np.asarray(q)
    def remaining_ticks(self):return 4-self._ticks
    def terminal(self):return self.remaining_ticks()==0
    def actuate_gripper(self,opening):self.requested_opening=opening;return True
    def tick(self,phase):
        before=self.absolute;tick=self._ticks;self.absolute+=.050003;self._ticks+=1;self.q=self.target.copy()
        self._last_tick_observation=dict(tick_before=tick,tick_after=self._ticks,
            physics_time_before=before,physics_time_after=self.absolute)
    def _bind_issued_training_reference(self,applied):
        self._issued_training_reference=deepcopy(dict(applied,physics_tick_before=self._ticks,actual_clock_before=self.clock()))
    def install_train_label_recorder(self,sink):
        self.recorder=TRAINLabelRecorder(self,sink);return self.recorder
    def _training_label_snapshot(self):
        return dict(physics_tick=self._ticks,actual_relative_physics_timestamp=self.clock(),physics_time_origin=self.origin,
            physics_time_absolute=self.absolute,task_time_absolute=self.absolute+2.,
            bodies={'body':dict(pose_world_xyzw=np.array([1,2,3,0,0,0,1]),rotation_world=np.eye(3))},
            q=self.q,dq=self.dq,finger_open_amounts=np.ones(2),
            last_physical_tick=self._last_tick_observation,issued_reference=self._issued_training_reference)


class Labels(unittest.TestCase):
    def test_original_execute_unchanged_and_unique_postcommit_true_clock(self):
        robot=Robot();rows=[];executor,_=make_executor(robot,train_label_sink=rows.append)
        self.assertIs(TRAINRecordingExecutor._execute,ShortHorizonExecutor._execute)
        receipt=executor.execute_joint_target(np.full(7,.1),opening=1.)
        self.assertEqual(receipt.executed_ticks,2);self.assertEqual(len(rows),3)
        self.assertEqual([r['physics_tick'] for r in rows],[0,1,2])
        self.assertNotEqual(rows[1]['actual_relative_physics_timestamp'],.05)
        self.assertAlmostEqual(rows[1]['observed_dt'],.050003)
        self.assertEqual(rows[1]['actuator_h']['reference_clock'],rows[1]['actual_relative_physics_timestamp'])
        self.assertEqual(rows[1]['issued_reference']['q_ref'],rows[1]['actuator_h']['reference_positions'])
        self.assertFalse(rows[1]['policy_use_allowed']);self.assertTrue(rows[1]['physical_timestamp_observed'])
        json.dumps(rows,allow_nan=False)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            robot.recorder.record_after_commit(executor.state())
        with self.assertRaisesRegex(RuntimeError,'permanently faulted'):
            robot.recorder.record_after_commit(executor.state())

    def test_held_initial_q_teacher_needs_no_NN_or_object_input(self):
        robot=Robot();rows=[];executor,_=make_executor(robot,train_label_sink=rows.append);logs=[]
        report=run_hold_teacher(robot,executor,log_private=logs.append)
        self.assertEqual(report['model_forward_calls'],0);self.assertEqual(len(rows),5)
        np.testing.assert_array_equal(robot.q,np.zeros(7));self.assertEqual(robot.requested_opening,1.)

    def test_nonTRAIN_and_skipped_clock_rows_refused(self):
        robot=Robot();robot.dataset_use='test'
        with self.assertRaises(ValueError):TRAINLabelRecorder(robot,lambda row:None)
        robot=Robot();rows=[];executor,_=make_executor(robot,train_label_sink=rows.append)
        robot._ticks=2;robot.absolute+=.1
        with self.assertRaisesRegex(ValueError,'skipped'):
            robot.recorder.record_after_commit(executor.state())


if __name__=='__main__':unittest.main()
