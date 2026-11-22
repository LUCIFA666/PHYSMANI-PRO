"""Full new snapshot merged by the real frozen TRAIN recorder; CPU only."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

from our_mani.cube_motion_capture_recovery_20261104_v1.backend import CubeMotionObservationBackendV1
from our_mani.cube_motion_capture_recovery_20261104_v2.backend import CubeMotionObservationBackendV2
from our_mani.generic_reference_mpc_20261104_v1.train_labels import TRAINLabelRecorder,TRAINRecordingExecutor
from our_mani.local_mpc_20261102.executor import JointLimits,ExecutionConfig
from our_mani.local_mpc_control_20261103.completed_gripper import CompletedGripperBackend


class CompleteSnapshotFixture(CubeMotionObservationBackendV2):
    def __init__(self):
        self.dataset_use='train';self.task_name='pick_moving_target_on_the_table'
        self.raw=SimpleNamespace(ledger=SimpleNamespace(ticks=0,time=0.))
        self._physics_origin=10.;self._task_origin=20.;self.phys=10.;self.task_time=20.
        self._physics_clock=lambda:self.phys
        self._issued_training_reference=None;self._last_tick_observation=None
        self.q=np.zeros(7);self.dq=np.zeros(7);self.target=np.zeros(7)
    def _check_live_clock(self):pass
    def _clock_tuple(self):return self.phys,self.task_time,self._ticks,self._ticks
    def read_state(self):
        return dict(object_position=np.array([.2,.1,.8]),object_quaternion=np.array([0.,0.,0.,1.]),
            object_causal_fd_velocity=np.array([.2,0.,0.]),object_causal_twist_valid=self._ticks>0,
            object_native_linear_velocity=np.zeros(3),tcp_position=np.array([.2,.1,1.]),
            tcp_quaternion=np.array([0.,1.,0.,0.]),q=self.q.copy(),dq=self.dq.copy(),
            finger_positions=np.array([.04,.04]),finger_velocities=np.zeros(2),
            finger_open_amounts=np.ones(2),servo_target_positions=self.target.copy(),
            requested_opening=1.,held=False,native_success=False,
            native_result=dict(success=False,terminate=False,terminal_failure=False))
    def get_joint_positions(self):return self.q.copy()
    def get_joint_velocities(self):return self.dq.copy()
    def set_joint_target_positions(self,q):self.target=np.asarray(q)
    def actuate_gripper(self,x):return True
    def terminal(self):return False
    def remaining_ticks(self):return 160-self._ticks
    def tick(self,phase):
        old=self.phys;tick=self._ticks;self.phys+=.050000012;self.task_time+=.05
        self.raw.ledger.ticks+=1;self.raw.ledger.time+=.05;self.q=self.target.copy()
        self._last_tick_observation=dict(tick_before=tick,tick_after=self._ticks,
            physics_time_before=old,physics_time_after=self.phys,
            task_time_before=self.task_time-.05,task_time_after=self.task_time,phase=phase)


class FullSnapshotContracts(unittest.TestCase):
    def test_original_failure_reproduced_without_Native(self):
        b=CompleteSnapshotFixture();snapshot=CubeMotionObservationBackendV1._training_label_snapshot(b)
        self.assertIn('contact_force_observed',snapshot)
        b._training_label_snapshot=lambda:snapshot
        recorder=TRAINLabelRecorder(b,lambda row:None)
        with self.assertRaisesRegex(TypeError,'contact_force_observed'):
            recorder.record_initial(dict(gripper_target=1.))

    def test_actual_full_v2_snapshot_merges_once_with_real_recorder(self):
        b=CompleteSnapshotFixture();before=b.clock();snapshot=b._training_label_snapshot()
        self.assertNotIn('contact_force_observed',snapshot)
        self.assertIn('target_position_world',snapshot);self.assertIn('target_rotation_world',snapshot)
        rows=[];recorder=TRAINLabelRecorder(b,rows.append)
        recorder.record_initial(dict(gripper_target=1.))
        self.assertEqual(len(rows),1);self.assertFalse(rows[0]['contact_force_observed'])
        self.assertTrue(rows[0]['label_only']);self.assertFalse(rows[0]['policy_use_allowed'])
        self.assertEqual(b.clock(),before)

    def test_real_executor_two_commits_complete_snapshot_after_each_tick(self):
        b=CompleteSnapshotFixture();rows=[];recorder=TRAINLabelRecorder(b,rows.append)
        e=TRAINRecordingExecutor(CompletedGripperBackend(b),JointLimits([-2]*7,[2]*7,[3]*7),
                                ExecutionConfig(),recorder=recorder)
        recorder.record_initial(e.state());e.execute_joint_target(np.zeros(7),opening=1.)
        self.assertEqual([r['physics_tick']for r in rows],[0,1,2])
        self.assertEqual(len(rows),3)
        for row in rows:
            self.assertFalse(row['contact_force_observed']);self.assertFalse(row['held'])
            self.assertEqual(row['tcp_pose_world_xyzw'],[.2,.1,1.,0.,1.,0.,0.])
        self.assertAlmostEqual(rows[-1]['observed_dt'],.050000012,places=12)
        np.testing.assert_array_equal(rows[-1]['issued_reference']['q_ref'],e.state()['reference_positions'])
        self.assertNotEqual(rows[-1]['actual_relative_physics_timestamp'],.1)

    def test_full_backend_AST_only_duplicate_field_removed(self):
        class Normalize(ast.NodeTransformer):
            def visit_ClassDef(self,node):node.name='Backend';return self.generic_visit(node)
            def visit_keyword(self,node):
                if node.arg=='contact_force_observed':return None
                return self.generic_visit(node)
        before=Normalize().visit(ast.parse(inspect.getsource(CubeMotionObservationBackendV1)))
        after=Normalize().visit(ast.parse(inspect.getsource(CubeMotionObservationBackendV2)))
        self.assertEqual(ast.dump(before,include_attributes=False),ast.dump(after,include_attributes=False))


if __name__=='__main__':unittest.main()
