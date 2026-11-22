"""Independent synthetic same-instance clocks/partials/public boundary tests."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np

from our_mani.generic_reference_mpc_20261104_v1.public_observation import pack_public_observation
from our_mani.official_dense_train16_20261104_v1.recording import DenseTrainingLedger,UnobservedPhysicalAdvance
from our_mani.official_dense_train16_20261104_v1.native_hooks import ScopedNativeStepObserver
from our_mani.official_dense_train16_20261104_v1.collector import one_attempt


class Fixture:
    def __init__(self):
        self.physics=100.;self.task_clock=7.;self.origin=100.;self.rows=[];self.qtarget=np.full(7,.1)
        parent=self
        class Physics:
            calls=0
            def step(self):self.calls+=1;parent.physics+=.05
        class Task:
            calls=0
            def step(self):self.calls+=1;parent.task_clock+=.05
        self.pyrep=Physics();self.task=Task()
        self.ledger=DenseTrainingLedger(self.clocks,self.public,self.private,
            lambda index,public,label:self.rows.append((index,public,deepcopy(label))),maximum_physics_steps=4)
    def clocks(self):return dict(physics=self.physics,task=self.task_clock)
    def public(self):
        return pack_public_observation(timestamp=self.physics-self.origin,
            rgb=np.zeros((4,256,256,3),np.uint8),depth_m=np.ones((4,256,256),np.float32),
            K=np.tile(np.diag([-200.,-200.,1.]),(4,1,1)),T=np.tile(np.eye(4),(4,1,1)),
            camera_clips=np.tile([.01,4.],(4,1)),q=np.zeros(7),tcp_pose=[0,0,1,0,0,0,1])
    def private(self):return dict(actual_q_target_positions=self.qtarget.copy(),Native_held_label_only=True,
        bodies={'ball':{'pose':[0,0,1,0,0,0,1]},'hoop':{'pose':[1,0,1,0,0,0,1]}})
    def observer(self):return ScopedNativeStepObserver(self.pyrep,self.task,self.ledger)


class Recording(unittest.TestCase):
    def test_scene_and_taskless_each_unique_clock_and_before_command_identity(self):
        f=Fixture();original_physics=f.pyrep.step;original_task=f.task.step
        with f.observer():
            f.ledger.initial();f.ledger.set_command({'intent':'A'})
            f.pyrep.step();f.task.step() # current scene path
            f.pyrep.step() # official gripper path without task driver
            f.ledger.set_command({'intent':'B'});f.qtarget.fill(.9)
            f.pyrep.step() # flush preceding taskless row before next physics
            f.task.step()
        self.assertEqual([row[2]['physical_step']for row in f.rows],[0,1,2,3])
        self.assertEqual((f.pyrep.calls,f.task.calls),(3,2))
        taskless=f.rows[2][2]
        self.assertFalse(taskless['transition']['task_driver_observed'])
        self.assertEqual(taskless['issued_expert_command_label_only'],{'intent':'A'})
        np.testing.assert_array_equal(taskless['transition']['before']['private_before']['actual_q_target_positions'],np.full(7,.1))
        self.assertEqual(f.rows[3][2]['issued_expert_command_label_only'],{'intent':'B'})
        self.assertAlmostEqual(f.rows[-1][2]['actual_relative_physics_time'],.15)
        self.assertAlmostEqual(f.rows[-1][2]['actual_relative_task_time'],.10)
        self.assertEqual(f.pyrep.step,original_physics);self.assertEqual(f.task.step,original_task)
        self.assertNotIn('step',vars(f.pyrep));self.assertNotIn('step',vars(f.task))
        for _,public,label in f.rows:
            self.assertNotIn('Native_held_label_only',public)
            self.assertTrue(label['snapshot']['Native_held_label_only'])
            self.assertIsNone(label['joint_velocity_reference']);self.assertIsNone(label['joint_acceleration_reference'])
            self.assertIsNone(label['actuator_h']);self.assertFalse(label['reference_observed_mask']['h'])

    def test_exception_retains_partial_no_retry_and_restores_bound_steps(self):
        f=Fixture();calls=[];original_physics=f.pyrep.step;original_task=f.task.step
        def command(index):
            calls.append(index);f.ledger.set_command({'intent':'A'})
            f.pyrep.step();f.task.step();f.pyrep.step()
            raise RuntimeError('synthetic official action failure')
        with tempfile.TemporaryDirectory()as d:
            result=one_attempt(command,f.ledger,f.observer(),d)
            self.assertTrue((Path(d)/'attempt.json').is_file())
        self.assertEqual(calls,[0]);self.assertEqual(len(f.rows),3)
        self.assertFalse(result['complete']);self.assertFalse(result['retried']);self.assertEqual(result['attempts'],1)
        self.assertEqual(result['command_calls_started'],1);self.assertEqual(result['command_calls_completed'],0)
        self.assertEqual(result['observed_physics_steps'],2);self.assertEqual(result['observed_task_step_calls'],1)
        self.assertEqual(f.pyrep.step,original_physics);self.assertEqual(f.task.step,original_task)

    def test_unobserved_direct_clock_gap_at_phase_return_faults_without_repair(self):
        f=Fixture();f.ledger.initial()
        with f.observer():
            f.pyrep.step();f.task.step();f.physics+=.05 # synthetic direct simExtStep bypass
            with self.assertRaises(UnobservedPhysicalAdvance):f.ledger.flush('phase_return')
            with self.assertRaisesRegex(RuntimeError,'permanently faulted'):f.ledger.before_physics()
        self.assertEqual(len(f.rows),2);self.assertEqual(f.ledger.physics_steps,1)
        self.assertAlmostEqual(f.physics-f.origin,.10) # unknown step neither erased nor fabricated

    def test_public_private_field_rejects_even_when_false_and_no_attempt_retry(self):
        f=Fixture();original=f.public
        f.ledger.read_public=lambda:dict(original(),Native_held=False)
        calls=[]
        with tempfile.TemporaryDirectory()as d:
            result=one_attempt(lambda i:calls.append(i),f.ledger,f.observer(),d)
        self.assertFalse(result['complete']);self.assertEqual(result['public_private_pairs'],0)
        self.assertEqual(calls,[]);self.assertFalse(result['retried'])
        self.assertNotIn('step',vars(f.pyrep));self.assertNotIn('step',vars(f.task))


if __name__=='__main__':unittest.main()
