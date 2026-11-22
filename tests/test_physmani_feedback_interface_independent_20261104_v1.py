"""Independent CPU feedback/metric boundaries; no Native/model/data execution."""
from copy import deepcopy
from pathlib import Path
import ast
import inspect
import unittest
from unittest.mock import patch
import numpy as np

from our_mani.physmani_feedback_interface_20261104_v1 import observation as obs
from our_mani.physmani_feedback_interface_20261104_v1 import facade
from our_mani.physmani_feedback_interface_20261104_v1.backend import PhysManiFeedbackBackendV1
from our_mani.generic_reference_mpc_20261104_v1.native_backend import GenericTaskNativeBackend
from our_mani.generic_reference_mpc_20261104_v1.reference_mpc import qualify_first_action
from our_mani.local_mpc_20261102.executor import ShortHorizonExecutor
from our_mani.local_mpc_control_20261103.completed_gripper import CompletedGripperBackend
from our_mani.multitask_execution_pipeline_20261104_v1.registry import TASKS


def measured(**updates):
    args=dict(physics_timestamp=10.,physics_time_origin=9.5,
        task_timestamp=.5,task_time_origin=0.,
        rgb=np.zeros((4,256,256,3),np.uint8),depth_m=np.ones((4,256,256),np.float32),
        K=np.repeat(np.eye(3)[None],4,axis=0),T=np.repeat(np.eye(4)[None],4,axis=0),
        camera_clips=np.tile([.01,10.],(4,1)),q=np.arange(7,dtype=float)*.01,
        dq=np.arange(7,dtype=float)*.001,tcp_pose=np.array([.2,.1,.4,0.,0.,0.,1.]),
        finger_joint_positions=np.array([.01,.03]),finger_joint_velocities=np.array([.001,-.002]),
        finger_open_amounts=np.array([.25,.75]),render_modes=['OPENGL3','OPENGL','OPENGL','OPENGL'])
    args.update(updates)
    return obs.pack_feedback_snapshot(**args)


class ReadOnlyToyBackend:
    """Explicit CPU sensor fixture; has no physics tick/Native service."""
    requested_opening=1.
    def __init__(self):
        self.snapshot=measured();self.metric_calls=0;self.t=.5
    def clock(self):return self.t
    def joint_limits(self):return dict(lower=[-3.]*7,upper=[3.]*7,velocity=[2.]*7)
    def get_joint_positions(self):return np.arange(7,dtype=float)*.01
    def get_joint_velocities(self):return np.arange(7,dtype=float)*.001
    def read_snapshot(self):return self.snapshot
    def metric_receipt(self):
        self.metric_calls+=1
        return dict(native_current=dict(success=False,terminate=False),metric_only=True,policy_use_allowed=False)


class FeedbackIndependent(unittest.TestCase):
    def test_actual_mechanics_preserved_but_old_NN_projection_stays_masked(self):
        s=measured();obs.validate_feedback_snapshot(s)
        p=obs.legacy_policy_packet(s)
        np.testing.assert_array_equal(s['finger_open_amounts'],[.25,.75])
        np.testing.assert_array_equal(s['finger_joint_positions'],[.01,.03])
        np.testing.assert_array_equal(s['dq'],np.arange(7)*.001)
        self.assertEqual(p['robot15'][-1],0.);self.assertFalse(p['robot15_observed_mask'][-1])
        self.assertNotIn('dq',p);self.assertNotIn('finger_open_amounts',p)
        self.assertFalse(s['q'].flags.writeable);self.assertFalse(s['rgb'].flags.writeable)
        for key in ('Native_held','object_pose','task_success','instance_masks'):
            with self.assertRaises(ValueError):obs.validate_feedback_snapshot(dict(s,**{key:False}))

    def test_both_actual_clock_deltas_and_nonnegative_origins_are_required(self):
        self.assertEqual(measured()['timestamp'],.5)
        with self.assertRaises(ValueError):measured(task_timestamp=99.)
        with self.assertRaises(ValueError):measured(task_timestamp=-.5)
        with self.assertRaises(ValueError):measured(physics_timestamp=True)
        wrong=dict(measured());wrong['task_timestamp']=99.
        with self.assertRaises(ValueError):obs.validate_feedback_snapshot(wrong)

    def test_all16_default_constructor_refuses_before_path_or_Native_import(self):
        with patch('importlib.import_module',side_effect=AssertionError('No Native import')):
            for i,task in enumerate(TASKS):
                with self.subTest(task=task),self.assertRaisesRegex(ValueError,'explicit Native'):
                    PhysManiFeedbackBackendV1(task,'/missing/data.h5',seed=261104960+i,
                        variation=0,budget_ticks=16,execute_native=False)

    def test_original_execution_and_metric_free_snapshot_bindings(self):
        self.assertIs(facade.ShortHorizonExecutor,ShortHorizonExecutor)
        self.assertIs(facade.CompletedGripperBackend,CompletedGripperBackend)
        self.assertIs(facade.qualify_first_action,qualify_first_action)
        self.assertIs(PhysManiFeedbackBackendV1.tick,GenericTaskNativeBackend.tick)
        tree=ast.parse(inspect.getsource(PhysManiFeedbackBackendV1))
        methods={n.name:n for n in tree.body[0].body if isinstance(n,ast.FunctionDef)}
        names=[n.func.attr for n in ast.walk(methods['read_snapshot'])
               if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('success',names);self.assertNotIn('metric_receipt',names)
        self.assertIn('_assert_unchanged',names)
        ctor=inspect.getsource(PhysManiFeedbackBackendV1.__init__)
        self.assertLess(ctor.index('set_simulation_timestep(DT)'),ctor.index('get_task('))
        self.assertLess(ctor.index('self.scene.init_task()'),ctor.index('self.task.disable_expert_plan()'))
        self.assertLess(ctor.index('self.task.disable_expert_plan()'),ctor.index('self.task_environment.reset()'))
        self.assertIn('NativeTask.success',ctor)

    def test_facade_snapshot_never_queries_or_embeds_native_metrics(self):
        backend=ReadOnlyToyBackend();f=facade.FeedbackFacadeV1(backend)
        s=f.read_snapshot();self.assertEqual(backend.metric_calls,0)
        self.assertEqual(set(s),obs.KEYS);self.assertNotIn('native_current',s)
        receipt=f.metric_receipt();self.assertEqual(backend.metric_calls,1)
        self.assertFalse(receipt['policy_use_allowed']);self.assertEqual(backend.clock(),.5)


if __name__=='__main__':unittest.main()
