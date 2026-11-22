"""Recovery adapter executes the author's actual AST-extracted _step_motion."""
import ast
import json
from pathlib import Path
import sys
from types import MethodType,ModuleType,SimpleNamespace
from typing import List
import unittest
from unittest.mock import patch
import numpy as np

from our_mani.multitask.rml_path_repair import NativeRMLHandleFactory,RMLVirtualProbeExhausted
from our_mani.multitask.rml_path_recovery import (recover_native_path,recovery_native_path_planner,
    recovery_diagnostic_evidence,NATIVE_DIAGNOSTIC)
from our_mani.multitask.protocol import MultitaskProtocolError
from our_mani.multitask.train_teacher import NativeRMLPath
from tests.test_multitask_rml_path_repair import Backend

AUTHOR=Path('/home/ubuntu/projects/PhysMani/third_party/PyRep/pyrep/robots/configuration_paths/arm_configuration_path.py')


class Arm:
    max_velocity=3.;max_acceleration=6.;max_jerk=12.;_num_joints=1;joints=(None,)
    def __init__(self):self.targets=[]
    def get_joint_upper_velocity_limits(self):return [1.]
    def set_joint_target_positions(self,q):self.targets.append(np.asarray(q).copy())


def native_path(backend):
    """Execute exactly the actual source functions without loading PyRep."""
    tree=ast.parse(AUTHOR.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ArmConfigurationPath')
    namespace=dict(sim=backend,np=np,List=List)
    functions=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('step','_step_motion','_get_path_point_lengths')]
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(AUTHOR),'exec'),namespace)
    p=SimpleNamespace(_arm=Arm(),_num_joints=1,_path_points=np.array([0.,1.]),
        _rml_handle=None,_path_done=False,_joint_position_action=None)
    for name in ('step','_step_motion','_get_path_point_lengths'):setattr(p,name,MethodType(namespace[name],p))
    p._get_rml_handle=lambda:(_ for _ in ()).throw(AssertionError('original initializer was called'))
    p.get_executed_joint_position_action=lambda:p._joint_position_action
    return p


def backend_modules(backend):
    parent=ModuleType('pyrep');parent.__path__=[];child=ModuleType('pyrep.backend');child.sim=backend
    return {'pyrep':parent,'pyrep.backend':child}


class RecoveryIntegrationChecks(unittest.TestCase):
    def test_actual_native_step_motion_initializes_once_checks_joint_speed_and_never_steps_physics(self):
        backend=Backend();path=native_path(backend);factory=NativeRMLHandleFactory(sim_backend=backend)
        recovered=recover_native_path(path,handle_factory=factory);previous=0.;done=False;physical_ticks=0;calls=0
        with patch.dict(sys.modules,backend_modules(backend)):
            while not done:
                done=recovered.step();calls+=1
                actual=recovered.get_executed_joint_position_action()[0]
                self.assertLessEqual(abs(actual-previous)/.1,1.001);previous=actual
                self.assertEqual(physical_ticks,calls-1) # path.step only writes the reference
                physical_ticks+=1 # caller-owned counted executor, not adapter
            self.assertEqual(recovered.recovery_receipt['initialization_count'],1)
            self.assertEqual(len(path._arm.targets),calls);self.assertEqual(physical_ticks,calls)
            self.assertEqual(backend.handles,{});self.assertIsNone(path._rml_handle)
            self.assertTrue(recovered.recovery_receipt['native_execution_completion_observed'])
            removes=list(backend.removed);recovered.close();recovered.close();self.assertEqual(backend.removed,removes)
            with self.assertRaises(MultitaskProtocolError):recovered.step()

    def test_failed_real_step_deletes_live_handle_and_cannot_reinitialize(self):
        backend=Backend();path=native_path(backend);factory=NativeRMLHandleFactory(sim_backend=backend)
        original=backend.simRMLStep
        def fail_execution(handle,*args):
            # Probe handles 1/2 succeed; the live execution handle is 3.
            if handle==3:raise ValueError('actual native execution failure')
            return original(handle,*args)
        backend.simRMLStep=fail_execution;recovered=recover_native_path(path,handle_factory=factory)
        with patch.dict(sys.modules,backend_modules(backend)):
            with self.assertRaisesRegex(ValueError,'execution failure'):recovered.step()
            self.assertEqual(backend.handles,{});self.assertEqual(backend.removed,[1,2,3])
            self.assertEqual(path._arm.targets,[])
            with self.assertRaises(MultitaskProtocolError):recovered.step()
            self.assertEqual(recovered.recovery_receipt['initialization_count'],1)

    def test_failed_initializer_cleans_probe_and_does_not_issue_target(self):
        backend=Backend();path=native_path(backend)
        recovered=recover_native_path(path,handle_factory=NativeRMLHandleFactory(sim_backend=backend,maximum_virtual_steps=1))
        with patch.dict(sys.modules,backend_modules(backend)):
            with self.assertRaises(RMLVirtualProbeExhausted):recovered.step()
        self.assertEqual(backend.handles,{});self.assertEqual(path._arm.targets,[])
        self.assertIsNone(path._rml_handle);self.assertTrue(recovered.recovery_receipt['close_called'])

    def test_manual_early_close_deletes_once_and_single_point_keeps_base_behavior(self):
        backend=Backend();path=native_path(backend);factory=NativeRMLHandleFactory(sim_backend=backend)
        recovered=recover_native_path(path,handle_factory=factory)
        with patch.dict(sys.modules,backend_modules(backend)):
            self.assertFalse(recovered.step());recovered.close();removed=list(backend.removed)
            recovered.close();self.assertEqual(backend.removed,removed);self.assertEqual(backend.handles,{})
            point=native_path(backend);point._path_points=np.array([.2])
            single=recover_native_path(point,handle_factory=factory);self.assertTrue(single.step())
            self.assertEqual(single.recovery_receipt['initialization_count'],0)
            self.assertEqual(single.recovery_receipt['native_step_calls'],0)
            np.testing.assert_array_equal(single.get_executed_joint_position_action(),[.2])

    def test_cleanup_error_remains_explicit_and_caller_may_retry_cleanup_without_reinitializing(self):
        backend=Backend();path=native_path(backend);factory=NativeRMLHandleFactory(sim_backend=backend)
        recovered=recover_native_path(path,handle_factory=factory)
        with patch.dict(sys.modules,backend_modules(backend)):
            self.assertFalse(recovered.step());backend.remove_failure=True
            with self.assertRaisesRegex(ValueError,'delete failure'):recovered.close()
            self.assertFalse(recovered.recovery_receipt['close_complete'])
            self.assertEqual(len(recovered.recovery_receipt['close_errors']),1)
            with self.assertRaises(MultitaskProtocolError):recovered.step()
            backend.remove_failure=False;recovered.close()
            self.assertTrue(recovered.recovery_receipt['close_complete']);self.assertEqual(backend.handles,{})
            self.assertEqual(recovered.recovery_receipt['initialization_count'],1)

    def test_adapter_planner_reuses_original_and_binds_truthful_native_diagnostic(self):
        backend=Backend();path=native_path(backend);original=NativeRMLPath(path)
        with patch('our_mani.multitask.rml_path_recovery.native_path_planner',return_value=lambda goal:original):
            recovered=recovery_native_path_planner(object(),handle_factory=NativeRMLHandleFactory(sim_backend=backend))(object())
        self.assertIs(recovered.path,path)
        evidence=recovery_diagnostic_evidence()
        self.assertEqual(evidence['diagnostic_report'],NATIVE_DIAGNOSTIC)
        self.assertFalse(evidence['historical_fault_reproduced_in_native_diagnostic'])
        self.assertFalse(evidence['native_velocity_correction_second_round_observed'])
        self.assertFalse(recovered.recovery_receipt['real_native_step_motion_replaced'])


if __name__=='__main__':unittest.main()
