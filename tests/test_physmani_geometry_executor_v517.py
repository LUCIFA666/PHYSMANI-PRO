from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import numpy as np
from our_mani.multitask.physmani_geometry_executor_v517 import (
    PhysManiGeometryExecutor,GuardedPath,GeometryTargetRejected,OriginalExecutor)


class Arm:
    def __init__(self): self.target=np.zeros(7)
    def get_joint_target_positions(self): return self.target.tolist()
    def set_joint_target_positions(self,q): self.target=np.asarray(q,float).copy()


class Path:
    def __init__(self,arm):
        self._arm=arm;self._joint_position_action=None;self._rml_handle=123
        self._path_points=np.zeros(14);self.calls=0
    def step(self):
        self.calls+=1
        self._joint_position_action=np.full(7,.01)
        self._arm.set_joint_target_positions(self._joint_position_action)
        return False


class Tests(unittest.TestCase):
    def fixture(self):
        arm=Arm();path=Path(arm)
        service=NS(plan_path=lambda *a,**kw:path)
        return NS(arm=arm,map_service=service,time=0.,ledger=NS(ticks=0)),path

    def adapter(self,**overrides):
        callbacks=dict(observe_public_frame=lambda *a:None,
                       admit_pose_action=lambda a:(True,{'public':True}),
                       admit_joint_target=lambda q:(True,{'public':True}))
        callbacks.update(overrides)
        return PhysManiGeometryExecutor(dict(dt=.05,baseline_decision_ticks=10),**callbacks)

    @patch('our_mani.multitask.physmani_geometry_executor_v517._check_clock')
    @patch('our_mani.multitask.physmani_geometry_executor_v517.physics_time',return_value=0.)
    def test_rejected_native_RML_target_is_restored_before_any_physics(self,*mocks):
        env,path=self.fixture();events=[]
        guarded=GuardedPath(path,env,lambda q:(False,{'collision':True}),events)
        with self.assertRaises(GeometryTargetRejected):guarded.step()
        np.testing.assert_array_equal(env.arm.target,np.zeros(7))
        self.assertEqual(env.ledger.ticks,0)
        self.assertFalse(events[0]['accepted'])
        self.assertEqual(path.calls,1)

    @patch('our_mani.multitask.physmani_geometry_executor_v517._check_clock')
    @patch('our_mani.multitask.physmani_geometry_executor_v517.physics_time',return_value=0.)
    def test_native_waypoint_and_RML_handle_assignments_reach_original_path(self,*mocks):
        env,path=self.fixture();guarded=GuardedPath(path,env,lambda q:(True,{}),[])
        guarded._path_points=np.ones(21);guarded._rml_handle=None
        self.assertEqual(len(path._path_points),21);self.assertIsNone(path._rml_handle)
        guarded.step();np.testing.assert_array_equal(env.arm.target,np.full(7,.01))
        self.assertEqual(env.ledger.ticks,0)

    def test_reuses_original_executor_without_constructing_or_predicting_neural_policy(self):
        env,path=self.fixture();original=env.map_service.plan_path;seen=[]
        action=np.r_[np.zeros(3),[0.,0.,0.,1.],1.]
        def execute(instance,current_env,current_action,observer=None):
            seen.append((current_env,current_action.copy()))
            self.assertIsInstance(current_env.map_service.plan_path(),GuardedPath)
            current_env.ledger.ticks+=10;current_env.time=.5
            return {'executor':'original_v7','ticks':10}
        with patch.object(OriginalExecutor,'__init__',side_effect=AssertionError('neural init forbidden')):
            adapter=self.adapter()
        with patch.object(OriginalExecutor,'execute',execute):adapter.execute(env,action)
        self.assertIs(env.map_service.plan_path,original)
        self.assertIs(seen[0][0],env);np.testing.assert_array_equal(seen[0][1],action)
        self.assertEqual(adapter.geometry_receipt()['actions'][0]['original_executor_receipt']['ticks'],10)
        with self.assertRaises(Exception):adapter.predict(None)

    def test_original_map_is_restored_after_executor_error_and_invalid_action_is_never_dispatched(self):
        env,path=self.fixture();original=env.map_service.plan_path;adapter=self.adapter()
        action=np.r_[np.zeros(3),[0.,0.,0.,1.],1.]
        with patch.object(OriginalExecutor,'execute',side_effect=RuntimeError('native test error')):
            with self.assertRaisesRegex(RuntimeError,'native test error'):adapter.execute(env,action)
        self.assertIs(env.map_service.plan_path,original)
        with patch.object(OriginalExecutor,'execute') as execute:
            with self.assertRaises(ValueError):adapter.execute(env,np.zeros(7))
            execute.assert_not_called()


if __name__=='__main__':unittest.main()
