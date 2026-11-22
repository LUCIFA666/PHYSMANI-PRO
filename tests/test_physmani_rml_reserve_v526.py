from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import numpy as np
from our_mani.multitask.physmani_rml_reserve_v526 import (
    PlanningPath,PhysManiGeometryExecutor,PreviousExecutor,direct)


class Factory:
    def __init__(self,**kw):self.last_receipt=None
    def __call__(self,path):
        self.last_receipt={'planning_acceleration':path._arm.max_acceleration}
        return 42


class Tests(unittest.TestCase):
    def fixture(self):
        arm=NS(max_acceleration=12.,max_velocity=3.,max_jerk=1000.,target=np.zeros(7))
        arm.get_joint_target_positions=lambda:arm.target.tolist()
        arm.set_joint_target_positions=lambda q:setattr(arm,'target',np.asarray(q).copy())
        arm.get_joint_positions=lambda:np.zeros(7)
        path=NS(_arm=arm,_rml_handle=None,_path_done=False,_joint_position_action=None)
        def step():
            if path._rml_handle is None:path._rml_handle=path._get_rml_handle()
            path._joint_position_action=np.full(7,.01)
            arm.set_joint_target_positions(path._joint_position_action)
            return False
        path.step=step
        env=NS(arm=arm,time=0.,map_service=NS(plan_path=lambda *a,**kw:path))
        return env,path

    def adapter(self):
        return PhysManiGeometryExecutor(dict(dt=.05,baseline_decision_ticks=10),
            observe_public_frame=lambda *a:None,admit_pose_action=lambda a:(True,{}),
            admit_joint_target=lambda q:(True,{}),rml_factory_class=Factory)

    def test_planning_view_does_not_modify_real_arm_parameters(self):
        env,path=self.fixture();view=PlanningPath(path)
        self.assertEqual(view._arm.max_acceleration,4.)
        self.assertEqual(env.arm.max_acceleration,12.)
        self.assertEqual(view._arm.max_velocity,env.arm.max_velocity)

    @patch('our_mani.multitask.physmani_geometry_executor_v517._check_clock')
    @patch('our_mani.multitask.physmani_geometry_executor_v517.physics_time',return_value=0.)
    def test_original_native_step_and_gripper_tick_get_the_reserved_RML_handle(self,*mocks):
        env,path=self.fixture();adapter=self.adapter();calls=[]
        original_plan=env.map_service.plan_path
        def apply(owned,joints,*,opening,phase):
            calls.append((np.asarray(joints).copy(),opening,phase));return joints
        def execute(instance,owned,action,observer=None):
            p=owned.map_service.plan_path();p.step()
            direct._apply_direct_target(owned,np.zeros(7),opening=1.,phase='physmani_gripper')
            return {'ticks':10}
        with patch.object(direct,'_apply_direct_target',apply):
            with patch.object(PreviousExecutor,'execute',execute):adapter.execute(env,np.zeros(8))
            self.assertIs(direct._apply_direct_target,apply)
        self.assertIs(env.map_service.plan_path,original_plan)
        self.assertEqual(path._rml_handle,42)
        self.assertEqual(adapter.planning_receipts[0]['planning_acceleration'],4.)
        np.testing.assert_array_equal(calls[0][0],np.full(7,.01))
        self.assertEqual(len(adapter.tenth_tick_events),1)
        self.assertEqual(env.arm.max_acceleration,12.)

    def test_scopes_restore_on_native_error_and_rejected_path_is_not_resumed(self):
        env,path=self.fixture();adapter=self.adapter();original_plan=env.map_service.plan_path
        original_apply=direct._apply_direct_target
        def execute(instance,owned,action,observer=None):
            owned.map_service.plan_path()
            raise RuntimeError('test owned native error')
        with patch.object(PreviousExecutor,'execute',execute):
            with self.assertRaisesRegex(RuntimeError,'test owned native error'):adapter.execute(env,np.zeros(8))
        self.assertIs(direct._apply_direct_target,original_apply)
        self.assertIs(env.map_service.plan_path,original_plan)


if __name__=='__main__':unittest.main()
