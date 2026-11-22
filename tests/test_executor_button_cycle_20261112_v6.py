import unittest
from unittest.mock import patch
from types import SimpleNamespace
from our_mani.executor_contract_repair_20261112_v6.press_cycle import PressState,plan

class ButtonCycleTests(unittest.TestCase):
    def test_does_not_retract_during_first_unfinished_approach(self):
        state=PressState()
        for now in (0.,1.,2.,3.):
            state,offset=plan(state,now=now,opening=0.,goal_z=.75,tcp_z=1.)
            self.assertEqual(state.phase,'press');self.assertEqual(offset,0.)

    def test_low_dwell_retracts_and_requires_actual_raise_before_repress(self):
        state,offset=plan(PressState(),now=0.,opening=0.,goal_z=.75,tcp_z=.76)
        state,offset=plan(state,now=.3,opening=0.,goal_z=.75,tcp_z=.76)
        self.assertEqual(offset,.08)
        state,offset=plan(state,now=2.,opening=0.,goal_z=.75,tcp_z=.76)
        self.assertEqual(state.phase,'raise')
        state,offset=plan(state,now=2.1,opening=0.,goal_z=.75,tcp_z=.83)
        state,offset=plan(state,now=2.2,opening=0.,goal_z=.75,tcp_z=.83)
        self.assertEqual(state.phase,'press');self.assertEqual(offset,0.)

    def test_open_cancels_all_contact_cycle_state(self):
        state,offset=plan(PressState('raise',1.),now=2.,opening=1.,goal_z=.75,tcp_z=.83)
        self.assertEqual(state,PressState());self.assertEqual(offset,0.)

    def test_typed_tuple_pose_is_preserved_in_execution_receipt(self):
        from our_mani.executor_contract_repair_20261112_v6.pose_executor import PoseGoalExecutor,PhysicalExecutor,WorldPoseGoal
        dispatch=object.__new__(PoseGoalExecutor)
        dispatch.backend=SimpleNamespace(clock=lambda:0.,arm=SimpleNamespace(
            get_tip=lambda:SimpleNamespace(get_position=lambda:[0.,0.,1.])))
        dispatch.executor=SimpleNamespace(config=SimpleNamespace(clock_tolerance=2e-5))
        dispatch.press_state=PressState()
        action=WorldPoseGoal(0.,[.2,.1,.75],[0.,1.,0.,0.],0.)
        with patch.object(PhysicalExecutor,'execute',return_value=dict(executed_ticks=2)):
            receipt=dispatch.execute(action)
        self.assertEqual(receipt['button_press_cycle']['raw_requested_position'],[.2,.1,.75])
        self.assertEqual(receipt['button_press_cycle']['dispatched_world_position'],[.2,.1,.75])

if __name__=='__main__':unittest.main()
