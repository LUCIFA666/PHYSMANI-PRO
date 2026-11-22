from types import SimpleNamespace
import unittest
import numpy as np
from our_mani.contracts import RobotState,Pose
from our_mani.multitask.joint16_block_episode_v98 import PublicCountedBlockEnvironment,shared_context_factory_scope


class FakeEnvironment:
    def __init__(self):
        self.q=np.zeros(7);self.ledger=SimpleNamespace(dt=.05,ticks=0);self.observations=0;self.actions=0
        self.arm=SimpleNamespace(get_joint_positions=lambda:self.q.copy())
        self.robot_model=SimpleNamespace(velocity_limits=np.full(7,10.),acceleration_limits=np.full(7,6.),
            lower=np.full(7,-10.),upper=np.full(7,10.))
    @property
    def time(self):return self.ledger.ticks*.05
    def observe(self):
        self.observations+=1
        return {},RobotState(self.time,Pose(np.zeros(3)),np.zeros(3),np.zeros(3),self.q,np.zeros(7),
            gripper_open=True,contact=False,task_success=True)
    def command_joints(self,q,**kwargs):
        self.q=np.array(q,float);self.ledger.ticks+=1;self.actions+=1;return 'actual-result'
    def advance(self,pose,**kwargs):return self.command_joints(self.q+.001)
    def hold_tick(self):return self.command_joints(self.q)


class Tests(unittest.TestCase):
    def test_each_action_is_forwarded_once_with_no_render_or_physics_probe(self):
        env=FakeEnvironment();wrapper=PublicCountedBlockEnvironment(env)
        self.assertEqual(wrapper.command_joints(np.full(7,.001)),'actual-result')
        wrapper.advance(Pose(np.zeros(3)));wrapper.hold_tick()
        self.assertEqual(env.actions,3);self.assertEqual(env.ledger.ticks,3);self.assertEqual(env.observations,0)
        receipt=wrapper.receipt();self.assertEqual(len(receipt['events']),3)
        self.assertTrue(receipt['all_stage_actual_window_checks_passed'])
        self.assertEqual(receipt['additional_camera_queries'],0)

    def test_actual_motion_violation_is_reported_without_modifying_command(self):
        env=FakeEnvironment();wrapper=PublicCountedBlockEnvironment(env)
        wrapper.command_joints(np.full(7,.005));wrapper.command_joints(np.full(7,.055))
        np.testing.assert_array_equal(env.q,np.full(7,.055))
        receipt=wrapper.receipt();self.assertFalse(receipt['all_stage_actual_window_checks_passed'])
        self.assertAlmostEqual(receipt['maximum_actual_window_acceleration'],18.)
        self.assertEqual(receipt['violations'],[2]);self.assertEqual(env.actions,2)

    def test_only_public_motion_fields_are_forwarded_from_the_existing_observation(self):
        env=FakeEnvironment();calls=[]
        context=SimpleNamespace(set_public_motion_state=lambda **kw:calls.append(kw))
        wrapper=PublicCountedBlockEnvironment(env,context);observation,robot=wrapper.observe()
        self.assertEqual(env.observations,1)
        self.assertEqual(calls,[dict(timestamp=0.,gripper_open=True,contact=False)])
        self.assertTrue(robot.task_success)

    def test_model_scope_restores_on_exception_and_refuses_value_construction(self):
        original_context=lambda *a:None;original_value=lambda *a:None
        evaluator=SimpleNamespace(ContextLatentDynamics=original_context,CandidateSuccessValue=original_value)
        with self.assertRaisesRegex(RuntimeError,'stop'):
            with shared_context_factory_scope(evaluator,None):
                with self.assertRaisesRegex(ValueError,'Value is prohibited'):evaluator.CandidateSuccessValue('unused')
                with self.assertRaisesRegex(ValueError,'separate Context'):evaluator.ContextLatentDynamics('old')
                raise RuntimeError('stop')
        self.assertIs(evaluator.ContextLatentDynamics,original_context);self.assertIs(evaluator.CandidateSuccessValue,original_value)

    def test_unaccounted_extra_tick_is_not_treated_as_a_valid_action(self):
        env=FakeEnvironment();original=env.command_joints
        def two(*args,**kwargs):
            result=original(*args,**kwargs);env.ledger.ticks+=1;return result
        env.command_joints=two;wrapper=PublicCountedBlockEnvironment(env)
        with self.assertRaisesRegex(ValueError,'exactly one'):wrapper.command_joints(np.zeros(7))
        self.assertEqual(env.actions,1)


if __name__=='__main__':unittest.main()
