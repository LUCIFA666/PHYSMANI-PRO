from types import SimpleNamespace as NS
import unittest
import numpy as np
import test_block_reference_v197 as prior
import test_block_reference_v186 as initial
from our_mani.intercept_planner import JointTrajectory
from our_mani.multitask.block_exact_reference_v201 import SmoothBlockEnvironment, public_decision_scope


class Tests(prior.Tests):
    def fixture(self):
        env=initial.Native();b=SmoothBlockEnvironment(env)
        b.hold_tick();b.hold_tick()
        return env,b

    def global_setup(self, duration=1.):
        env,b=self.fixture()
        self.decision(env,b)
        robot,body=b.public_robot,b.public_body
        def original(*args,**kwargs):
            workflow=NS(dynamics=b.local_dynamics,selector=NS(),collision_scene=NS(origin=None))
            workflow.plan=JointTrajectory(env.time,env.time+duration,np.zeros(7),np.zeros(7),np.full(7,.04),np.zeros(7))
            workflow.step=lambda robot,obj,remaining:workflow.joint_command(robot)
            return workflow
        return env,b,robot,body,NS(CandidateInterception=original)

    def test_global_reference_reuses_deadline_and_acknowledged_boundary_without_servo_lead(self):
        env,b,robot,body,evaluator=self.global_setup()
        with public_decision_scope(evaluator,b):
            workflow=evaluator.CandidateInterception('dynamics',env.robot_model,None)
            deadline=workflow.plan.end;decision=workflow.step(robot,body,10.)
            self.assertEqual(decision['command'],'joint_trajectory')
            self.assertEqual(decision['trajectory_end'],deadline)
            b.command_joints(decision['joints'],opening=1.)
            self.assertGreater(np.linalg.norm(env.target),0.)
            self.assertEqual(env.actions,3)
            self.assertTrue(b.reference_events[-1]['evidence']['exact_next_tick_original_collision_gate_passed'])

    def test_global_original_collision_gate_still_returns_hold(self):
        env,b,robot,body,evaluator=self.global_setup();env.robot_model.block=True
        with public_decision_scope(evaluator,b):
            workflow=evaluator.CandidateInterception('dynamics',env.robot_model,None)
            self.assertEqual(workflow.step(robot,body,10.)['command'],'hold')
            self.assertEqual(env.actions,2)

    def test_exact_next_tick_collision_cannot_be_missed_between_full_path_samples(self):
        env,b,robot,body,evaluator=self.global_setup(.9)
        env.robot_model.collision=lambda q,offset:abs(offset-.05)<1e-9
        with public_decision_scope(evaluator,b):
            workflow=evaluator.CandidateInterception('dynamics',env.robot_model,None)
            decision=workflow.step(robot,body,10.)
            self.assertEqual(decision['command'],'joint_trajectory')
            b.command_joints(decision['joints'],opening=1.)
            np.testing.assert_array_equal(env.target,np.zeros(7))
            self.assertEqual(env.actions,3)
            self.assertEqual(b.reference_events[-1]['reason'],'exact_next_reference_collision_rejected')


if __name__=='__main__':unittest.main()
