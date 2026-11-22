from types import SimpleNamespace as NS
import unittest
import numpy as np
import test_block_reference_v186 as prior
from our_mani.intercept_planner import JointTrajectory
from our_mani.multitask.block_reference_v197 import SmoothBlockEnvironment, clipped_terminal, ReferenceModel, public_decision_scope


class Tests(prior.Tests):
    def fixture(self):
        env=prior.Native();boundary=SmoothBlockEnvironment(env)
        boundary.hold_tick();boundary.hold_tick()
        return env,boundary

    def test_joint_delta_saturation_preserves_unsaturated_motion_components(self):
        current=np.array([.1,-1.,0.]);desired=np.array([.2,2.,0.])
        terminal=clipped_terminal(current,desired,.15,1.)
        self.assertAlmostEqual(terminal[0],.2)
        self.assertAlmostEqual(terminal[1],-.4)
        self.assertLessEqual(max(abs(terminal-current))/.15,4.+1e-8)

    def test_reference_model_forwards_original_collision_binding_and_does_not_set_native_four(self):
        actual=prior.Model();reference=ReferenceModel(actual);callback=lambda offset:True
        reference.collision_query=callback
        self.assertIs(actual.collision_query,callback)
        np.testing.assert_array_equal(actual.acceleration_limits,np.full(7,6.))
        np.testing.assert_array_equal(reference.acceleration_limits,np.full(7,4.))

    def test_decision_scope_restores_original_constructor_and_preserves_policy_output(self):
        env,b=self.fixture();_,robot=b.observe();body=NS();decision=dict(command='hold')
        original=lambda *args,**kwargs:NS(step=lambda *a:decision,dynamics='original-model',selector=NS(),plan=None)
        evaluator=NS(CandidateInterception=original)
        with public_decision_scope(evaluator,b):
            workflow=evaluator.CandidateInterception('dynamics',env.robot_model,None)
            self.assertIs(workflow.step(robot,body,10.),decision)
            self.assertIs(b.public_body,body)
            self.assertEqual(b.local_dynamics,'original-model')
        self.assertIs(evaluator.CandidateInterception,original)

    def global_workflow(self, env, boundary):
        def original(*args, **kwargs):
            workflow=NS(dynamics='original-model',selector=NS(),collision_scene=NS(origin=None))
            workflow.plan=JointTrajectory(env.time,env.time+1.,np.zeros(7),np.zeros(7),np.full(7,.04),np.zeros(7))
            workflow.step=lambda robot,obj,remaining:workflow.joint_command(robot)
            return workflow
        return NS(CandidateInterception=original)

    def test_global_reference_reuses_deadline_and_acknowledged_boundary_without_servo_lead(self):
        env,b=self.fixture();_,robot=b.observe();evaluator=self.global_workflow(env,b)
        with public_decision_scope(evaluator,b):
            workflow=evaluator.CandidateInterception('dynamics',env.robot_model,None)
            deadline=workflow.plan.end
            decision=workflow.step(robot,NS(),10.)
            self.assertEqual(decision['command'],'joint_trajectory')
            self.assertEqual(decision['trajectory_end'],deadline)
            np.testing.assert_array_equal(decision['joints'],decision['reference_joints'])
            self.assertEqual(env.actions,2)
            b.command_joints(decision['joints'],opening=1.)
            self.assertEqual(env.actions,3)
            self.assertTrue(b.receipt()['all_stage_actual_window_checks_passed'])

    def test_global_original_collision_gate_still_returns_hold(self):
        env,b=self.fixture();_,robot=b.observe();evaluator=self.global_workflow(env,b);env.robot_model.block=True
        with public_decision_scope(evaluator,b):
            workflow=evaluator.CandidateInterception('dynamics',env.robot_model,None)
            decision=workflow.step(robot,NS(),10.)
            self.assertEqual(decision['command'],'hold')
            self.assertEqual(env.actions,2)


if __name__=='__main__':unittest.main()
