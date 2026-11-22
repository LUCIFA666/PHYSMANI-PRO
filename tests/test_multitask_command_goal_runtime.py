"""CPU controller boundaries, with synthetic policy observations and no native."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from our_mani.contracts import Pose
from our_mani.multitask.interaction import (InteractionController,GripperTimingConfig,
    _observed_reference,_selected_reference_at)
from our_mani.multitask.reference_time import COMMAND_GOAL
from tests.test_multitask_interaction import synthetic_library,state,candidates_config


def controller(phase='acquire'):
    template=replace(synthetic_library().templates[0],phase=phase,reference_time_basis=COMMAND_GOAL,
        reference_time_rule_sha256='a'*64,completion_geometry_xyzw=Pose([0.,0.,.1]).xyzw(),
        duration_seconds=.5,duration_budget_seconds=1.)
    actor=InteractionController.__new__(InteractionController)
    actor.library=SimpleNamespace(phase_order=(phase,));actor.phase_index=0
    actor.goal=SimpleNamespace(template=template,reference_id='button',manipulated_ids=('button',),
        candidate_id='fixture',selected_at=0.,arrival_time=.2,phase_budget=1.,reference_pose=Pose([0.,0.,0.]))
    actor.origin=Pose([0.,0.,0.]);actor.trajectory=None;actor.committed_roles={}
    actor.measured_attachments={};actor.events=[];actor.last_diagnostics={};actor.gripper_timing_events=[]
    actor._close_edge_issued_at=.1;actor._gripper_goal_token=None
    actor.gripper_timing_config=GripperTimingConfig(.01,.05)
    actor.attempts=SimpleNamespace(skill_completed=lambda *args:None,abandon=lambda *args:None)
    actor.generator=SimpleNamespace(config=candidates_config(),bind_candidate_scene=lambda *args:None,
        collision_check=lambda **kwargs:dict(feasible=False),
        observed_reference=lambda observed,body,t,origin:_observed_reference(observed,body,t,origin,candidates_config()))
    return actor


def observed(time=.6,*,body_x=0.,tcp_x=0.,contact=True,observable=True,visible=True):
    value=state(time,robot_z=.1,observable=observable)
    info=value.perception_metadata['bodies']['button']
    info.update(motion_mode='attached' if contact else 'external',
        mode_provenance='caller_controller_touch_action_proprioception',directly_visible=visible)
    return replace(value,robot=replace(value.robot,pose=Pose([tcp_x,0.,.1]),contact=contact,gripper_open=not contact),
        bodies={'button':replace(value.bodies['button'],pose=Pose([body_x,0.,0.]),visible=visible)})


class NoFutureQueries:
    metadata={}
    def at(self,*args):raise AssertionError('fixed execution anchor queried a future model')


class CommandGoalRuntimeTests(unittest.TestCase):
    def test_planned_target_cannot_be_completion_evidence_even_with_public_contact(self):
        actor=controller();value=observed()
        self.assertFalse(actor._phase_done(value,Pose([0.,0.,.1])))
        self.assertTrue(actor._phase_done(value,Pose([0.,0.,.1]),public_geometry=True))

    def test_contact_and_old_planned_arrival_do_not_bypass_current_wrong_geometry(self):
        actor=controller();action=actor.step(observed(body_x=.2),NoFutureQueries(),9.)
        self.assertEqual(actor.phase_index,0)
        self.assertEqual(action['reason'],'fresh_dynamic_collision_rejected')
        self.assertAlmostEqual(actor.last_diagnostics['public_completion_geometry']['position_error_m'],.2)

    def test_fresh_current_geometry_can_verify_contact_and_unique_own_close(self):
        actor=controller();action=actor.step(observed(body_x=.2,tcp_x=.2),NoFutureQueries(),9.)
        self.assertEqual(actor.phase_index,1)
        self.assertEqual(action['reason'],'advance_to_next_TRAIN_fitted_skill')
        self.assertFalse(actor.events[-1]['official_task_success_claimed'])

    def test_completion_does_not_accept_contact_without_own_close(self):
        actor=controller();actor._close_edge_issued_at=None
        actor.step(observed(),NoFutureQueries(),9.)
        self.assertEqual(actor.phase_index,0)

    def test_none_orientation_remains_unsupported_and_timeout_does_not_hang(self):
        actor=controller();value=observed(observable=False)
        self.assertEqual(actor.step(value,NoFutureQueries(),9.)['reason'],'fresh_observed_reference_or_forecast_required')
        self.assertEqual(actor.phase_index,0)
        action=actor.step(observed(1.1,observable=False),NoFutureQueries(),8.)
        self.assertEqual(action['opening'],1.)
        self.assertIn('unverified_acquire_geometry_timeout',action['reason'])

    def test_measured_attachment_does_not_replace_required_fresh_RGBD_geometry(self):
        actor=controller();actor.measured_attachments['button']=dict(local_pose_xyzw=Pose([0.,0.,-.1]).xyzw(),
            observed_at=.2,source='current_RGBD_body_pose_and_robot_proprioception_at_public_contact')
        action=actor.step(observed(visible=False),NoFutureQueries(),9.)
        self.assertEqual(action['reason'],'fresh_observed_reference_or_forecast_required')
        self.assertEqual(actor.phase_index,0)

    def test_other_observed_phase_uses_current_geometry_not_planned_arrival(self):
        actor=controller('follow');actor.step(observed(body_x=.2),NoFutureQueries(),9.)
        self.assertEqual(actor.phase_index,0)

    def test_selected_causal_pose_is_fixed_for_all_future_control_times(self):
        actor=controller();actor.goal.reference_pose=Pose([.3,.1,0.])
        for time in (.2,.6,1.):
            pose,v,w=_selected_reference_at(actor.goal,NoFutureQueries(),time,actor.origin)
            np.testing.assert_array_equal(pose.position,[.3,.1,0.]);np.testing.assert_array_equal(v,np.zeros(3))


if __name__=='__main__':unittest.main()
