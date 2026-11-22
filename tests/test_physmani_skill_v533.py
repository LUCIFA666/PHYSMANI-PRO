from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import json
import unittest
import numpy as np
from our_mani.contracts import Pose,RobotState,ObjectState
from our_mani.multitask.interaction import InteractionLibrary,PolicyState,_observed_reference,CandidateConfig
from our_mani.multitask.physmani_skill_policy_v533 import SkillGeometryIntents
from our_mani.multitask.physmani_geometry_goals_v529 import current_completion_pose
from our_mani.multitask.physmani_skill_executor_v534 import PhysManiSkillExecutor,OriginalGeometry,direct

ROOT=Path(__file__).resolve().parents[1]


class Forecast:
    def __init__(self,state):
        self.state=state;self.times=np.array([state.robot.timestamp,state.robot.timestamp+1.1])
        self.metadata=dict(observed_until=state.robot.timestamp)
    def at(self,identity,time):
        b=self.state.bodies[identity]
        return Pose(b.pose.position+np.array([.1*(time-self.state.robot.timestamp),0.,0.]),b.pose.rotation),np.array([.1,0,0]),np.zeros(3)


class Tests(unittest.TestCase):
    def fixture(self,phase='acquire'):
        config=json.loads((ROOT/'artifacts/framework/multitask_r10/bin_physical_smoke_v240_setup/configuration.json').read_text())
        library=InteractionLibrary.load(ROOT/config['learned_artifacts']['put_rubbish_in_moving_bin']['interaction']['path'])
        robot=RobotState(1.,Pose(np.array([1.,0.,1.])),np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7))
        bodies={k:ObjectState(k,1.,Pose(np.array([.3,0.,.8])),np.zeros((3,3)),np.zeros(3),np.zeros(3),1.,True)
                for k in ('rubbish','bin')}
        meta=dict(bodies={k:dict(directly_visible=True,registration_valid=True,orientation_observable=True,
            motion_mode='external',mode_provenance='caller_controller_touch_action_proprioception') for k in bodies})
        state=PolicyState(robot,bodies,meta,{k:(k,) for k in bodies})
        cfg=CandidateConfig(**config['candidate_config'])
        generator=NS(observed_reference=lambda s,i,t,o:_observed_reference(s,i,t,o,cfg),bind_candidate_scene=lambda *a:None)
        c=NS(library=library,generator=generator,gripper_timing_config=NS(close_position_tolerance_m=.02,
            close_orientation_tolerance_rad=.1),committed_roles={},measured_attachments={},origin=None,
            _causal_attachment_pose=lambda *a:None)
        actor=NS(controller=c,predictor='context',ranking='geometry',robot_model=NS(lower=np.full(7,-3.),
            upper=np.full(7,3.),collision=lambda *a:False),command_segments=NS(issue=lambda *a:None),requested_opening=1.)
        f=NS(actor=actor,state=state,forecast=Forecast(state),ready=lambda phase:True)
        p=SkillGeometryIntents(f);p.phase_index=p.phase_order.index(phase)
        p.select()
        return p,f

    def move(self,f,pose,**kwargs):
        f.state=PolicyState(replace(f.state.robot,pose=pose,**kwargs),f.state.bodies,
                            f.state.perception_metadata,f.state.role_bindings)

    def test_future_pose_arrival_is_not_a_physical_close_or_acquire_witness(self):
        p,f=self.fixture();goal=p.selected;current=current_completion_pose(f,goal,origin=p.origin)
        self.move(f,goal['pose'])
        opening,_,proposed=p.qualify_opening(0.,1.)
        self.assertEqual(opening,1.);self.assertIsNone(proposed)
        self.move(f,current)
        opening,proof,proposed=p.qualify_opening(0.,1.)
        self.assertEqual(opening,0.);self.assertIsNotNone(proposed)
        self.assertIsNone(p.close_witness)
        self.assertFalse(p._public_capture(goal))
        p.confirm_opening(opening,proof,proposed,confirmed_after=1.05)
        self.assertIsNotNone(p.close_witness)
        self.assertFalse(p._public_capture(goal))
        self.move(f,current,timestamp=1.5,contact=True,gripper_open=False)
        f.state.perception_metadata['bodies']['rubbish']['motion_mode']='attached'
        f.actor.controller.measured_attachments['rubbish']={}
        self.assertTrue(p._public_capture(goal))
        f.state.perception_metadata['bodies']['bin']['motion_mode']='attached'
        self.assertFalse(p._public_capture(goal))

    def test_replanning_does_not_reset_phase_epoch_and_stale_forecast_cannot_issue(self):
        p,f=self.fixture();epoch=p.phase_epoch
        self.move(f,f.state.robot.pose,timestamp=1.05)
        f.forecast=Forecast(f.state);p.select()
        self.assertEqual(p.phase_epoch,epoch)
        f.forecast.metadata['observed_until']=1.
        self.assertIsNone(p.select())

    def test_release_requires_current_canonical_pose_and_observed_release_before_completion(self):
        p,f=self.fixture('deposit');goal=p.selected
        self.move(f,goal['pose'],contact=True,gripper_open=False)
        self.assertEqual(p.qualify_opening(1.,0.)[0],0.)
        current=current_completion_pose(f,goal,origin=p.origin)
        self.move(f,current)
        self.assertEqual(p.qualify_opening(1.,0.)[0],1.)
        self.assertFalse(p._done(goal))

    def test_owned_gripper_scope_restores_after_native_failure_and_does_not_confirm_failed_tick(self):
        p,f=self.fixture();adapter=PhysManiSkillExecutor(dict(dt=.05,baseline_decision_ticks=10),
            skill_policy=p,observe_public_frame=lambda *a:None,admit_pose_action=lambda a:(True,{}),
            admit_joint_target=lambda q:(True,{}))
        env=NS(time=1.,_requested_opening=1.)
        def run(instance,owned,action,observer=None):
            return direct._apply_direct_target(owned,np.zeros(7),opening=0.,phase='physmani_gripper')
        with patch.object(direct,'_apply_direct_target',side_effect=RuntimeError('native failed')) as original:
            with patch.object(OriginalGeometry,'execute',run):
                with self.assertRaisesRegex(RuntimeError,'native failed'):adapter.execute(env,np.zeros(8))
            self.assertIs(direct._apply_direct_target,original)
        self.assertEqual(p.gripper_events,[]);self.assertIsNone(p.close_witness)


if __name__=='__main__':unittest.main()
