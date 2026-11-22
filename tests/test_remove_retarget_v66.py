"""Actual source-qualified library and manager, CPU kinematics only."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from copy import deepcopy
import json,unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.contracts import Pose,RobotState,ObjectState
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.reachability import TimedReachability
from our_mani.multitask.interaction import (InteractionLibrary,CandidateGenerator,CandidateConfig,InteractionController,
    SE3InteractionMPC,MPCConfig,GripperTimingConfig,PolicyState)
from our_mani.multitask.feedback_interaction import ObservedFeedbackController
from our_mani.multitask.causal_retarget_interaction import CausalRetargetManager
from our_mani.multitask.remove_retarget_v66 import binding,install

ROOT=Path(__file__).resolve().parents[1]


class Cartesian:
    name='CPU_only_remove_retarget_kinematics'
    lower,upper=np.full(7,-10.),np.full(7,10.)
    velocity_limits,acceleration_limits=np.full(7,100.),np.full(7,10000.)
    def inverse(self,pose,seed):return np.r_[pose.position,Rotation.from_matrix(pose.rotation.copy()).as_rotvec(),0.]
    def forward(self,q):return Pose(q[:3],Rotation.from_rotvec(q[3:6]).as_matrix())
    def jacobian(self,q):return np.c_[np.eye(6),np.zeros(6)]
    def collision(self,q,t):self.queries+=1;return False
    def __init__(self):self.queries=0


class Scene:
    def __init__(self):self.blocked=False;self.bindings=[]
    def bind_candidate(self,**kwargs):self.bindings.append(kwargs)
    def __call__(self,**kwargs):return dict(feasible=not self.blocked,reason='explicit CPU scene fixture',evidence={})


class Tests(unittest.TestCase):
    def fixture(self):
        c=json.loads((ROOT/'artifacts/framework/multitask_r10/remove_rigid_source_v59_setup/configuration.json').read_text())
        task='remove_cups_from_rotating_frame';library=InteractionLibrary.load(ROOT/c['learned_artifacts'][task]['interaction']['path'])
        model=Cartesian();scene=Scene();config=dict(c['candidate_config']);config.update(maximum_linear_speed=100.,maximum_angular_speed=100.)
        generator=CandidateGenerator(library,TimedReachability(model),scene,CandidateConfig(**config))
        controller=ObservedFeedbackController.from_controller(InteractionController(library,generator,SE3InteractionMPC(MPCConfig(**c['mpc_config'])),
            predictor='cv',ranking='geometry',continuation_sha256='CPU_fixture_only',gripper_timing_config=GripperTimingConfig(**c['gripper_timing_config'])),
            {'phase_tracking_slack_seconds':0.})
        local=library.for_phase('approach')[0].at(0.)[0];self.q=np.r_[local.position,Rotation.from_matrix(local.rotation.copy()).as_rotvec(),0.]
        state,forecast=self.observed(0.);controller.step(state,forecast,10.);self.assertIsNotNone(controller.goal)
        manager=CausalRetargetManager(controller,dict(minimum_issue_interval_seconds=.1,scheduled_issue_interval_seconds=.1,
            reference_shift_threshold_m=.0001,arrival_delays_seconds=(.3,.15,.05)))
        actor=SimpleNamespace(continuous_reference_manager=SimpleNamespace(retarget=manager),metrics_report=lambda:{})
        c['remove_retarget_v66']={task:binding(ROOT,c,task)};install(actor,c,ROOT,task)
        return controller,manager,model,scene

    def observed(self,time):
        q=self.q;robot=RobotState(time,Pose(q[:3],Rotation.from_rotvec(q[3:6]).as_matrix()),np.zeros(3),np.zeros(3),q,np.zeros(7))
        bodies={role:ObjectState(role,time,Pose([.1*time,0.,0.]),np.tile([.1*time,0.,0.],(3,1)),np.array([.1,0.,0.]),np.zeros(3))
            for role in ('cup0','spoke0')}
        state=PolicyState(robot,bodies,{'bodies':{role:dict(directly_visible=True,registration_valid=True,orientation_observable=True,motion_mode='external') for role in bodies}},
            {'selected_cup':('cup0',),'selected_spoke':('spoke0',)})
        f=ConstantVelocityDynamics().predict(list(bodies.values()),1.5,.05);f=replace(f,metadata={**f.metadata,'observed_until':float(time)})
        return state,f

    def test_actual_reissue_keeps_attempt_epoch_deadline_and_uses_full_original_gates(self):
        controller,manager,model,scene=self.fixture();old=controller.goal;records=deepcopy(controller.attempts.to_report())
        state,forecast=self.observed(.15);event=manager.before_step(state,forecast,9.85)
        self.assertEqual(event['action'],'new_candidate_issue',event)
        self.assertIs(controller.goal.template,old.template)
        self.assertEqual(controller.goal.selected_at,old.selected_at)
        self.assertEqual(controller.goal.phase_budget,old.phase_budget)
        self.assertEqual(controller.goal.suffix_budget,old.suffix_budget)
        self.assertEqual(controller.attempts.to_report(),records)
        self.assertFalse(event['phase_completion_claimed']);self.assertGreater(model.queries,0);self.assertTrue(scene.bindings)

    def test_rejected_dynamic_scene_preserves_original_candidate_and_budget(self):
        controller,manager,model,scene=self.fixture();old=controller.goal;scene.blocked=True
        state,forecast=self.observed(.15);event=manager.before_step(state,forecast,9.85)
        self.assertEqual(event['action'],'reissue_rejected_original_candidate_retained',event)
        self.assertIs(controller.goal,old);self.assertFalse(event['phase_clock_reset'])


if __name__=='__main__':unittest.main()
