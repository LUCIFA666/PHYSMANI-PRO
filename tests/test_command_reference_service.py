from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import json
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import test_remove_retarget_v66 as ports
from our_mani.contracts import Pose,RobotState,ObjectState
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.reachability import TimedReachability
from our_mani.multitask.interaction import (InteractionLibrary,CandidateGenerator,CandidateConfig,
    InteractionController,InteractionCandidate,SE3InteractionMPC,MPCConfig,PolicyState,_world_target,GripperTimingConfig)
from our_mani.multitask.feedback_interaction import ObservedFeedbackController
from our_mani.multitask.causal_retarget_interaction import CausalRetargetManager
from our_mani.multitask.candidate_success import FEATURE_NAMES
from our_mani.multitask.command_reference_service import install,binding
ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def fixture(self,task,phase):
        config=json.loads((ROOT/'artifacts/framework/multitask_r10/registered_tasks_v408_setup'/
            (task+'_configuration.json')).read_text())
        library=InteractionLibrary.load(ROOT/config['learned_artifacts'][task]['interaction']['path'])
        t=library.for_phase(phase)[0]; model,scene=ports.Cartesian(),ports.Scene()
        gate=TimedReachability(model)
        generator=CandidateGenerator(library,gate,scene,CandidateConfig(**dict(config['candidate_config'],
            maximum_linear_speed=100.,maximum_angular_speed=100.)))
        c=ObservedFeedbackController.from_controller(InteractionController(library,generator,
            SE3InteractionMPC(MPCConfig(**config['mpc_config'])),predictor='cv',ranking='geometry',
            continuation_sha256='CPU_buzz_command_reference',
            gripper_timing_config=GripperTimingConfig(**config['gripper_timing_config'])),{'phase_tracking_slack_seconds':0.})
        c.phase_index=library.phase_order.index(phase)
        target=_world_target(Pose(np.zeros(3)),np.zeros(3),np.zeros(3),t,1.)[0]
        self.q=model.inverse(Pose(target.position+[.05,0.,0.],target.rotation),np.zeros(7))
        state,forecast=self.observed(0.,t.reference_role)
        checked=gate.check_joints(state.robot,model.inverse(target,self.q),np.zeros(7),.3)
        c.goal=InteractionCandidate('CPU_buzz_command',phase,c.phase_index,t.reference_role,('wand',),t,
            0.,.3,target,np.zeros(3),np.zeros(3),Pose(np.zeros(3)),np.zeros(3),np.zeros(3),
            state.bodies[t.reference_role],checked,0.,t.duration_seconds,9.,0.,True,{})
        c.attempts.begin(phase,c.goal.candidate_id,0.,np.zeros(len(FEATURE_NAMES)))
        m=CausalRetargetManager(c,dict(minimum_issue_interval_seconds=.1,scheduled_issue_interval_seconds=.1,
            arrival_delays_seconds=(1.1,.9,.7,.5,.3,.15,.05)))
        actor=NS(controller=c,continuous_reference_manager=NS(retarget=m),metrics_report=lambda:{})
        config['command_reference_service']=binding(ROOT,config,task)
        install(actor,config,ROOT,task)
        return actor,c,m,model,scene

    def observed(self,time,role):
        robot=RobotState(time,Pose(self.q[:3],Rotation.from_rotvec(self.q[3:6]).as_matrix()),
            np.zeros(3),np.zeros(3),self.q.copy(),np.zeros(7),gripper_open=False,contact=True)
        body=ObjectState(role,time,Pose([.1*time,0.,0.]),np.tile([.1*time,0.,0.],(3,1)),
            np.array([.1,0.,0.]),np.zeros(3))
        state=PolicyState(robot,{role:body},{'bodies':{role:dict(directly_visible=True,
            registration_valid=True,orientation_observable=True,motion_mode='external')}},{role:(role,)})
        forecast=ConstantVelocityDynamics().predict([body],1.1,.05)
        forecast.metadata['observed_until']=float(time)
        return state,forecast

    def test_all_actual_buzz_phase_templates_keep_fresh_canonical_completion_and_budget(self):
        for task in ('beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed'):
            for phase in ('approach','acquire','follow','release'):
                with self.subTest(task=task,phase=phase):
                    actor,c,m,_,_=self.fixture(task,phase);old=c.goal;attempts=deepcopy(c.attempts.to_report())
                    canonical=old.template.completion_geometry_xyzw.copy()
                    state,forecast=self.observed(.15,old.reference_id)
                    event=m.before_step(state,forecast,9.85)
                    self.assertEqual(event['action'],'new_candidate_issue',event)
                    self.assertIs(c.goal.template,old.template)
                    np.testing.assert_array_equal(c.goal.template.completion_geometry_xyzw,canonical)
                    self.assertEqual((c.goal.selected_at,c.goal.phase_budget,c.goal.suffix_budget),
                        (old.selected_at,old.phase_budget,old.suffix_budget))
                    self.assertEqual(c.attempts.to_report(),attempts)
                    self.assertEqual(m.metrics()['normalized_config']['maximum_reissues_per_phase'],16)

    def test_collision_stale_forecast_and_attachment_retain_original_command(self):
        for condition in ('collision','stale','attached'):
            actor,c,m,_,scene=self.fixture('beat_the_rotating_buzz','approach');old=c.goal
            state,forecast=self.observed(.15,old.reference_id)
            if condition=='collision':scene.blocked=True
            if condition=='stale':forecast.metadata['observed_until']=-.2
            if condition=='attached':state.perception_metadata['bodies'][old.reference_id]['motion_mode']='attached'
            event=m.before_step(state,forecast,9.85)
            self.assertEqual(event['action'],'reissue_rejected_original_candidate_retained')
            self.assertIs(c.goal,old)


if __name__=='__main__':unittest.main()
