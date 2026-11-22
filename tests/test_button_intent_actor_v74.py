"""Actual factory with CPU shape ports; source commands are not completion labels."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import json,unittest
import numpy as np
import tempfile
from unittest.mock import patch
from test_multitask_training_geometry import fixture as geometry_fixture
from test_multitask_interaction import CartesianRobot
from our_mani.multitask import training_geometry
from our_mani.contracts import Pose,RobotState,ObjectState
from our_mani.multitask.interaction import PolicyState,_world_target
from our_mani.multitask.button_intent_library_v71 import ButtonIntentLibrary
from our_mani.multitask.button_intent_actor_v74 import binding,create_actor

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/framework/multitask_r10'


class ScenePort:
    required_dynamic_role_ids=('button','plate','wrap')
    def __init__(self):self.registrations={};self.attachments={}
    def register_estimated_geometry(self,identity,vertices,triangles,**provenance):self.registrations[identity]=provenance
    def check(self,poses=None,**kwargs):return False
    def bind_estimated_attachment(self,identity,pose,**kwargs):
        if pose is None:self.attachments.pop(identity,None)
        else:self.attachments[identity]=pose


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='button_author_factory_CPU_',dir=ROOT/'runtime/framework/tmp')
        self.path=Path(self.tmp.name);self.env,source,parts=geometry_fixture()
        self.env.map_service=ScenePort();self.env.robot_model=CartesianRobot()
        asset=training_geometry.export_train_geometry(self.env,source,surface_samples=24,shape_type='shape')
        (self.path/'geometry.json').write_text(json.dumps(asset))
        for part in parts:part.forbid_pose=True;part.handle+=100
        original=training_geometry.declared_instance_bindings
        self.binding_patch=patch.object(training_geometry,'declared_instance_bindings',
            side_effect=lambda env,asset,**kw:original(env,asset,shape_type='shape',**kw))
        self.binding_patch.start()
    def tearDown(self):self.binding_patch.stop();self.tmp.cleanup()
    def pointer(self,name):
        from our_mani.multitask.evaluation import file_sha
        p=self.path/name;return dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p))
    def actor(self):
        task='push_moving_button';c=json.loads((BASE/'task_executor_smoke_v69_setup/configuration.json').read_text())
        c['learned_artifacts'][task]['geometry']=self.pointer('geometry.json')
        c['learned_artifacts'][task]['rigid_layout']=None
        p=BASE/'button_author_action_intents_v70'/task/'model.json'
        from our_mani.multitask.evaluation import file_sha
        c['learned_artifacts'][task]['interaction']=dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p))
        c['policy_overlay']['configurations']['causal_retarget']['maximum_reissues_per_phase']=16
        c['continuation_identities'].setdefault(task,{})['cv']=binding(ROOT,c,task,'cv')['sha256']
        return create_actor(ROOT,self.env,c,'cv_geometry',['push the moving red button'],purpose='train_collection')[0]

    def test_actual_factory_installs_final_controller_guard_and_press_prefix_with_original_executor(self):
        actor=self.actor();c=actor.controller;m=actor.continuous_reference_manager
        self.assertIs(c._phase_done.__self__,c);self.assertIs(m.controller,c)
        self.assertEqual(m.approach_transit.normalized['maximum_TCP_step_m'],.08)
        self.assertEqual(m.config.dt,.05);self.assertEqual(m.retarget.config.maximum_reissues_per_phase,16)
        self.assertFalse(c.library.metadata['completed_skill_support_claimed'])
        self.assertFalse(c.library.metadata['public_press_completion_granted'])
        self.assertEqual(c.library.metadata['independent_command_parent_support']['press'],95)

    def test_source_distance_guard_changes_state_without_claiming_press_or_completed_geometry(self):
        actor=self.actor();c=actor.controller;template=c.library.for_phase('approach')[0]
        body=ObjectState('button',1.,Pose([.2,0.,.8]),np.tile([.2,0.,.8],(3,1)),np.zeros(3),np.zeros(3))
        target=_world_target(body.pose,np.zeros(3),np.zeros(3),template,1.)[0]
        pose=Pose(target.position+np.array([.29,0.,0.]),target.rotation)
        robot=RobotState(1.,pose,np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7),gripper_open=True,contact=False)
        state=PolicyState(robot,{'button':body},{'bodies':{'button':dict(directly_visible=True,registration_valid=True,motion_mode='external')}},{'button':('button',)})
        c.goal=SimpleNamespace(phase='approach',template=template,reference_id='button')
        self.assertTrue(c._phase_done(state,target,public_geometry=True))
        c._complete_phase(state);self.assertEqual(c.phase_index,1)
        self.assertFalse(actor.metrics_report()['button_author_intent_v74']['source_state_events'][0]['verified_completed_skill_claimed'])
        c.goal=SimpleNamespace(phase='press',template=c.library.for_phase('press')[0],reference_id='button')
        self.assertFalse(c._phase_done(state,target,public_geometry=True))

    def test_candidate_model_refuses_formal_purpose_before_any_native_or_policy_use(self):
        path=BASE/'button_author_action_intents_v70/push_moving_button/model.json'
        with self.assertRaisesRegex(ValueError,'cannot enter formal'):
            ButtonIntentLibrary(ROOT,path,purpose='evaluation')


if __name__=='__main__':unittest.main()
