"""Real factory/actor wiring with explicit CPU production-shaped fake artifacts.

These temporary artifacts exercise loaders, not real robot training. They
remain under desktop runtime and must never enter formal TRAIN/VAL manifests.
No native simulator/GPU is imported or launched by the fake shape/map ports.
"""
from copy import deepcopy
from dataclasses import asdict,replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from our_mani.contracts import Pose,RobotState
from our_mani.perception import CameraCalibration
from our_mani.multitask import factory,training_geometry
from our_mani.multitask.candidate_success import (ContinuationAttempts,FEATURE_NAMES,
    FEATURE_SCHEMA_SHA256,LABEL_SCOPE,CONTINUATION_POLICY)
from our_mani.multitask.candidate_training import (CriticFitConfig,MANIFEST_SCHEMA,
    fit_phase_head,load_executed_manifest)
from our_mani.multitask.interaction import fit_interaction_library,InteractionFitConfig
from our_mani.multitask.public_roles import TRAINColourSample,ColourFitConfig,fit_colour_artifact
from our_mani.multitask.protocol import MultitaskProtocolError
from test_multitask_interaction import synthetic_sources,candidates_config,mpc_config,CartesianRobot
from test_multitask_training_geometry import fixture
from test_multitask_completed_skill_fitting import completed_source

ROOT=Path(__file__).resolve().parents[1]


class ScenePort:
    required_dynamic_role_ids=('button','plate','wrap')
    def __init__(self):
        self.registrations={}
        self.attachments={}
    def register_estimated_geometry(self,identity,vertices,triangles,**provenance):
        self.registrations[identity]=provenance
    def check(self,poses=None,**kwargs):
        return False
    def bind_estimated_attachment(self,identity,pose,**kwargs):
        if pose is None:self.attachments.pop(identity,None)
        else:self.attachments[identity]=pose


def write(path,payload):
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')


class FactoryWiringChecks(unittest.TestCase):
    def setUp(self):
        runtime=ROOT/'runtime/framework/tmp'
        runtime.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='factory_wiring_CPU_only_',dir=runtime)
        self.path=Path(self.temporary.name)
        self.env,source,parts=fixture()
        self.env.map_service=ScenePort()
        self.env.robot_model=CartesianRobot()
        self.parts=parts
        self.asset=training_geometry.export_train_geometry(self.env,source,surface_samples=24,shape_type='shape')
        write(self.path/'geometry.json',self.asset)
        sources=[]
        for index in range(8):
            original=completed_source(index)
            raw='CPU_fake_parent_'+str(index)
            sources.append(replace(original,source_kind='TRAIN_demonstration',source_id=raw,
                raw_episode_id=raw,parent_raw_episode_id=None,augmentation_id=None,
                parent_source_group='train:push_moving_button:'+raw))
        self.train_entries=[dict(split='train',task='push_moving_button',source_id=s.source_id,
                                 raw_episode_id=s.raw_episode_id,CPU_fixture_only=True) for s in sources]
        write(self.path/'interaction_TRAIN_manifest.json',dict(entries=self.train_entries,CPU_fixture_only=True))
        self.library=fit_interaction_library(sources,InteractionFitConfig(3,(0.,),1.,.9,2,.01,.05,1.,
            minimum_completed_parent_sources=8,maximum_position_tolerance_m=.03,
            maximum_orientation_tolerance_rad=.15,development_only=False),
            TRAIN_source_manifest_path=self.path/'interaction_TRAIN_manifest.json')
        self.library.save(self.path/'interaction.json')
        pixels={'red':np.tile([220.,10.,10.],(16,1)),'blue':np.tile([10.,10.,220.],(16,1))}
        samples=[TRAINColourSample('push_moving_button',source.parent_source_group,'button',
                    ('push_button_target',),name,values) for source in sources for name,values in pixels.items()]
        colour=fit_colour_artifact(samples,ColourFitConfig(2,.9,.05,.05,5,.1),
            TRAIN_manifest_path=self.path/'interaction_TRAIN_manifest.json')
        write(self.path/'colour.json',colour)
        self.protocol=dict(dt=.05,simulation_seconds=10.,gripper_velocity=.2,global_joint_acceleration=6.,
            attachment_association_distance_m=.15,shared_semantic_instance_binding=True,
            gripper_timing_config=dict(close_position_tolerance_m=.02,close_orientation_tolerance_rad=.10),
            joint_limit_recovery_margin_rad=.005,
            semantic_binding_available_to_baseline=True,shared_TRAIN_CAD_geometry=True,
            information_setting='CPU fixture only; shared semantic/CAD ports',
            body_orientation_observability={'push_moving_button':{'button':'full','plate':'full','wrap':'full'}},
            candidate_config=asdict(candidates_config()),mpc_config=asdict(mpc_config()),
            context_checkpoint={'path':'runtime/NEVER_READ_CPU_context.pt','sha256':'0'*64},
            learned_artifacts={'push_moving_button':{name:self.pointer(name+'.json')
                for name in ('geometry','interaction','colour')}})
        # Simulate a different fresh process's current instance IDs. Every
        # pose access now raises; name/handle binding alone remains legal.
        self.env.development_split='validation'
        for part in self.parts:
            part.forbid_pose=True
            part.handle+=100
            part.get_pose=lambda: (_ for _ in ()).throw(AssertionError('native body pose leaked into policy factory'))
            part.get_velocity=lambda: (_ for _ in ()).throw(AssertionError('native body velocity leaked into policy factory'))
        original=training_geometry.declared_instance_bindings
        self.binding_patch=patch.object(training_geometry,'declared_instance_bindings',
            side_effect=lambda env,asset,**kwargs:original(env,asset,shape_type='shape',**kwargs))
        self.binding_patch.start()

    def tearDown(self):
        self.binding_patch.stop()
        self.temporary.cleanup()

    def pointer(self,name):
        path=self.path/name
        return dict(path=str(path.relative_to(ROOT)),sha256=factory.file_sha(path))

    def create(self,method='cv_geometry',**kwargs):
        return factory.create_actor(ROOT,self.env,self.protocol,method,['push the moving red button'],**kwargs)

    def fit_fake_critic(self,continuation):
        train=[];validation=[]
        for split,output in (('train',train),('validation',validation)):
            for index in range(8):
                x=np.zeros(len(FEATURE_NAMES));x[0]=1. if index%2 else -1.
                output.append(dict(x=x,y=index%2,group=f'{split}:push_moving_button:episode{index}',phase='approach'))
        head,_=fit_phase_head(train,validation,CriticFitConfig(regularization_grid=(.1,),TRAIN_folds=2))
        train_entries=[dict(split='train',task='push_moving_button',raw_episode_id=f'episode{i}',CPU_fixture_only=True) for i in range(8)]
        val_entries=[dict(split='validation',task='push_moving_button',raw_episode_id=f'episode{i}',CPU_fixture_only=True) for i in range(8)]
        write(self.path/'critic_TRAIN.json',dict(entries=train_entries,CPU_fixture_only=True))
        write(self.path/'critic_VAL.json',dict(entries=val_entries,CPU_fixture_only=True))
        metadata=dict(training_complete=True,classifier_fit_split='train',classifier_selection_split='train',
            normalization_fit_split='train',TEST_used_for_fitting=False,label_scope=LABEL_SCOPE,
            continuation_selection_policy=CONTINUATION_POLICY,feature_schema_sha256=FEATURE_SCHEMA_SHA256,
            continuation_sha256=continuation['sha256'],interaction_library_sha256=self.library.sha256,
            TRAIN_source_manifest_path=str(self.path/'critic_TRAIN.json'),TRAIN_source_manifest_sha256=factory.file_sha(self.path/'critic_TRAIN.json'),
            TRAIN_parent_source_grouping=['split','task','parent_raw_episode_id_or_raw_episode_id'],
            TRAIN_parent_source_groups=[f'train:push_moving_button:episode{i}' for i in range(8)],
            VAL_calibration_manifest_path=str(self.path/'critic_VAL.json'),VAL_calibration_manifest_sha256=factory.file_sha(self.path/'critic_VAL.json'),
            classifier_selection_uses_VAL=False,calibration_selector_used_VAL=True,synthetic=False,CPU_fixture_only=True)
        write(self.path/'critic.json',dict(metadata=metadata,heads={'push_moving_button|cv|approach':head}))
        return self.pointer('critic.json')

    def test_real_factory_constructs_CV_actor_without_native_pose_or_context_reads(self):
        actor,identity=self.create()
        self.assertIsNone(actor.bridge.context)
        self.assertNotIn('context',identity['artifact_pointers'])
        self.assertEqual(actor.first_selection_policy,'geometry')
        self.assertEqual(set(self.env.map_service.registrations),{'button','plate','wrap'})
        self.assertEqual(actor.templates[0].components[0].instance_ids[0] >= 121,True)
        # Actual tracker/bridge/actor ports receive an empty but complete current
        # RGB-D frame and hold; no fake visible pose is substituted.
        camera=CameraCalibration(np.array([[100.,0.,8.],[0.,100.,8.],[0.,0.,1.]]),Pose(np.zeros(3)))
        names=('front','8','16','36')
        observation={'timestamp':0.,'observation/depths':{n:np.zeros((16,16)) for n in names},
            'observation/images':{n:np.zeros((16,16,3),np.uint8) for n in names},
            'observation/instance_masks':{n:np.zeros((16,16),np.int32) for n in names},
            'observation/calibrations':{n:camera for n in names}}
        robot=RobotState(0.,Pose(np.zeros(3)),np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7))
        command=actor.act(observation,robot,10.)
        self.assertEqual(command['command'],'hold')
        self.assertFalse(actor.bridge.last_forecast.object_ids)
        self.assertEqual(set(actor.bridge.last_forecast.metadata['missing_body_ids']),{'button','plate','wrap'})

    def test_critic_added_after_fit_keeps_full_continuation_identity_and_value_loads(self):
        geometry,identity=self.create()
        self.protocol['continuation_identities']={'push_moving_button':{'cv':identity['sha256']}}
        self.protocol['learned_artifacts']['push_moving_button']['cv_value']=self.fit_fake_critic(identity)
        value,loaded=self.create('cv_value')
        self.assertEqual(loaded['sha256'],identity['sha256'])
        self.assertEqual(value.first_selection_policy,'value_once')
        self.assertIsNotNone(value.controller.success_model)
        self.protocol['attachment_association_distance_m']=.16
        with self.assertRaisesRegex(MultitaskProtocolError,'continuation identity changed'):
            self.create()

    def test_full_identity_binds_effective_sphere_fit_prior_but_not_critic_pointer(self):
        first=factory.continuation_identity(ROOT,self.protocol,'push_moving_button','cv',
            self.library,candidates_config(),mpc_config())
        self.protocol['body_sphere_radii']={'push_moving_button':{'button':.03}}
        changed=factory.continuation_identity(ROOT,self.protocol,'push_moving_button','cv',
            self.library,candidates_config(),mpc_config())
        self.assertNotEqual(first['sha256'],changed['sha256'])
        self.protocol['learned_artifacts']['push_moving_button']['cv_value']={'path':'not-yet-fitted.json','sha256':'f'*64}
        critic_added=factory.continuation_identity(ROOT,self.protocol,'push_moving_button','cv',
            self.library,candidates_config(),mpc_config())
        self.assertEqual(changed['sha256'],critic_added['sha256'])

    def test_full_identity_binds_actual_child_and_shared_sensor_sources(self):
        first=factory.continuation_identity(ROOT,self.protocol,'push_moving_button','cv',
            self.library,candidates_config(),mpc_config())
        required=('scripts/evaluate_multitask_case_r10.py',
                  'our_mani/sensor_alignment/author_rgbd.py',
                  'our_mani/multitask/actuation.py',
                  'our_mani/multitask/evaluation.py')
        for name in required:
            self.assertIn(name,first['observation_control_sources'])
        actual_sha=factory.file_sha
        for changed_source in required:
            with self.subTest(source=changed_source):
                with patch.object(factory,'file_sha',side_effect=lambda path:
                        'f'*64 if Path(path)==ROOT/changed_source else actual_sha(path)):
                    changed=factory.continuation_identity(ROOT,self.protocol,'push_moving_button','cv',
                        self.library,candidates_config(),mpc_config())
                self.assertNotEqual(first['sha256'],changed['sha256'])

    def test_collector_rigid_layout_digest_loads_and_tamper_rejected(self):
        spec=importlib.util.spec_from_file_location('collector_factory_fixture',ROOT/'scripts/collect_multitask_training_r10.py')
        collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
        arrays=dict(times=np.arange(6)*.05,body_ids=np.asarray(['button','wrap']),
            target_positions=np.zeros((6,2,3)),target_rotations=np.tile(np.eye(3),(6,2,1,1)),target_valid=np.ones((6,2),bool))
        arrays['target_positions'][:,1,0]=.1
        layout=collector.fit_rigid_layout(arrays,task='push_moving_button',source_group='train:push_moving_button:CPU_中文_parent')
        path=self.path/'layout.json';write(path,layout)
        loaded=factory.load_rigid_layout(path,'push_moving_button')
        self.assertEqual(loaded[0].child_role,'wrap')
        np.testing.assert_allclose(loaded[0].child_to_parent.position,[.1,0.,0.])
        layout['accepted'][0]['child_to_parent_position'][0]=.2
        write(path,layout)
        with self.assertRaises(MultitaskProtocolError):factory.load_rigid_layout(path,'push_moving_button')

    def test_executed_critic_loader_rejects_value_once_policy_class(self):
        ledger=ContinuationAttempts(task_name='push_moving_button',predictor='cv',
            continuation_sha256='controller',library_sha256='library')
        ledger.begin('approach','candidate',0.,np.zeros(len(FEATURE_NAMES)))
        attempts=ledger.finish(False,.2,'native_budget_failure')
        case=dict(task='push_moving_button',predictor='cv',split='train',raw_episode_id='CPU_case',
            complete=True,infrastructure_error=None,official_task_success=False,
            continuation_sha256='controller',interaction_library_sha256='library',
            continuation_selection_policy=CONTINUATION_POLICY,downstream_selection_policy='geometry',
            unexecuted_candidates_labelled=False,first_selection_policy='value_once',continuation_attempts=attempts)
        path=self.path/'value_case.json';write(path,case)
        entry=dict(split='train',task='push_moving_button',predictor='cv',raw_episode_id='CPU_case',
            path=str(path),sha256=factory.file_sha(path))
        manifest=self.path/'executed.json';write(manifest,dict(schema=MANIFEST_SCHEMA,entries=[entry]))
        with self.assertRaisesRegex(ValueError,'frozen geometry downstream'):
            load_executed_manifest(manifest,split='train',task_name='push_moving_button',predictor='cv',
                continuation_sha256='controller',library_sha256='library')


if __name__ == '__main__':
    unittest.main()
