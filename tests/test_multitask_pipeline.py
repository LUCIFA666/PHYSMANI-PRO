"""CPU actor wiring mocks; no native simulator, teacher or task outcome."""
from types import SimpleNamespace
import threading
import unittest

import numpy as np

from our_mani.contracts import ObjectState,Pose,RobotState
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.multitask.forecasting import MultiBodyForecastBridge
from our_mani.multitask.interaction import (PolicyState,TRAINUniformFirstCandidateSampler,
    TRAINReferenceBinding,TRAINBodyTrack,TRAINInteractionSource,fit_interaction_library,
    InteractionFitConfig,CandidateConfig)
from our_mani.multitask.pipeline import PipelineActor,PipelineInputError,PhaseCollisionPort
from our_mani.multitask.specs import get_task_spec
from tests.test_multitask_interaction import (synthetic_library,CartesianRobot,candidates_config,mpc_config)


class SceneMap:
    required_dynamic_role_ids=('button','plate','wrap')
    def __init__(self):
        self.checks,self.attachments=[],{}
        self.failure=None
    def check(self,poses=None,*,ignored_object_ids=(),allowed_robot_contact_ids=()):
        if self.failure:
            raise self.failure
        self.checks.append(dict(body_ids=tuple(sorted(poses)),ignored=tuple(ignored_object_ids),
                               permitted=tuple(allowed_robot_contact_ids)))
        return False
    def bind_estimated_attachment(self,identity,pose,*,provenance):
        if pose is None:
            self.attachments.pop(identity,None)
        else:
            self.attachments[identity]=pose


class ProbeRobot(CartesianRobot):
    collision_query=None
    def collision(self,joints,future_offset):
        return bool(self.collision_query(future_offset)) if self.collision_query else True


class Tracker:
    def __init__(self):
        self.missing=set()
    def observe(self,observation,*,modes):
        now=observation['timestamp'];bodies=[];metadata={}
        for index,body in enumerate(SceneMap.required_dynamic_role_ids):
            if body in self.missing:
                continue
            pose=Pose(np.zeros(3) if index == 0 else np.array([index*.3,.3,0.]))
            bodies.append(ObjectState(body,now,pose,np.zeros((3,3)),np.zeros(3),np.zeros(3)))
            metadata[body]=dict(visible=True,directly_visible=True,registration_valid=True,
                orientation_observable=True,orientation_observability='full',motion_mode=modes[body])
        return bodies,dict(bodies=metadata,rigid_groups=[])


class PublicResolver:
    def resolve(self,descriptions,state,observation,templates):
        return {**state.role_bindings,'selected_button':('button',)},dict(source='public_language_RGBD_mock')


class ContextMock:
    metadata=dict(training_complete=True,TEST_used_for_fitting=False,state_schema='synthetic_fixture')
    checkpoint_sha256='synthetic_context_fixture'
    maximum_validated_horizon=1.1
    def __init__(self):
        self.history={};self.lock=threading.RLock()
    def reset(self):
        self.history.clear()
    def observe(self,objects,**kwargs):
        for body in objects:
            self.history.setdefault(body.object_id,[]).append(body)
    def predict(self,objects,horizon,dt):
        result=ConstantVelocityDynamics().predict(objects,horizon,dt)
        result.metadata['contexts']={body.object_id:[0.]*16 for body in objects}
        return result


def make_actor(*,method='cv',purpose='evaluation',sampler=None,joint_margin=None):
    spec=get_task_spec('push_moving_button');tracker=Tracker();scene=SceneMap()
    templates=[SimpleNamespace(role_id=body) for body in scene.required_dynamic_role_ids]
    bridge=MultiBodyForecastBridge(scene.required_dynamic_role_ids,method=method,
        context=ContextMock() if method == 'context' else None,task_spec=spec)
    actor=PipelineActor(spec,templates,tracker,bridge,synthetic_library(),ProbeRobot(),scene,
        candidates_config(),mpc_config(),predictor=method,ranking='geometry',descriptions=['push red button'],
        role_resolver=PublicResolver(),attachment_association_distance_m=.1,
        purpose=purpose,first_candidate_sampler=sampler,allow_synthetic=True,
        joint_limit_recovery_margin_rad=joint_margin)
    return actor,scene,tracker


def observation(time):
    return {'timestamp':time,'observation/images':{},'observation/depths':{},
            'observation/instance_masks':{},'observation/calibrations':{}}


def robot(time,z=.1,contact=False):
    pose=Pose(np.array([0.,0.,z]))
    return RobotState(time,pose,np.zeros(3),np.zeros(3),np.r_[pose.position,np.zeros(4)],
                      np.zeros(7),True,contact,False)


class PipelineChecks(unittest.TestCase):
    def test_only_frozen_TRAIN_noncolliding_roles_can_omit_collision_forecast(self):
        spec=get_task_spec('push_moving_button');scene=SceneMap();model=ProbeRobot()
        scene.noncolliding_geometry={'wrap':dict(provenance='TRAIN_geometry_model',source_sha256='a'*64,
            geometry_rule='no owned collidable OR respondable component in TRAIN asset')}
        port=PhaseCollisionPort(spec,scene,model,maximum_observation_age=.1)
        states,meta=Tracker().observe(observation(0.),modes={name:'external' for name in scene.required_dynamic_role_ids})
        state=PolicyState(robot(0.),{body.object_id:body for body in states},meta,{})
        forecast=ConstantVelocityDynamics().predict([body for body in states if body.object_id!='wrap'],1.,.05)
        port.configure_frame(state,forecast)
        self.assertTrue(port.readiness(state,forecast,'approach')['ready'])
        self.assertIn('wrap',port.required)
        self.assertFalse(port.check_future(.1))
        self.assertEqual(set(scene.checks[-1]['body_ids']),{'button','plate'})
        self.assertIn('wrap',port.receipt()['TRAIN_noncolliding_geometry'])
        scene.noncolliding_geometry={}
        self.assertTrue(port.check_future(.1)) # a missing physical role is unsafe
        scene.noncolliding_geometry={'unknown':dict(provenance='TRAIN_geometry_model',source_sha256='a'*64,
            geometry_rule='no owned collidable OR respondable component in TRAIN asset')}
        self.assertTrue(port.check_future(.1)) # foreign declaration cannot erase required roles

    def test_established_RGBD_attachment_can_complete_acquire_after_occlusion(self):
        spec=get_task_spec('insert_onto_rotating_peg')
        stages=tuple(stage for stage in ('wp0','wp1-grasp','wp2','wp3','wp4') for _ in range(2))
        times=np.arange(len(stages))*.05;poses=np.tile([0.,0.,0.,0.,0.,0.,1.],(len(times),1))
        binding=TRAINReferenceBinding('ring','ring','observed_se3','fixture_anchor','ring')
        sources=[]
        for index in range(2):
            track=TRAINBodyTrack('ring',times,poses,('external',)*len(times),np.ones(len(times),bool))
            sources.append(TRAINInteractionSource(spec.task_name,str(index),'train','synthetic',times,
                poses,np.r_[np.ones(2),np.zeros(8)],poses,stages,(binding,)*len(times),{'ring':track}))
        library=fit_interaction_library(sources,InteractionFitConfig(3,(0.,),1.,.9,2,.01,.05,1.))
        scene=SceneMap();scene.required_dynamic_role_ids=('ring','frame','peg0','peg1','peg2')
        class RingTracker:
            hide=False
            def observe(self,obs,*,modes):
                now=obs['timestamp'];bodies=[];metadata={}
                for identity in scene.required_dynamic_role_ids:
                    visible=not(identity == 'ring' and self.hide)
                    bodies.append(ObjectState(identity,now,Pose(np.zeros(3)),np.zeros((3,3)),
                                              np.zeros(3),np.zeros(3),1.,visible))
                    metadata[identity]=dict(visible=visible,directly_visible=visible,
                        registration_valid=visible,orientation_observable=True,
                        orientation_observability='full',motion_mode=modes[identity])
                return bodies,dict(bodies=metadata,rigid_groups=[])
        bridge=MultiBodyForecastBridge(scene.required_dynamic_role_ids,task_spec=spec)
        tracker=RingTracker()
        resolver=SimpleNamespace(resolve=lambda descriptions,state,obs,templates:
            ({**state.role_bindings,'selected_peg':('peg0',)},{}))
        config=CandidateConfig((.05,.1),.05,1.,2.,.1,.2,.1,.02,.02,1.)
        actor=PipelineActor(spec,[SimpleNamespace(role_id=body) for body in scene.required_dynamic_role_ids],
            tracker,bridge,library,ProbeRobot(),scene,config,mpc_config(),predictor='cv',ranking='geometry',
            descriptions=['put ring on red spoke'],role_resolver=resolver,
            attachment_association_distance_m=.1,allow_synthetic=True)
        contact=False
        for tick in range(15):
            now=tick*.05
            tracker.hide=contact and bool(scene.attachments)
            public=RobotState(now,Pose(np.zeros(3)),np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7),
                              not contact,contact,False)
            command=actor.act(observation(now),public,10.-now)
            if command.get('phase') == 'acquire' and command.get('opening') == 0.:
                contact=True
            if actor.controller.phase_index == 2:
                break
        self.assertEqual(actor.controller.phase_index,2)
        self.assertIn('ring',scene.attachments)
        self.assertNotIn('ring',bridge.last_forecast.object_ids)
        self.assertFalse(next(body for body in actor.previous_states if body.object_id == 'ring').visible)
        self.assertEqual(actor.controller.last_diagnostics['acquire_completion_reference_source'],
                         'causal_measured_attachment_and_current_robot_proprioception')
        self.assertTrue(all(row['official_task_success'] is None for row in actor.controller.attempts.to_report()))

    def test_CV_and_Context_share_five_real_frames_and_missing_body_rewarms(self):
        for method in ('cv','context'):
            actor,scene,tracker=make_actor(method=method)
            for tick in range(4):
                command=actor.act(observation(tick*.05),robot(tick*.05),10.-tick*.05)
                self.assertEqual(command['command'],'hold')
            command=actor.act(observation(.2),robot(.2),9.8)
            self.assertEqual(command['command'],'joint_trajectory')
            self.assertTrue(scene.checks)
            self.assertTrue(all(value >= 5 for value in actor.last_diagnostics['forecast_metadata']['history_frames'].values()))
            tracker.missing.add('wrap')
            command=actor.act(observation(.25),robot(.25),9.75)
            self.assertEqual(command['command'],'hold')
            tracker.missing.clear()
            for tick in range(6,10):
                command=actor.act(observation(tick*.05),robot(tick*.05),10.-tick*.05)
                self.assertEqual(command['command'],'hold')
            actor.act(observation(.5),robot(.5),9.5)
            self.assertGreaterEqual(actor.last_diagnostics['forecast_metadata']['history_frames']['wrap'],5)

    def test_press_controlled_plate_does_not_deadlock_or_become_external_prediction(self):
        actor,scene,tracker=make_actor()
        saw_press=saw_controlled=False
        for tick in range(19):
            time=tick*.05
            z=.1 if actor.controller.phase_index == 0 else .05
            command=actor.act(observation(time),robot(time,z),10.-time)
            saw_press |= command['public_command_kind'] == 'press'
            ready=actor.last_diagnostics.get('collision_scene_readiness',{})
            if ready.get('controlled_body_bookkeeping_ids') == ['plate']:
                saw_controlled=True
                self.assertNotIn('plate',actor.bridge.last_forecast.object_ids)
                self.assertEqual(ready['ready'],True)
        self.assertTrue(saw_press and saw_controlled)
        self.assertTrue(any(set(check['permitted']) == {'button','plate'} and not check['ignored'] for check in scene.checks))
        self.assertTrue(all(row['official_task_success'] is None for row in actor.controller.attempts.to_report()))

    def test_each_candidate_rebinds_scene_and_backend_clock_failure_propagates(self):
        actor,scene,tracker=make_actor()
        for tick in range(5):
            actor.act(observation(tick*.05),robot(tick*.05),10.-tick*.05)
        self.assertGreaterEqual(actor.collision_port.counts['candidate_scene_bindings'],4)
        self.assertGreater(actor.collision_port.counts['sampled_collision_queries'],20)
        scene.failure=ValueError('mock counted physics drift')
        with self.assertRaisesRegex(ValueError,'physics drift'):
            actor.collision_port.check_future(.01)

    def test_insert_robot_permission_preserves_held_world_and_buzz_frame_is_not_exempt(self):
        scene=SceneMap();scene.required_dynamic_role_ids=('ring','frame','peg0','peg1','peg2')
        spec=get_task_spec('insert_onto_rotating_peg');model=ProbeRobot()
        port=PhaseCollisionPort(spec,scene,model,maximum_observation_age=.1)
        bodies={identity:ObjectState(identity,0.,Pose(np.zeros(3)),np.zeros((3,3)),np.zeros(3),np.zeros(3))
                for identity in scene.required_dynamic_role_ids}
        state=PolicyState(robot(0.),bodies,{'bodies':{identity:dict(directly_visible=True,
            registration_valid=True) for identity in bodies}}, {'selected_peg':('peg1',)})
        forecast=ConstantVelocityDynamics().predict(list(bodies.values()),1.,.05)
        port.configure_frame(state,forecast,attachment_ids=('ring',))
        port.bind_candidate(state=state,forecast=forecast,phase='insert',reference_id='peg1',
            manipulated_ids=('ring',),arrival_time=.2)
        self.assertFalse(port.check_future(.2))
        self.assertEqual(scene.checks[-1]['permitted'],('peg1',))
        self.assertEqual(scene.checks[-1]['ignored'],())
        self.assertIn('peg1',scene.checks[-1]['body_ids'])
        port.bind_candidate(state=state,forecast=forecast,phase='follow',reference_id='frame',
            manipulated_ids=('ring',),arrival_time=.2)
        port.check_future(.2)
        self.assertEqual(scene.checks[-1]['permitted'],())

    def test_TRAIN_uniform_selection_is_once_and_evaluation_refuses_it(self):
        sampler=TRAINUniformFirstCandidateSampler(seed=0,purpose='train')
        with self.assertRaises(PipelineInputError):
            make_actor(sampler=sampler)
        actor,scene,tracker=make_actor(purpose='train_collection',sampler=sampler)
        for tick in range(19):
            time=tick*.05;z=.1 if actor.controller.phase_index == 0 else .05
            actor.act(observation(time),robot(time,z),10.-time)
        self.assertEqual(sampler.draw_count,1)
        self.assertEqual(actor.first_selection_policy,'TRAIN_explore_once')
        self.assertEqual(actor.metrics_report()['TRAIN_first_candidate_sampler']['seed'],0)
        self.assertTrue(all(row['official_task_success'] is None for row in actor.controller.attempts.to_report()))

    def test_privileged_observation_and_unknown_mode_are_fail_closed(self):
        actor,scene,tracker=make_actor()
        with self.assertRaises(PipelineInputError):
            actor.act(dict(observation(0.),native_result=True),robot(0.),10.)
        actor.mode_observer=SimpleNamespace(observe=lambda previous,current:
            ({body:'unknown' for body in scene.required_dynamic_role_ids},set(scene.required_dynamic_role_ids)))
        command=actor.act(observation(0.),robot(0.),10.)
        self.assertEqual(command['command'],'hold')
        self.assertEqual(command['reason'],'unknown_or_ambiguous_public_motion_mode')
        self.assertFalse(scene.checks)


if __name__ == '__main__':
    unittest.main()
