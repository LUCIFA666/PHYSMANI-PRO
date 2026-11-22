"""CPU transparency checks; no Native imports, getters or project jobs."""
from types import SimpleNamespace as NS
import json
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

from our_mani.contracts import Forecast, ObjectState, Pose, RobotState
from our_mani.multitask.interaction import PhaseTemplate, PolicyState
from our_mani.multitask import train_public_readiness_probe_v1134 as module


def robot():
    return RobotState(3.75,Pose([0.,0.,1.]),np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7),False,True)


def body():
    return ObjectState('peg0',3.75,Pose([.1,.2,.9]),[[0.,0.,0.],[.01,0.,0.],[0.,.01,0.]],np.zeros(3),np.zeros(3))


def template():
    return PhaseTemplate('insert','TRAIN_anchor','peg','observed_se3','ring',0.,np.array([0.,1.]),
        np.zeros((2,3)),np.repeat(np.eye(3)[None],2,axis=0),np.zeros(2),.5,1.,.01,.1,'external')


class Generator:
    def __init__(self): self.calls=[]; self.error=None; self.result=body()
    def observed_reference(self,state,body_id,template,origin):
        self.calls.append((state,body_id,template,origin))
        if self.error is not None: raise self.error
        return self.result


class Port:
    def __init__(self): self.calls=[]; self.error=None; self.result={'ready':True,'reason':'existing original report'}
    def readiness(self,state,forecast,phase):
        self.calls.append((state,forecast,phase))
        if self.error is not None: raise self.error
        return self.result


class Frontend:
    def __init__(self, actor, state, forecast):
        self.actor=actor; self.state=state; self.forecast=forecast; self.role_ready=True; self.calls=[]
    def ready(self,phase):
        self.calls.append(phase)
        if not self.role_ready:return False
        return self.actor.collision_port.readiness(self.state,self.forecast,phase)['ready']


class Source:
    def __init__(self, frontend):
        self.frontend=frontend; self.phase_index=4; self.phase_order=('approach','acquire','lift','align','insert','release')
        self.selected=dict(phase='insert',template=template(),reference_id='peg0',manipulated_ids=('ring',),
            opening=0.,selected_at=3.2,predicted_reference_time=3.7)
        self.committed=dict(self.selected); self.calls=[]; self.result=[self.selected]; self.error=None
    def candidates(self,phase):
        self.calls.append(phase)
        try:self.frontend.actor.controller.generator.observed_reference(self.frontend.state,'peg0',self.selected['template'],None)
        except (ValueError,KeyError):return []
        if self.error is not None:raise self.error
        return self.result


class Tests(unittest.TestCase):
    def fixture(self):
        state=PolicyState(robot(),{'peg0':body()},{'bodies':{'peg0':dict(motion_mode='external',directly_visible=True,
            registration_valid=True,orientation_observable=True,pose_evidence_until=3.75)},'invalid_mode_ids':[]},
            {'peg':('peg0',),'ring':('ring',)})
        forecast=Forecast(('peg0',),np.array([3.75,4.25]),np.zeros((2,1,3)),
            np.repeat(np.eye(3)[None,None],2,axis=0),np.zeros((2,1,3)),np.zeros((2,1,3)),metadata={'observed_until':3.75})
        generator=Generator(); port=Port(); controller=NS(generator=generator,committed_roles={'peg':('peg0',),'ring':('ring',)})
        actor=NS(controller=controller,collision_port=port,last_diagnostics=dict(invalid_mode_ids=[],mode_estimates={'peg0':'external','ring':'attached'}))
        frontend=Frontend(actor,state,forecast); source=Source(frontend); policy=NS(source_policy=source)
        executor=NS(skill_policy=policy); result=(actor,{},frontend,policy,executor)
        factory=Mock(return_value=result); runtime=NS(_geometry=factory)
        return NS(runtime=runtime,factory=factory,components=result,frontend=frontend,source=source,generator=generator,port=port)

    def test_original_calls_returns_identity_and_restoration(self):
        f=self.fixture(); sink={}; old_geometry=f.runtime._geometry
        with module.readiness_probe_scope(f.runtime,sink):
            self.assertIs(f.runtime._geometry(None,None,[]),f.components)
            self.assertTrue(f.frontend.ready('insert'))
            self.assertIs(f.port.readiness(f.frontend.state,f.frontend.forecast,'insert'),f.port.result)
            self.assertIs(f.source.candidates('insert'),f.source.result)
        self.assertIs(f.runtime._geometry,old_geometry); self.assertNotIn('ready',f.frontend.__dict__)
        self.assertNotIn('readiness',f.port.__dict__); self.assertNotIn('candidates',f.source.__dict__)
        self.assertNotIn('observed_reference',f.generator.__dict__)
        f.factory.assert_called_once(); self.assertEqual(len(f.frontend.calls),1); self.assertEqual(len(f.port.calls),2)
        self.assertEqual(len(f.source.calls),1); self.assertEqual(len(f.generator.calls),1)
        receipt=sink['readiness_probe']; self.assertTrue(receipt['restored']); self.assertEqual(receipt['recording_errors'],0)
        self.assertEqual(receipt['method_call_counts']['collision_port.readiness'],dict(calls=2,returns=2,exceptions=0))
        candidate=next(x for x in receipt['events'] if x['method']=='source_policy.candidates')
        ref=next(x for x in receipt['events'] if x['method']=='generator.observed_reference')
        self.assertEqual(candidate['original_return']['candidate_count'],1)
        self.assertEqual(ref['context']['kind'],'candidate'); self.assertEqual(ref['parent_call_sequence'],candidate['call_sequence'])
        snapshot=candidate['snapshot']; self.assertEqual(snapshot['public_state']['timestamp'],3.75)
        self.assertEqual(snapshot['forecast']['time_range'],dict(first=3.75,last=4.25,count=2))
        self.assertEqual(snapshot['committed_roles']['peg'],['peg0'])
        json.dumps(receipt,allow_nan=False)

    def test_swallowed_reference_exception_is_preserved_and_not_observer_error(self):
        f=self.fixture(); failure=ValueError('required reference rotation is unobservable'); f.generator.error=failure; sink={}
        with module.readiness_probe_scope(f.runtime,sink):
            f.runtime._geometry(); self.assertEqual(f.source.candidates('insert'),[])
        receipt=sink['readiness_probe']; ref=next(x for x in receipt['events'] if x['method']=='generator.observed_reference')
        self.assertEqual(ref['original_exception']['reason'],str(failure)); self.assertEqual(ref['context']['kind'],'candidate')
        self.assertEqual(receipt['method_call_counts']['generator.observed_reference'],dict(calls=1,returns=0,exceptions=1))
        self.assertEqual(receipt['recording_errors'],0); self.assertEqual(len(f.generator.calls),1)

    def test_direct_reference_exception_object_and_scope_exception_are_unchanged(self):
        f=self.fixture(); failure=KeyError('original reference missing'); f.generator.error=failure; sink={}
        with self.assertRaises(KeyError) as caught:
            with module.readiness_probe_scope(f.runtime,sink):
                f.runtime._geometry(); f.generator.observed_reference(f.frontend.state,'peg0',template(),None)
        self.assertIs(caught.exception,failure); self.assertTrue(sink['restored'])
        self.assertEqual(sink['readiness_probe']['scope_exception_type'],'KeyError')

    def test_ready_false_does_not_invent_port_or_candidate_calls(self):
        f=self.fixture(); f.frontend.role_ready=False; sink={}
        with module.readiness_probe_scope(f.runtime,sink):
            f.runtime._geometry(); self.assertFalse(f.frontend.ready('insert'))
        self.assertEqual(len(f.port.calls),0); self.assertEqual(len(f.source.calls),0)
        receipt=sink['readiness_probe']; row=receipt['events'][0]
        self.assertFalse(row['snapshot']['role_ready']); self.assertFalse(row['original_return'])
        self.assertEqual(receipt['method_call_counts']['collision_port.readiness']['calls'],0)

    def test_port_original_report_is_a_copy_and_ready_false_is_not_capture_failure(self):
        f=self.fixture(); f.port.result={'ready':False,'reason':'original geometry unavailable'}; sink={}
        with module.readiness_probe_scope(f.runtime,sink):
            f.runtime._geometry(); self.assertFalse(f.frontend.ready('insert'))
            f.port.result['reason']='changed after original call'
        row=next(x for x in sink['readiness_probe']['events'] if x['method']=='collision_port.readiness')
        self.assertEqual(row['original_return']['reason'],'original geometry unavailable')
        self.assertEqual(sink['readiness_probe']['recording_errors'],0)

    def test_body_visible_and_direct_rgbd_visibility_are_distinct_with_rigid_group_provenance(self):
        f=self.fixture(); sink={}; metadata=f.frontend.state.perception_metadata
        metadata['bodies']['peg0'].update(directly_visible=False,rigid_root='frame',
            pose_source='current_group_RGBD_and_declared_rigid_layout',
            points_source='rigid_prior_geometry_transformed_by_observed_group')
        metadata['rigid_groups']=[dict(root='frame',members=['frame','peg0'],registration_valid=True,
                                      TRAIN_constraints=[dict(parent='frame',child='peg0',evidence_sha256='TRAIN_layout')])]
        with module.readiness_probe_scope(f.runtime,sink):
            f.runtime._geometry(); f.frontend.ready('insert')
        state=sink['readiness_probe']['events'][0]['snapshot']['public_state']; peg=state['bodies']['peg0']
        self.assertTrue(peg['visible']); self.assertFalse(peg['directly_visible']); self.assertTrue(peg['registration_valid'])
        self.assertEqual(peg['rigid_root'],'frame'); self.assertEqual(peg['pose_source'],'current_group_RGBD_and_declared_rigid_layout')
        self.assertEqual(state['rigid_groups'][0]['TRAIN_constraints'][0]['child'],'peg0')
        self.assertEqual(peg['mode_provenance']['status'],'unknown')

    def test_candidate_and_port_exceptions_propagate_same_objects(self):
        for target in ('source','port'):
            with self.subTest(target=target):
                f=self.fixture(); sink={}; failure=RuntimeError('original method failure'); getattr(f,target).error=failure
                with self.assertRaises(RuntimeError) as caught:
                    with module.readiness_probe_scope(f.runtime,sink):
                        f.runtime._geometry()
                        if target=='source':f.source.candidates('insert')
                        else:f.frontend.ready('insert')
                self.assertIs(caught.exception,failure); self.assertTrue(sink['restored'])
                self.assertEqual(sink['readiness_probe']['recording_errors'],0)

    def test_done_context_is_only_code_frames_and_done_is_never_patched(self):
        f=self.fixture(); sink={}; namespace={}
        exec(compile('def _completion_target(generator,state,template):\n return generator.observed_reference(state,"peg0",template,None)\n'
             'def _done(generator,state,template):\n return _completion_target(generator,state,template)\n',
             '/fixed/our_mani/multitask/physmani_skill_policy_v548.py','exec'),namespace)
        function=namespace['_done']; f.source._done=Mock(side_effect=AssertionError('extra done forbidden'))
        with module.readiness_probe_scope(f.runtime,sink):
            f.runtime._geometry(); self.assertIs(function(f.generator,f.frontend.state,template()),f.generator.result)
        ref=sink['readiness_probe']['events'][0]; self.assertEqual(ref['context']['kind'],'done')
        f.source._done.assert_not_called()

    def test_completion_target_from_qualification_is_not_mislabeled_as_done(self):
        f=self.fixture(); sink={}; namespace={}
        exec(compile('def _completion_target(generator,state,template):\n return generator.observed_reference(state,"peg0",template,None)\n'
             'def qualify_opening(generator,state,template):\n return _completion_target(generator,state,template)\n',
             '/fixed/our_mani/multitask/physmani_skill_policy_v548.py','exec'),namespace)
        with module.readiness_probe_scope(f.runtime,sink):
            f.runtime._geometry(); namespace['qualify_opening'](f.generator,f.frontend.state,template())
        context=sink['readiness_probe']['events'][0]['context']
        self.assertEqual(context['kind'],'other_or_unknown')
        self.assertEqual(context['evidence'],'completion_target_outside_known_done_frame')

    def test_properties_native_ducks_and_unknown_values_are_never_queried(self):
        f=self.fixture(); sink={}; hits=[]
        class Duck:
            @property
            def timestamp(self):hits.append('timestamp'); raise AssertionError('getter forbidden')
            @property
            def object_ids(self):hits.append('object_ids'); raise AssertionError('getter forbidden')
            @property
            def __dict__(self):hits.append('__dict__'); raise AssertionError('getter forbidden')
        duck=Duck(); f.frontend.state=duck; f.frontend.forecast=duck
        f.source.selected['template']=duck; f.source.selected['opening']=duck
        f.generator.result=duck
        with module.readiness_probe_scope(f.runtime,sink):
            f.runtime._geometry(); self.assertIs(f.generator.observed_reference(duck,'peg0',duck,None),duck)
        self.assertEqual(hits,[]); receipt=sink['readiness_probe']; self.assertEqual(receipt['recording_errors'],0)
        self.assertEqual(receipt['events'][0]['snapshot']['public_state']['status'],'unknown')
        self.assertEqual(receipt['events'][0]['original_return']['status'],'unknown')
        self.assertEqual(module._storage(duck),{}); self.assertEqual(hits,[])

    def test_recording_failure_does_not_alter_original_return_or_exception(self):
        f=self.fixture(); sink={}
        with patch.object(module.ReadinessProbe,'snapshot',side_effect=ValueError('observer copy failed')):
            with module.readiness_probe_scope(f.runtime,sink):
                f.runtime._geometry(); self.assertIs(f.source.candidates('insert'),f.source.result)
        self.assertEqual(len(f.source.calls),1); self.assertEqual(len(f.generator.calls),1)
        self.assertGreater(sink['readiness_probe']['recording_errors'],0); self.assertTrue(sink['restored'])

    def test_bound_drops_trace_without_repeating_original_calls(self):
        f=self.fixture(); sink={}
        with module.readiness_probe_scope(f.runtime,sink,max_events=2):
            f.runtime._geometry()
            for _ in range(3):f.source.candidates('insert')
        receipt=sink['readiness_probe']; self.assertEqual(len(receipt['events']),2)
        self.assertEqual(receipt['dropped_events'],4); self.assertEqual(len(f.source.calls),3); self.assertEqual(len(f.generator.calls),3)
        self.assertEqual(receipt['recording_errors'],0); self.assertTrue(receipt['restored'])

    def test_repeated_geometry_factory_does_not_double_wrap_owned_components(self):
        f=self.fixture(); sink={}
        with module.readiness_probe_scope(f.runtime,sink):
            f.runtime._geometry(); f.runtime._geometry(); f.source.candidates('insert')
        receipt=sink['readiness_probe']; self.assertEqual(receipt['geometry_calls'],2)
        self.assertEqual(receipt['method_call_counts']['source_policy.candidates']['calls'],1)
        self.assertIn('same_owned_components_already_instrumented',receipt['unknown_reasons'])

    def test_real_pose_goals_filters_keep_original_reason_without_extra_reference_queries(self):
        from our_mani.multitask.physmani_geometry_goals_v529 import pose_goals
        for failure_kind in ('reference_rotation','forecast_cutoff'):
            with self.subTest(failure_kind=failure_kind):
                f=self.fixture(); sink={}; t=template(); controller=f.frontend.actor.controller
                controller.library=NS(for_phase=lambda phase:[t])
                f.source.candidates=lambda phase:pose_goals(f.frontend,phase,origin=None,committed=f.source.committed)
                if failure_kind=='reference_rotation':
                    f.generator.error=ValueError('required reference rotation is unobservable')
                else:
                    f.frontend.forecast.metadata['observed_until']=3.7
                with module.readiness_probe_scope(f.runtime,sink):
                    f.runtime._geometry(); self.assertEqual(f.source.candidates('insert'),[])
                receipt=sink['readiness_probe']; self.assertEqual(receipt['recording_errors'],0)
                candidate=next(x for x in receipt['events'] if x['method']=='source_policy.candidates')
                self.assertEqual(candidate['original_return']['candidate_count'],0)
                if failure_kind=='reference_rotation':
                    self.assertEqual(len(f.generator.calls),1)
                    ref=next(x for x in receipt['events'] if x['method']=='generator.observed_reference')
                    self.assertEqual(ref['original_exception']['reason'],'required reference rotation is unobservable')
                    self.assertEqual(ref['context']['kind'],'candidate')
                else:
                    self.assertEqual(len(f.generator.calls),0)
                    self.assertEqual(candidate['snapshot']['forecast']['observed_until'],3.7)

    def test_nested_original_candidates_keep_parent_and_original_counts(self):
        f=self.fixture(); sink={}; original=f.source.candidates; calls=[]
        def outer(phase):
            calls.append(phase)
            return original(phase)
        f.source.candidates=outer
        with module.readiness_probe_scope(f.runtime,sink):
            f.runtime._geometry(); self.assertIs(f.source.candidates('insert'),f.source.result)
        self.assertEqual(calls,['insert']); self.assertEqual(len(f.source.calls),1)
        self.assertEqual(len(f.generator.calls),1)
        self.assertEqual(sink['readiness_probe']['method_call_counts']['source_policy.candidates']['calls'],1)

    def test_scope_stacks_after_public_trace_and_restores_that_parent_factory(self):
        from our_mani.multitask.train_public_execution_trace_v1126 import public_execution_trace_scope
        f=self.fixture(); sink={}; trace_sink={}; observations=[]; old_factory=f.runtime._geometry
        def observe(observation,public_robot):
            observations.append((observation,public_robot)); return 'original public observation'
        f.frontend.observe=observe; f.components[-1].observe_public_frame=observe
        env=NS(map_service=NS(plan_path=lambda *args,**kwargs:NS(_path_points=np.arange(14.),_num_joints=7,_path_done=False)))
        with public_execution_trace_scope(f.runtime,trace_sink):
            trace_factory=f.runtime._geometry
            with module.readiness_probe_scope(f.runtime,sink):
                self.assertIs(f.runtime._geometry(None,env,[]),f.components)
                self.assertTrue(f.frontend.ready('insert')); self.assertIs(f.source.candidates('insert'),f.source.result)
                self.assertEqual(f.frontend.observe({},robot()),'original public observation')
                env.map_service.plan_path([0.,0.,0.])
            self.assertIs(f.runtime._geometry,trace_factory)
        self.assertIs(f.runtime._geometry,old_factory); self.assertEqual(len(observations),1)
        self.assertEqual(len(f.port.calls),1); self.assertEqual(len(f.generator.calls),1)
        self.assertTrue(sink['restored']); self.assertTrue(trace_sink['restored'])
        self.assertEqual((trace_sink['trace_receipt']['frame_calls'],trace_sink['trace_receipt']['plan_calls']),(1,1))
        self.assertEqual(sink['readiness_probe']['recording_errors'],0)

    def test_parent_factory_and_external_profiler_are_preserved(self):
        f=self.fixture(); sink={}; old=sys.getprofile(); profiler=lambda *args:None
        parent=Mock(return_value=f.components)
        try:
            sys.setprofile(profiler)
            with patch.object(f.runtime,'_geometry',parent):
                with module.readiness_probe_scope(f.runtime,sink):
                    f.runtime._geometry(); f.frontend.ready('insert')
                self.assertIs(f.runtime._geometry,parent); self.assertIs(sys.getprofile(),profiler)
        finally:sys.setprofile(old)
        self.assertTrue(sink['restored']); parent.assert_called_once()

    def test_original_factory_exception_object_is_not_rewritten(self):
        f=self.fixture(); sink={}; failure=RuntimeError('original factory failed'); f.factory.side_effect=failure
        with self.assertRaises(RuntimeError) as caught:
            with module.readiness_probe_scope(f.runtime,sink):f.runtime._geometry()
        self.assertIs(caught.exception,failure); receipt=sink['readiness_probe']; self.assertTrue(receipt['restored'])
        self.assertEqual((receipt['geometry_calls'],receipt['geometry_returns'],receipt['geometry_exceptions']),(1,0,1))

    def test_bounds_and_fresh_sink_required(self):
        f=self.fixture()
        for maximum in (0,True,20001):
            with self.assertRaises(ValueError):
                with module.readiness_probe_scope(f.runtime,{},max_events=maximum):pass
        with self.assertRaises(ValueError):
            with module.readiness_probe_scope(f.runtime,{'readiness_probe':{}}):pass


if __name__=='__main__':unittest.main()
