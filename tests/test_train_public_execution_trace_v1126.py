"""Transparent CPU observation/planning wrappers, with no Native imports/jobs."""
from contextlib import nullcontext
from enum import Enum
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np

from our_mani.contracts import Pose, RobotState, ObjectState
from our_mani.multitask.interaction import PhaseTemplate
from our_mani.multitask import train_public_execution_trace_v1126 as module


def robot(timestamp=.05):
    return RobotState(timestamp, Pose([0.,0.,1.]), np.zeros(3), np.zeros(3),
                      np.arange(7,dtype=float), np.zeros(7))


def body(identity):
    return ObjectState(identity,.05,Pose([.1,.2,.9]),[[0.,0.,0.],[.01,0.,0.],[0.,.01,0.]],
                       np.zeros(3),np.zeros(3))


def template():
    return PhaseTemplate('approach','TRAIN_anchor','wand','observed_se3','wand',0.,
        np.array([0.,1.]),np.zeros((2,3)),np.repeat(np.eye(3)[None],2,axis=0),np.ones(2),.5,1.,.01,.1,'external')


class Service:
    def __init__(self):
        self.calls=[]; self.error=None
        self.path=NS(_path_points=np.arange(21,dtype=float),_num_joints=7,_path_done=False,
                     step=Mock(side_effect=AssertionError('extra path step forbidden')))

    def plan_path(self,*args,**kwargs):
        self.calls.append((args,kwargs))
        if self.error is not None:raise self.error
        return self.path


class Tests(unittest.TestCase):
    def fixture(self, *, observer_error=None):
        calls=[]; frontend=NS(state=None,predict=Mock(side_effect=AssertionError('extra prediction forbidden')))
        result=object()
        def observe(observation, public_robot):
            calls.append((observation,public_robot))
            if observer_error is not None:raise observer_error
            frontend.state=NS(robot=public_robot,bodies={'frame':body('frame'),'wand':body('wand')})
            return result
        frontend.observe=observe
        policy=NS(source_policy=NS(selected=dict(phase='approach',reference_id='wand',manipulated_ids=('wand',),
            selected_at=.0,opening=1.,predicted_reference_time=.0,pose=Pose([.1,.2,.9]),
            present_pose=Pose([.1,.2,.9]),template=template())))
        executor=NS(observe_public_frame=frontend.observe)
        service=Service(); env=NS(map_service=service,
            observe=Mock(side_effect=AssertionError('extra env observation forbidden')),
            arm=NS(get_tip=Mock(side_effect=AssertionError('Native FK/getter forbidden'))))
        components=(object(),{'original_model':True},frontend,policy,executor)
        runtime=NS(_geometry=Mock(return_value=components))
        return NS(frontend=frontend,policy=policy,executor=executor,env=env,service=service,runtime=runtime,
                  calls=calls,observe_result=result,components=components)

    def test_original_factory_observe_plan_objects_and_calls_are_unchanged(self):
        fixture=self.fixture(); events={}; original_geometry=fixture.runtime._geometry
        old_observe=fixture.frontend.observe; old_callback=fixture.executor.observe_public_frame
        class Algorithm(Enum):RRTConnect='RRTConnect'
        with module.public_execution_trace_scope(fixture.runtime,events):
            self.assertIs(fixture.runtime._geometry(None,fixture.env,[]),fixture.components)
            current=robot()
            self.assertIs(fixture.frontend.observe({'existing':'frame'},current),fixture.observe_result)
            self.assertIs(fixture.executor.observe_public_frame({'next':'frame'},current),fixture.observe_result)
            path=fixture.service.plan_path([.1,.2,.9],quaternion=[0.,0.,0.,1.],ignore_collisions=True,
                                          algorithm=Algorithm.RRTConnect)
            self.assertIs(path,fixture.service.path)
            fixture.service.path._path_points[:]=-1  # Saved raw planner output must be a copy.
        original_geometry.assert_called_once()
        self.assertEqual(len(fixture.calls),2); self.assertEqual(len(fixture.service.calls),1)
        fixture.env.observe.assert_not_called(); fixture.env.arm.get_tip.assert_not_called()
        fixture.frontend.predict.assert_not_called(); fixture.service.path.step.assert_not_called()
        self.assertIs(fixture.runtime._geometry,original_geometry)
        self.assertIs(fixture.frontend.observe,old_observe); self.assertIs(fixture.executor.observe_public_frame,old_callback)
        self.assertNotIn('plan_path',fixture.service.__dict__)
        receipt=events['trace_receipt']; self.assertTrue(receipt['restored'])
        self.assertEqual(receipt['recording_errors'],0)
        self.assertEqual(receipt['frames'][0]['bodies']['frame']['pose']['position'],[.1,.2,.9])
        self.assertEqual(receipt['frames'][0]['selected']['anchor_id'],'TRAIN_anchor')
        self.assertEqual(receipt['plans'][0]['path_points'],list(np.arange(21,dtype=float)))
        self.assertEqual(receipt['plans'][0]['original_kwargs']['algorithm']['name'],'RRTConnect')
        self.assertFalse(receipt['sysprofile_installed'])

    def test_strict_dtos_do_not_call_duck_pose_body_or_robot_getters(self):
        fixture=self.fixture(); events={}
        class NativeDuck:
            @property
            def pose(self):raise AssertionError('Native pose getter forbidden')
            @property
            def timestamp(self):raise AssertionError('Native timestamp getter forbidden')
        bad=NativeDuck()
        def observe(observation,public_robot):
            fixture.calls.append((observation,public_robot))
            fixture.frontend.state=NS(robot=public_robot,bodies={'frame':bad})
            return fixture.observe_result
        fixture.frontend.observe=fixture.executor.observe_public_frame=observe
        fixture.policy.source_policy.selected['pose']=bad
        with module.public_execution_trace_scope(fixture.runtime,events):
            fixture.runtime._geometry(None,fixture.env,[])
            self.assertIs(fixture.frontend.observe({},bad),fixture.observe_result)
        row=events['trace_receipt']['frames'][0]
        self.assertEqual(row['robot']['status'],'unknown')
        self.assertEqual(row['bodies']['frame']['reason'],'not_original_ObjectState_DTO')
        self.assertEqual(row['selected']['pose']['reason'],'not_original_Pose_DTO')
        self.assertEqual(events['trace_receipt']['recording_errors'],0)

    def test_original_corrected_template_anchor_is_copied_without_calling_template_methods(self):
        from dataclasses import fields
        from our_mani.multitask.grasp_frame_compensation_native_compat import CorrectedTemplate
        fixture=self.fixture(); events={}; original=template()
        corrected=CorrectedTemplate(**{field.name:getattr(original,field.name) for field in fields(PhaseTemplate)})
        fixture.policy.source_policy.selected['template']=corrected
        with patch.object(CorrectedTemplate,'at',side_effect=AssertionError('extra template evaluation forbidden')):
            with module.public_execution_trace_scope(fixture.runtime,events):
                fixture.runtime._geometry(None,fixture.env,[]); fixture.frontend.observe({},robot())
        self.assertEqual(events['trace_receipt']['frames'][0]['selected']['anchor_id'],'TRAIN_anchor')

    def test_limits_drop_only_trace_without_repeating_original_calls(self):
        fixture=self.fixture(); events={}
        with module.public_execution_trace_scope(fixture.runtime,events,max_frames=1,max_plans=1):
            fixture.runtime._geometry(None,fixture.env,[])
            for _ in range(3):
                fixture.frontend.observe({},robot()); fixture.service.plan_path([.1,.2,.9])
        receipt=events['trace_receipt']
        self.assertEqual((receipt['frame_calls'],receipt['frame_returns'],receipt['frame_dropped']),(3,3,2))
        self.assertEqual((receipt['plan_calls'],receipt['plan_returns'],receipt['plan_dropped']),(3,3,2))
        self.assertEqual(len(fixture.calls),3); self.assertEqual(len(fixture.service.calls),3)
        self.assertTrue(receipt['restored'])

    def test_original_observe_exception_object_propagates_and_all_hooks_restore(self):
        failure=ValueError('original observation failed'); fixture=self.fixture(observer_error=failure); events={}
        original_geometry=fixture.runtime._geometry; original_observe=fixture.frontend.observe
        with self.assertRaises(ValueError) as caught:
            with module.public_execution_trace_scope(fixture.runtime,events):
                fixture.runtime._geometry(None,fixture.env,[]); fixture.frontend.observe({},robot())
        self.assertIs(caught.exception,failure)
        self.assertIs(fixture.runtime._geometry,original_geometry); self.assertIs(fixture.frontend.observe,original_observe)
        self.assertEqual(len(fixture.calls),1)
        receipt=events['trace_receipt']; self.assertTrue(receipt['restored'])
        self.assertEqual((receipt['frame_calls'],receipt['frame_returns'],receipt['frame_exceptions']),(1,0,1))
        self.assertEqual(receipt['scope_exception_type'],'ValueError')

    def test_original_plan_exception_object_is_recorded_without_getter_or_rewrite(self):
        failure=RuntimeError('original planner failed'); fixture=self.fixture(); fixture.service.error=failure; events={}
        with self.assertRaises(RuntimeError) as caught:
            with module.public_execution_trace_scope(fixture.runtime,events):
                fixture.runtime._geometry(None,fixture.env,[]); fixture.service.plan_path([.1,.2,.9])
        self.assertIs(caught.exception,failure); self.assertEqual(len(fixture.service.calls),1)
        receipt=events['trace_receipt']; self.assertTrue(receipt['restored'])
        self.assertEqual(receipt['plan_exceptions'],1)
        self.assertEqual(receipt['plans'][0]['exception_type'],'RuntimeError')
        self.assertEqual(receipt['plans'][0]['path_points']['status'],'unknown')

    def test_recording_error_does_not_change_original_observe_return(self):
        fixture=self.fixture(); events={}
        with patch.object(module.PublicExecutionTrace,'frame',side_effect=ValueError('copy failed')):
            with module.public_execution_trace_scope(fixture.runtime,events):
                fixture.runtime._geometry(None,fixture.env,[])
                self.assertIs(fixture.frontend.observe({},robot()),fixture.observe_result)
        self.assertEqual(len(fixture.calls),1)
        self.assertEqual(events['trace_receipt']['recording_errors'],1)
        self.assertTrue(events['restored'])

    def test_missing_or_oversized_stored_path_is_unknown_and_never_queries_a_property(self):
        fixture=self.fixture(); events={}
        class PathWithoutStorage:
            @property
            def _path_points(self):raise AssertionError('path getter forbidden')
        with module.public_execution_trace_scope(fixture.runtime,events):
            fixture.runtime._geometry(None,fixture.env,[])
            fixture.service.path=PathWithoutStorage(); fixture.service.plan_path([0.,0.,0.])
            fixture.service.path=NS(_path_points=np.zeros(module.MAX_PLAN_VALUES+1))
            fixture.service.plan_path([0.,0.,0.])
        receipt=events['trace_receipt']
        self.assertEqual(receipt['recording_errors'],0); self.assertEqual(receipt['plan_value_truncations'],1)
        self.assertTrue(all(row['path_points']['status']=='unknown' for row in receipt['plans']))

    def test_different_existing_observer_callbacks_are_not_rebound(self):
        fixture=self.fixture(); events={}; alternate=Mock(return_value='alternate')
        fixture.executor.observe_public_frame=alternate; original=fixture.frontend.observe
        with module.public_execution_trace_scope(fixture.runtime,events):
            fixture.runtime._geometry(None,fixture.env,[])
            self.assertIs(fixture.frontend.observe,original)
            self.assertEqual(fixture.executor.observe_public_frame({},robot()),'alternate')
        alternate.assert_called_once()
        self.assertIn('different_observe_callback_not_rebound',events['trace_receipt']['unknown_reasons'])

    def test_profile_is_untouched_and_scope_restores_the_parent_installed_factory(self):
        fixture=self.fixture(); events={}; old=sys.getprofile(); external=lambda *args:None
        parent=Mock(return_value=fixture.components)
        try:
            sys.setprofile(external)
            with patch.object(fixture.runtime,'_geometry',parent):
                with module.public_execution_trace_scope(fixture.runtime,events):
                    fixture.runtime._geometry(None,fixture.env,[])
                self.assertIs(fixture.runtime._geometry,parent)
                self.assertIs(sys.getprofile(),external)
        finally:sys.setprofile(old)
        self.assertTrue(events['restored'])


if __name__=='__main__':unittest.main()
