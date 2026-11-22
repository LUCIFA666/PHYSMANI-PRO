"""CPU-only v56 producer and exact-dispatch wiring; no Native or fitting."""
import ast,builtins,dis,inspect
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import CodeType,FunctionType,SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

import our_mani.local_mpc_control_20261103.public_candidate_goal_maintenance_v56 as v56
import our_mani.local_mpc_control_20261103.public_grasp_candidates_v5 as v5
from our_mani.contracts import Pose
from our_mani.benchmark_perception import box_surface
from our_mani.local_mpc_control_20261103.active_visual_attachment import ActiveVisualAttachment,ActiveVisualConfig
from our_mani.local_mpc_control_20261103.public_grasp_v41 import ConfirmedAttemptRecovery
from our_mani.local_mpc_control_20261103.public_grasp_state import ApprovedTrackerCacheBinding
from our_mani.local_mpc_control_20261103.goal_aware_binding_v42 import ApprovedGoalAwareTrackerCacheBinding
from our_mani.local_mpc_p4_20261103.runtime import pack_public_row
from tests.test_local_mpc_active_visual_attachment_phase_scope_20261103 import sensor_frame,actual_h
from tests.test_local_mpc_public_motion_bridge_20261103 import observation
from tests.test_local_mpc_goal_aware_binding_v42_20261103 import raw_metadata
from tests.test_local_mpc_goal_aware_perception_20261103 import metadata
from tests.test_local_mpc_public_grasp_20261103 import RawBackend
from tests.test_local_mpc_public_pregrasp_20261103 import Executor

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'artifacts/framework/local_mpc_20261102/P4_candidate_training_v1/final.pt'
SHA='61e5827b4838649b8e477048bea56fcace18697f906fb610a3f963bba2437f60'


class V56Tests(unittest.TestCase):
    def wrapper(self,goal=False,maintenance=True):
        return v56.PublicCandidateBridgeV56(goal_sphere_filter=goal,postconfirmed_maintenance=maintenance,
            active_config=ActiveVisualConfig(enforce_probe_anchor_orientation_after_confirmation=False))

    def feed(self,wrapper,i,*,height=0.,shift=(0.,0.,0.),close=True):
        target,robot=sensor_frame(i,height=height,relative_shift=shift)
        tracked=wrapper.tracker.tracker if wrapper.goal_sphere_filter else wrapper.tracker
        with patch.object(tracked,'observe',return_value=(target,raw_metadata())),\
             patch.object(wrapper.bridge.goal_marker,'observe',return_value=(Pose([.3,.1,1.]),metadata())):
            result=wrapper.observe(observation(robot.timestamp),robot,close_requested=close)
        return result,robot,target

    def confirmed(self):
        wrapper=self.wrapper();self.feed(wrapper,0,close=False)
        result,robot,target=self.feed(wrapper,1)
        wrapper.active.begin_probe(target,robot,actual_h(robot))
        for i in range(2,7):result,robot,target=self.feed(wrapper,i,height=.012*(i-1)/5)
        self.assertTrue(wrapper.maintenance.started)
        return wrapper,result,robot

    def test_flags_default_false_exact_caps_and_no_broad_injection(self):
        self.assertFalse(v56.v56_enabled())
        for bad in (0,1,None,np.bool_(False)):
            with self.assertRaises(ValueError):v56.v56_enabled(goal_sphere_filter=bad)
        raw=self.wrapper(False,False);goal=self.wrapper(True,False)
        self.assertIs(type(raw.active),ActiveVisualAttachment)
        self.assertIs(type(raw.binding),ApprovedTrackerCacheBinding)
        self.assertIs(type(goal.binding),ApprovedGoalAwareTrackerCacheBinding)
        self.assertFalse(v56.PublicCandidateStateAssemblerV56(raw.binding).allow_test_doubles)
        self.assertFalse(v56.PublicCandidateStateAssemblerV56(goal.binding).allow_test_doubles)
        with self.assertRaises(TypeError):v56.PublicCandidateStateAssemblerV56(SimpleNamespace())

    def test_first_real_bridge_confirmation_precedes_helper_admission_and_ledger(self):
        wrapper=self.wrapper();bridge=wrapper.bridge
        with patch.object(bridge,'observe',wraps=bridge.observe) as calls,\
             patch.object(wrapper.maintenance,'start',wraps=wrapper.maintenance.start) as admission:
            self.feed(wrapper,0,close=False);_,robot,target=self.feed(wrapper,1)
            wrapper.active.begin_probe(target,robot,actual_h(robot))
            for i in range(2,7):result,robot,_=self.feed(wrapper,i,height=.012*(i-1)/5)
        self.assertEqual(calls.call_count,7);self.assertEqual(admission.call_count,1)
        admitted=admission.call_args.args[0]
        self.assertIs(admitted,result);self.assertTrue(admitted['available'])
        self.assertTrue(admitted['controlled_mode_available'])
        self.assertEqual(result['attachment_estimate']['protocol'],'active_vertical_visual_comotion_v4')
        recovery=ConfirmedAttemptRecovery();recovery.observe(result,robot)
        self.assertTrue(recovery.preserves_closed_attempt(wrapper.active.attempt_id))
        pack_public_row(result,robot,True,6,'rgbd/0006.npz')

    def test_later_9mm_maintains_current_visual_with_explicit_new_protocol_raw_failure_retained(self):
        wrapper,result,robot=self.confirmed();recovery=ConfirmedAttemptRecovery();recovery.observe(result,robot)
        result,robot,target=self.feed(wrapper,7,height=.05,shift=(.009,0.,0.))
        self.assertTrue(wrapper.active.failed)
        self.assertEqual(wrapper.active.latest['protocol'],'active_vertical_visual_comotion_v4')
        self.assertEqual(result['mode'],'ATTACHED_PUBLIC')
        self.assertEqual(result['attachment_estimate']['protocol'],v56.MAINTENANCE_PROTOCOL)
        self.assertIs(result['original_probe_estimator_diagnostic'],wrapper.active.latest)
        self.assertIs(result['object_observation'],target)
        self.assertFalse(result['attachment_estimate']['evidence'].orientation_observable)
        recovery.observe(result,robot);self.assertTrue(recovery.preserves_closed_attempt(wrapper.active.attempt_id))
        pack_public_row(result,robot,True,7,'rgbd/0007.npz')

    def test_helper_failure_stays_unknown_after_old_deadline_no_reprobe_or_fake_free_forecast(self):
        wrapper,result,robot=self.confirmed();identity=wrapper.active.spec['identity']
        result,robot,_=self.feed(wrapper,7,height=.05,shift=(.011,0.,0.))
        self.assertEqual(result['mode'],'UNRESOLVED');self.assertTrue(wrapper.maintenance.failed)
        for i in range(8,20):result,robot,_=self.feed(wrapper,i,height=.05)
        self.assertEqual(result['mode'],'UNRESOLVED')
        self.assertFalse(result['free_forecast_available']);self.assertFalse(result['controlled_mode_available'])
        self.assertEqual(wrapper.active.spec['identity'],identity)
        self.assertFalse(result['attachment_estimate']['maintenance']['current_position_maintenance_passed'])

    def test_goal_fit_failure_has_no_unfiltered_policy_fallback(self):
        wrapper=self.wrapper(True,True);target,robot=sensor_frame(0)
        with patch.object(wrapper.tracker.marker,'observe',side_effect=ValueError('no sphere')),\
             patch.object(wrapper.tracker.tracker,'observe',return_value=(target,raw_metadata())),\
             patch.object(wrapper.bridge.goal_marker,'observe',return_value=(Pose([.3,.1,1.]),metadata())):
            result=wrapper.observe(observation(0.),robot,close_requested=False)
        self.assertIsNone(wrapper.binding.latest_visual_target)
        self.assertFalse(result['available']);self.assertFalse(wrapper.maintenance.started)
        self.assertTrue(result['goal_aware_filter_trace']['strict_initial_fit_unavailable_output_rejected_for_policy'])

    def test_runner_AST_preserves_current_v5_full_probe_onpolicy_and_dispatch(self):
        original=ast.parse(inspect.getsource(v5.run_public_grasp_candidates_v5))
        source=inspect.getsource(v56.run_public_candidate_goal_maintenance_v56)
        for a,b in [('run_public_candidate_goal_maintenance_v56','run_public_grasp_candidates_v5'),
            ('ObservedPublicCandidateV56Backend','ObservedPublicCandidateV5Backend'),
            ('CandidateGraspV56Coordinator','CandidateGraspV5Coordinator'),
            ('public_candidate_goal_maintenance_v56_provenance.json','public_grasp_candidates_v5_provenance.json'),
            ('public_candidate_goal_maintenance_v56_decisions.jsonl','public_grasp_candidates_v5_decisions.jsonl'),
            ('P4_candidate_transformer_v56_','P4_candidate_transformer_v5_'),
            ('P4_public_candidate_goal_maintenance_v56_development','P4_public_candidate_transformer_grasp_v5_development')]:source=source.replace(a,b)
        class Normalize(ast.NodeTransformer):
            def visit_Expr(self,node):
                call=node.value
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='_validate_v56_runner':return None
                if (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='update'
                        and any(k.arg=='optional_public_producer' for k in call.keywords)):return None
                return self.generic_visit(node)
            def visit_Call(self,node):
                node.keywords=[k for k in node.keywords if k.arg not in ('optional_public_producer','effective_attachment_final')]
                return self.generic_visit(node)
        self.assertEqual(ast.dump(original),ast.dump(Normalize().visit(ast.parse(source))))
        self.assertIn('probe_onpolicy_joint_response',source)

    def test_all_added_function_globals_resolve(self):
        missing=[]
        def scan(code):
            for op in dis.get_instructions(code):
                if op.opname=='LOAD_GLOBAL' and op.argval not in vars(v56) and not hasattr(builtins,op.argval):missing.append(op.argval)
            for value in code.co_consts:
                if isinstance(value,CodeType):scan(value)
        for value in vars(v56).values():
            if isinstance(value,FunctionType) and value.__module__==v56.__name__:scan(value.__code__)
            if isinstance(value,type) and value.__module__==v56.__name__:
                for member in vars(value).values():
                    if isinstance(member,FunctionType):scan(member.__code__)
        self.assertEqual(missing,[])

    def test_actual_outer_tracker_alias_cannot_escape_its_exact_authenticated_binding(self):
        from our_mani.benchmark_perception import RedBoxTracker
        wrapper=self.wrapper();wrapper.tracker=RedBoxTracker()
        _,robot=sensor_frame(0)
        with self.assertRaisesRegex(ValueError,'authenticated'):wrapper.observe(observation(0.),robot,close_requested=False)
        wrapper=self.wrapper();wrapper.postconfirmed_maintenance=False
        with self.assertRaisesRegex(ValueError,'modifiers'):wrapper.receipt()

    def test_observer_real50ms_and_runner_exact_first_action(self):
        wrapper=self.wrapper(False,True);raw=RawBackend(budget=2);selected=np.arange(7)*.013
        with TemporaryDirectory() as directory:
            observer=v56.ObservedPublicCandidateV56Backend(raw,Path(directory)/'public',checkpoint=CHECKPOINT,
                checkpoint_sha256=SHA,public_bridge=wrapper,postconfirmed_maintenance=True,save_frames=False)
            executor=Executor(observer)
            def track(obs):
                obj,_=sensor_frame(0,timestamp=obs['timestamp'])
                return replace(obj,points_world=box_surface(obj.pose,(.05,.05,.05))),raw_metadata()
            helper=SimpleNamespace(qualify_action=lambda backend,state,action,**kwargs:(np.asarray(action['joint_goal']).copy(),None,{}))
            with patch.object(wrapper.tracker,'observe',side_effect=track) as tracks,\
                 patch.object(wrapper.bridge.goal_marker,'observe',return_value=(Pose([.3,.1,1.]),metadata())),\
                 patch.object(v56,'build_response_planner',return_value=(SimpleNamespace(),helper,{})),\
                 patch.object(v56.CandidateGraspV56Coordinator,'decide',return_value=dict(mode='FREE_GRU',state={},action=dict(joint_goal=selected,opening=1.))):
                observer.sample();executor.h=actual_h(observer.latest_robot);executor.h['gripper_target']=1.
                result=v56.run_public_candidate_goal_maintenance_v56(observer,executor,directory,dict(
                    public_grasp_candidate_transformer_v5=True,postconfirmed_position_maintenance_v56=True,simulation_seconds=.1))
            self.assertEqual(raw.ticks,2);self.assertEqual(raw.captures,3);self.assertEqual(tracks.call_count,3)
            np.testing.assert_array_equal(executor.commands[0]['q'],selected);self.assertEqual(executor.commands[0]['opening'],1.)
            self.assertEqual(result['cycles'],1);self.assertFalse(result['Native_success_exit_used'])


if __name__=='__main__':unittest.main()
