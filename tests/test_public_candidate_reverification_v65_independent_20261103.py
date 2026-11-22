"""Synthetic V65 integration contracts; no fake passing production constructor.

Functional methods are isolated from the Native/model capability constructor.
The prototype/committed owner and all robot/body samples remain real typed
objects. P0 separately covers the genuine constructor and saved-public pack.
"""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from our_mani.local_mpc_control_20261103 import public_candidate_reverification_v65 as subject
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors, PublicMotionEvidence, CHECKPOINT_SHA256
from our_mani.local_mpc_control_20261103.active_visual_attachment import ActiveVisualAttachment
from our_mani.local_mpc_control_20261103.public_reverified_maintenance_owner_20261103_v1 import PublicReverifiedMaintenanceOwner
from our_mani.local_mpc_p4_20261103.runtime import pack_public_row
from tests.test_public_reverification_epoch_prototype_20261103 import fixture, frame, actual_h
from tests.test_public_reverified_maintenance_owner_20261103_v1 import committed


class Binding:
    def __init__(self):
        self.latest_visual_target=None;self.latest_result=None;self.last_frame_timestamp=None
    def approve(self,result):
        if result is not self.latest_result:raise ValueError('fixture exact current result required')
    def bind_current(self,result,target):
        self.latest_result=result;self.latest_visual_target=target
    def receipt(self):return dict(binding_version='isolated_CPU_method_fixture')


class Harness:
    def __init__(self):
        _,self.old,self.baseline=fixture()
        bridge=object.__new__(subject.PublicCandidateReverificationBridgeV65)
        self.bridge=bridge;self.target=None;self.track_calls=0;self.consume_calls=0;self.active_calls=0
        self.robot=None
        primitive=SimpleNamespace(_last_open_visual=self.baseline,_last_close_requested=True)
        delegate=SimpleNamespace(maintenance=self.old,close_visual_baseline=self.baseline,
            delegate=primitive,retired_maintenance=[],last_initial_admission=None)
        bridge.delegate=delegate;bridge.binding=Binding()
        bridge.tracker=SimpleNamespace(sphere=object(),last_current_target=None,_last_observed_timestamp=None)
        bridge.cache=object();bridge.estimator=SimpleNamespace(observe=lambda *args,**kwargs:dict(reason='diagnostic_only'))
        bridge.active=SimpleNamespace(attempt_id=1,attempt_active=True,observe=self.active_observe)
        delegate._track_once=self.track
        bridge.bridge=SimpleNamespace(observe=self.consume)
        bridge.goal_sphere_filter=bridge.postconfirmed_maintenance=True
        bridge.witness=bridge.epoch_owner=None;bridge.used_attempt_ids=set();bridge.retired_reverification=[]
        bridge.latest_result=bridge.latest_attachment=bridge.latest_raw_active=None
        bridge._frame=bridge._inner_robot=bridge._last_close=bridge.last_admission=None
        bridge._source_stats={}

    def track(self,observation,robot):
        self.track_calls+=1;self.robot=robot
        self.bridge.tracker.last_current_target=self.target
        self.bridge.tracker._last_observed_timestamp=robot.timestamp
        self.bridge.binding.last_frame_timestamp=robot.timestamp
        return self.target,{},None

    def active_observe(self,target,robot,**kwargs):
        self.active_calls+=1
        return dict(timestamp=robot.timestamp,protocol='active_vertical_visual_comotion_v4',attempt_id=1,
            attempt_active=True,mode='UNRESOLVED',probe=None,
            evidence=PublicMotionEvidence('manipulation_unresolved',robot.timestamp,('public_robot_pose','public_gripper_opening')))

    def consume(self,observation,robot,*,close_requested,evidence):
        self.consume_calls+=1
        attached=type(evidence)is PublicMotionEvidence and evidence.event=='attachment_confirmed'
        return dict(timestamp=robot.timestamp,object_observation=self.target,
            mode='ATTACHED_PUBLIC'if attached else'UNRESOLVED',available=attached,
            free_forecast_available=False,controlled_mode_available=attached,forecast=None,
            Native_inputs_used=False,future_observations_used=False,observations_fabricated=False,
            forecaster_is_injected_test_double=False,goal_marker_dependency_injected=False,
            frozen_checkpoint_sha256=CHECKPOINT_SHA256)

    def observe(self,i,height=.012,*,missing=False):
        raw,robot=frame(i,height);self.target=None if missing else raw['object_observation']
        with patch.object(subject.PublicCandidateReverificationBridgeV65,'_validate_components',lambda owner:None):
            result=self.bridge.observe({},robot,close_requested=True)
        return result,robot

    def admit(self,result,robot):
        outer=PublicRobotSensors.from_mapping(dict(timestamp=robot.timestamp,
            tcp_position=robot.tcp_position,tcp_quaternion=robot.tcp_quaternion,q=robot.q,dq=robot.dq,
            finger_open_amounts=robot.finger_open_amounts))
        assert outer is not robot
        with patch.object(subject.PublicCandidateReverificationBridgeV65,'_validate_components',lambda owner:None):
            answer=self.bridge.try_begin_natural_witness(result,outer,actual_h(outer))
        return answer,outer


class V65Independent(unittest.TestCase):
    def test_equal_outer_robot_admits_using_exact_inner_once(self):
        h=Harness();result,inner=h.observe(8)
        admission,outer=h.admit(result,inner)
        self.assertTrue(admission['admitted']);self.assertIsNot(inner,outer)
        self.assertIs(h.bridge.witness.last_robot,inner)
        self.assertIs(h.bridge.witness.last_result,h.bridge._frame)
        self.assertEqual(h.bridge.witness.witness_mode,'natural')
        with patch.object(subject.PublicCandidateReverificationBridgeV65,'_validate_components',lambda owner:None):
            self.assertIsNone(h.bridge.try_begin_natural_witness(result,outer,actual_h(outer)))
        self.assertEqual(h.bridge.used_attempt_ids,{1})

    def test_missing_or_different_outer_robot_cannot_start(self):
        h=Harness();result,robot=h.observe(8,missing=True)
        admission,_=h.admit(result,robot);self.assertIsNone(admission)
        self.assertIsNone(h.bridge.witness.spec)
        h=Harness();result,robot=h.observe(8)
        bad=PublicRobotSensors(robot.timestamp,robot.tcp_position,robot.tcp_quaternion,robot.q+.001,robot.dq,robot.finger_open_amounts)
        with patch.object(subject.PublicCandidateReverificationBridgeV65,'_validate_components',lambda owner:None):
            with self.assertRaisesRegex(ValueError,'inner robot'):
                h.bridge.try_begin_natural_witness(result,bad,actual_h(bad))
            with self.assertRaisesRegex(ValueError,'exact current result'):
                h.bridge.try_begin_natural_witness(deepcopy(result),robot,actual_h(robot))
        self.assertIsNone(h.bridge.witness.spec)

    def test_static_visible_reappearance_never_commits(self):
        h=Harness();result,robot=h.observe(8);h.admit(result,robot)
        for i in range(9,14):h.observe(i)
        self.assertEqual(h.bridge.verification_epoch,0);self.assertIsNone(h.bridge.epoch_owner)
        self.assertFalse(h.bridge.witness.latest['current_fresh_witness_passed'])
        self.assertTrue(h.old.failed)

    def test_commit_frame_single_consume_then_50ms_failure_latches(self):
        h=Harness();original_r0=h.old.r0.copy();original_baseline=h.baseline.pose.position.copy()
        result,robot=h.observe(8);h.admit(result,robot)
        for i in range(9,14):result,robot=h.observe(i,.012+.012*(i-8)/5)
        self.assertEqual(h.bridge.verification_epoch,1)
        owner=h.bridge.epoch_owner
        self.assertFalse(owner.failed)
        self.assertEqual(owner.last_timestamp,robot.timestamp)
        self.assertTrue(owner.latest['current_reverified_position_maintenance_passed'])
        self.assertEqual((h.track_calls,h.active_calls,h.consume_calls),(6,6,6))
        result,robot=h.observe(14,.026,missing=True)
        self.assertTrue(owner.failed);self.assertEqual(result['attachment_estimate']['mode'],'UNRESOLVED')
        h.observe(15,.028)
        self.assertTrue(owner.failed);self.assertEqual(h.bridge.verification_epoch,1)
        self.assertTrue(h.old.failed);np.testing.assert_array_equal(h.old.r0,original_r0)
        np.testing.assert_array_equal(h.bridge.close_visual_baseline.pose.position,original_baseline)

    def test_public_summary_small_and_pack_does_not_receive_full_proof(self):
        h=Harness();result,robot=h.observe(8);h.admit(result,robot)
        for i in range(9,14):result,robot=h.observe(i,.012+.012*(i-8)/5)
        summary=result['public_reverification']
        self.assertEqual(set(summary),{'protocol','timestamp','physical_attempt_id','verification_epoch','admission_attempted',
            'witness_started','witness_failed','epoch_owner_failed','current_reverified_position_maintenance_passed',
            'natural_motion_only','active_test_action_executed','old_failed_owner_reset','original_references_rolled',
            'initial_probe_changed','force_or_contact_qualified'})
        self.assertNotIn('witness',result);self.assertNotIn('epoch_owner',result)
        self.assertNotIn('spec',summary);self.assertNotIn('committed',summary)
        self.assertIn('spec',h.bridge.private_diagnostics()['witness'])
        packed=pack_public_row(result,robot,True,13,'rgbd/0013.npz')
        self.assertEqual(packed['result']['mode'],'ATTACHED_PUBLIC')

    def test_critical_witness_and_owner_method_replacement_rejected(self):
        for name in ('observe','begin_witness','commit_verified_epoch'):
            with self.subTest(witness_method=name):
                p,_,_=fixture();bridge=object.__new__(subject.PublicCandidateReverificationBridgeV65)
                delegate=object.__new__(subject.base.PublicCandidatePositionBridgeV64)
                bridge.delegate=delegate;bridge.goal_sphere_filter=bridge.postconfirmed_maintenance=True
                for key in ('tracker','cache','bridge','binding','estimator'):
                    value=object();setattr(bridge,key,value);setattr(delegate,key,value)
                bridge.active=delegate.active=ActiveVisualAttachment();bridge.witness=p;bridge.epoch_owner=None;bridge._source_stats={}
                setattr(p,name,lambda *args,**kwargs:None)
                with patch.object(subject.base.PublicCandidatePositionBridgeV64,'_validate_components',lambda owner:None):
                    with self.assertRaises(ValueError):bridge._validate_components()
        p,_,_=committed();owner=PublicReverifiedMaintenanceOwner(p)
        bridge.witness=p;bridge.epoch_owner=owner;owner.observe=lambda *args,**kwargs:None
        with patch.object(subject.base.PublicCandidatePositionBridgeV64,'_validate_components',lambda owner:None):
            with self.assertRaises(ValueError):bridge._validate_components()

    def test_runner_loop_is_v64_math_exact_after_declared_log_names(self):
        import inspect
        old=ast.parse(inspect.getsource(subject.base.run_public_candidate_position_v64))
        new=ast.parse(inspect.getsource(subject.run_public_candidate_reverification_v65))
        oldloop=next(node for node in ast.walk(old) if isinstance(node,ast.While))
        newloop=next(node for node in ast.walk(new) if isinstance(node,ast.While))
        class Names(ast.NodeTransformer):
            def visit_Constant(self,node):
                if isinstance(node.value,str):node.value=node.value.replace('v65','v64')
                return node
        self.assertEqual(ast.dump(oldloop,include_attributes=False),ast.dump(Names().visit(newloop),include_attributes=False))


if __name__=='__main__':unittest.main()
