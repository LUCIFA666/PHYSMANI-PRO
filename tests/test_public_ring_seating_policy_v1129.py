"""CPU-only public geometry/dispatch tests; no environment or Native imports."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import Pose, RobotState, ObjectState
from our_mani.multitask.interaction import PolicyState
from our_mani.multitask import public_ring_seating_policy_v1129 as module

ROOT = Path(__file__).resolve().parents[1]


def config(task):
    return json.loads((ROOT/('artifacts/framework/multitask_r10/post_training_qualification_v680_setup/'+task+'_ours_configuration.json')).read_text())


def body(identity, pose, time=5.):
    return ObjectState(identity,time,pose,np.zeros((3,3)),np.zeros(3),np.zeros(3),1.,True)


class Source:
    def __init__(self,rule,*,phase='insert',ring_pose=None):
        self.task=rule.task;self.phase_order=('approach','acquire','lift','align','insert','release')
        self.phase_index=self.phase_order.index(phase);self.done_calls=[]
        self.selected=dict(phase=phase,reference_id='peg0',manipulated_ids=('ring',),selected_at=5.,
            pose=Pose([0.,0.,1.]),opening=0. if phase=='insert' else 1.)
        self.close_witness=dict(phase_index=1,manipulated_ids=['ring'],confirmed_after=1.)
        robot=RobotState(5.,Pose([0.,0.,1.]),np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7),False,True)
        state=PolicyState(robot,{'ring':body('ring',ring_pose or rule.nominal_ring_to_peg),'peg0':body('peg0',Pose([0.,0.,0.]))},
            {'bodies':{'ring':{'motion_mode':'attached'},'peg0':{'motion_mode':'external'}},'invalid_mode_ids':[]},{})
        self.frontend=NS(state=state)

    def _done(self,goal):
        self.done_calls.append(goal['phase']);return True


class Parent:
    def __init__(self,source,*,qualified=None):
        self.source_policy=source;self.qualified=qualified;self.qualifications=[];self.confirmations=[];self.receipts=0
    def __getattr__(self,name):return getattr(self.source_policy,name)
    def qualify_opening(self,requested,current_requested):
        self.qualifications.append((requested,current_requested))
        return float(requested>=.5) if self.qualified is None else self.qualified,{'parent1121':True},None
    def confirm_opening(self,*args,**kwargs):self.confirmations.append((args,kwargs));return 'original_confirm'
    def receipt(self):self.receipts+=1;return {'parent_receipt':True}


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule=module.derive_ring_seat_rule(ROOT,config(module.TASKS[0]),module.TASKS[0])

    def evaluate(self,source):
        with patch.object(module,'_current_evidence_matches',return_value=True):
            return module.evaluate_public_ring_seating(source,self.rule)

    def test_exact_existing_train_inputs_derive_both_tasks_and_nominal_geometry(self):
        for task in module.TASKS:
            rule=module.derive_ring_seat_rule(ROOT,config(task),task)
            self.assertTrue(module._clip_peg(rule,'peg0',rule.nominal_ring_to_peg,Pose([0.,0.,0.]))[0])
            self.assertLess(rule.epsilon,1e-12)
            self.assertTrue(rule.provenance['nominal_geometric_estimate_only'])
            self.assertFalse(rule.provenance['native_success_certificate'])
            self.assertFalse(rule.provenance['template_tolerances_used_as_calibrated_public_seating_bounds'])
            self.assertFalse(rule.provenance['public_pose_uncertainty_calibrated'])
            self.assertAlmostEqual(rule.aperture[0][1],.015,places=5)
            self.assertFalse(rule.peg_vertices['peg0'].flags.writeable)
            json.dumps(module.binding(ROOT,config(task),task),allow_nan=False)

    def test_wrong_config_or_changed_frozen_bytes_are_rejected_without_fitting(self):
        value=config(self.rule.task);value['learned_artifacts'][self.rule.task]['interaction']['sha256']='changed'
        with self.assertRaises(ValueError):module.derive_ring_seat_rule(ROOT,value,self.rule.task)
        with TemporaryDirectory() as directory:
            root=Path(directory);name=module.INPUTS[self.rule.task]['interaction'][0]
            path=root/name;path.parent.mkdir(parents=True);path.write_text('changed')
            with self.assertRaisesRegex(ValueError,'frozen TRAIN input changed'):
                module.derive_ring_seat_rule(root,config(self.rule.task),self.rule.task)

    def test_nominal_seating_accepted_but_hovering_or_lateral_miss_is_not(self):
        self.assertTrue(self.evaluate(Source(self.rule))['seated'])
        nominal=self.rule.nominal_ring_to_peg
        for delta in ([0.,0.,.02],[.006,0.,0.]):
            source=Source(self.rule,ring_pose=Pose(nominal.position+delta,nominal.rotation))
            result=self.evaluate(source)
            self.assertTrue(result['known']);self.assertFalse(result['seated'])
        hover=self.evaluate(Source(self.rule,ring_pose=Pose(nominal.position+[0.,0.,.02],nominal.rotation)))
        self.assertFalse(all(hover['geometry']['axial_faces_intersect']))

    def test_large_tilt_rejected_without_arbitrary_orientation_threshold(self):
        nominal=self.rule.nominal_ring_to_peg
        tilted=Pose(nominal.position,Rotation.from_rotvec([.3,0.,0.]).as_matrix()@nominal.rotation)
        result=self.evaluate(Source(self.rule,ring_pose=tilted))
        self.assertFalse(result['seated'])
        self.assertNotIn('angle_threshold',result)

    def test_all_d4_equivalent_representatives_preserve_nominal_decision(self):
        cert=json.loads((ROOT/module.INPUTS[self.rule.task]['D4'][0]).read_text());center=np.array(cert['center_body'])
        nominal=self.rule.nominal_ring_to_peg
        for rotation in cert['group_rotations_body']:
            symmetry=np.array(rotation)
            pose=Pose(nominal.position+nominal.rotation@(center-symmetry@center),nominal.rotation@symmetry)
            self.assertTrue(self.evaluate(Source(self.rule,ring_pose=pose))['seated'])

    def test_missing_stale_invisible_held_or_own_close_evidence_stays_unknown(self):
        for bad in ('missing','stale','invisible','contact','mode','witness','future','wrong_phase_witness'):
            source=Source(self.rule);state=source.frontend.state;bodies=dict(state.bodies);metadata=deepcopy(state.perception_metadata)
            if bad=='missing':bodies.pop('ring')
            if bad=='stale':bodies['ring']=replace(bodies['ring'],timestamp=4.95)
            if bad=='invisible':bodies['ring']=replace(bodies['ring'],visible=False)
            if bad=='contact':state=replace(state,robot=replace(state.robot,contact=False))
            if bad=='mode':metadata['bodies']['ring']['motion_mode']='unknown'
            if bad=='witness':source.close_witness=None
            if bad=='future':source.close_witness['confirmed_after']=6.
            if bad=='wrong_phase_witness':source.close_witness['phase_index']=4
            source.frontend.state=replace(state,bodies=bodies,perception_metadata=metadata)
            result=self.evaluate(source);self.assertFalse(result['known']);self.assertFalse(result['seated'])

    def test_fake_rotation_evidence_never_invokes_a_duck_getter_or_match_method(self):
        source=Source(self.rule)
        class NativeDuck:
            @property
            def certificate_sha256(self):raise AssertionError('Native getter forbidden')
            def matches(self,**kwargs):raise AssertionError('duck Native method forbidden')
        source.frontend.state.perception_metadata['bodies']['ring']['rotation_equivalence_evidence']=NativeDuck()
        result=module.evaluate_public_ring_seating(source,self.rule)
        self.assertFalse(result['known'])


class DispatchScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rule=module.derive_ring_seat_rule(ROOT,config(module.TASKS[0]),module.TASKS[0])

    def test_insert_scope_replaces_only_done_and_restores_even_on_exception(self):
        source=Source(self.rule);original=source._done;before=deepcopy(source.selected)
        with patch.object(module,'_current_evidence_matches',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'body error'):
                with module.owned_ring_seating_scope(source,self.rule):
                    self.assertTrue(source._done(source.selected));self.assertEqual(source.done_calls,[])
                    self.assertTrue(source._done(dict(phase='release')));self.assertEqual(source.done_calls,['release'])
                    np.testing.assert_array_equal(source.selected['pose'].xyzw(),before['pose'].xyzw())
                    self.assertEqual(source.selected['selected_at'],before['selected_at'])
                    raise RuntimeError('body error')
        self.assertNotIn('_done',source.__dict__);self.assertEqual(source._done,original)

    def test_3cm_parent_done_does_not_advance_hovering_insert(self):
        nominal=self.rule.nominal_ring_to_peg
        source=Source(self.rule,ring_pose=Pose(nominal.position+[0,0,.02],nominal.rotation))
        self.assertTrue(source._done(source.selected))
        with patch.object(module,'_current_evidence_matches',return_value=True),module.owned_ring_seating_scope(source,self.rule):
            self.assertFalse(source._done(source.selected))

    def test_release_closed_to_open_defers_then_passes_with_one_parent_call(self):
        source=Source(self.rule,phase='release');parent=Parent(source);wrapper=module.RingGuardedReleaseWitness(parent,self.rule)
        with patch.object(module,'evaluate_public_ring_seating',return_value={'known':True,'seated':False}):
            opening,proof,proposed=wrapper.qualify_opening(1.,0.)
        self.assertEqual(opening,0.);self.assertEqual(parent.qualifications,[(1.,0.)])
        self.assertEqual(proof['reason'],'defer_first_release_open_until_public_ring_seating');self.assertIsNone(proposed)
        self.assertEqual(wrapper.confirm_opening(opening,proof,proposed,confirmed_after=5.05),'original_confirm')
        self.assertEqual(len(parent.confirmations),1);self.assertEqual(parent.confirmations[0][0][0],0.)
        with patch.object(module,'evaluate_public_ring_seating',return_value={'known':True,'seated':True}):
            self.assertEqual(wrapper.qualify_opening(1.,0.)[0],1.)
        self.assertEqual(len(parent.qualifications),2)

    def test_already_open_parent_defer_preclose_and_nonrelease_are_not_overridden(self):
        source=Source(self.rule,phase='release');parent=Parent(source);wrapper=module.RingGuardedReleaseWitness(parent,self.rule)
        with patch.object(module,'evaluate_public_ring_seating',side_effect=AssertionError('must not evaluate')):
            self.assertEqual(wrapper.qualify_opening(1.,1.)[0],1.)
            self.assertEqual(wrapper.qualify_opening(0.,0.)[0],0.)
            parent.qualified=0.;self.assertEqual(wrapper.qualify_opening(1.,0.)[0],0.)
            source.selected['phase']='acquire';self.assertEqual(wrapper.qualify_opening(0.,1.)[0],0.)

    def test_integrated_scope_owns_final_dispatch_and_does_not_change_targets_or_observer(self):
        source=Source(self.rule);parent=Parent(source);executor=NS(skill_policy=parent);frontend=source.frontend
        with module.integrated_ring_seating_scope(ROOT,config(self.rule.task),self.rule.task,frontend,parent,executor) as result:
            self.assertIs(result[0],frontend);self.assertIs(result[2],executor)
            self.assertIs(executor.skill_policy,result[1]);self.assertIs(result[1].source_policy,source)
            self.assertIs(result[1].parent_dispatch,parent)
            self.assertFalse(result[1].receipt()['public_ring_seating_v1129']['native_sensor_or_task_success_input'])
        self.assertIs(executor.skill_policy,parent);self.assertNotIn('_done',source.__dict__)

    def test_receipt_is_bounded_and_parent_chain_called_once(self):
        parent=Parent(Source(self.rule,phase='release'));wrapper=module.RingGuardedReleaseWitness(parent,self.rule,max_events=1)
        with patch.object(module,'evaluate_public_ring_seating',return_value={'known':False,'seated':False}):
            wrapper.qualify_opening(1.,0.);wrapper.qualify_opening(1.,0.)
        receipt=wrapper.receipt();self.assertEqual(parent.receipts,1)
        self.assertEqual(receipt['public_ring_seating_v1129']['dropped_events'],1)
        json.dumps(receipt,allow_nan=False)


if __name__=='__main__':unittest.main()
