"""CPU-only semantic-witness contracts; no Native, model or job construction."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import numpy as np

from our_mani.contracts import ObjectState,Pose,RobotState
from our_mani.multitask.interaction import PolicyState
from our_mani.multitask.perception import BodyTemplate,GeometryComponent,PriorProvenance
from our_mani.multitask.public_roles import PublicRoleResolver,PublicRoleResolutionError
from our_mani.multitask.specs import TASK_SPECS
from our_mani.multitask import committed_public_role_integration_v1137 as subject

IDS=('ring','frame','peg0','peg1','peg2')
ORDER=('approach','acquire','lift','align','insert','release')


def fixture():
    prior=PriorProvenance('declared_cad','frozen TRAIN CAD','body',True)
    templates=tuple(BodyTemplate(name,(GeometryComponent('colour_'+name,(index+100,),np.zeros((6,3))),),prior)
                    for index,name in enumerate(IDS))
    resolver=PublicRoleResolver.__new__(PublicRoleResolver)
    resolver.spec=TASK_SPECS[subject.TASKS[0]]
    resolver.artifact=dict(artifact_sha256='original-public-colour',colours={'red':{'aliases':['red']},'blue':{'aliases':['blue']}})
    resolver.config=NS(maximum_observation_age=.2)
    resolver._role_colour=Mock(side_effect=lambda role,*args:dict(valid=True,colour_name='red' if role=='peg0' else 'blue',
        colour_artifact_sha256='original-public-colour'))
    controller=NS(library=NS(sha256='frozen-library'),committed_roles={'selected_peg':('peg0',)},
        measured_attachments={'ring':dict(local_pose_xyzw=[0.,0.,0.,0.,0.,0.,1.],observed_at=.55,
            source='current_RGBD_body_pose_and_robot_proprioception_at_public_contact')})
    actor=NS(role_resolver=resolver,controller=controller,requested_opening=0.)
    source=NS(task=subject.TASKS[0],phase_index=4,phase_order=ORDER,selected=None,close_witness=dict(
        phase_index=1,manipulated_ids=['ring'],issued_at=.5,confirmed_after=.55),frontend=NS(state='old frame must not be read'))
    original=Mock(wraps=resolver.resolve)
    continuation=subject.PublicCommitmentContinuation(source,actor,original)
    return NS(actor=actor,source=source,resolver=resolver,templates=templates,original=original,continuation=continuation)


def state(now=1.,*,direct=True,contact=True,roles=None):
    robot=RobotState(now,Pose([0.,0.,0.]),np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7),False,contact)
    bodies={identity:ObjectState(identity,now,Pose([0.,0.,0.]),np.zeros((6,3)),np.zeros(3),np.zeros(3),visible=direct)
            for identity in IDS}
    info={identity:dict(motion_mode='attached' if identity=='ring' else 'external',
        mode_provenance='caller_controller_touch_action_proprioception',directly_visible=direct,registration_valid=True,
        pose_source='current_group_RGBD_and_declared_rigid_layout',pose_evidence_until=now) for identity in IDS}
    return PolicyState(robot,bodies,dict(bodies=info),roles or {identity:(identity,) for identity in IDS})


def call(f,now,*,direct=True,description='put ring onto the red peg',current=None):
    return f.continuation.resolve((description,),state(now,direct=direct) if current is None else current,
        {'timestamp':now},f.templates)


class WitnessContracts(unittest.TestCase):
    def test_real_original_resolver_success_then_colour_absence_keeps_only_current_semantics(self):
        f=fixture();first=call(f,1.)
        self.assertEqual(first[0]['selected_peg'],('peg0',));self.assertEqual(f.original.call_count,1)
        current=state(10.,direct=False,roles={'fresh_role':('frame',)})
        roles,proof=call(f,10.,direct=False,current=current)
        self.assertEqual(f.original.call_count,2)
        self.assertEqual(roles,{'fresh_role':('frame',),'selected_peg':('peg0',)})
        self.assertFalse(proof['current_colour_revalidated']);self.assertFalse(proof['current_pose_or_geometry_qualification_granted'])
        self.assertEqual(proof['identity_qualification_at'],1.)
        self.assertEqual(f.continuation.uses,1)
        self.assertEqual(f.source.frontend.state,'old frame must not be read')

    def test_no_initial_original_success_or_wrong_success_proof_never_creates_witness(self):
        f=fixture()
        with self.assertRaises(PublicRoleResolutionError):call(f,1.,direct=False)
        self.assertIsNone(f.continuation.witness)
        f=fixture();f.original=Mock(return_value=({'selected_peg':('peg0',)},dict(body_id='peg0')))
        f.continuation.original=f.original;call(f,1.)
        self.assertIsNone(f.continuation.witness)

    def test_only_precise_colour_error_can_continue_and_other_errors_keep_identity(self):
        for error in (PublicRoleResolutionError('misaligned current RGB/instance colour pixels'),
                      PublicRoleResolutionError('public task language is required'),ValueError('original helper failed')):
            f=fixture();call(f,1.);f.original.side_effect=error
            with self.assertRaises(type(error)) as raised:call(f,2.,direct=False)
            self.assertIs(raised.exception,error);self.assertEqual(f.continuation.uses,0)

    def test_post_acquire_phase_changes_with_selected_none_keep_original_close_token_valid(self):
        f=fixture();f.source.phase_index=2;call(f,1.)
        for phase,now in ((3,2.),(4,3.),(5,4.)):
            f.source.phase_index=phase;f.source.selected=None
            self.assertEqual(call(f,now,direct=False)[0]['selected_peg'],('peg0',))
        self.assertEqual(f.continuation.uses,3)

    def test_phase_acquire_uncommitted_or_different_committed_target_never_continues(self):
        for change in ('acquire','uncommitted','other_target'):
            f=fixture();call(f,1.)
            if change=='acquire':f.source.phase_index=1
            if change=='uncommitted':f.actor.controller.committed_roles={}
            if change=='other_target':f.actor.controller.committed_roles['selected_peg']=('peg1',)
            with self.assertRaises(PublicRoleResolutionError):call(f,2.,direct=False)
            self.assertEqual(f.continuation.uses,0)

    def test_held_mode_close_and_actual_measured_attachment_are_required(self):
        for bad in ('contact','open','mode','multiple_attached','close_phase','close_ids','future_close',
                    'missing_measure','measure_source','measure_before_close','invalid_mode'):
            f=fixture();call(f,1.);current=state(2.,direct=False)
            if bad=='contact':current=state(2.,direct=False,contact=False)
            if bad=='open':f.actor.requested_opening=1.
            if bad=='mode':current.perception_metadata['bodies']['ring']['motion_mode']='external'
            if bad=='multiple_attached':current.perception_metadata['bodies']['peg1']['motion_mode']='attached'
            if bad=='close_phase':f.source.close_witness['phase_index']=4
            if bad=='close_ids':f.source.close_witness['manipulated_ids']=['frame']
            if bad=='future_close':f.source.close_witness['confirmed_after']=3.
            if bad=='missing_measure':f.actor.controller.measured_attachments={}
            if bad=='measure_source':f.actor.controller.measured_attachments['ring']['source']='native_ID'
            if bad=='measure_before_close':f.actor.controller.measured_attachments['ring']['observed_at']=.1
            if bad=='invalid_mode':current.perception_metadata['invalid_mode_ids']=['ring']
            with self.subTest(bad=bad),self.assertRaises(PublicRoleResolutionError):call(f,2.,current=current)
            self.assertEqual(f.continuation.uses,0)

    def test_new_original_success_conflicting_with_commit_does_not_authorize_old_target(self):
        f=fixture();call(f,1.)
        f.resolver._role_colour.side_effect=lambda role,*args:dict(valid=True,colour_name='red' if role=='peg1' else 'blue',
            colour_artifact_sha256='original-public-colour')
        self.assertEqual(call(f,2.)[0]['selected_peg'],('peg1',))
        self.assertIsNone(f.continuation.witness)
        with self.assertRaises(PublicRoleResolutionError):call(f,3.,direct=False)

    def test_clock_and_phase_rewind_cannot_requalify_old_close_with_new_colour_success(self):
        for rewind in ('clock','phase'):
            f=fixture();call(f,5.);call(f,5.1,direct=False)
            if rewind=='phase':f.source.phase_index=3
            call(f,4. if rewind=='clock' else 5.2)
            with self.assertRaises(PublicRoleResolutionError):call(f,4.1 if rewind=='clock' else 5.3,direct=False)
            self.assertEqual(f.continuation.uses,1)
            self.assertTrue(f.continuation.scope_invalidated)

    def test_changed_language_topology_library_or_artifact_cannot_reuse_old_close(self):
        for change in ('description','topology','library','artifact'):
            f=fixture();call(f,1.)
            description='place ring onto the red peg' if change=='description' else 'put ring onto the red peg'
            if change=='topology':
                first=f.templates[0];f.templates=(BodyTemplate(first.role_id,(GeometryComponent('new',(999,),np.zeros((6,3))),),first.provenance),*f.templates[1:])
            if change=='library':f.actor.controller.library.sha256='new-library'
            if change=='artifact':f.resolver.artifact['extra']='changed'
            call(f,2.,description=description)
            with self.assertRaises(PublicRoleResolutionError):call(f,3.,direct=False,description=description)
            self.assertEqual(f.continuation.uses,0)

    def test_open_or_loss_invalidates_prior_close_even_if_later_public_contact_returns(self):
        for change in ('open','lost'):
            f=fixture();call(f,1.)
            if change=='open':f.actor.requested_opening=1.
            current=state(2.,direct=False,contact=change!='lost')
            with self.assertRaises(PublicRoleResolutionError):call(f,2.,current=current)
            f.actor.requested_opening=0.
            with self.assertRaises(PublicRoleResolutionError):call(f,3.,direct=False)

    def test_verify_is_expected_inactive_with_no_recording_errors_or_continuation(self):
        f=fixture();call(f,1.);before=deepcopy(f.continuation.witness)
        f.source.phase_index=len(ORDER)
        call(f,2.)
        self.assertEqual(f.continuation.witness,before)
        with self.assertRaises(PublicRoleResolutionError):call(f,3.,direct=False)
        self.assertEqual(f.continuation.recording_errors,0);self.assertEqual(f.continuation.uses,0)

    def test_bounded_receipt_does_not_change_original_call_count(self):
        f=fixture();f.continuation.max_events=1;call(f,1.)
        call(f,2.,direct=False);call(f,3.,direct=False)
        receipt=f.continuation.receipt()
        self.assertEqual(f.original.call_count,3);self.assertEqual(receipt['continuation_uses'],2)
        self.assertEqual(len(receipt['events']),1);self.assertEqual(receipt['dropped_events'],2)


class ScopeContracts(unittest.TestCase):
    def test_ring_patches_only_owned_resolver_keeps_ready_guard_and_executor_and_restores(self):
        f=fixture();original=f.resolver.resolve;ready=object();frontend=NS(ready=ready)
        dispatched=NS(source_policy=f.source,receipt=lambda:{'parent1131':True});executor=NS(skill_policy=dispatched)
        @contextmanager
        def parent(*args):yield frontend,dispatched,executor
        events={}
        with patch.object(subject.parent,'integrated_components',side_effect=parent):
            with subject.integrated_components(None,{},subject.TASKS[0],f.actor,events=events) as triple:
                self.assertEqual(triple,(frontend,dispatched,executor));self.assertIs(frontend.ready,ready)
                f.resolver.resolve(('red peg',),state(1.),{'timestamp':1.},f.templates)
                self.assertEqual(f.resolver.resolve(('red peg',),state(2.,direct=False),{'timestamp':2.},f.templates)[0]['selected_peg'],('peg0',))
                self.assertIn('committed_public_role_continuation_v1137',dispatched.receipt())
                self.assertIs(executor.skill_policy,dispatched)
        self.assertEqual(f.resolver.resolve,original);self.assertTrue(events['restored']);self.assertTrue(events['witness_cleared'])

    def test_other_tasks_are_exact_parent_triples_and_original_exception_restores(self):
        for task in ('beat_the_rotating_buzz','push_moving_button','remove_cups_from_rotating_frame'):
            triple=(object(),object(),object())
            @contextmanager
            def parent(*args):yield triple
            with patch.object(subject.parent,'integrated_components',side_effect=parent):
                with subject.integrated_components(None,{},task,object()) as actual:self.assertEqual(actual,triple)
        f=fixture();original=f.resolver.resolve;dispatched=NS(source_policy=f.source,receipt=lambda:{})
        @contextmanager
        def parent(*args):yield object(),dispatched,object()
        error=RuntimeError('original actuator error');events={}
        with patch.object(subject.parent,'integrated_components',side_effect=parent):
            with self.assertRaises(RuntimeError) as raised:
                with subject.integrated_components(None,{},subject.TASKS[0],f.actor,events=events):raise error
        self.assertIs(raised.exception,error);self.assertEqual(f.resolver.resolve,original);self.assertTrue(events['restored'])


if __name__=='__main__':unittest.main()
