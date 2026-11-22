"""Pure fake-call contract checks; no frozen executor/model/native imports."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from our_mani.multitask import rml_component_domain_v640 as capabilities
from our_mani.multitask import train_public_capture_diagnostics_v815 as module


def capability(split='train'):
    # Synthetic sealed domain for CPU tests; no for_train loader/preflight.
    return capabilities.ComponentDomain('train_qualification' if split=='train' else 'evaluation',
        'remove_cups_from_rotating_frame','context_geometry',split,'CPU_fake_only',
        None if split=='train' else 'a'*64,capabilities._TOKEN)


class Poison:
    def __getattribute__(self,name): raise AssertionError('forbidden extra access: '+name)


class Port:
    def __init__(self):self.calls=0;self.proof={'ready':False,'reason':'already_computed_public_forecast_unavailable'}
    def readiness(self,state,forecast,phase):
        self.calls+=1
        return self.proof


class Frontend:
    def __init__(self):
        self.role_ready=True;self.port=Port();self.ready_calls=0
        pose=SimpleNamespace(position=[0.,0.,.8],rotation=[[1,0,0],[0,1,0],[0,0,1]])
        self.state=SimpleNamespace(robot=SimpleNamespace(timestamp=.2,contact=True,gripper_open=False,
            joints=[0.1,0.2],pose=pose),role_bindings={'selected_cup':('cup1',)},
            perception_metadata={'invalid_mode_ids':[], 'bodies':{
                'cup1':dict(motion_mode='attached',mode_provenance='caller_controller_touch_action_proprioception',
                    registration_valid=True,directly_visible=True,pose_evidence_until=.2)}})
        self.forecast=SimpleNamespace(object_ids=('frame',),times=[.2,.25],metadata={'observed_until':.2})
        self.actor=SimpleNamespace(collision_port=self.port,
            controller=SimpleNamespace(measured_attachments={'cup1':dict(local_pose_xyzw=[0,0,.02,0,0,0,1],observed_at=.2)}),
            last_diagnostics={'invalid_mode_ids':[], 'mode_estimates':{'cup1':'attached'}})
        # Any accidental native/env/truth inspection must fail a test.
        self.actor.env=Poison();self.actor.native_task=Poison()

    def ready(self,phase):
        self.ready_calls+=1
        if not self.role_ready:return False
        return self.port.readiness(self.state,self.forecast,phase)['ready']


class Policy:
    task='remove_cups_from_rotating_frame'
    def __init__(self):
        self.frontend=Frontend();self.phase_index=1;self.phase_epoch=.1
        self.close_witness={'phase_index':1,'manipulated_ids':['cup1'],'confirmed_after':.15}
        template=SimpleNamespace(manipulated_role='selected_cup',duration_seconds=.5,
            reference_kind='observed_se3',reference_time_basis='command_goal')
        self.selected={'phase':'acquire','template':template,'reference_id':'spoke1',
            'manipulated_ids':('cup1',),'selected_at':.2,'predicted_reference_time':.7}
        self.calls={name:0 for name in ('select','_done','_public_capture','_completion_target','qualify_opening','confirm_opening')}
        self.target=self.frontend.state.robot.pose;self.capture_result=object();self.done_result=False
        self.capture_error=None;self.action=None
        self.opening_return=(0.,{'current_position_error_m':.012,'current_orientation_error_rad':.04,
            'reason':'already_computed_original_proof'},None)

    def select(self):
        self.calls['select']+=1
        self._done(self.selected)
        if not self.frontend.ready('acquire'):return None
        return self.action
    def _done(self,goal):
        self.calls['_done']+=1
        self._completion_target(goal)
        self._public_capture(goal)
        return self.done_result
    def _public_capture(self,goal):
        self.calls['_public_capture']+=1
        if self.capture_error is not None:raise self.capture_error
        return self.capture_result
    def _completion_target(self,goal):
        self.calls['_completion_target']+=1
        return self.target
    def qualify_opening(self,requested,current_requested):
        self.calls['qualify_opening']+=1
        return self.opening_return
    def confirm_opening(self,opening,proof,proposed,*,confirmed_after):
        self.calls['confirm_opening']+=1
        self.close_witness={'phase_index':1,'manipulated_ids':['cup1'],'confirmed_after':confirmed_after}


class ObserverTests(unittest.TestCase):
    def fixture(self,**kwargs):
        policy=Policy()
        observer=module.TrainPublicCaptureDiagnostics(policy,domain=capability(),enabled=True,
            development_only=True,**kwargs)
        return policy,observer

    def shadows(self,policy):
        return [(obj,dict(vars(obj))) for obj in (policy,policy.frontend,policy.frontend.port)]

    def assert_restored(self,before):
        methods=('select','_done','_public_capture','_completion_target','qualify_opening','confirm_opening','ready','readiness')
        for obj,raw in before:
            for name in methods:
                self.assertEqual(name in vars(obj),name in raw)
                if name in raw:self.assertIs(vars(obj)[name],raw[name])

    def test_default_inert_and_no_capability_or_policy_access(self):
        with module.TrainPublicCaptureDiagnostics(Poison(),domain=Poison()) as observer:
            self.assertFalse(observer.bind_context(tick=-1,action_id=None,action_kind='invalid'))
        self.assertEqual(observer.events,[]);self.assertIsNone(observer.receipt()['source_sha256'])

    def test_VAL_TEST_and_missing_development_optin_rejected_without_calls(self):
        policy=Policy();before=self.shadows(policy)
        for split,development in [('validation',True),('test',True),('train',False)]:
            with self.subTest(split=split,development=development),self.assertRaises(ValueError):
                with module.TrainPublicCaptureDiagnostics(policy,domain=capability(split),enabled=True,development_only=development):pass
        self.assertTrue(all(n==0 for n in policy.calls.values()));self.assert_restored(before)

    def test_original_nested_calls_once_and_original_side_effectful_ready_not_repeated(self):
        policy,observer=self.fixture();before=self.shadows(policy)
        with observer:
            observer.bind_context(tick=4,action_id='A7',action_kind='geometry_execute',action_tick=0)
            self.assertIsNone(policy.select())
        self.assertEqual({k:policy.calls[k] for k in ('select','_done','_completion_target','_public_capture')},
            {'select':1,'_done':1,'_completion_target':1,'_public_capture':1})
        self.assertEqual(policy.frontend.ready_calls,1);self.assertEqual(policy.frontend.port.calls,1)
        self.assert_restored(before)
        by_name={event['method']:event for event in observer.events}
        self.assertEqual(by_name['_public_capture']['parent_call_id'],by_name['_done']['call_id'])
        self.assertEqual(by_name['_done']['parent_call_id'],by_name['select']['call_id'])
        self.assertEqual(by_name['readiness']['parent_call_id'],by_name['ready']['call_id'])
        self.assertEqual(by_name['readiness']['original_return'],policy.frontend.port.proof)
        capture=by_name['_public_capture']
        self.assertEqual(capture['public_operands']['exact_tick'],4)
        self.assertEqual(capture['public_operands']['exact_action_id'],'A7')
        self.assertEqual(capture['public_operands']['public_frame']['measured_attachment_ids'],['cup1'])
        self.assertEqual(observer.diagnostic_errors,[])

    def test_full_subpredicate_vector_is_explicitly_unknown_not_recomputed(self):
        policy,observer=self.fixture()
        with observer:policy._done(policy.selected)
        value=observer.receipt();self.assertFalse(value['full_subpredicate_vector_available'])
        for event in value['events']:
            self.assertEqual(event['extra_predicate_or_geometry_calls'],0)
            for condition in event['original_subpredicate_evaluations'].values():
                self.assertIsNone(condition['value']);self.assertEqual(condition['status'],'not_exposed_by_original_call')
        done=next(e for e in value['events'] if e['method']=='_done')
        self.assertNotIn('computed_distance',done)
        self.assertTrue(any('distance/angle' in line for line in value['limitations']))

    def test_original_readiness_short_circuit_does_not_create_missing_child_call(self):
        policy,observer=self.fixture();policy.frontend.role_ready=False
        with observer:self.assertFalse(policy.frontend.ready('acquire'))
        self.assertEqual(policy.frontend.port.calls,0)
        self.assertEqual([e['method'] for e in observer.events],['ready'])
        self.assertEqual(observer.events[0]['public_operands']['role_ready'],False)

    def test_exact_result_objects_and_existing_geometry_or_proof_forwarded(self):
        policy,observer=self.fixture()
        with observer:
            self.assertIs(policy._public_capture(policy.selected),policy.capture_result)
            self.assertIs(policy._completion_target(policy.selected),policy.target)
            self.assertIs(policy.qualify_opening(0.,1.),policy.opening_return)
        target=next(e for e in observer.events if e['method']=='_completion_target')
        self.assertEqual(target['original_returned_public_target']['position'],[0.,0.,.8])
        proof=next(e for e in observer.events if e['method']=='qualify_opening')['original_return'][1]
        self.assertEqual(proof['current_position_error_m'],.012)
        self.assertEqual(policy.calls['_completion_target'],1)

    def test_original_exception_and_all_hooks_restored(self):
        policy,observer=self.fixture();problem=RuntimeError('exact original capture error')
        policy.capture_error=problem;before=self.shadows(policy)
        with self.assertRaises(RuntimeError) as caught:
            with observer:policy.select()
        self.assertIs(caught.exception,problem);self.assert_restored(before)
        self.assertEqual(observer._calls,[])
        self.assertTrue(all(e.get('original_exception',{}).get('message')==str(problem)
            for e in observer.events if 'original_exception' in e))

    def test_snapshot_failure_cannot_replace_original_result(self):
        policy,observer=self.fixture()
        with observer,patch.object(observer,'_snapshot',side_effect=RuntimeError('diagnostic snapshot unavailable')):
            self.assertIs(policy._public_capture(policy.selected),policy.capture_result)
        self.assertEqual(policy.calls['_public_capture'],1)
        self.assertTrue(observer.diagnostic_errors)

    def test_missing_or_stale_tick_context_is_unknown(self):
        policy,observer=self.fixture()
        with observer:
            policy._public_capture(policy.selected)
            observer.bind_context(tick=5,action_id='stale',action_kind='public_hold')
            policy._public_capture(policy.selected)
        for event in observer.events:
            self.assertIsNone(event['public_operands']['exact_tick'])
            self.assertIsNone(event['public_operands']['exact_action_id'])
            self.assertEqual(event['public_operands']['tick_context_status'],'missing_or_stale_unknown')

    def test_reentry_overlap_and_partial_install_restore(self):
        policy,observer=self.fixture();before=self.shadows(policy)
        with observer:
            with self.assertRaisesRegex(RuntimeError,'reentered'):observer.__enter__()
            other=module.TrainPublicCaptureDiagnostics(policy,domain=capability(),enabled=True,development_only=True)
            with self.assertRaisesRegex(RuntimeError,'overlapping'):other.__enter__()
        self.assert_restored(before)
        original=observer._patch
        def install(obj,name,wrapper):
            if name=='ready':raise RuntimeError('cannot shadow this method')
            original(obj,name,wrapper)
        with patch.object(observer,'_patch',side_effect=install):
            with self.assertRaisesRegex(RuntimeError,'cannot shadow'):observer.__enter__()
        self.assert_restored(before)
        with observer:self.assertIs(policy._public_capture(policy.selected),policy.capture_result)

    def test_confirmation_only_observes_original_state_change(self):
        policy,observer=self.fixture()
        with observer:policy.confirm_opening(0.,{'existing':True},None,confirmed_after=.25)
        self.assertEqual(policy.calls['confirm_opening'],1)
        event=observer.events[0]
        self.assertEqual(event['public_operands']['close_witness']['confirmed_after'],.15)
        self.assertEqual(event['public_operands_after']['close_witness']['confirmed_after'],.25)

    def test_truthful_in_memory_receipt_bounded_and_no_fix_claim(self):
        policy,observer=self.fixture(max_events=1)
        with observer:
            policy._public_capture(policy.selected);policy._public_capture(policy.selected)
        value=observer.receipt();json.dumps(value,allow_nan=False)
        self.assertEqual(value['dropped_events'],1);self.assertEqual(len(value['events']),1)
        self.assertFalse(value['native_validated']);self.assertFalse(value['fix_claimed'])
        self.assertFalse(value['participates_in_formal_qualification']);self.assertFalse(value['participates_in_VAL_TEST'])
        self.assertTrue(value['restored']);self.assertFalse(value['class_or_module_hooks_installed'])
        self.assertEqual(value['source_sha256'],hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
        value['events'].clear();self.assertEqual(len(observer.events),1)


if __name__=='__main__':unittest.main()
