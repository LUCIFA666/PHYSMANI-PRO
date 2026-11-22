"""Pure CPU passive-call fakes: no native, data, lease or resource probing."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from our_mani.contracts import Pose, RobotState, ObjectState, Forecast
from our_mani.multitask import rml_component_domain_v640 as capabilities
from our_mani.multitask import train_public_collision_tap_v934 as m


class Poison:
    def __getattribute__(self,name):raise AssertionError('forbidden hidden getter '+name)
    def __bool__(self):raise AssertionError('extra truth coercion')
    def __repr__(self):raise AssertionError('extra repr')
    def __str__(self):raise AssertionError('extra str')


def domain(method='cv_geometry',split='train',token=None):
    return capabilities.ComponentDomain('train_qualification' if split=='train' else 'evaluation',
        'insert_onto_rotating_peg',method,split,'CPU_episode0',None if split=='train' else 'a'*64,
        capabilities._TOKEN if token is None else token)


class Scene:
    def __init__(self,fixture):self.fixture=fixture;self.yielded=object();self.suppress=False;self.errors=[]
    def __enter__(self):self.fixture.operations.append('CM.enter');return self.yielded
    def __exit__(self,kind,error,tb):
        self.fixture.operations.append('CM.exit');self.errors.append(error);return self.suppress


class Ghost:
    def __init__(self,fixture,name):self.fixture=fixture;self.name=name
    def check_collision(self,other,**kwargs):
        self.fixture.operations.append('ghost.check_collision');self.fixture.arguments.append(('ghost',self,other,kwargs))
        if self.fixture.error is not None:raise self.fixture.error
        return self.fixture.result
    def get_handle(self):raise AssertionError('extra get_handle')
    def get_matrix(self):raise AssertionError('extra get_matrix')
    def is_collidable(self):raise AssertionError('extra is_collidable')
    def check_distance(self,*args):raise AssertionError('extra collision probe')


class Arm:
    def __init__(self,fixture):self.fixture=fixture
    def check_arm_collision(self,other,**kwargs):
        self.fixture.operations.append('arm.check_arm_collision');self.fixture.arguments.append(('arm',other,kwargs))
        return self.fixture.static.check_collision(other,**kwargs)


class Map:
    def __init__(self,fixture):
        self.fixture=fixture;self.static_ghosts=[fixture.static]
        self.estimated_bodies={'target':SimpleNamespace(ghost=fixture.primary,fallback_ghost=fixture.fallback),
            'tool':SimpleNamespace(ghost=fixture.static,fallback_ghost=None)}
        self.estimated_attachments={'tool':dict(local_pose_xyzw=[0.,0.,.1,0.,0.,0.,1.],observed_at=.2)}
    @property
    def env(self):raise AssertionError('observer read map.env property')
    def public_scene(self,*args,**kwargs):
        self.fixture.operations.append('map.public_scene');self.fixture.arguments.append(('scene',args,kwargs))
        return self.fixture.scene
    def check(self,q,time=0.,**kwargs):
        self.fixture.operations.append('map.check');self.fixture.arguments.append(('map',q,time,kwargs))
        with self.public_scene({'target':'already_public_pose'},include_attachments=True):
            return self.fixture.arm.check_arm_collision(self.fixture.primary,**kwargs)


class Model:
    def __init__(self,fixture):
        self.fixture=fixture;self.arm=fixture.arm
        def query(q,time=0.,**kwargs):
            fixture.operations.append('model.collision_query');fixture.arguments.append(('query',q,time,kwargs))
            return fixture.map.check(q,time,**kwargs)
        self.collision_query=query
    def get_joints(self):
        self.fixture.original_getter_calls+=1;return np.zeros(7)
    def collision(self,q,time=0.,**kwargs):
        self.fixture.operations.append('model.collision');self.fixture.arguments.append(('model',q,time,kwargs))
        self.get_joints()  # An original backend getter; the observer must not repeat it.
        return self.collision_query(q,time,**kwargs)
    def forward_kinematics(self,*args):raise AssertionError('extra FK')


class Fixture:
    def __init__(self):
        self.operations=[];self.arguments=[];self.original_getter_calls=0;self.result=Poison();self.error=None
        self.static=Ghost(self,'static');self.primary=Ghost(self,'primary');self.fallback=Ghost(self,'fallback')
        self.arm=Arm(self);self.scene=Scene(self);self.map=Map(self);self.model=Model(self)
        robot=RobotState(.2,Pose([0.,0.,.8]),np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7))
        body=ObjectState('target',.2,Pose([.1,0.,.8]),np.zeros((3,3)),np.zeros(3),np.zeros(3))
        forecast=Forecast(('target',),np.array([.2,.25]),np.zeros((2,1,3)),np.tile(np.eye(3),(2,1,1,1)),
            np.zeros((2,1,3)),np.zeros((2,1,3)),metadata={'observed_until':.2})
        self.port=SimpleNamespace(map=self.map,model=self.model,query_callable=self.model.collision_query,
            bound={'phase':'lift','allowed_robot_contact_ids':('tool',)},last_geometry_modes={'target':'primary'},
            last_geometry_reasons={'target':'existing public registration'},
            frame=(SimpleNamespace(robot=robot,role_bindings={'target':('target',)},bodies={'target':body}),forecast))
        self.permit=object();self.root=Path('/CPU-only/our-mani')
    def tap(self,**kwargs):
        return m.TrainPublicCollisionTap(self.map,port=self.port,domain=domain(),permit=self.permit,root=self.root,
            enabled=True,development_only=True,**kwargs)
    def methods(self):
        return [(self.model,'collision'),(self.model,'collision_query'),(self.map,'check'),(self.map,'public_scene'),
            (self.arm,'check_arm_collision'),*( (g,'check_collision') for g in (self.static,self.primary,self.fallback))]
    def shadows(self):return [(obj,name,name in vars(obj),vars(obj).get(name)) for obj,name in self.methods()]


class TapTests(unittest.TestCase):
    def setUp(self):
        self.fixture=Fixture()
        self.lease=Mock(return_value=dict(complete=True,CPU_fake_permit=True))
        scope=patch.object(m,'_require_lease',self.lease);scope.start();self.addCleanup(scope.stop)

    def restored(self,saved):
        for obj,name,present,value in saved:
            self.assertEqual(name in vars(obj),present)
            if present:self.assertIs(vars(obj)[name],value)

    def test_disabled_adversarial_objects_do_not_authorize_import_or_inspect(self):
        with patch.object(m,'_authorize',side_effect=AssertionError('disabled domain touched')), \
                patch.object(m,'_public_types',side_effect=AssertionError('disabled imports')):
            with m.TrainPublicCollisionTap(Poison(),port=Poison(),domain=Poison(),permit=Poison(),root=Poison()) as tap:
                self.assertEqual(tap.events,[])
        proof=tap.receipt();self.assertTrue(proof['restored']);self.assertFalse(proof['enabled'])
        self.lease.assert_not_called()
        with self.assertRaisesRegex(ValueError,'explicit boolean'):
            m.TrainPublicCollisionTap(Poison(),enabled=Poison())

    def test_authentic_synthetic_TRAIN_domain_only_and_explicit_development(self):
        for method in ('cv_geometry','context_geometry'):
            with m.TrainPublicCollisionTap(self.fixture.map,port=self.fixture.port,domain=domain(method),
                    permit=self.fixture.permit,root=self.fixture.root,enabled=True,development_only=True):pass
        for d,dev in ((SimpleNamespace(split='train'),True),(domain(split='validation'),True),(domain(split='test'),True),
                      (domain(method='physmani'),True),(domain(token=object()),True),(domain(),False)):
            before=self.lease.call_count
            with self.subTest(domain_type=type(d),dev=dev),self.assertRaises(ValueError):
                with m.TrainPublicCollisionTap(self.fixture.map,port=self.fixture.port,domain=d,
                        permit=self.fixture.permit,root=self.fixture.root,enabled=True,development_only=dev):pass
            self.assertEqual(self.lease.call_count,before)

    def test_bad_permit_or_limit_never_patches_or_queries(self):
        saved=self.fixture.shadows()
        with patch.object(m,'_require_lease',side_effect=ValueError('CPU permit rejected')),self.assertRaises(ValueError):
            with self.fixture.tap():pass
        self.restored(saved);self.assertEqual(self.fixture.operations,[])
        for limit in (0,False,10001,'10'):
            with self.subTest(limit=limit),self.assertRaises(ValueError):
                with self.fixture.tap(max_events=limit):pass
        self.restored(saved)

    def test_entering_same_instance_cannot_reenter_during_lease_validation(self):
        tap=self.fixture.tap();calls=[]
        def lease(*args):
            calls.append(1)
            with self.assertRaisesRegex(ValueError,'reentered'):tap.__enter__()
            return {'complete':True,'CPU_fake_permit':True}
        with patch.object(m,'_require_lease',side_effect=lease):
            with tap:self.assertTrue(tap._entered)
        self.assertEqual(calls,[1]);self.assertFalse(tap._entering);self.assertTrue(tap.receipt()['restored'])

    def test_production_lease_helper_never_accepts_test_callback_permits(self):
        from our_mani.multitask import post_TEST_resource_guard_v924 as resource
        # Restore just this helper from its source globals without calling its
        # real validator: the validator itself is strictly a Mock.
        import ast
        path=Path(m.__file__);tree=ast.parse(path.read_bytes())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_require_lease')
        namespace={'__package__':'our_mani.multitask'}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        marker=object()
        with patch.object(resource,'require_permit',return_value=marker) as validator:
            self.assertIs(namespace['_require_lease'](self.fixture.permit,self.fixture.root),marker)
        validator.assert_called_once_with(self.fixture.permit,self.fixture.root,allow_test_callbacks=False)

    def test_all_layers_forward_same_objects_once_without_truth_coercion(self):
        f=self.fixture;q=np.zeros(7);marker=Poison();saved=f.shadows();before_query=f.model.collision_query
        with f.tap() as tap:self.assertIs(f.model.collision(q,.05,marker=marker),f.result)
        self.restored(saved);self.assertIs(f.model.collision_query,before_query)
        self.assertEqual(f.operations,['model.collision','model.collision_query','map.check','map.public_scene',
            'CM.enter','arm.check_arm_collision','ghost.check_collision','CM.exit'])
        for call in f.arguments:
            if call[0] in ('model','query','map'):
                self.assertIs(call[1],q);self.assertIs(call[3]['marker'],marker)
            elif call[0]=='arm':self.assertIs(call[1],f.primary);self.assertIs(call[2]['marker'],marker)
        proof=tap.receipt();self.assertEqual(proof['original_calls_seen'],6)
        self.assertEqual(f.original_getter_calls,1)
        self.assertEqual(proof['extra_native_getter_calls'],0)
        self.assertEqual(proof['original_native_getter_call_count'],'unknown_not_instrumented')
        events={e['operation']:e for e in proof['events']}
        self.assertEqual(events['model.collision_query']['parent_call_id'],events['model.collision']['call_id'])
        self.assertEqual(events['registered_ghost.check_collision']['parent_call_id'],events['arm.check_arm_collision']['call_id'])
        self.assertTrue(all(e['original_calls_performed']==1 and e['extra_queries_performed']==0 for e in proof['events']))
        self.assertEqual(events['model.collision']['result']['status'],'unknown')

    def test_registered_roles_remain_descriptions_not_native_pair_claims(self):
        f=self.fixture
        with f.tap() as tap:f.model.collision(np.zeros(7))
        event=next(e for e in tap.events if e['operation']=='registered_ghost.check_collision')
        receiver=event['registered_receiver'];arg=event['arguments']['args'][0]
        self.assertEqual(receiver['kind'],'registered_public_ghost')
        self.assertEqual(len(receiver['registered_roles']),2)
        self.assertEqual(arg['registered_roles'],[dict(role='target',variant='primary')])
        for field in ('native_handle','current_native_matrix','collidable_flag'):self.assertEqual(receiver[field],'unknown_not_queried')
        proof=tap.receipt();self.assertEqual(proof['registered_ghost_count'],3)
        self.assertFalse(proof['robot_collision_member_known']);self.assertFalse(proof['active_scene_observed'])

    def test_rotation_uncertainty_ghost_is_labelled_only_from_registration(self):
        f=self.fixture
        with f.tap() as tap:self.assertIs(f.fallback.check_collision(f.primary),f.result)
        self.assertEqual(len(tap.events),1)
        roles=tap.events[0]['registered_receiver']['registered_roles']
        self.assertEqual(roles,[dict(role='target',variant='rotation_uncertainty')])
        self.assertEqual(tap.events[0]['registered_receiver']['active_scene_membership'],'unknown_not_observed')

    def test_original_exception_object_and_all_hooks_survive(self):
        f=self.fixture;problem=RuntimeError('CPU original collision failure');f.error=problem;saved=f.shadows()
        with self.assertRaises(RuntimeError) as caught:
            with f.tap() as tap:f.model.collision(np.zeros(7))
        self.assertIs(caught.exception,problem);self.restored(saved)
        self.assertIs(f.scene.errors[-1],problem)
        self.assertTrue(tap.receipt()['restored'])
        exceptions=[e for e in tap.events if 'exception' in e]
        self.assertTrue(exceptions);self.assertTrue(all(e['exception']['original_exception_rethrown'] for e in exceptions))

    def test_logging_failures_cannot_replace_result_or_original_exception(self):
        for failing in ('_snapshot','_copy','_begin'):
            f=Fixture();problem=RuntimeError('CPU actual backend error')
            for error in (None,problem):
                f.error=error
                with f.tap() as tap,patch.object(tap,failing,side_effect=MemoryError('CPU recorder failure')):
                    if error:
                        with self.assertRaises(RuntimeError) as caught:f.model.collision(np.zeros(7))
                        self.assertIs(caught.exception,problem)
                    else:self.assertIs(f.model.collision(np.zeros(7)),f.result)
                self.assertTrue(tap.receipt()['restored']);self.assertGreater(tap.error_count,0)

    def test_public_scene_CM_identity_yield_and_suppression_untouched(self):
        f=self.fixture;problem=RuntimeError('CPU inside original contextmanager')
        with f.tap() as tap:
            cm=f.map.public_scene({'CPU':'pose'})
            self.assertIs(cm,f.scene);self.assertEqual(f.operations,['map.public_scene'])
            f.scene.suppress=True
            with cm as value:
                self.assertIs(value,f.scene.yielded);raise problem
            self.assertIs(f.scene.errors[-1],problem)
            f.scene.suppress=False
            with self.assertRaises(RuntimeError) as caught:
                with cm as value:raise problem
            self.assertIs(caught.exception,problem)
        event=tap.events[0]
        self.assertTrue(event['contextmanager_returned_unchanged']);self.assertTrue(event['enter_exit_yield_not_observed'])
        self.assertEqual(len(tap.events),1);self.assertFalse(tap.receipt()['active_scene_observed'])

    def test_bounded_events_copy_truncation_and_unknown_values_are_honest(self):
        f=self.fixture;f.port.bound={'many':list(range(100)),'array':np.zeros(300),'opaque':Poison(),'text':'x'*400}
        with f.tap(max_events=2) as tap:self.assertIs(f.model.collision(np.zeros(7)),f.result)
        proof=tap.receipt();self.assertEqual(len(proof['events']),2);self.assertEqual(proof['dropped_events'],4)
        self.assertGreater(proof['copy_truncations'],0);self.assertGreater(proof['unknown_values'],0)
        copied=proof['events'][0]['public_context']['candidate_binding']
        self.assertEqual(len(copied['many']),m.MAX_ITEMS);self.assertEqual(copied['array']['status'],'unknown')
        self.assertEqual(len(copied['text']),m.MAX_TEXT)
        for flag in ('native_validated','qualification','participates_in_VAL_TEST','formal_score_contribution',
                     'physical_equivalence_claimed','planning_wall_clock_invariance_claimed'):
            self.assertIs(proof[flag],False)

    def test_missing_public_fields_are_unknown_while_explicit_None_is_retained(self):
        f=self.fixture;del f.port.bound;del f.port.frame[0].role_bindings
        with f.tap() as tap:f.map.public_scene({})
        context=tap.events[0]['public_context']
        self.assertEqual(context['candidate_binding']['status'],'unknown')
        self.assertEqual(context['role_bindings']['status'],'unknown')
        f.port.bound=None;f.port.frame[0].role_bindings=None
        with f.tap() as tap:f.map.public_scene({})
        self.assertIsNone(tap.events[0]['public_context']['candidate_binding'])
        self.assertIsNone(tap.events[0]['public_context']['role_bindings'])

    def test_error_records_are_bounded_and_never_trigger_extra_forecast(self):
        f=self.fixture;forecast=f.port.frame[1]
        object.__setattr__(forecast,'at',Mock(side_effect=AssertionError('extra forecast query')))
        with f.tap(max_events=2) as tap,patch.object(tap,'_copy',side_effect=MemoryError('CPU copy failure')):
            self.assertIs(f.model.collision(np.zeros(7)),f.result)
        self.assertLessEqual(len(tap.errors),2)
        self.assertEqual(tap.dropped_errors,tap.error_count-len(tap.errors))
        self.assertEqual(len(tap.events),2);self.assertEqual(tap.dropped_events,4)
        forecast.at.assert_not_called();self.assertEqual(f.original_getter_calls,1)

    def test_non_owner_thread_forwards_without_observing_context(self):
        f=self.fixture
        with f.tap() as tap:
            with patch.object(m.threading,'get_ident',return_value=tap._thread+1), \
                    patch.object(tap,'_snapshot',side_effect=AssertionError('other thread observed public context')):
                self.assertIs(f.model.collision(np.zeros(7)),f.result)
        self.assertEqual(tap.events,[]);self.assertEqual(tap.non_owner_thread_calls_unobserved,6)
        self.assertEqual(f.original_getter_calls,1)

    def test_overlap_active_reentry_rejected_and_serial_reuse_restores(self):
        f=self.fixture;tap=f.tap();saved=f.shadows()
        with tap:
            with self.assertRaises(ValueError):tap.__enter__()
            with self.assertRaises(ValueError):
                with f.tap():pass
            self.assertIs(f.model.collision(np.zeros(7)),f.result)
        self.restored(saved)
        with tap:self.assertIs(f.model.collision(np.zeros(7)),f.result)
        self.restored(saved);self.assertTrue(tap.receipt()['restored'])

    def test_restore_order_is_reverse_of_instance_patch_order(self):
        f=self.fixture;tap=f.tap();tap.__enter__();expected=[id(p[0]) for p in reversed(tap._patches)];seen=[];original=m._namespace
        def namespace(value):seen.append(id(value));return original(value)
        with patch.object(m,'_namespace',side_effect=namespace):tap.__exit__(None,None,None)
        self.assertEqual(seen,expected);self.assertTrue(tap.receipt()['restored'])

    def test_partial_install_failure_restores_every_earlier_instance_binding(self):
        f=self.fixture;tap=f.tap();saved=f.shadows();original=tap._patch;count=0
        def failing(*args):
            nonlocal count
            original(*args);count+=1
            if count==3:raise RuntimeError('CPU partial install failure')
        with patch.object(tap,'_patch',side_effect=failing),self.assertRaises(RuntimeError):tap.__enter__()
        self.restored(saved);self.assertTrue(tap.receipt()['restored'])
        with f.tap():pass

    def test_property_callable_rejected_without_running_its_getter(self):
        f=self.fixture;saved=f.shadows()
        class UnsafeModel:
            @property
            def collision(self):raise AssertionError('property getter was executed')
        unsafe=UnsafeModel();unsafe.arm=f.arm;unsafe.collision_query=f.port.query_callable;f.port.model=unsafe
        with self.assertRaisesRegex(ValueError,'plain Python method'):
            with f.tap():pass
        self.restored(saved);self.assertEqual(f.operations,[])


if __name__=='__main__':unittest.main()
