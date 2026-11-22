"""Pure CPU contract tests; no sim, GPU, source manifests or formal jobs."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from our_mani.multitask import rml_component_domain_v640 as domains
from our_mani.multitask.maps import SharedMultiObjectMap
from our_mani.multitask.train_public_collision_diagnostics_v783 import TrainPublicCollisionDiagnostics


def domain(split='train', purpose=None):
    # Synthetic sealed capability only; do not call loaders/read any dataset.
    return domains.ComponentDomain(purpose or ('train_qualification' if split=='train' else 'evaluation'),
        'insert_onto_rotating_peg', 'cv_geometry', split, 'synthetic_train_only',
        None if split=='train' else 'a'*64, domains._TOKEN)


class Ghost:
    def __init__(self, handle):
        self.handle, self.collidable = handle, False
        self.hits, self.calls = set(), []

    def get_handle(self): return self.handle
    def is_collidable(self): return self.collidable
    def set_collidable(self, flag): self.collidable = flag
    def get_matrix(self): return [[1,0,0,self.handle],[0,1,0,0],[0,0,1,0]]
    def check_collision(self, obj=None):
        self.calls.append(obj)
        return obj.handle in self.hits


class NativeTruthTrap:
    def __getattribute__(self, name):
        raise AssertionError('native task/target truth accessed: '+name)


class Arm:
    _collision_collection = 717

    def __init__(self, ghosts):
        self.ghosts, self.calls = ghosts, []
        self.aggregate_result, self.positive_handles = False, set()
        self.probe_error = None
        self.q = [0.25, -0.5]

    def get_joint_positions(self): return self.q[:]
    def get_tip(self): return SimpleNamespace(get_pose=lambda:[0.1,0.2,0.3,0,0,0,1])
    def check_arm_collision(self, obj=None):
        self.calls.append((obj, [g.is_collidable() for g in self.ghosts]))
        if obj is None: return self.aggregate_result
        assert any(obj is item for item in self.ghosts), 'non-public target queried'
        if self.probe_error: raise self.probe_error
        return obj.handle in self.positive_handles


class Map:
    # Exercise the actual frozen map branch/short circuit/contact semantics.
    check = SharedMultiObjectMap.check

    def __init__(self):
        self.static_ghosts = [Ghost(10)]
        self.ring, self.peg = Ghost(20), Ghost(30)
        self.estimated_bodies = {
            role:SimpleNamespace(ghost=ghost, geometry_sha256=role+'mesh',
                fallback_ghost=None, fallback_geometry_sha256=None,
                source_sha256='b'*64, provenance='TRAIN_geometry_model')
            for role, ghost in [('ring',self.ring),('peg',self.peg)]}
        self.estimated_attachments = {'ring':SimpleNamespace(position=[0,0,.03],rotation=[[1,0,0],[0,1,0],[0,0,1]])}
        self.noncolliding_geometry = {}
        self.map = [dict(name='snapshot_table',mesh_sha256='staticmesh',pose=[0,0,0,0,0,0,1])]
        self.ghosts = self.static_ghosts+[self.ring,self.peg]
        self.env = SimpleNamespace(arm=Arm(self.ghosts),
            gripper=SimpleNamespace(get_joint_positions=lambda:[.02,.02],get_open_amount=lambda:[.5,.5]),
            task=NativeTruthTrap())
        self.static = [NativeTruthTrap()]
        self._live_nonrobot = [NativeTruthTrap()]
        self.queries, self.robot_contact_queries = 0, 0
        self.allowed_robot_contact_ids_seen = set()

    @contextmanager
    def public_scene(self, poses=None, **kwargs):
        try:
            for ghost in self.ghosts: ghost.set_collidable(True)
            yield {'ring':self.ring, 'peg':self.peg}
        finally:
            for ghost in self.ghosts: ghost.set_collidable(False)


class Port:
    def __init__(self, service):
        self.map = service
        self.bound = dict(phase='lift',allowed_robot_contact_ids=(),arrival_time=2.)
        self.last_geometry_reasons = {'ring':{'reason':'public typed TRAIN mesh'}}
        self.poses = {'peg':SimpleNamespace(position=[1,2,3],rotation=[[1,0,0],[0,1,0],[0,0,1]])}
        public_robot = SimpleNamespace(timestamp=1.,joints=[.1,.2],gripper_open=False,
            contact=True,pose=SimpleNamespace(position=[0,0,0],rotation=[[1,0,0],[0,1,0],[0,0,1]]))
        state=SimpleNamespace(robot=public_robot,role_bindings={'selected_peg':['peg']},
            bodies={'peg':SimpleNamespace(pose=self.poses['peg'],timestamp=1.)},
            perception_metadata={'bodies':{'peg':{'registration_valid':True}}})
        forecast=SimpleNamespace(object_ids=('peg',),times=[1.,2.],metadata={'observed_until':1.})
        self.frame=(state,forecast)
        self.query_callable=self.check_future
        self.model=SimpleNamespace(collision_query=self.query_callable)

    def check_future(self, future_offset):
        return self.map.check(self.poses,allowed_robot_contact_ids=self.bound['allowed_robot_contact_ids'],
            geometry_modes={'ring':'primary','peg':'primary'})


class DiagnosticsTests(unittest.TestCase):
    def fixture(self, **kwargs):
        service=Map();port=Port(service)
        observer=TrainPublicCollisionDiagnostics(service,port=port,domain=domain(),
            enabled=True,development_only=True,**kwargs)
        return service,port,observer

    def shadows(self, service, port):
        return [(obj,dict(vars(obj))) for obj in [service,service.env.arm,port.model,*service.ghosts]]

    def assert_restored(self, before):
        # Check actual shadows; no inherited bound-method shadows may survive.
        for obj, namespace in before:
            for name in ('check','public_scene','check_arm_collision','check_collision','collision_query'):
                self.assertEqual(name in vars(obj),name in namespace)
                if name in namespace: self.assertIs(vars(obj)[name],namespace[name])

    def test_disabled_default_touches_nothing_even_invalid_objects(self):
        with TrainPublicCollisionDiagnostics(NativeTruthTrap(),port=NativeTruthTrap(),domain=NativeTruthTrap()) as observer:
            self.assertEqual(observer.events,[])
        receipt=observer.receipt()
        self.assertFalse(receipt['enabled']);self.assertIsNone(receipt['source_sha256'])
        self.assertTrue(receipt['restored'])

    def test_explicit_development_and_authentic_train_only_domain_required(self):
        service=Map();port=Port(service)
        cases=[dict(domain=domain(),development_only=False),
            dict(domain=domain('validation'),development_only=True),
            dict(domain=domain('test'),development_only=True),
            dict(domain=domain('validation','train_qualification'),development_only=True),
            dict(domain=SimpleNamespace(split='train'),development_only=True)]
        for args in cases:
            with self.subTest(args=args):
                before=self.shadows(service,port)
                with self.assertRaises(ValueError):
                    with TrainPublicCollisionDiagnostics(service,port=port,enabled=True,**args): pass
                self.assert_restored(before)
        self.assertEqual(service.env.arm.calls,[])

    def test_exact_result_and_original_arguments_forwarded(self):
        service,port,observer=self.fixture()
        result=object();args=(object(),);kwargs={'custom':object()};calls=[]
        def original(*a,**k):calls.append((a,k));return result
        service.check=original
        before=self.shadows(service,port)
        with observer:
            self.assertIs(service.check(*args,**kwargs),result)
        self.assertEqual(calls,[(args,kwargs)])
        self.assert_restored(before)
        self.assertEqual(service.env.arm.calls,[])

    def test_exact_exception_forwarded_and_all_hooks_restored(self):
        service,port,observer=self.fixture();error=RuntimeError('original exact exception')
        def original(*a,**k):raise error
        service.check=original;before=self.shadows(service,port)
        with self.assertRaises(RuntimeError) as caught:
            with observer:port.model.collision_query(.25)
        self.assertIs(caught.exception,error)
        self.assert_restored(before)
        self.assertEqual(observer.events[0]['exception']['message'],str(error))
        self.assertIsNone(observer._future_offset)

    def test_arm_positive_resolves_only_public_targets_inside_original_contact_flags(self):
        service,port,observer=self.fixture();arm=service.env.arm
        arm.aggregate_result=True;arm.positive_handles={30}
        before=self.shadows(service,port)
        with observer:
            self.assertIs(port.model.collision_query(.25),True)
        self.assert_restored(before)
        # One original aggregate, then static+peg. Held ring remains disabled.
        self.assertEqual([None if obj is None else obj.handle for obj,_ in arm.calls],[None,10,30])
        self.assertTrue(all(flags==[True,False,True] for _,flags in arm.calls))
        event=observer.events[0];branch=event['arm_world'][0]
        self.assertEqual(branch['resolved_public_collision_targets'][0]['role'],'peg')
        self.assertEqual(branch['resolved_public_collision_targets'][0]['shape_handle'],30)
        self.assertEqual(branch['robot']['member_shape'],'unresolved')
        self.assertEqual(event['held_world_calls'],[])
        public=event['public_context']
        self.assertEqual(public['future_offset'],.25)
        self.assertEqual(public['forecast']['sampled_timestamp'],1.25)
        self.assertEqual(public['query_robot_proprioception']['arm_joints'],[.25,-.5])
        self.assertEqual(public['current_observation']['joints'],[.1,.2])
        self.assertEqual(public['supplied_estimated_poses']['peg']['position'],[1,2,3])
        self.assertTrue(event['scenes'][0]['inventory_complete'])
        self.assertTrue(all(not ghost.collidable for ghost in service.ghosts))
        self.assertTrue(all(not ghost.calls for ghost in service.ghosts))
        self.assertEqual(observer.errors,[])

    def test_held_world_observes_actual_pairs_and_preserves_insert_contact(self):
        service,port,observer=self.fixture();port.bound['phase']='insert'
        port.bound['allowed_robot_contact_ids']=('peg',);service.ring.hits={30}
        with observer:self.assertTrue(port.model.collision_query(0.))
        self.assertEqual(len(service.env.arm.calls),1)  # negative aggregate, no probes
        self.assertEqual(service.env.arm.calls[0][1],[True,False,False])
        pairs=observer.events[0]['held_world_calls']
        self.assertEqual([p['second']['shape_handle'] for p in pairs],[10,30])
        self.assertEqual([p['result'] for p in pairs],[False,True])
        self.assertEqual([p['first']['role'] for p in pairs],['ring','ring'])
        self.assertEqual(service.ring.calls,[service.static_ghosts[0],service.peg])
        self.assertTrue(all(p['observation']=='original_pair_call' for p in pairs))
        self.assertEqual(observer.errors,[])

    def test_additional_probe_error_never_replaces_original_positive_result(self):
        service,port,observer=self.fixture();arm=service.env.arm
        arm.aggregate_result=True;arm.probe_error=RuntimeError('read-only query unavailable')
        with observer:self.assertIs(port.model.collision_query(0.),True)
        branch=observer.events[0]['arm_world'][0]
        self.assertTrue(all('exception' in p for p in branch['public_target_probes']))
        self.assertEqual(branch['resolution'],'unresolved_no_positive_public_target_no_self_collision_claim')
        self.assertEqual(branch['resolved_public_collision_targets'],[])

    def test_held_pair_exception_and_scene_unwind_keep_original_map_exception(self):
        service,port,observer=self.fixture();error=RuntimeError('original held query failed')
        def failing(obj=None):raise error
        service.ring.check_collision=failing
        before=self.shadows(service,port)
        # The original frozen map wraps backend errors; the diagnostic must
        # leave that native contract/cause intact, rather than replacing it.
        with self.assertRaises(Exception) as caught:
            with observer:port.model.collision_query(0.)
        self.assertIs(caught.exception.__cause__,error)
        self.assertEqual(observer.events[0]['held_world_calls'][0]['exception']['message'],str(error))
        self.assertTrue(all(not ghost.collidable for ghost in service.ghosts))
        self.assertIsNone(observer._active);self.assertIsNone(observer._query)
        self.assert_restored(before)

    def test_diagnostic_base_exception_does_not_mask_original_return(self):
        service,port,observer=self.fixture();service.env.arm.aggregate_result=True
        service.env.arm.probe_error=KeyboardInterrupt('diagnostic query interrupted')
        with observer:self.assertIs(port.model.collision_query(0.),True)
        self.assertTrue(all(p['exception']['type']=='KeyboardInterrupt'
            for p in observer.events[0]['arm_world'][0]['public_target_probes']))

    def test_snapshot_error_is_diagnostic_and_does_not_change_map_result(self):
        service,port,observer=self.fixture()
        service.env.gripper.get_open_amount=lambda:(_ for _ in ()).throw(RuntimeError('read failure'))
        with observer:self.assertIs(port.model.collision_query(0.),False)
        self.assertTrue(observer.errors)
        self.assertEqual(observer.errors[0]['operation'],'public_snapshot')
        self.assertEqual(len(service.env.arm.calls),1)

    def test_nested_overlapping_scope_rejected_without_disturbing_first_scope(self):
        service,port,observer=self.fixture();before=self.shadows(service,port)
        with observer:
            installed=service.check
            with self.assertRaisesRegex(RuntimeError,'reentered'):observer.__enter__()
            other=TrainPublicCollisionDiagnostics(service,port=port,domain=domain(),
                enabled=True,development_only=True)
            with self.assertRaisesRegex(RuntimeError,'overlapping'):other.__enter__()
            self.assertIs(service.check,installed)
            self.assertFalse(port.model.collision_query(0.))
        self.assert_restored(before)
        # Reservations released and future independent scopes can run.
        with TrainPublicCollisionDiagnostics(service,port=port,domain=domain(),
                enabled=True,development_only=True):self.assertFalse(port.model.collision_query(0.))

    def test_partial_install_failure_restores_prior_hooks_and_reservation(self):
        service,port,observer=self.fixture();before=self.shadows(service,port)
        original_patch=observer._patch
        def install(obj,name,wrapper):
            if name=='check_arm_collision':raise RuntimeError('cannot shadow native method')
            return original_patch(obj,name,wrapper)
        with patch.object(observer,'_patch',side_effect=install):
            with self.assertRaisesRegex(RuntimeError,'cannot shadow'):observer.__enter__()
        self.assert_restored(before)
        with observer:self.assertFalse(port.model.collision_query(0.))
        self.assert_restored(before)

    def test_receipt_is_truthful_bounded_json_copy_and_own_source_hash(self):
        service,port,observer=self.fixture(max_events=1)
        with observer:
            port.model.collision_query(0.);port.model.collision_query(.1)
        receipt=observer.receipt();json.dumps(receipt,allow_nan=False)
        self.assertEqual(receipt['dropped_events'],1)
        self.assertEqual(len(receipt['events']),1)
        self.assertTrue(receipt['development_only']);self.assertFalse(receipt['native_validated'])
        self.assertFalse(receipt['participates_in_formal_qualification'])
        self.assertFalse(receipt['participates_in_VAL_TEST'])
        import our_mani.multitask.train_public_collision_diagnostics_v783 as module
        self.assertEqual(receipt['source_sha256'],hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
        receipt['events'].clear();self.assertEqual(len(observer.events),1)


if __name__=='__main__':unittest.main()
