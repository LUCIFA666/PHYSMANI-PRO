"""Pure CPU public-cache/frozen-inference/runtime contracts; no Native or fit."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from our_mani.contracts import ObjectState,Pose,Forecast
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from our_mani.local_mpc_p4_20261103.data import CHECKPOINT_SHA256,public_features
from our_mani.local_mpc_p4_20261103.runtime import (
    PublicRowCache,FrozenPilotInference,LearnedCandidates,ProposalUnavailable,pack_public_row)


ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'artifacts/framework/local_mpc_20261102/P4_candidate_training_v1/final.pt'
EXPECTED='61e5827b4838649b8e477048bea56fcace18697f906fb610a3f963bba2437f60'


def row(index,*,timestamp=None,mode='FREE',close=False,n=None):
    t=index*.05 if timestamp is None else timestamp
    p=np.array([.25,-.15+.02*t,.775]);points=p+np.array([[x,y,z] for x in (-.025,.025) for y in (-.025,.025) for z in (-.025,.025)])
    n=min(index+1,20) if n is None else n;times=t+np.arange(17)*.05
    ready=mode=='FREE' and n>=5
    f=None if not ready else dict(object_ids=['target'],times=times.tolist(),positions=(p+np.arange(17)[:,None]*np.array([0.,.001,0.]))[:,None,:].tolist(),
        rotations=np.broadcast_to(np.eye(3),(17,1,3,3)).tolist(),linear_velocities=np.broadcast_to([0.,.02,0.],(17,1,3)).tolist(),
        angular_velocities=np.zeros((17,1,3)).tolist(),points_world=np.zeros((17,1,256,3)).tolist(),
        metadata=dict(checkpoint_sha256=CHECKPOINT_SHA256,observed_until=t,history_frames={'target':n},contexts={'target':[.125]*16},unused_large_public_metadata=[0]*200))
    return dict(index=index,close_requested=close,rgbd=f'rgbd/{index:04d}.npz',
        public_robot=dict(timestamp=t,tcp_position=[.3,-.12,.81],tcp_quaternion=[0.,1.,0.,0.],q=[0.]*7,dq=[0.]*7,finger_open_amounts=[1.,1.]),
        result=dict(timestamp=t,mode=mode,available=ready or mode=='ATTACHED_PUBLIC',free_forecast_available=ready,
            controlled_mode_available=mode=='ATTACHED_PUBLIC',Native_inputs_used=False,future_observations_used=False,
            observations_fabricated=False,forecaster_is_injected_test_double=False,frozen_checkpoint_sha256=CHECKPOINT_SHA256,
            object_observation=dict(object_id='target',timestamp=t,pose=dict(position=p.tolist(),rotation=np.eye(3).tolist()),points_world=points.tolist(),confidence=1.,visible=True),
            forecast=f,unused_binding_provenance={'cameras':{'a':[0]*200}}))


def ready_cache():
    cache=PublicRowCache()
    for i in range(5):cache.observe(row(i))
    return cache


def state(cache,*,held=False,free=True):
    r=cache.latest;robot=r['public_robot'];o=r['result']['object_observation']
    return dict(time=robot['timestamp'],**{key:np.array(robot[key]) for key in ('q','dq','tcp_position','tcp_quaternion','finger_open_amounts')},
        object_position=np.array(o['pose']['position']),object_quaternion=Rotation.from_matrix(o['pose']['rotation']).as_quat(),
        held=held,free_motion_asserted=free,grasp_motion_mode='FREE_GRU' if free else ('ATTACHED_PUBLIC' if held else 'PENDING_CLOSE_UNKNOWN_CONTACT'),
        Native_object_state_used=False,Native_held_used=False,Native_success_used=False)


class PublicCacheTests(unittest.TestCase):
    def test_projection_preserves_real_history_and_original_feature_values(self):
        cache=PublicRowCache();original=[]
        for i in range(25):
            r=row(i);original.append(r);cache.observe(r)
        self.assertEqual(cache.receipt()['stored_history_frames'],20)
        self.assertEqual(cache.receipt()['history_timestamps'],[r['public_robot']['timestamp'] for r in original[-20:]])
        self.assertEqual(cache.receipt()['history_RGBD_references'],[r['rgbd'] for r in original[-20:]])
        compact=cache.latest
        self.assertNotIn('points_world',compact['result']['forecast'])
        self.assertNotIn('unused_binding_provenance',compact['result'])
        direct,_=public_features(original[-1],original[-20:]);cached,_=cache.features(current_time=1.2)
        for key in direct:np.testing.assert_array_equal(direct[key],cached[key])
        self.assertEqual(cache.receipt()['observations_fabricated'],False)

    def test_gap_clears_history_without_changing_GRU_count_or_inventing_frames(self):
        cache=ready_cache();cache.observe(row(6,timestamp=.4))
        self.assertEqual(cache.receipt()['stored_history_frames'],1)
        self.assertEqual(cache.receipt()['gap_resets'],1)
        with self.assertRaises(ProposalUnavailable):cache.features(current_time=.4)
        self.assertEqual(cache.latest['result']['forecast']['metadata']['history_frames']['target'],7)

    def test_duplicate_same_frame_is_idempotent_only_if_entire_input_is_identical(self):
        cache=ready_cache();latest=row(4);cache.observe(latest)
        self.assertEqual(len(cache.history),5)
        changed=deepcopy(latest);changed['result']['unused_binding_provenance']['cameras']['a'][0]=1
        with self.assertRaisesRegex(ValueError,'same-time'):cache.observe(changed)
        with self.assertRaisesRegex(ValueError,'past'):cache.observe(row(3))

    def test_unknown_or_closed_frame_refuses_new_features_without_fake_continuation(self):
        for mode in ('UNRESOLVED','STALE'):
            cache=ready_cache();r=row(5,mode=mode,close=True)
            if mode=='STALE':r['result']['object_observation']=None
            cache.observe(r);self.assertEqual(len(cache.history),0)
            with self.assertRaises(ProposalUnavailable):cache.features(current_time=.25)
            self.assertEqual(cache.latest['rgbd'],'rgbd/0005.npz')

    def test_GT_handles_bad_provenance_and_future_real_anchor_are_rejected(self):
        for variant in ('GT','handle','flag','future_observed','bad_robot','nan'):
            r=row(4)
            if variant=='GT':r['result']['ground_truth']={'object_position':[1.,2.,3.]}
            elif variant=='handle':r['result']['unused_binding_provenance']['backend']=object()
            elif variant=='flag':r['result']['Native_inputs_used']=True
            elif variant=='future_observed':r['result']['forecast']['metadata']['observed_until']=.25
            elif variant=='bad_robot':r['public_robot']['q']=[0.]*6
            else:r['result']['forecast']['points_world'][0][0][0][0]=float('nan')
            with self.subTest(variant=variant),self.assertRaises((TypeError,ValueError)):PublicRowCache().observe(r)
        cache=ready_cache()
        with self.assertRaises(ProposalUnavailable):cache.features(current_time=.15)

    def test_optional_file_binding_requires_each_real_NPZ_timestamp(self):
        with TemporaryDirectory() as directory:
            p=Path(directory)/'rgbd';p.mkdir();np.savez(p/'0000.npz',timestamp=0.)
            cache=PublicRowCache(rgbd_root=directory);cache.observe(row(0))
            self.assertTrue(cache.receipt()['actual_RGBD_file_timestamps_verified'])
            np.savez(p/'0001.npz',timestamp=.2)
            with self.assertRaisesRegex(ValueError,'NPZ timestamp'):cache.observe(row(1))
            r=row(1);r['rgbd']='../labels.npz'
            with self.assertRaises(ValueError):cache.observe(r)

    def test_typed_packer_keeps_small_whitelist_and_never_copies_Forecast_points(self):
        r=row(4);o=r['result']['object_observation'];f=r['result']['forecast']
        typed=deepcopy(r['result']);typed['object_observation']=ObjectState('target',.2,Pose(**o['pose']),o['points_world'],[0.,.02,0.],[0.,0.,0.])
        typed['forecast']=Forecast(tuple(f['object_ids']),f['times'],f['positions'],f['rotations'],f['linear_velocities'],f['angular_velocities'],
            points_world=np.zeros((17,1,4096,3)),metadata=f['metadata'])
        packed=pack_public_row(typed,PublicRobotSensors.from_mapping(r['public_robot']),False,4,r['rgbd'])
        self.assertEqual(set(packed),set(r));self.assertNotIn('points_world',packed['result']['forecast'])
        self.assertLess(len(json.dumps(packed)),12000)
        typed['native_state']={'held':True}
        with self.assertRaises(ValueError):pack_public_row(typed,PublicRobotSensors.from_mapping(r['public_robot']),False,4,r['rgbd'])


class FrozenRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.inference=FrozenPilotInference(CHECKPOINT,expected_sha256=EXPECTED)
    def test_real_frozen_checkpoint_inference_is_CPU_eval_and_unknown_refuses_NN(self):
        inference=self.inference;cache=ready_cache();before=inference.calls
        output,provenance=inference.predict(cache,current_time=.2)
        self.assertEqual(output['positions'].shape,(1,8,3));self.assertEqual(output['positions'].device.type,'cpu')
        self.assertFalse(output['positions'].requires_grad);self.assertFalse(inference._model.training)
        self.assertTrue(all(not p.requires_grad for p in inference._model.parameters()))
        self.assertEqual(provenance['model_checkpoint_sha256'],EXPECTED)
        self.assertEqual(inference.calls,before+1)
        with self.assertRaises(ProposalUnavailable):inference.predict(cache,current_time=.25)
        self.assertEqual(inference.calls,before+1)

    def test_wrong_weight_identity_source_map_spec_and_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'SHA'):FrozenPilotInference(CHECKPOINT,expected_sha256='0'*64)
        original=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        for variant in ('sources','updates','spec','parameter','fixed_physical_scale'):
            with self.subTest(variant=variant),TemporaryDirectory() as directory:
                payload=deepcopy(original)
                if variant=='sources':payload['source_sha256'].pop(next(iter(payload['source_sha256'])))
                elif variant=='updates':payload['training_updates']=99
                elif variant=='spec':payload['model_spec']['output_position_bounds_m']=[-.2,.2]
                elif variant=='fixed_physical_scale':payload['model_state']['future_input_scale'][1]=1000.
                else:
                    key=next(k for k,v in payload['model_state'].items() if v.numel()>10)
                    payload['model_state'][key].reshape(-1)[0]=float('nan')
                p=Path(directory)/'synthetic_invalid_checkpoint.pt';torch.save(payload,p)
                with self.assertRaises(ValueError):FrozenPilotInference(p,expected_sha256=hashlib.sha256(p.read_bytes()).hexdigest())

    def test_in_memory_eval_mode_or_parameter_mutation_cannot_pass_as_frozen_inference(self):
        for variant in ('train_mode','weights'):
            with self.subTest(variant=variant):
                inference=FrozenPilotInference(CHECKPOINT,expected_sha256=EXPECTED)
                if variant=='train_mode':inference._model.train()
                else:
                    with torch.no_grad():next(inference._model.parameters()).reshape(-1)[0].add_(.1)
                with self.assertRaisesRegex(ValueError,'frozen inference'):
                    inference.predict(ready_cache(),current_time=.2)

    def test_initial_and_all_expired_call_NN_once_while_active_deadlines_stay_fixed(self):
        cache=ready_cache();inference=self.inference;start=inference.calls;adapter=LearnedCandidates(inference,cache)
        first=adapter.update(state(cache));self.assertEqual(inference.calls,start+1)
        cache.observe(row(5));second=adapter.update(state(cache))
        self.assertEqual(inference.calls,start+1)
        self.assertEqual([(c['id'],c['deadline'],c['object_tcp_position']) for c in first],[(c['id'],c['deadline'],c['object_tcp_position']) for c in second])
        last=max(c['deadline'] for c in first);end_index=int(np.floor(last/.05))+1
        for i in range(6,end_index+1):cache.observe(row(i))
        third=adapter.update(state(cache));self.assertEqual(inference.calls,start+2)
        self.assertEqual(adapter.generation,1);self.assertFalse({c['id'] for c in first}&{c['id'] for c in third})
        self.assertFalse(adapter.last_snapshot['availability_is_reachability'])

    def test_pending_or_publicly_attached_pool_never_regenerates_even_after_deadlines(self):
        for held,mode in ((False,'UNRESOLVED'),(True,'ATTACHED_PUBLIC')):
            with self.subTest(held=held):
                cache=ready_cache();inference=self.inference;adapter=LearnedCandidates(inference,cache)
                first=adapter.update(state(cache));calls=inference.calls
                for i in range(5,30):cache.observe(row(i,mode=mode,close=True))
                frozen=adapter.update(state(cache,held=held,free=False))
                self.assertEqual(inference.calls,calls);self.assertEqual(adapter.generation,0)
                self.assertEqual([c['deadline'] for c in first],[c['deadline'] for c in frozen])
                self.assertTrue(adapter.last_snapshot['phase_pool_frozen'])

    def test_current_body_robot_time_and_Native_state_poison_do_not_reuse_pool(self):
        cache=ready_cache();adapter=LearnedCandidates(self.inference,cache);adapter.update(state(cache));calls=self.inference.calls
        for variant in ('time','body','q','Native'):
            s=state(cache)
            if variant=='time':s['time']-=.05
            elif variant=='body':s['object_position'][1]+=.01
            elif variant=='q':s['q'][1]+=.01
            else:s['Native_object_state_used']=True
            with self.subTest(variant=variant),self.assertRaises(ValueError):adapter.update(s)
        self.assertEqual(self.inference.calls,calls)

    def test_cache_epoch_reset_retires_old_IDs_preserves_events_and_increments_lifetime_generation(self):
        cache=ready_cache();adapter=LearnedCandidates(self.inference,cache);first=adapter.update(state(cache));calls=self.inference.calls
        old_generation=adapter.generation;old_events=adapter.events
        cache.reset()
        for i in range(5):cache.observe(row(i))
        second=adapter.update(state(cache));self.assertGreater(adapter.generation,old_generation)
        self.assertEqual(self.inference.calls,calls+1);self.assertEqual(first[0]['prediction_anchor_time'],second[0]['prediction_anchor_time'])
        self.assertFalse({c['id'] for c in first}&{c['id'] for c in second})
        self.assertEqual(adapter.events[:len(old_events)],old_events)
        retirement=next(e for e in adapter.events if e['reason']=='cache_epoch_changed_retire_all_previous_candidates')
        self.assertEqual(retirement['retired'],[dict(id=c['id'],deadline=c['deadline']) for c in first])
        self.assertTrue(retirement['warm_start_clear_required'])
        self.assertEqual(adapter.last_snapshot['events'],adapter.events)


if __name__=='__main__':unittest.main()
