"""CPU symmetry-representative contracts; no Native, labels, or fitting."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import Forecast
from our_mani.local_mpc_p4_20261103.runtime import PublicRowCache,FrozenPilotInference
from our_mani.local_mpc_p4_20261103.symmetry_runtime import (
    SymmetricLearnedCandidates,choose_pi_representative)
from tests.test_local_mpc_p4_20261103_runtime import row,state,CHECKPOINT,EXPECTED


ZPI=Rotation.from_euler('z',np.pi).as_matrix()
RXPI=Rotation.from_euler('x',np.pi).as_matrix()
BASE=Rotation.from_euler('xyz',[.41,-.63,.28]).as_matrix()
WORLD_W=np.array([.3,-.2,.4])


def predicted(t0=10.):
    delta=np.arange(17)*.05
    rotations=(Rotation.from_rotvec(delta[:,None]*WORLD_W).as_matrix()@BASE)[:,None,:,:]
    return Forecast(('target',),t0+delta,np.zeros((17,1,3)),rotations,
        np.zeros((17,1,3)),np.broadcast_to(WORLD_W,(17,1,3)))


def nonaxis_row(i,*,tcp_matrix=None):
    r=row(i);o=r['result']['object_observation'];p=np.array(o['pose']['position'])
    o['pose']['rotation']=BASE.tolist()
    o['points_world']=(p+(np.array(o['points_world'])-p)@BASE.T).tolist()
    tcp_matrix=BASE@RXPI@ZPI if tcp_matrix is None else tcp_matrix
    r['public_robot']['tcp_quaternion']=Rotation.from_matrix(tcp_matrix).as_quat().tolist()
    if r['result']['forecast'] is not None:
        r['result']['forecast']['rotations']=predicted(r['public_robot']['timestamp']).rotations.tolist()
        r['result']['forecast']['angular_velocities']=np.broadcast_to(WORLD_W,(17,1,3)).tolist()
    return r


def nonaxis_cache():
    cache=PublicRowCache()
    for i in range(5):cache.observe(nonaxis_row(i))
    return cache


class RepresentativeMathTests(unittest.TestCase):
    def test_nonaxis_world_orientation_uses_target_local_Z_right_multiplication(self):
        forecast=predicted();deadline=10.075
        future=(Rotation.from_rotvec(WORLD_W*.075).as_matrix()@BASE)
        tcp=future@RXPI@ZPI
        raw=Rotation.from_matrix(RXPI).as_quat()
        record=choose_pi_representative(raw,forecast,deadline,Rotation.from_matrix(tcp).as_quat())
        self.assertTrue(record['available']);self.assertEqual(record['selection_index'],1)
        self.assertAlmostEqual(record['angles_rad'][0],np.pi,places=10)
        self.assertLess(record['angles_rad'][1],1e-10)
        np.testing.assert_allclose(record['object_world_rotation_at_deadline'],future,atol=1e-12)
        chosen=Rotation.from_quat(record['selected_quaternion_xyzw']).as_matrix()
        np.testing.assert_allclose(chosen,RXPI@ZPI,atol=1e-12)
        wrong_world=ZPI@future@RXPI
        self.assertGreater((Rotation.from_matrix(tcp).inv()*Rotation.from_matrix(wrong_world)).magnitude(),.3)
        self.assertEqual(record['sample_time'],deadline)

    def test_strict_domain_masks_even_one_outside_ulp_and_never_clamps_deadline(self):
        forecast=predicted();q=Rotation.from_matrix(RXPI).as_quat()
        for deadline in (np.nextafter(forecast.times[0],-np.inf),np.nextafter(forecast.times[-1],np.inf)):
            record=choose_pi_representative(q,forecast,deadline,q)
            self.assertFalse(record['available']);self.assertIsNone(record['sample_time'])
            self.assertIsNone(record['selection_index']);self.assertEqual(record['original_absolute_deadline'],deadline)
            self.assertFalse(record['deadline_or_sample_clamped'])
            np.testing.assert_array_equal(record['selected_quaternion_xyzw'],q)
        edge=choose_pi_representative(q,forecast,float(forecast.times[-1]),q)
        self.assertTrue(edge['available']);self.assertEqual(edge['sample_time'],forecast.times[-1])

    def test_half_turn_tie_keeps_raw_and_invalid_numbers_are_rejected(self):
        times=np.arange(17)*.05
        forecast=Forecast(('target',),times,np.zeros((17,1,3)),np.broadcast_to(np.eye(3),(17,1,3,3)),np.zeros((17,1,3)),np.zeros((17,1,3)))
        q=np.array([0.,0.,0.,1.]);tcp=Rotation.from_euler('z',np.pi/2).as_quat()
        record=choose_pi_representative(q,forecast,.3,tcp)
        self.assertEqual(record['selection_index'],0)
        self.assertAlmostEqual(record['angles_rad'][0],record['angles_rad'][1],places=12)
        for invalid in (float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):choose_pi_representative(q,forecast,invalid,tcp)
        with self.assertRaises(ValueError):choose_pi_representative([0.,0.,0.,0.],forecast,.3,tcp)


class SymmetricCandidateRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.inference=FrozenPilotInference(CHECKPOINT,expected_sha256=EXPECTED)

    def test_new_generation_returns_equivalent_clones_preserving_point_time_ID_and_raw_ledger(self):
        cache=nonaxis_cache();adapter=SymmetricLearnedCandidates(self.inference,cache)
        candidates=adapter.update(state(cache));raw=adapter.raw_candidates
        record=adapter.last_snapshot['symmetry_generation_record']
        self.assertTrue(any(c['selection_index']==1 for c in record['slots']))
        for chosen,original,choice in zip(candidates,raw,record['slots']):
            for key in ('id','object_tcp_position','deadline','relative_time_seconds','prediction_anchor_time','proposal_logit','proposal_weight'):
                self.assertEqual(chosen[key],original[key])
            np.testing.assert_array_equal(original['object_tcp_quaternion'],choice['raw_quaternion_xyzw'])
            actual=Rotation.from_quat(chosen['object_tcp_quaternion']).as_matrix()
            expected=Rotation.from_quat(original['object_tcp_quaternion']).as_matrix()@(ZPI if choice['selection_index'] else np.eye(3))
            np.testing.assert_allclose(actual,expected,atol=1e-12)
            np.testing.assert_allclose(actual.T@actual,np.eye(3),atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(actual),1.)
            self.assertLessEqual(choice['angles_rad'][choice['selection_index']],min(choice['angles_rad'])+1e-12)
        self.assertTrue(adapter.last_snapshot['raw_model_pose_unmodified_in_ledger'])

    def test_replans_cannot_reselect_representative_after_current_TCP_or_forecast_changes(self):
        cache=nonaxis_cache();adapter=SymmetricLearnedCandidates(self.inference,cache)
        first=adapter.update(state(cache));original=deepcopy(first);calls=self.inference.calls
        raw0=Rotation.from_quat(adapter.raw_candidates[0]['object_tcp_quaternion']).as_matrix()
        # This real-time synthetic row makes the other branch preferable for
        # slot0. The active generation must still retain its original choice.
        changed=nonaxis_row(5,tcp_matrix=BASE@raw0)
        cache.observe(changed);second=adapter.update(state(cache))
        self.assertEqual(self.inference.calls,calls)
        for a,b in zip(original,second):
            self.assertEqual(a['object_tcp_quaternion'],b['object_tcp_quaternion'])
            self.assertEqual(a['deadline'],b['deadline']);self.assertEqual(a['object_tcp_position'],b['object_tcp_position'])
            self.assertEqual(b['symmetry_choice_anchor_time'],.2)
        choice_events=[e for e in adapter.events if e['reason']=='parallel_gripper_pi_representatives_selected_once_at_generation_creation']
        self.assertEqual(len(choice_events),1)

    def test_epoch_reset_retains_old_events_retires_IDs_and_selects_once_for_new_generation(self):
        cache=nonaxis_cache();adapter=SymmetricLearnedCandidates(self.inference,cache)
        first=adapter.update(state(cache));first_gen=adapter.generation;first_events=adapter.events
        cache.reset()
        for i in range(5):cache.observe(nonaxis_row(i))
        second=adapter.update(state(cache))
        self.assertGreater(adapter.generation,first_gen)
        self.assertFalse({c['id'] for c in first}&{c['id'] for c in second})
        self.assertEqual(adapter.events[:len(first_events)],first_events)
        self.assertEqual(len([e for e in adapter.events if e['reason']=='parallel_gripper_pi_representatives_selected_once_at_generation_creation']),2)
        retirement=next(e for e in adapter.events if e['reason']=='cache_epoch_changed_retire_all_previous_candidates')
        self.assertTrue(retirement['warm_start_clear_required'])
        self.assertEqual(retirement['retired'],[dict(id=c['id'],deadline=c['deadline']) for c in first])

    def test_unavailable_creation_forecast_masks_without_new_times_or_later_reselection(self):
        cache=nonaxis_cache();adapter=SymmetricLearnedCandidates(self.inference,cache)
        small=Forecast(('target',),[.2,.25],np.zeros((2,1,3)),np.broadcast_to(BASE,(2,1,3,3)),np.zeros((2,1,3)),np.zeros((2,1,3)))
        with patch('our_mani.local_mpc_p4_20261103.symmetry_runtime._forecast',return_value=small):
            first=adapter.update(state(cache))
        raw=adapter.raw_candidates
        invalid=[i for i,c in enumerate(first) if c['deadline']>.25]
        self.assertTrue(invalid)
        for i in invalid:
            self.assertFalse(first[i]['proposal_available'])
            self.assertEqual(first[i]['deadline'],raw[i]['deadline'])
            self.assertEqual(first[i]['object_tcp_quaternion'],raw[i]['object_tcp_quaternion'])
        cache.observe(nonaxis_row(5));second=adapter.update(state(cache))
        for i in invalid:
            self.assertFalse(second[i]['proposal_available']);self.assertIsNone(second[i]['parallel_gripper_symmetry_choice'])
        self.assertEqual(len([e for e in adapter.events if e['reason']=='parallel_gripper_pi_representatives_selected_once_at_generation_creation']),1)


if __name__=='__main__':unittest.main()
