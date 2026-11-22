"""Saved public failure + pure arithmetic; no model forward or Native calls."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import unittest
import numpy as np
from our_mani.contracts import Forecast,ObjectState,Pose
from our_mani.local_mpc_control_20261103.forecast_control import BoundForecastExecutionModel,ANCHOR_QUANTIZATION_TOLERANCE
from our_mani.cube_corrected_anchor_live_20261104_v2.binding import _bind_free_anchor_derivatives,AnchorConsistentPublicStateAssemblerV2
from our_mani.cube_corrected_live_20261104_v1.binding import CorrectedPublicStateAssembler
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/framework/local_mpc_20261102/Cube_same_public_OLS_anchor_repair_CPU_v2'
SHA='703bc89d4ff6f1460ed90530926fc8a869bf7df2d0530d3bb96161da04e5cf0c'


def saved(record=-1):
    freeze=json.loads((ART/'saved_public_input_freeze.json').read_text())
    p=ROOT/freeze['saved_public_inputs']['path']
    if hashlib.sha256(p.read_bytes()).hexdigest()!=freeze['saved_public_inputs']['sha256']:raise ValueError('Saved public freeze changed')
    row=json.loads(p.read_text())['records'][record];s=row['result'];o=s['object_observation'];f=s['forecast']
    obj=ObjectState(o['object_id'],o['timestamp'],Pose(o['pose']['position'],o['pose']['rotation']),
        o['points_world'],o['linear_velocity'],o['angular_velocity'],confidence=o['confidence'],visible=o['visible'])
    forecast=Forecast(tuple(f['object_ids']),f['times'],f['positions'],f['rotations'],f['linear_velocities'],f['angular_velocities'],metadata=f['metadata'])
    r=dict(s,object_observation=obj,forecast=forecast)
    state=dict(time=s['timestamp'],held=False,grasp_motion_mode='FREE_GRU',
        object_position=obj.pose.position.copy(),object_quaternion=obj.pose.xyzw()[3:],
        object_linear_velocity=obj.linear_velocity.copy(),object_angular_velocity=obj.angular_velocity.copy())
    return r,state


class NeverExecutedModel:
    def initialize_batch(self,*a,**k):raise AssertionError('No rollout/NN in arithmetic regression')
    def step_batch(self,*a,**k):raise AssertionError('No rollout/NN in arithmetic regression')


class SavedPublicAnchor(unittest.TestCase):
    def test_actual_saved_failing_anchor_old_rejects_new_derivative_passes(self):
        r,state=saved();bound=BoundForecastExecutionModel.from_public_bridge(NeverExecutedModel(),r)
        with self.assertRaisesRegex(ValueError,'Forecast index0'):bound.validate_anchor(state)
        self.assertGreater(bound.last_anchor_errors['linear_velocity_m_per_s'],.00043)
        changed,trace=_bind_free_anchor_derivatives(state,r,expected_sha256=SHA,now=state['time'])
        bound.validate_anchor(changed)
        self.assertLessEqual(max(bound.last_anchor_errors.values()),1e-6)
        self.assertEqual(trace['current_derivative_source'],'same_frozen_Forecast_index0_causal_public_last5_OLS')

    def test_all_saved_forecasts_future_pose_cloud_and_original_measurement_bitexact(self):
        for index in range(3):
            r,state=saved(index);f=r['forecast'];obj=r['object_observation']
            before=[x.copy()for x in(f.times,f.positions,f.rotations,f.linear_velocities,f.angular_velocities)]
            p=state['object_position'].copy();v=state['object_linear_velocity'].copy();q=state['object_quaternion'].copy();cloud=obj.points_world.copy()
            updated,trace=_bind_free_anchor_derivatives(state,r,expected_sha256=SHA,now=state['time'])
            self.assertIs(r['forecast'],f);self.assertIs(r['object_observation'],obj)
            for a,b in zip(before,(f.times,f.positions,f.rotations,f.linear_velocities,f.angular_velocities)):np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(state['object_linear_velocity'],v)
            np.testing.assert_array_equal(updated['object_position'],p);np.testing.assert_array_equal(updated['object_quaternion'],q)
            np.testing.assert_array_equal(obj.points_world,cloud)
            self.assertFalse(trace['forecast_samples_or_future_learned_outputs_changed'])

    def test_same_frame_source_and_original_pose_tolerance_remain_strict(self):
        r,state=saved()
        for edit in ('time','sha','position','rotation'):
            bad=dict(state);rr=dict(r);expected=SHA
            if edit=='time':bad['time']+=.05
            if edit=='sha':expected='0'*64
            if edit=='position':bad['object_position']=bad['object_position']+[.00001,0,0]
            if edit=='rotation':bad['object_quaternion']=np.array([0,0,.01,np.sqrt(1-.0001)])
            with self.assertRaises(ValueError):_bind_free_anchor_derivatives(bad,rr,expected_sha256=expected,now=r['timestamp'])
        # Alter the Forecast pose, preserving current measured pose: old1um
        # tolerance must reject, not silently relabel its anchor to fit.
        f=r['forecast'];p=f.positions.copy();p[0,0,0]+=.00001
        rr=dict(r,forecast=Forecast(f.object_ids,f.times,p,f.rotations,f.linear_velocities,f.angular_velocities,metadata=f.metadata))
        with self.assertRaisesRegex(ValueError,'original quantization'):_bind_free_anchor_derivatives(state,rr,expected_sha256=SHA,now=r['timestamp'])

    def test_nonfree_pending_and_held_unchanged_and_old_methods_retained(self):
        r,state=saved();pending=dict(state,grasp_motion_mode='PENDING_CLOSE_UNKNOWN_CONTACT',held=False)
        actual,trace=_bind_free_anchor_derivatives(pending,r,expected_sha256=SHA,now=state['time'])
        self.assertIs(actual,pending);self.assertIsNone(trace)
        self.assertIs(AnchorConsistentPublicStateAssemblerV2.build_pending,CorrectedPublicStateAssembler.build_pending)
        self.assertIs(AnchorConsistentPublicStateAssemblerV2._inputs,CorrectedPublicStateAssembler._inputs)
        self.assertIs(AnchorConsistentPublicStateAssemblerV2.observe_robot,CorrectedPublicStateAssembler.observe_robot)
        self.assertEqual(ANCHOR_QUANTIZATION_TOLERANCE,dict(position_m=1e-6,orientation_rad=1e-6,linear_velocity_m_per_s=1e-6,angular_velocity_rad_per_s=1e-6))

if __name__=='__main__':unittest.main()
