"""Genuine V10 certificate, fixed closed V66 public sequence, no Native/GT.

The fixed 00 development sequence ends at the predeclared 6.6 s scope gap.
This is an observer/packing replay, with no new action, probe admission,
counterfactual attachment claim or calibration scoring.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import unittest

import numpy as np
import torch

from our_mani.contracts import Pose, ObjectState
from our_mani.perception import CameraCalibration
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from our_mani.local_mpc_control_20261103 import public_candidate_scope12_v67 as main
from our_mani.local_mpc_control_20261103 import public_candidate_scope12_reverification_v67 as implementation
from our_mani.local_mpc_control_20261103 import depth_scope12_goal_perception_v67 as tracker_type
from our_mani.local_mpc_control_20261103 import goal_aware_scope12_binding_v67 as binding_type
from our_mani.local_mpc_p4_20261103.runtime import pack_public_row, PublicRowCache, _FORBIDDEN

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT/'runtime/framework/local_mpc_20261102/public_candidate_local_hold_v66_development_v1/00_val_static'
CHECKPOINT = ROOT/'artifacts/framework/near_goal_segmentation_20261103/position_measurement_scope12_fresh8_v10/calibrated.pt'
CHECKPOINT_SHA = 'ee079d1f0ea302f65ed4123c5137328005f582f5347e282497dc2d2e07dc0884'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forbidden(value, path=()):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in _FORBIDDEN or key.casefold().startswith('gt_'):
                found.append('.'.join(path+(key,)))
            found.extend(forbidden(item, path+(key,)))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            found.extend(forbidden(item, path+(str(i),)))
    return found


def replay():
    started = time.perf_counter()
    if any(os.environ.get(k) != '4' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')):
        raise ValueError('Actual Native policy environment4 required')
    torch.set_num_threads(2)
    exit_record = json.loads((CASE/'worker_exit.json').read_text())
    if exit_record.get('exit_code') != 0 or exit_record.get('native_closed') is not True:
        raise ValueError('Only the fixed actually closed development case')
    public = CASE/'public_shadow'
    rows_path = public/'public_outputs.jsonl'
    cal_path = public/'camera_calibration.json'
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows = [row for row in rows if row['public_robot']['timestamp'] <= 6.6+1e-8]
    assert len(rows) == 133 and rows[0]['public_robot']['timestamp'] == 0.
    calibrations = json.loads(cal_path.read_text())
    cameras = {name: CameraCalibration(np.asarray(value['intrinsics']), Pose.from_xyzw(value['camera_to_world']))
               for name, value in calibrations.items()}
    sources = main.position_producer_source_map()
    inputs = {str(p): sha(p) for p in (rows_path, cal_path, CASE/'worker_exit.json')}
    producer = implementation.PublicCandidateScope12ReverificationBridgeV67(
        refiner_checkpoint=CHECKPOINT, refiner_checkpoint_sha256=CHECKPOINT_SHA)
    packed_cache = PublicRowCache()
    ledger = []
    internal_route_changes = 0
    previous_source = previous_route = None
    initial_sphere_identity = None
    for index, row in enumerate(rows):
        raw = public/row['rgbd']
        inputs[str(raw)] = sha(raw)
        with np.load(raw, allow_pickle=False) as z:
            now = float(z['timestamp'])
            observation = {'timestamp': now,
                'observation/images': {name: z['images_'+name].copy() for name in ('front', '8', '16', '36')},
                'observation/depths': {name: z['depths_'+name].copy() for name in ('front', '8', '16', '36')},
                'observation/calibrations': cameras}
        robot = PublicRobotSensors.from_mapping(row['public_robot'])
        assert now == robot.timestamp
        result = producer.observe(observation, robot, close_requested=row['close_requested'])
        producer.binding.approve(result)
        assert producer.binding.latest_result is result and producer.latest_result is result
        assert producer.binding.latest_visual_target is producer.tracker.last_current_target
        assert producer._frame['object_observation'] is producer.tracker.last_current_target
        assert producer.cache._pending is None or producer.cache._pending['consumed'] is True
        assert producer.cache._last_staged == now and producer.cache.discarded_unconsumed_frames == 0
        assert not forbidden(result)
        packed = pack_public_row(result, robot, row['close_requested'], index, row['rgbd'])
        packed_cache.observe(packed)
        assert packed_cache.latest['index'] == index
        sphere = producer.tracker.sphere
        assert sphere is not None and sphere.observed_at == 0.
        if initial_sphere_identity is None:
            initial_sphere_identity = id(sphere)
        assert id(sphere) == initial_sphere_identity
        trace = result['depth_refiner_trace']
        private = producer.tracker.last_measurement_metadata
        assert 'NN_current_measurement' in private and 'NN_current_measurement' not in trace
        proposal = trace.get('position_proposals', {})
        source, route = trace.get('accepted_source'), proposal.get('measurement_route')
        if source == previous_source == 'NN_CURRENT_POSITION_F' and previous_route != route:
            internal_route_changes += 1
            assert trace['accepted_history_reset_for_route_change'] is False
        previous_source, previous_route = source, route
        target = producer.binding.latest_visual_target
        ledger.append(dict(index=index, timestamp=now, mode=result['mode'],
            current_target_available=target is not None, accepted_source=source, measurement_route=route,
            geometry_profile=private['NN_current_measurement']['geometry_profile'],
            measurement_available=trace['measurement_available'],
            tracker_error=result.get('depth_refiner_tracker_error'),
            measured_points=private['NN_current_measurement']['measured_points'],
            visible_views=private['NN_current_measurement']['visible_views'],
            motion_history_reset=trace['accepted_history_reset_for_route_change']))
    assert all(sha(name) == digest for name, digest in inputs.items())
    assert sources == main.position_producer_source_map()
    assert all(sha(name) == digest for name, digest in sources.items())
    return dict(schema='V67_genuine_scope12_closed_V66_public_pack_smoke_v1',
        fixed_case='00_val_static', fixed_end_timestamp=6.6, frames=len(rows), ledger=ledger,
        checkpoint=dict(path=str(CHECKPOINT), sha256=CHECKPOINT_SHA),
        source_sha256=sources, public_input_sha256=inputs,
        initial_sphere=dict(center=sphere.center.tolist(), observed_at=sphere.observed_at),
        internal_NN_route_changes_without_motion_reset=internal_route_changes,
        final_target=producer.binding.latest_visual_target is not None,
        final_old_public_target_available=rows[-1]['result']['observed_target_available'],
        final_binding_same_object=producer.binding.latest_result is producer.latest_result,
        final_current_target_type=type(producer.binding.latest_visual_target).__name__,
        final_tracker_type=type(producer.tracker).__name__, final_binding_type=type(producer.binding).__name__,
        final_scope_qualified=producer.tracker.refiner.scope12_EF_qualified,
        composite_global_coverage_qualified=producer.tracker.refiner.composite_global_coverage_qualified,
        final_private_diagnostic_present='NN_current_measurement' in producer.tracker.last_measurement_metadata,
        stage_discard_count=producer.cache.discarded_unconsumed_frames,
        model_forward_calls=len(rows), calibration_quality_scored=False, GT_read=False, TEST_read=False,
        Native_started=False, new_actions=0, new_probe_admissions=0, attachment_claim_from_replay=False,
        wall_seconds=time.perf_counter()-started), producer


class GenuineV67PublicPack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report, cls.producer = replay()

    def test_all_actual_frames_once_bound_consumed_packed(self):
        self.assertEqual(self.report['frames'], 133)
        self.assertEqual(self.report['stage_discard_count'], 0)
        self.assertTrue(self.report['final_binding_same_object'])
        self.assertIs(type(self.producer.tracker), tracker_type.CurrentDepthScope12TrackerV67)
        self.assertIs(type(self.producer.binding), binding_type.ApprovedDepthScope12TrackerCacheBindingV67)

    def test_predeclared_old_scope_gap_is_real_current_scope_measurement(self):
        final = self.report['ledger'][-1]
        self.assertAlmostEqual(final['timestamp'], 6.6)
        self.assertFalse(self.report['final_old_public_target_available'])
        self.assertTrue(self.report['final_target'])
        self.assertIs(type(self.producer.binding.latest_visual_target), ObjectState)
        self.assertEqual(final['measurement_route'], 'scope12_EF')
        self.assertEqual(final['accepted_source'], 'NN_CURRENT_POSITION_F')
        self.assertIsNone(final['tracker_error'])
        self.assertFalse(self.report['composite_global_coverage_qualified'])

    def test_exact_result_and_private_projection_fail_closed(self):
        self.assertTrue(self.report['final_private_diagnostic_present'])
        with self.assertRaises(ValueError):
            self.producer.binding.approve(deepcopy(self.producer.latest_result))
        bad = deepcopy(self.producer.tracker.last_measurement_metadata)
        bad['unexpected_private_tree'] = False
        with self.assertRaises(ValueError):
            tracker_type.project_public_position_evidence(bad)


if __name__ == '__main__':
    unittest.main()
