"""Typed/source-only contracts, with synthetic public packets and zero NN calls."""
import ast
from copy import deepcopy
import inspect
import textwrap
import unittest
import numpy as np

from our_mani.contracts import ObjectState,Pose,Forecast
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicMotionBridge,PublicRobotSensors
from our_mani.local_mpc_control_20261103 import public_candidate_pixel_centre_measurement_v70 as measurement
from our_mani.local_mpc_control_20261103 import public_candidate_pixel_centre_reverification_v70 as reverification
from our_mani.local_mpc_control_20261103.public_grasp_state import PublicGraspStateAssembler
from our_mani.local_mpc_p4_20261103 import runtime as old_runtime
from our_mani.local_mpc_p4_20261103.runtime import PublicRowCache,ProposalUnavailable
from our_mani.local_mpc_p4_20261103.symmetry_runtime import SymmetricLearnedCandidates
from our_mani.cube_v72_live_20261104_v1 import producer as previous
from our_mani.cube_corrected_live_20261104_v1.motion_bridge import CorrectedPublicMotionBridge
from our_mani.cube_corrected_live_20261104_v1.measurement_bridge import CorrectedPixelPrimitive,CorrectedPixelMeasurementBridge
from our_mani.cube_corrected_live_20261104_v1.reverification_bridge import CorrectedReverificationBridge
from our_mani.cube_corrected_live_20261104_v1.binding import CorrectedPublicStateAssembler
from our_mani.cube_corrected_live_20261104_v1.candidate_runtime import (
    CorrectedPublicRowCache,CorrectedFrozenPilotInference,CorrectedLearnedCandidates,
    CorrectedSymmetricLearnedCandidates,pack_public_row,LEGACY_FORECAST_SHA)
from our_mani.cube_corrected_live_20261104_v1.producer import (
    ObservedCorrectedCubeV72Backend,CorrectedCubeV72Coordinator,run_corrected_cube_v72)

NEW_SHA='a'*64


def packet(index,*,available=True):
    now=index*.05
    centre=np.array([.2+now*.1,0.,.8])
    corners=np.array([[x,y,z] for x in (-.025,.025) for y in (-.025,.025) for z in (-.025,.025)])+centre
    obj=ObjectState('target',now,Pose(centre,np.eye(3)),corners,np.array([.1,0.,0.]),np.zeros(3),.9,True)
    robot=PublicRobotSensors(now,np.array([.3,0.,1.]),np.array([0.,0.,0.,1.]),np.zeros(7),np.zeros(7),np.ones(2))
    times=now+np.arange(17)*.05
    positions=centre[None,None,:]+np.arange(17)[:,None,None]*.005*np.array([1.,0.,0.])
    forecast=Forecast(('target',),times,positions,np.tile(np.eye(3),(17,1,1,1)),
        np.tile(np.array([.1,0.,0.]),(17,1,1)),np.zeros((17,1,3)),
        metadata=dict(checkpoint_sha256=NEW_SHA,observed_until=now,
            history_frames={'target':min(index+1,20)},contexts={'target':[.125]*16}))
    result=dict(timestamp=now,mode='FREE' if available else 'STALE',available=available,
        free_forecast_available=available,controlled_mode_available=False,
        object_observation=obj if available else None,forecast=forecast if available else None,
        frozen_checkpoint_sha256=NEW_SHA,Native_inputs_used=False,future_observations_used=False,
        observations_fabricated=False,forecaster_is_injected_test_double=False)
    return result,robot


class CorrectedContracts(unittest.TestCase):
    def test_new_authority_public_pack_and_real_causal_features(self):
        cache=CorrectedPublicRowCache(expected_forecast_sha256=NEW_SHA)
        for i in range(5):
            r,robot=packet(i)
            row=pack_public_row(r,robot,False,i,f'rgbd/{i:04d}.npz',expected_forecast_sha256=NEW_SHA)
            cache.observe(row)
        features,provenance=cache.features(current_time=.2)
        self.assertEqual(features['future_tokens'].shape,(17,16))
        self.assertEqual(features['robot'].shape,(25,))
        self.assertTrue(np.array_equal(features['future_tokens'][0,1:4],np.zeros(3)))
        self.assertEqual(provenance['checkpoint_sha256'],NEW_SHA)
        self.assertEqual(cache.receipt()['forecast_checkpoint_sha256'],NEW_SHA)
        self.assertFalse(cache.receipt()['legacy_ebec_authority_used'])
        self.assertEqual(set(cache.latest['result']['forecast']['metadata']),{'checkpoint_sha256','observed_until','history_frames','contexts'})
        # The measured centre remains current, while future positions remain predictions.
        self.assertNotEqual(cache.latest['result']['forecast']['positions'][-1][0],cache.latest['result']['object_observation']['pose']['position'])

    def test_old_authority_and_private_keys_not_retagged(self):
        r,robot=packet(4)
        with self.assertRaises(ValueError):old_runtime.pack_public_row(r,robot,False,4,'rgbd/0004.npz')
        with self.assertRaises(ValueError):pack_public_row(r,robot,False,4,'rgbd/0004.npz',expected_forecast_sha256=LEGACY_FORECAST_SHA)
        r['labels']={'target_position':[0.,0.,0.]}
        with self.assertRaises(ValueError):pack_public_row(r,robot,False,4,'rgbd/0004.npz',expected_forecast_sha256=NEW_SHA)
        with self.assertRaises(ValueError):CorrectedPublicRowCache(expected_forecast_sha256=LEGACY_FORECAST_SHA)

    def test_gap_unknown_and_mutated_current_frame_do_not_make_measurements(self):
        cache=CorrectedPublicRowCache(expected_forecast_sha256=NEW_SHA)
        for i in range(5):
            r,robot=packet(i); cache.observe(pack_public_row(r,robot,False,i,f'rgbd/{i:04d}.npz',expected_forecast_sha256=NEW_SHA))
        changed=cache.latest;changed['public_robot']['q'][0]=.1
        with self.assertRaises(ValueError):cache.observe(changed)
        r,robot=packet(5,available=False)
        cache.observe(pack_public_row(r,robot,False,5,'rgbd/0005.npz',expected_forecast_sha256=NEW_SHA))
        self.assertEqual(cache.receipt()['stored_history_frames'],0)
        with self.assertRaises(ProposalUnavailable):cache.features(current_time=.25)
        r,robot=packet(8)
        cache.observe(pack_public_row(r,robot,False,8,'rgbd/0008.npz',expected_forecast_sha256=NEW_SHA))
        self.assertEqual(cache.receipt()['stored_history_frames'],1)
        self.assertEqual(cache.receipt()['gap_resets'],1)
        with self.assertRaises(ProposalUnavailable):cache.features(current_time=.4)

    def test_new_provider_rejects_old_cache_before_any_weight_or_forward_access(self):
        provider=CorrectedFrozenPilotInference.__new__(CorrectedFrozenPilotInference)
        with self.assertRaises(TypeError):provider.predict(PublicRowCache(),current_time=0.)
        with self.assertRaises(ValueError):CorrectedPublicMotionBridge('/unread',expected_forecast_sha256=None,tracker=None)
        with self.assertRaises(ValueError):ObservedCorrectedCubeV72Backend(None,'/unused',checkpoint=None,checkpoint_sha256=None,
            refiner_checkpoint=None,refiner_checkpoint_sha256=None,forecast_checkpoint=None,forecast_checkpoint_sha256=None)

    def test_original_measurement_attachment_phase_and_ledger_methods_are_original_objects(self):
        self.assertIs(CorrectedPublicMotionBridge.observe,PublicMotionBridge.observe)
        self.assertIs(CorrectedPublicMotionBridge.reset,PublicMotionBridge.reset)
        self.assertIs(CorrectedPixelPrimitive.observe,measurement.PixelCentreGoalAwareActiveVisualPublicBridgeV70.observe)
        self.assertIs(CorrectedPixelMeasurementBridge._track_once,measurement.PublicCandidatePixelCentreMeasurementBridgeV70._track_once)
        self.assertIs(CorrectedPixelMeasurementBridge.observe,measurement.PublicCandidatePixelCentreMeasurementBridgeV70.observe)
        self.assertIs(CorrectedReverificationBridge.observe,reverification.PublicCandidatePixelCentreReverificationBridgeV70.observe)
        self.assertIs(CorrectedReverificationBridge.try_begin_natural_witness,reverification.PublicCandidatePixelCentreReverificationBridgeV70.try_begin_natural_witness)
        self.assertIs(CorrectedPublicStateAssembler.build,PublicGraspStateAssembler.build)
        self.assertIs(CorrectedLearnedCandidates.update,old_runtime.LearnedCandidates.update)
        self.assertIs(CorrectedSymmetricLearnedCandidates.update,SymmetricLearnedCandidates.update)
        self.assertIn(reverification.CandidatePixelCentreReverificationV70Coordinator,CorrectedCubeV72Coordinator.__mro__)

    def test_runner_is_original_v72_control_math_with_only_typed_names_changed(self):
        renames={'run_corrected_cube_v72':'run_cube_v72_live','ObservedCorrectedCubeV72Backend':'ObservedCubeV72Backend',
                 'CorrectedCubeV72Coordinator':'CubeV72Coordinator','_validate_corrected_runner':'_validate_v72_runner'}
        class Names(ast.NodeTransformer):
            def visit_Name(self,node):node.id=renames.get(node.id,node.id);return node
            def visit_FunctionDef(self,node):node.name=renames.get(node.name,node.name);return self.generic_visit(node)
        a=ast.parse(textwrap.dedent(inspect.getsource(run_corrected_cube_v72)))
        b=ast.parse(textwrap.dedent(inspect.getsource(previous.run_cube_v72_live)))
        self.assertEqual(ast.dump(Names().visit(a),include_attributes=False),ast.dump(b,include_attributes=False))


if __name__=='__main__':unittest.main()
