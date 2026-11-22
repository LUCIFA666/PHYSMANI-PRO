"""Synthetic model/masked-time/type contracts; no real weights or fitting."""
import ast
import inspect
import json
import textwrap
import unittest

import numpy as np
import torch

from our_mani.cube_matched_candidate_20261104_v1 import CHECKPOINT_SCHEMA
from our_mani.cube_matched_candidate_20261104_v1.model import (
    CandidateConfig,CubeMatchedCandidateTransformer,model_from_spec,parameter_count)
from our_mani.cube_matched_candidate_20261104_v1.losses import candidate_loss
from our_mani.candidate_transformer_cube_20261104_v1.model import CubeFutureCandidateTransformer
from our_mani.cube_candidate_matched_20261104_v2 import FRAME
from our_mani.cube_candidate_matched_20261104_v2.losses import candidate_loss as reference_loss
from our_mani.local_mpc_p4_20261103.runtime import PublicRowCache,LearnedCandidates
from our_mani.cube_corrected_live_20261104_v1.candidate_runtime import CorrectedPublicRowCache
from our_mani.cube_corrected_live_20261104_v1.matched_candidate_runtime import MatchedCorrectedCandidateInference,MatchedCorrectedLearnedCandidates,_validate_epoch_metadata
from our_mani.cube_corrected_live_20261104_v1 import producer as previous
from our_mani.cube_corrected_live_20261104_v1.matched_producer import (
    ObservedMatchedCorrectedCubeV72Backend,MatchedCorrectedCubeV72Coordinator,run_matched_corrected_cube_v72)


def features(batch=2):
    f=torch.zeros(batch,17,16)
    f[:,:,0]=torch.arange(17)/16.
    f[:,:,4:10]=torch.tensor([1.,0.,0.,0.,1.,0.])
    robot=torch.zeros(batch,25);robot[:,-2:]=1.
    return dict(future_tokens=f,robot=robot,geometry=torch.tensor([[.05,.05,.05,.9]]).expand(batch,4),
        context_c=torch.full((batch,16),.125),task_e=torch.full((batch,16),.125))


class MatchedContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(72)
        self.config=CandidateConfig(d_model=32,num_layers=1,num_heads=4,dropout=0.)

    def test_count_and_canonical_public_frame_spec(self):
        self.assertEqual(parameter_count(),3190538)
        model=CubeMatchedCandidateTransformer(self.config)
        spec=model.spec()
        self.assertEqual(spec['schema'],CHECKPOINT_SCHEMA)
        self.assertEqual(spec['target_frame'],FRAME)
        self.assertFalse(spec['semantic_checkpoint_compatible_with_old'])
        self.assertEqual(model_from_spec(json.loads(json.dumps(spec))).spec(),spec)
        with self.assertRaises(ValueError):model_from_spec(CubeFutureCandidateTransformer(self.config).spec())

    def test_same_forward_state_shapes_and_unranked_zero_score(self):
        old=CubeFutureCandidateTransformer(self.config)
        new=CubeMatchedCandidateTransformer(self.config)
        new.load_state_dict(old.state_dict(),strict=True)
        a=old(**features());b=new(**features())
        for key in a:
            if isinstance(a[key],torch.Tensor):self.assertTrue(torch.equal(a[key],b[key]),key)
            else:self.assertEqual(a[key],b[key])
        self.assertTrue(torch.equal(b['proposal_logits'],torch.zeros(2,8)))
        self.assertFalse(new.spec()['proposal_logits_trainable'])
        with self.assertRaises(TypeError):new(**features(),target_dt=torch.ones(2))

    def test_masked_time_math_exact_and_no_far_tau_clamp(self):
        model=CubeMatchedCandidateTransformer(self.config)
        output=model(**features())
        p=torch.tensor([[.01,0.,-.015],[.02,0.,-.015]])
        rotation=output['rotation_matrices'][:,0].detach().clone()
        dt=torch.tensor([.3,2.5]);mask=torch.tensor([True,False])
        actual=candidate_loss(output,p,rotation,dt,mask)
        expected=reference_loss(output,p,rotation,dt,mask,compatibility_logits=None,ranking_weight=0.)
        self.assertTrue(torch.equal(actual['per_example'],expected['per_example']))
        self.assertFalse(actual['far_time_was_clipped'])
        self.assertFalse(actual['ranking_head_present'])
        self.assertFalse(actual['unmatched_candidates_are_negative'])
        output['times'].retain_grad();actual['per_example'].sum().backward()
        self.assertTrue(torch.equal(output['times'].grad[1],torch.zeros(8)))
        self.assertGreater(output['times'].grad[0].abs().sum(),0.)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()for p in model.parameters()))

    def test_new_provider_type_and_forecast_identity_before_forward(self):
        provider=MatchedCorrectedCandidateInference.__new__(MatchedCorrectedCandidateInference)
        with self.assertRaises(TypeError):provider.predict(PublicRowCache(),current_time=0.)
        provider.forecast_sha256='a'*64
        cache=CorrectedPublicRowCache(expected_forecast_sha256='b'*64)
        with self.assertRaises(ValueError):provider.predict(cache,current_time=0.)
        with self.assertRaises(ValueError):MatchedCorrectedCandidateInference(None,expected_sha256=None,forecast_sha256=None)

    def test_original_v72_observer_decide_and_fixed_ledger_methods(self):
        self.assertIs(ObservedMatchedCorrectedCubeV72Backend.sample,previous.ObservedCorrectedCubeV72Backend.sample)
        self.assertIs(MatchedCorrectedCubeV72Coordinator.decide,previous.CorrectedCubeV72Coordinator.decide)
        self.assertIs(MatchedCorrectedLearnedCandidates.update,LearnedCandidates.update)
        with self.assertRaises(ValueError):ObservedMatchedCorrectedCubeV72Backend(None,'/unused',checkpoint=None,checkpoint_sha256=None,
            refiner_checkpoint=None,refiner_checkpoint_sha256=None,forecast_checkpoint=None,forecast_checkpoint_sha256=None)

    def test_matched_runner_only_names_changed(self):
        names={'run_matched_corrected_cube_v72':'run_corrected_cube_v72',
            'ObservedMatchedCorrectedCubeV72Backend':'ObservedCorrectedCubeV72Backend',
            'MatchedCorrectedCubeV72Coordinator':'CorrectedCubeV72Coordinator',
            '_validate_matched_runner':'_validate_corrected_runner'}
        class Rename(ast.NodeTransformer):
            def visit_Name(self,n):n.id=names.get(n.id,n.id);return n
            def visit_FunctionDef(self,n):n.name=names.get(n.name,n.name);return self.generic_visit(n)
        a=ast.parse(textwrap.dedent(inspect.getsource(run_matched_corrected_cube_v72)))
        b=ast.parse(textwrap.dedent(inspect.getsource(previous.run_corrected_cube_v72)))
        self.assertEqual(ast.dump(Rename().visit(a),include_attributes=False),ast.dump(b,include_attributes=False))

    def test_one_based_epoch_accepts_first_and_final_but_rejects_zero(self):
        metadata=dict(training_updates=1400,epochs_completed=200,selected_epoch=1,selected_epoch_indexing='one_based')
        _validate_epoch_metadata(metadata)
        _validate_epoch_metadata(dict(metadata,selected_epoch=200))
        with self.assertRaises(ValueError):_validate_epoch_metadata(dict(metadata,selected_epoch=0))
        with self.assertRaises(ValueError):_validate_epoch_metadata(dict(metadata,selected_epoch_indexing='zero_based'))


if __name__=='__main__':unittest.main()
