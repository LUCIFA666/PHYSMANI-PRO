import unittest

import numpy as np
import torch

from tests.test_world_tcp_command_warmstart_20261104_v1 import inputs
from our_mani.world_tcp_command_warmstart_20261104_v2.model import (
    CandidateConfig,WorldTCPCommandTransformerV2,command_imitation_loss,parameter_count,
)
from our_mani.world_tcp_command_warmstart_20261104_v2.data import select_parent_runs
from our_mani.world_tcp_command_warmstart_20261104_v2.runtime_adapter import (
    DemoCommandBankV2,bank_from_model_output,generic_v1_proposals,select_generic_reference,
)


class WorldCommandV2Contracts(unittest.TestCase):
    def test_exact_extra_LN_parameters_and_checkpoint_family(self):
        self.assertEqual(parameter_count(),85977611)
        model=WorldTCPCommandTransformerV2(CandidateConfig(width=32,layers=1,heads=4,dropout=0.))
        self.assertIn('v2_only',model.spec()['checkpoint_family'])
        self.assertIn('visual_projection.1.weight',model.state_dict())

    def test_rotation_is_measured_TCP_times_residual(self):
        model=WorldTCPCommandTransformerV2(CandidateConfig(width=32,layers=1,heads=4,dropout=0.)).eval()
        with torch.no_grad():model.rotation_head.weight.zero_()
        batch=inputs();batch['current_robot'][:,10:14]=torch.tensor([1.,0,0,0])
        with torch.no_grad():result=model(**batch)
        expected=torch.diag(torch.tensor([1.,-1,-1])).expand(1,8,3,3)
        self.assertTrue(torch.allclose(result['world_tcp_rotations'],expected,atol=1e-6,rtol=0))
        self.assertTrue(torch.allclose(result['world_tcp_positions'][:,0],batch['current_robot'][:,7:10]))

    def test_PI_target_has_finite_nonzero_FP32_gradient(self):
        torch.manual_seed(43)
        model=WorldTCPCommandTransformerV2(CandidateConfig(width=32,layers=1,heads=4,dropout=0.))
        result=model(**inputs());target=torch.diag(torch.tensor([1.,-1,-1]))[None]
        loss=command_imitation_loss(result,torch.tensor([[.2,.3,.4]]),target,torch.zeros(1))['loss']
        loss.backward();grad=model.rotation_head.weight.grad
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.norm()),1e-4)

    def test_PI_target_has_finite_nonzero_BF16_gradient(self):
        torch.manual_seed(44)
        model=WorldTCPCommandTransformerV2(CandidateConfig(width=32,layers=1,heads=4,dropout=0.))
        with torch.autocast('cpu',dtype=torch.bfloat16):
            result=model(**inputs())
            loss=command_imitation_loss(result,torch.tensor([[.2,.3,.4]]),
                torch.diag(torch.tensor([1.,-1,-1]))[None],torch.ones(1))['loss']
        loss.backward();grad=model.rotation_head.weight.grad
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.norm()),1e-4)
        self.assertIsNone(result['handoff_tau_seconds'])

    def test_stage_first_last_and_midpoint_fill_are_deterministic(self):
        indices=list(range(100,180));stages=['s'+str(i//12)for i in range(80)]
        selected=select_parent_runs(indices,stages);again=select_parent_runs(indices,stages)
        self.assertEqual(selected,again);self.assertEqual(len(selected['selected_global_indices']),16)
        self.assertEqual(selected['stage_count'],7)
        for stage in set(stages):
            group=[i for i,s in zip(indices,stages)if s==stage]
            self.assertIn(group[0],selected['selected_global_indices']);self.assertIn(group[-1],selected['selected_global_indices'])

    def test_more_than8_stages_cannot_silently_drop_a_stage(self):
        with self.assertRaises(ValueError):select_parent_runs(list(range(18)),[str(i//2)for i in range(18)])

    def test_explicit_demo_bank_maps_names_and_keeps_opening_unissued(self):
        model=WorldTCPCommandTransformerV2(CandidateConfig(width=32,layers=1,heads=4,dropout=0.)).eval()
        with torch.no_grad():raw=model(**inputs())
        bank=bank_from_model_output(raw)
        self.assertEqual(type(bank),DemoCommandBankV2)
        self.assertFalse(bank.demo_logits.flags.writeable)
        converted=generic_v1_proposals(bank)
        self.assertTrue(np.array_equal(converted['proposal_logits'],bank.demo_logits))
        ref=select_generic_reference(bank,timestamp=1.,policy_sha256='a'*64,e_sha256='b'*64,selection_rule='highest_demo_likelihood')
        self.assertIn('unresolved_not_immediate',ref.opening_intent_condition)
        bad=dict(raw,private_GT=False)
        with self.assertRaises(ValueError):bank_from_model_output(bad)
        with self.assertRaises(ValueError):generic_v1_proposals(converted)


if __name__=='__main__':unittest.main()
