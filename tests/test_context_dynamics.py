import unittest

import numpy as np
import torch

from our_mani.context_dynamics import ContextDynamicsNetwork, history_features


class ContextDynamicsTests(unittest.TestCase):
    def test_simnorm_groups_and_finite_multistep_forecast(self):
        model = ContextDynamicsNetwork().eval()
        history = torch.zeros(2,20,8)
        history[:,:,6:] = 1.
        with torch.inference_mode():
            prediction = model(history,torch.tensor([5,20]),torch.zeros(2,6))
        self.assertTrue(torch.isfinite(prediction['positions']).all())
        self.assertTrue((prediction['context']>=0).all())
        torch.testing.assert_close(prediction['context'].reshape(2,2,8).sum(-1),torch.ones(2,2))
        torch.testing.assert_close(prediction['latents'].reshape(2,23,8,8).sum(-1),torch.ones(2,23,8))

    def test_padded_future_rows_cannot_change_context(self):
        torch.manual_seed(1)
        model = ContextDynamicsNetwork().eval()
        history = torch.randn(1,20,8)
        other = history.clone();other[:,5:]=1000.
        with torch.inference_mode():
            a=model.encode_context(history,torch.tensor([5]))
            b=model.encode_context(other,torch.tensor([5]))
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_missing_time_steps_are_rejected(self):
        times=np.array([0.,.05,.1,.2,.25])
        with self.assertRaisesRegex(ValueError,'missing'):
            history_features(times,np.zeros((5,3)),np.zeros((5,3)),np.ones(5))

    def test_context_fingerprint_tracks_encoder_weights(self):
        model=ContextDynamicsNetwork()
        before=model.encoder_sha256()
        with torch.no_grad():
            model.history_encoder.weight_ih_l0[0,0]+=1.
        self.assertNotEqual(before,model.encoder_sha256())


if __name__=='__main__':
    unittest.main()
