"""Synthetic pooling/backward at actual micro2 spatial and model dimensions."""
import ast
import inspect
import unittest
import torch
from torch.nn import functional as F
from our_mani.current_cube_rgbd_centre_large_20261104_v1.model import LargeDualCropCubeCentreV1
from our_mani.current_cube_rgbd_centre_large_20261104_v1.config import fixed_config as old_config
from our_mani.current_cube_rgbd_centre_large_20261104_v2.model import (
    LargeDualCropCubeCentreV2,fixed_average_pool2d,parameter_count)
from our_mani.current_cube_rgbd_centre_large_20261104_v2.config import fixed_config


class Pooling(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(261104901)
        self.old_deterministic=torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
    def tearDown(self):torch.use_deterministic_algorithms(self.old_deterministic)

    def test_actual_bins_outputs_and_backward_FP32_BF16(self):
        # B2 x two crops x4views=16; encoder64ch8x8, metric3ch64x64,
        # validity2ch64x64. The binary denominator is identical in both ops.
        for dtype in (torch.float32,torch.bfloat16):
            for shape in ((16,64,8,8),(16,3,64,64),(16,2,64,64)):
                x=torch.randn(shape,dtype=dtype,requires_grad=True);y=x.detach().clone().requires_grad_()
                old=F.adaptive_avg_pool2d(x,(2,2));new=fixed_average_pool2d(y,2)
                torch.testing.assert_close(new,old,atol=2e-6 if dtype==torch.float32 else .002,
                    rtol=2e-6 if dtype==torch.float32 else .01)
                cotangent=torch.randn_like(old);old.backward(cotangent);new.backward(cotangent)
                torch.testing.assert_close(y.grad,x.grad,atol=0,rtol=0)
        with self.assertRaises(ValueError):fixed_average_pool2d(torch.ones(1,1,8,8),3)

    def test_parameters_state_keys_numeric_config_and_delta(self):
        self.assertEqual(fixed_config(),old_config());self.assertEqual(parameter_count(),25409695)
        with torch.device('meta'):
            old=LargeDualCropCubeCentreV1();new=LargeDualCropCubeCentreV2()
        self.assertEqual({k:tuple(v.shape)for k,v in old.state_dict().items()},
            {k:tuple(v.shape)for k,v in new.state_dict().items()})
        # Check complete forward AST after mechanically restoring only3 calls.
        import textwrap
        a=ast.parse(textwrap.dedent(inspect.getsource(LargeDualCropCubeCentreV1.forward)))
        b=ast.parse(textwrap.dedent(inspect.getsource(LargeDualCropCubeCentreV2.forward)))
        class Restore(ast.NodeTransformer):
            count=0
            def visit_Call(self,node):
                self.generic_visit(node)
                if isinstance(node.func,ast.Name)and node.func.id=='fixed_average_pool2d':
                    self.count+=1;node.func=ast.Attribute(ast.Name('F',ast.Load()),'adaptive_avg_pool2d',ast.Load())
                    node.args[1]=ast.Tuple([ast.Name('grid',ast.Load()),ast.Name('grid',ast.Load())],ast.Load())
                return node
        restore=Restore();b=restore.visit(b)
        self.assertEqual(restore.count,3);self.assertEqual(ast.dump(a),ast.dump(b))

    def test_actual_full25M_micro2_BF16_backward_all_encoder_trainable(self):
        model=LargeDualCropCubeCentreV2()
        # Synthetic fixture only, not an actual transferred or qualified model.
        # The production guard/prefix-transfer function is unchanged.
        model._encoder_initialization={'synthetic_CPU_fixture_only':True}
        features=torch.randn(2,4,9,64,64);features[:,:,7:9]=1.
        wide=torch.randn_like(features);wide[:,:,7:9]=1.
        K=torch.diag(torch.tensor([-200.,-200.,1.])).expand(2,4,3,3).clone()
        R=torch.eye(3).expand(2,4,3,3).clone();sphere=torch.zeros(2,3)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            value=model(features=features,wide_features=wide,K=K,camera_rotations=R,initial_sphere=sphere)
        loss=(value['centre_world'].float().square().mean()+value['uncertainty_score_m'].float().mean()
            +value['measurement_availability_logit'].float().square().mean())
        loss.backward()
        self.assertEqual(sum(p.numel()for p in model.parameters()),25409695)
        self.assertTrue(torch.isfinite(loss))
        for p in model.parameters():self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all())
        self.assertGreater(sum(float(p.grad.float().abs().sum())for n,p in model.named_parameters()if n.startswith(('e0.','e1.','e2.','bottom.'))),0.)


if __name__=='__main__':unittest.main()
