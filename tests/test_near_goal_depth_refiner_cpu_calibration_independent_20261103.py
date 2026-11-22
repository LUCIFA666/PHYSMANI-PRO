"""CPU2 calibration/inference contract only: no GPU, Native, optimizer or fit."""
from copy import deepcopy
from unittest.mock import patch
import unittest

import numpy as np
import torch
from torch import nn

from our_mani.near_goal_depth_refiner_20261103 import data as frozen_data
from our_mani.near_goal_depth_refiner_20261103.model import TinyDepthInfluenceUNet, predict_probabilities
from our_mani.near_goal_depth_refiner_20261103_v2 import calibration as cpu
from our_mani.near_goal_depth_refiner_20261103_v2 import runtime, training


def calibration_samples():
    values = []
    for parent in frozen_data.PARTITION_IDS[28:]:
        for shot in frozen_data.SHOT_IDS:
            x = np.zeros((4,9,64,64), np.float32)
            x[:,0,0,0] = len(values)/32
            values.append(dict(parent_id=parent, scene_id=parent+'/'+shot, role='calib', features=x))
    return values


def public():
    return dict(rgb=np.zeros((4,256,256,3),np.uint8), depth_m=np.ones((4,256,256),np.float32),
        K=np.repeat(np.array([[-100.,0,128],[0,-100.,128],[0,0,1]])[None],4,0),
        T=np.repeat(np.eye(4)[None],4,0), public_sphere_center=np.array([0.,0.,1.]),
        camera_names=np.array(frozen_data.CAMERAS), timestamp=np.array(0.))


class CPUCalibrationIndependentTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self): torch.set_num_threads(self.previous_threads)

    def test_selected_model_one32shot_CPU2_pass_then_original5grid_only(self):
        torch.manual_seed(261103700)
        model = TinyDepthInfluenceUNet().train()
        parameters = {k:v.detach().clone() for k,v in model.state_dict().items()}
        samples = calibration_samples(); calls = []; evaluated = []
        def forward(selected, features):
            self.assertIs(selected, model)
            self.assertEqual(torch.get_num_threads(), 2)
            self.assertFalse(selected.training)
            self.assertTrue(all(p.device.type == 'cpu' for p in selected.parameters()))
            self.assertEqual(features.shape, (4,9,64,64))
            self.assertEqual(features.dtype, np.float32)
            calls.append(features)
            return predict_probabilities(selected, features)
        def evaluate(fixed_samples, probabilities, threshold):
            self.assertEqual(len(probabilities), 32)
            self.assertTrue(all(p.shape == (4,2,64,64) and np.isfinite(p).all() for p in probabilities))
            self.assertTrue(all(a is b for a,b in zip(fixed_samples,samples)))
            evaluated.append((threshold, probabilities))
            return dict(threshold=threshold, synthetic_only=True)
        torch.set_num_threads(4)
        with patch.object(model, 'cpu', wraps=model.cpu) as moved, \
             patch.object(cpu, 'predict_probabilities', side_effect=forward), \
             patch.object(cpu, 'evaluate_probabilities', side_effect=evaluate), \
             patch.object(cpu, 'choose_calibrated_threshold', return_value=dict(selected_threshold=None,calibration_passed=False)) as choose:
            result = cpu.calibrate_cpu_selected_model(model, samples)
        moved.assert_called_once(); choose.assert_called_once()
        self.assertEqual(len(calls), 32)
        self.assertTrue(all(features is sample['features'] for features,sample in zip(calls,samples)))
        self.assertEqual([x[0] for x in evaluated], [.5,.7,.9,.95,.99])
        self.assertEqual(torch.get_num_threads(), 2)
        self.assertEqual(result['calibration_device'], 'cpu')
        self.assertEqual(result['calibration_num_threads'], 2)
        self.assertFalse(result['execution']['GPU_CPU_selection_performed'])
        self.assertEqual(result['execution']['torch_version'], '2.8.0+cu128')
        self.assertEqual(result['execution']['numpy_version'], '1.23.5')
        self.assertTrue(all(torch.equal(parameters[k],v) for k,v in model.state_dict().items()))
        # Same selected CPU model, same four-view path as runtime, bit-exact.
        for index in (0,15,31):
            np.testing.assert_array_equal(evaluated[0][1][index], predict_probabilities(model,samples[index]['features']))

    def test_incomplete_or_reordered_calibration_rejected_before_model_or_threads(self):
        samples = calibration_samples(); swapped = samples.copy(); swapped[0],swapped[1]=swapped[1],swapped[0]
        for bad in (samples[:-1], swapped):
            model = nn.Linear(1,1).eval(); torch.set_num_threads(4)
            with patch.object(model, 'cpu', side_effect=AssertionError('model should not move')), \
                 patch.object(cpu, 'predict_probabilities', side_effect=AssertionError('no prediction')):
                with self.assertRaisesRegex(ValueError, '32ordered'):
                    cpu.calibrate_cpu_selected_model(model, bad)
            self.assertEqual(torch.get_num_threads(), 4)

    def test_contract_rejects_trainmode_float64_nonCPUbuffer_and_library_drift(self):
        model = nn.Linear(1,1).eval()
        cpu.assert_cpu_prediction_contract(model)
        model.train()
        with self.assertRaisesRegex(ValueError, 'eval/float32'): cpu.assert_cpu_prediction_contract(model)
        model.eval().double()
        with self.assertRaisesRegex(ValueError, 'eval/float32'): cpu.assert_cpu_prediction_contract(model)
        model.float().register_buffer('not_cpu', torch.empty(1, device='meta'))
        with self.assertRaisesRegex(ValueError, 'all model tensors onCPU'): cpu.assert_cpu_prediction_contract(model)
        environment = cpu.calibration_environment(); environment['numpy_version'] = 'wrong-version'
        with self.assertRaisesRegex(ValueError, 'Torch/NumPy'):
            cpu.assert_cpu_prediction_contract(nn.Linear(1,1).eval(), environment)

    def test_qualified_thread_mismatch_refuses_before_crop_without_global_correction(self):
        obj = runtime.CurrentCubeDepthRefinerV2.__new__(runtime.CurrentCubeDepthRefinerV2)
        obj.qualified, obj.model = True, nn.Linear(1,1).eval()
        obj.calibration_environment = cpu.calibration_environment()
        torch.set_num_threads(4)
        with patch.object(runtime, 'make_near_goal_crops', side_effect=AssertionError('crop forbidden')) as crops, \
             patch.object(runtime, 'predict_probabilities', side_effect=AssertionError('forward forbidden')) as predict:
            with self.assertRaisesRegex(ValueError, 'Torchthreads2'): obj.predict({})
        crops.assert_not_called(); predict.assert_not_called()
        self.assertEqual(torch.get_num_threads(), 4)

    def test_qualified_device_mismatch_refuses_before_crop_without_GPU(self):
        obj = runtime.CurrentCubeDepthRefinerV2.__new__(runtime.CurrentCubeDepthRefinerV2)
        obj.qualified, obj.model = True, nn.Linear(1,1).eval()
        obj.model.register_buffer('not_cpu', torch.empty(1, device='meta'))
        obj.calibration_environment = cpu.calibration_environment()
        with patch.object(runtime, 'make_near_goal_crops', side_effect=AssertionError('crop forbidden')):
            with self.assertRaisesRegex(ValueError, 'all model tensors onCPU'): obj.predict({})

    def test_diagnostic_is_unknown_even_when_policy_thread_contract_not_met(self):
        obj = runtime.CurrentCubeDepthRefinerV2.__new__(runtime.CurrentCubeDepthRefinerV2)
        obj.qualified, obj.model, obj.checkpoint_sha256 = False, object(), 'synthetic-not-weights'
        p = np.zeros((4,2,64,64),np.float32);p[:,1]=1.
        torch.set_num_threads(4)
        with patch.object(runtime, 'predict_probabilities', return_value=p), \
             patch.object(runtime, 'probability_to_observation', side_effect=AssertionError('measurement forbidden')):
            result = obj.predict(public())
        self.assertFalse(result['refiner_available']); self.assertIsNone(result['centre'])
        self.assertFalse(result['completion']); self.assertEqual(torch.get_num_threads(),4)

    def test_source_contract_matches_baseline_CPU2_and_retains_budget(self):
        cfg=training.fixed_config()
        self.assertEqual(cfg['calibration_device'],'cpu'); self.assertEqual(cfg['calibration_num_threads'],2)
        self.assertEqual(cfg['epochs'],160); self.assertEqual(cfg['actual_optimizer_steps_required'],1920)
        self.assertIs(cpu.predict_probabilities, runtime.predict_probabilities)
        self.assertEqual(cpu.THRESHOLD_GRID,(.5,.7,.9,.95,.99))


if __name__ == '__main__': unittest.main()
