"""One fixed selected-model CPU2 calibration, matching policy inference."""
import sys

import numpy as np
import torch

from ..near_goal_depth_refiner_20261103.data import PARTITION_IDS, SHOT_IDS
from ..near_goal_depth_refiner_20261103.model import predict_probabilities
from .evaluation import (
    evaluate_probabilities, choose_calibrated_threshold, THRESHOLD_GRID,
    BASELINE_ELIGIBILITY_PROTOCOL, CALIBRATION_BASELINE_ELIGIBLE,
)
from ..near_goal_cube_geometry_20261103_v1 import VERSION as GEOMETRY_VERSION, geometry_source_map

CALIBRATION_DEVICE = 'cpu'
CALIBRATION_NUM_THREADS = 2
EXPECTED_TORCH_VERSION = '2.8.0+cu128'
EXPECTED_NUMPY_VERSION = '1.23.5'


def calibration_environment():
    return dict(torch_version=str(torch.__version__), numpy_version=np.__version__,
                python_executable=sys.executable)


def check_library_environment(expected_environment=None):
    actual = calibration_environment()
    expected = expected_environment or dict(torch_version=EXPECTED_TORCH_VERSION,
                                            numpy_version=EXPECTED_NUMPY_VERSION)
    if any(actual[key] != expected.get(key) for key in ('torch_version', 'numpy_version')):
        raise ValueError('Runtime Torch/NumPy must match the fixed NativePython CPU calibration environment')
    return actual


def assert_cpu_prediction_contract(model, expected_environment=None):
    """No device/thread correction at inference: reject mismatched execution."""
    if torch.get_num_threads() != CALIBRATION_NUM_THREADS:
        raise ValueError('Qualified measurement requires actual policy Torchthreads2')
    tensors = list(model.parameters())+list(model.buffers())
    if not tensors or any(t.device.type != CALIBRATION_DEVICE for t in tensors):
        raise ValueError('CPU-calibrated measurement requires all model tensors onCPU')
    if model.training or any(t.is_floating_point() and t.dtype != torch.float32 for t in tensors):
        raise ValueError('Exact eval/float32 CPU inference contract required')
    return check_library_environment(expected_environment)


def calibrate_cpu_selected_model(model, samples):
    """Exactly32 CPU2 forwards once, followed by the unchanged five-grid report."""
    samples = list(samples)
    expected = [(p, p+'/'+shot, 'calib') for p in PARTITION_IDS[28:] for shot in SHOT_IDS]
    actual = [(s.get('parent_id'), s.get('scene_id'), s.get('role')) for s in samples]
    if actual != expected:
        raise ValueError('Exactly4fixedcalibration parents/all32ordered samples required')
    model.cpu().eval()
    torch.set_num_threads(CALIBRATION_NUM_THREADS)
    environment = assert_cpu_prediction_contract(model)
    probabilities = []
    for sample in samples:
        features = sample['features']
        if (not isinstance(features, np.ndarray) or features.dtype != np.float32
                or features.shape != (4, 9, 64, 64) or not np.isfinite(features).all()):
            raise ValueError('Unchanged current float32 four-view calibration crop required')
        assert_cpu_prediction_contract(model, environment)
        probabilities.append(predict_probabilities(model, features))
    assert_cpu_prediction_contract(model, environment)
    reports = [evaluate_probabilities(samples, probabilities, threshold) for threshold in THRESHOLD_GRID]
    for report in reports:
        if (report.get('geometry_version') != GEOMETRY_VERSION
                or report.get('geometry_source_sha256') != geometry_source_map()
                or report.get('baseline_eligibility_protocol') != BASELINE_ELIGIBILITY_PROTOCOL
                or report.get('positions', {}).get('oracle_geometry_eligible') != CALIBRATION_BASELINE_ELIGIBLE):
            raise ValueError('Every CPU2 threshold report must retain fixedold20 and exactF geometry identity')
    execution = dict(calibration_device=CALIBRATION_DEVICE,
        calibration_num_threads=CALIBRATION_NUM_THREADS, model_eval=True,
        model_dtype='float32', forward_calls=32, samples=32, parent_count=4,
        threshold_grid=list(THRESHOLD_GRID), GPU_CPU_selection_performed=False, **environment)
    return dict(selection=choose_calibrated_threshold(reports), reports=reports,
        calibration_device=CALIBRATION_DEVICE, calibration_num_threads=CALIBRATION_NUM_THREADS,
        execution=execution)
