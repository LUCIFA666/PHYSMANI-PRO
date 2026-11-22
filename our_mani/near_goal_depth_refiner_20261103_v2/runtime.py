"""SHA-bound schedule160 runtime, separate from the immutable40epoch runtime."""
from pathlib import Path
import json

import numpy as np
import torch

from ..near_goal_depth_refiner_20261103.data import (
    CLASS_NAMES, FEATURE_NAMES, LABEL_PROTOCOL, PARTITION_IDS, SHOT_IDS, sha256, make_near_goal_crops,
)
from ..near_goal_depth_refiner_20261103.model import TinyDepthInfluenceUNet, predict_probabilities
from ..near_goal_depth_refiner_20261103.evaluation import (
    QUALITY_PROTOCOL, THRESHOLD_GRID, choose_calibrated_threshold, probability_to_observation,
)
from .cache import canonical_sha
from .training import (VERSION, EPOCHS, OPTIMIZER_STEPS, PROTOCOL_SHA256,
                       fixed_config, check_frozen_definitions, implementation_source_map,
                       CPU_CALIBRATION_ADDENDUM_SHA256)
from .calibration import (CALIBRATION_DEVICE, CALIBRATION_NUM_THREADS,
                           assert_cpu_prediction_contract, check_library_environment)


def _ledger(records, parents, roles):
    expected = [(p, p+'/'+shot, role) for p, role in zip(parents, roles) for shot in SHOT_IDS]
    actual = [(r.get('parent_id'), r.get('scene_id'), r.get('role')) for r in records]
    if actual != expected:
        raise ValueError('Entire fixed ordered parent/shot ledger required, no filtered or pooled substitute')


def _cache_evidence(before, after, identity):
    if (before != after or before.get('total_bytes') != 135790592
            or before.get('calibration_cached') is not False
            or set(before.get('roles', {})) != {'fit', 'selection'}):
        raise ValueError('Exact unchanged fit/selection-only cache evidence required')
    roles = before['roles']
    if before.get('roles_content_sha256') != canonical_sha(roles):
        raise ValueError('Cache role evidence content hash mismatch')
    for role, parents, count in (('fit', PARTITION_IDS[:24], 192), ('selection', PARTITION_IDS[24:28], 32)):
        row = roles[role]
        expected = [[p, p+'/'+shot, role] for p in parents for shot in SHOT_IDS]
        if (row.get('role') != role or row.get('samples') != count
                or row.get('fields_only') != ['features', 'semantic']
                or row.get('ordered_records') != expected
                or row.get('ordered_records_sha256') != canonical_sha(expected)
                or row.get('features', {}).get('shape') != [count, 4, 9, 64, 64]
                or row.get('features', {}).get('dtype') != '<f4'
                or row.get('semantic', {}).get('shape') != [count, 4, 64, 64]
                or row.get('semantic', {}).get('dtype') != '|u1'
                or row.get('calibration_cached') is not False
                or row.get('privileged_metadata_or_robot_or_CF_cached_as_features') is not False
                or row.get('source_manifest_sha256') != identity.get('manifest_sha256')
                or row.get('source_binding_sha256') != canonical_sha(identity['bound_data_sha256'])):
            raise ValueError('Cache order/role/dtype/source semantics differ from fixed full32 protocol')
        for digest in (row.get('ordered_content_sha256'), row['features'].get('sha256'), row['semantic'].get('sha256')):
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError('Ordered cache content hashes required')


class CurrentCubeDepthRefinerV2:
    def __init__(self, checkpoint, expected_sha256, *, allow_unqualified_diagnostic=False):
        if type(allow_unqualified_diagnostic) is not bool:
            raise ValueError('Explicit diagnostic bool required')
        check_frozen_definitions()
        path = Path(checkpoint).resolve(strict=True)
        if sha256(path) != expected_sha256:
            raise ValueError('Checkpoint SHA mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        meta = payload.get('metadata', {})
        if (payload.get('version') != VERSION or meta.get('scope') != 'full32'
                or meta.get('dataset_split') != 'TRAIN' or meta.get('TEST_read') is not False
                or meta.get('training_complete') is not True
                or type(meta.get('epochs_completed')) is not int or meta['epochs_completed'] != EPOCHS
                or type(meta.get('actual_optimizer_steps')) is not int or meta['actual_optimizer_steps'] != OPTIMIZER_STEPS
                or meta.get('schedule_protocol_sha256') != PROTOCOL_SHA256
                or meta.get('label_protocol') != LABEL_PROTOCOL or meta.get('quality_protocol') != QUALITY_PROTOCOL
                or meta.get('completion') is not False or meta.get('current_depth_influence_only') is not True
                or meta.get('complete_visible_instance_claimed') is not False
                or meta.get('geometry_erosions') != 1 or meta.get('calibration_cached') is not False
                or meta.get('TCP_GRU_GT_input') is not False or meta.get('absolute_worldXYZ_input') is not False
                or tuple(meta.get('feature_names', ())) != FEATURE_NAMES
                or tuple(meta.get('class_names', ())) != CLASS_NAMES
                or meta.get('model_config') != dict(input_channels=9, classes=2, base_channels=8)
                or type(meta.get('selected_epoch')) is not int or not 1 <= meta['selected_epoch'] <= EPOCHS
                or type(meta.get('model_qualified')) is not bool
                or meta.get('calibration_parent_ids') != list(PARTITION_IDS[28:])):
            raise ValueError('Only exact completed full32 schedule160/1920 binary model contract accepted')
        identity = meta.get('identity')
        if not isinstance(identity, dict) or identity.get('config') != fixed_config():
            raise ValueError('Exact independent160/cache configuration identity required')
        if identity.get('code_sha256') != implementation_source_map():
            raise ValueError('Exact current frozen v1/v2/CLI implementation source SHA map required')
        if (meta.get('calibration_device') != CALIBRATION_DEVICE
                or type(meta.get('calibration_num_threads')) is not int
                or meta['calibration_num_threads'] != CALIBRATION_NUM_THREADS
                or meta.get('calibration_addendum_sha256') != CPU_CALIBRATION_ADDENDUM_SHA256):
            raise ValueError('Fixed CPU2 calibration metadata/addendum required; GPU calibration cannot qualify CPU')
        check_library_environment(meta.get('calibration_environment'))
        status = json.loads(path.with_name('status.json').read_text())
        if (status.get('version') != VERSION or status.get('complete') is not True
                or status.get('training_started') is not True or status.get('training_complete') is not True
                or type(status.get('completed_epochs')) is not int or status['completed_epochs'] != EPOCHS
                or type(status.get('actual_optimizer_steps')) is not int or status['actual_optimizer_steps'] != OPTIMIZER_STEPS
                or status.get('best_checkpoint_sha256') != expected_sha256 or status.get('identity') != identity
                or status.get('selected_epoch') != meta['selected_epoch'] or status.get('scope') != 'full32'
                or status.get('model_qualified') is not meta['model_qualified']):
            raise ValueError('Actual completed160epoch/1920step same-identity status required')
        if (status.get('calibration_device') != CALIBRATION_DEVICE
                or type(status.get('calibration_num_threads')) is not int
                or status['calibration_num_threads'] != CALIBRATION_NUM_THREADS
                or status.get('calibration_addendum_sha256') != CPU_CALIBRATION_ADDENDUM_SHA256):
            raise ValueError('Completed status must bind the same CPU2 calibration contract')
        evidence = {}
        names = {'calibration', 'oracle_before_fit', 'cache_before', 'cache_after', 'source_verification', 'history'}
        if set(meta.get('evidence', {})) != names:
            raise ValueError('Complete oracle/cache/source/history/calibration evidence required')
        for name, pointer in meta['evidence'].items():
            source = (path.parent/pointer['path']).resolve(strict=True)
            if source.parent != path.parent or sha256(source) != pointer['sha256']:
                raise ValueError('Accompanying evidence changed')
            evidence[name] = json.loads(source.read_text())
        _cache_evidence(evidence['cache_before'], evidence['cache_after'], identity)
        verification = evidence['source_verification']
        if (verification.get('complete') is not True
                or verification.get('raw_data_and_sources_verified_before_after') is not True
                or verification.get('cache_content_unchanged') is not True
                or verification.get('calibration_cached') is not False
                or verification.get('actual_optimizer_steps') != OPTIMIZER_STEPS
                or verification.get('original_identity_sha256') != canonical_sha(identity)
                or verification.get('source_bindings_sha256') != canonical_sha(identity['bound_data_sha256'])):
            raise ValueError('Raw data/source/cache start-end verification required')
        history = evidence['history']
        if not isinstance(history, list) or len(history) != EPOCHS:
            raise ValueError('All160 epoch records required')
        for epoch, row in enumerate(history, 1):
            if (type(row.get('epoch')) is not int or row['epoch'] != epoch
                    or type(row.get('actual_optimizer_steps')) is not int or row['actual_optimizer_steps'] != epoch*12
                    or not np.isfinite([row.get('fit_loss'), row.get('selection_loss')]).all()):
                raise ValueError('Actual epoch/optimizer/loss ledger inconsistent')
        selected_epoch = min(range(EPOCHS), key=lambda i: history[i]['selection_loss'])+1
        if selected_epoch != meta['selected_epoch']:
            raise ValueError('Only selection epoch loss may choose the checkpoint')
        oracle = evidence['oracle_before_fit']
        _ledger(oracle['records'], PARTITION_IDS, ['fit']*24+['selection']*4+['calib']*4)
        if oracle.get('ready_for_segmentation_fit') is not True:
            raise ValueError('Original full256 oracle must have passed before optimization')
        calibration = evidence['calibration']
        execution = calibration.get('execution', {})
        if (calibration.get('calibration_device') != CALIBRATION_DEVICE
                or type(calibration.get('calibration_num_threads')) is not int
                or calibration['calibration_num_threads'] != CALIBRATION_NUM_THREADS
                or execution != meta.get('calibration_environment')
                or execution.get('calibration_device') != CALIBRATION_DEVICE
                or execution.get('calibration_num_threads') != CALIBRATION_NUM_THREADS
                or execution.get('model_eval') is not True or execution.get('model_dtype') != 'float32'
                or execution.get('forward_calls') != 32 or execution.get('samples') != 32
                or execution.get('parent_count') != 4 or execution.get('threshold_grid') != list(THRESHOLD_GRID)
                or execution.get('GPU_CPU_selection_performed') is not False
                or verification.get('calibration_device') != CALIBRATION_DEVICE
                or verification.get('calibration_num_threads') != CALIBRATION_NUM_THREADS
                or verification.get('calibration_environment') != execution):
            raise ValueError('Exactly32 one-pass CPU2 calibration forwards and unchanged5grid evidence required')
        check_library_environment(execution)
        if tuple(r.get('threshold') for r in calibration['reports']) != THRESHOLD_GRID:
            raise ValueError('Original5threshold grid required')
        for report in calibration['reports']:
            _ledger(report['records'], PARTITION_IDS[28:], ['calib']*4)
            if set(report.get('per_parent_pixels', {})) != set(PARTITION_IDS[28:]):
                raise ValueError('All4calibration parents must appear in threshold evidence')
        recalculated = choose_calibrated_threshold(calibration['reports'])
        if (calibration.get('selection') != recalculated
                or recalculated['selected_threshold'] != meta.get('selected_threshold')
                or recalculated['calibration_passed'] is not meta['model_qualified']):
            raise ValueError('Qualified flag/threshold must equal unchanged conservative calibration gates')
        self.qualified = meta['model_qualified']
        if not self.qualified and not allow_unqualified_diagnostic:
            raise ValueError('Unqualified model cannot provide a current-depth measurement')
        self.model = TinyDepthInfluenceUNet().cpu()
        self.model.load_state_dict(payload['model_state_dict'], strict=True)
        self.model.eval()
        self.threshold, self.metadata, self.checkpoint_sha256 = meta['selected_threshold'], meta, expected_sha256
        self.calibration_environment = execution

    def predict(self, inputs):
        if self.qualified:
            assert_cpu_prediction_contract(self.model, self.calibration_environment)
        crops = make_near_goal_crops(inputs)
        probabilities = predict_probabilities(self.model, crops['features'])
        if not self.qualified:
            return dict(refiner_available=False, centre=None, probabilities=probabilities,
                reason='unqualified_schedule160_diagnostic_only_no_threshold_fallback', completion=False,
                checkpoint_sha256=self.checkpoint_sha256, Native_control_qualified=False)
        result = probability_to_observation(inputs, crops, probabilities, self.threshold)
        result.update(refiner_available=result['measurement']['refiner_available'],
                      centre=result['measurement']['centre'], checkpoint_sha256=self.checkpoint_sha256)
        return result
