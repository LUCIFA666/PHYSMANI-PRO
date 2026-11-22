"""Qualified v5 current-depth runtime; never promotes selected-only weights."""
from pathlib import Path
import json

import numpy as np
import torch

from ..near_goal_depth_refiner_20261103.data import make_near_goal_crops, sha256
from ..near_goal_depth_refiner_20261103.model import TinyDepthInfluenceUNet, predict_probabilities
from ..near_goal_depth_refiner_20261103_v4.calibration import (
    assert_cpu_prediction_contract, check_library_environment,
)
from ..near_goal_depth_refiner_20261103_v4.evaluation import (
    THRESHOLD_GRID, QUALITY_PROTOCOL, BASELINE_ELIGIBILITY_PROTOCOL, probability_to_observation,
)
from ..near_goal_fresh4_calibration_20261103.data import PARTITION_IDS
from ..near_goal_fresh4_calibration_20261103.evaluation import choose_fresh_calibrated_threshold
from .qualification import (
    VERSION, PROTOCOL_SHA256, qualification_source_map, validate_selected_checkpoint,
    state_digest, checked_sha, _local_evidence,
)
from .training import checked_geometry_sources


class CurrentCubeDepthRefinerV5:
    def __init__(self, checkpoint, expected_sha256, *, allow_unqualified_diagnostic=False):
        if type(allow_unqualified_diagnostic) is not bool:
            raise ValueError('Explicit diagnostic bool required')
        path = Path(checkpoint).resolve(strict=True)
        if sha256(path) != checked_sha(expected_sha256):
            raise ValueError('Calibrated checkpoint SHA mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        meta = payload.get('metadata', {})
        if (payload.get('version') != VERSION or meta.get('qualification_version') != VERSION
                or meta.get('stage') != 'fresh4_calibrated' or type(meta.get('model_qualified')) is not bool
                or meta.get('protocol_sha256') != PROTOCOL_SHA256 or meta.get('completion') is not False
                or meta.get('current_depth_influence_only') is not True
                or meta.get('complete_visible_instance_claimed') is not False
                or meta.get('geometry_erosions') != 1 or meta.get('TCP_GRU_GT_input') is not False
                or meta.get('quality_protocol') != QUALITY_PROTOCOL
                or meta.get('baseline_eligibility_protocol') != BASELINE_ELIGIBILITY_PROTOCOL
                or meta.get('absolute_worldXYZ_input') is not False
                or meta.get('Native_control_qualified') is not False or meta.get('grasp_qualified') is not False
                or meta.get('calibration_parent_ids') != list(PARTITION_IDS)
                or meta.get('calibration_scope') != 'fresh4_calibration'
                or meta.get('calibration_device') != 'cpu' or meta.get('calibration_num_threads') != 2
                or meta.get('calibration_forward_calls') != 32
                or meta.get('geometry_source_sha256') != checked_geometry_sources()):
            raise ValueError('Only completed fresh4-qualified v5 artifact accepted; selected-only is not runtime')
        fresh = meta.get('fresh_identity', {})
        sources = qualification_source_map()
        if fresh.get('method_source_sha256') != sources:
            raise ValueError('Exact current explicit qualification/fit/geometry source map required')
        selected_ref = fresh.get('selected_evidence', {}).get('selected', {})
        selected, selected_evidence = validate_selected_checkpoint(selected_ref['path'], selected_ref['sha256'])
        changed_fit_fields = {'stage', 'model_qualified', 'selected_threshold', 'calibration_state',
                              'calibration_forward_calls'}
        for key, value in selected['metadata'].items():
            if key not in changed_fit_fields and meta.get(key) != value:
                raise ValueError('Original selected-fit metadata must remain unchanged')
        if (selected_evidence != fresh['selected_evidence'] or meta.get('identity') != selected['metadata']['identity']
                or meta.get('selected_checkpoint_sha256') != selected_ref['sha256']
                or meta.get('selected_weights_sha256') != state_digest(selected['model_state_dict'])
                or state_digest(payload['model_state_dict']) != meta['selected_weights_sha256']):
            raise ValueError('Calibrated artifact must preserve exact selected weights and fit identity')
        evidence = _local_evidence(path.parent, meta.get('qualification_evidence', {}),
            ('fresh_identity', 'request', 'eligibility_before_predictions', 'prediction_freeze', 'qualification'))
        if evidence['fresh_identity'] != fresh:
            raise ValueError('Qualification identity changed')
        request = fresh.get('request', {})
        request_path = Path(request['path']).resolve(strict=True)
        if (request_path.parent != Path(fresh['capture_root']).resolve(strict=True)
                or request_path.name != 'calibration_consumption.request.json'
                or sha256(request_path) != request['sha256']
                or json.loads(request_path.read_text()) != evidence['request']):
            raise ValueError('Actual exclusive cohort consumption request changed')
        sibling = fresh.get('dataset_request', {})
        sibling_path = Path(sibling['path']).resolve(strict=True)
        if (sibling_path.parent != Path(fresh['manifest']['path']).resolve().parent
                or sibling_path.name != 'calibration_consumption.request.json'
                or sibling.get('sha256') != request['sha256'] or sha256(sibling_path) != request['sha256']):
            raise ValueError('Dataset sibling consumption record changed')
        claimed = evidence['request']
        if (claimed.get('manifest') != fresh['manifest']
                or claimed.get('selected_checkpoint_sha256') != selected_ref['sha256']
                or claimed.get('qualification_version') != VERSION
                or claimed.get('protocol_sha256') != PROTOCOL_SHA256
                or claimed.get('method_source_sha256') != sources
                or claimed.get('capture_root') != fresh['capture_root']
                or Path(claimed.get('output', '')).resolve() != path.parent
                or claimed.get('no_retry') is not True or claimed.get('request_consumes_cohort_even_if_failure') is not True):
            raise ValueError('This qualification must match the original one-use request')
        freeze = evidence['prediction_freeze']
        if (freeze.get('complete') is not True or freeze.get('frozen_before_scores') is not True
                or freeze.get('selected') != selected_ref or freeze.get('fresh_manifest') != fresh['manifest']
                or freeze.get('method_source_sha256') != sources or freeze.get('request') != request):
            raise ValueError('Pre-score probability freeze required')
        for key in ('probabilities', 'eligibility'):
            ref = freeze[key]
            target = Path(ref['path']).resolve(strict=True)
            if target.parent != path.parent or sha256(target) != ref['sha256']:
                raise ValueError('Frozen predictions or eligibility changed')
        if freeze['eligibility']['sha256'] != meta['qualification_evidence']['eligibility_before_predictions']['sha256']:
            raise ValueError('Frozen eligibility differs from accompanying ledger')
        execution = freeze.get('execution', {})
        if (execution.get('calibration_device') != 'cpu' or execution.get('calibration_num_threads') != 2
                or execution.get('model_eval') is not True or execution.get('model_dtype') != 'float32'
                or execution.get('forward_calls') != 32 or execution.get('samples') != 32
                or execution.get('parent_count') != 4 or execution.get('threshold_grid') != list(THRESHOLD_GRID)
                or execution.get('GPU_CPU_selection_performed') is not False
                or execution != meta.get('calibration_environment')):
            raise ValueError('Exactly32 four-view CPU2 forwards in the matching environment required')
        check_library_environment()
        check_library_environment(execution)
        report = evidence['qualification']
        if (report.get('version') != VERSION or report.get('complete') is not True
                or report.get('fresh_identity') != fresh or report.get('execution') != execution
                or report.get('source_before_after_verified') is not True
                or report.get('old_selected_artifact_unchanged') is not True
                or report.get('predictions_frozen_before_scores') is not True
                or report.get('qualification_is_internal_static_known_50mm_axisCube_only') is not True
                or report.get('Native_control_qualified') is not False or report.get('grasp_qualified') is not False
                or report.get('TEST_read') is not False):
            raise ValueError('Complete source-bound fresh4 qualification evidence required')
        selection = choose_fresh_calibrated_threshold(report['reports'], evidence['eligibility_before_predictions'])
        if (selection != report.get('selection') or selection['calibration_passed'] is not meta['model_qualified']
                or selection['selected_threshold'] != meta.get('selected_threshold')
                or meta.get('calibration_state') != ('passed' if meta['model_qualified'] else 'failed_raw995')):
            raise ValueError('Qualification/threshold must equal unchanged raw995/five-grid gates')
        status = json.loads(path.with_name('status.json').read_text())
        if (status.get('version') != VERSION or status.get('complete') is not True
                or status.get('qualification_complete') is not True
                or status.get('calibrated_checkpoint_sha256') != expected_sha256
                or status.get('selected_checkpoint_sha256') != selected_ref['sha256']
                or status.get('selected_weights_sha256') != meta['selected_weights_sha256']
                or status.get('model_qualified') is not meta['model_qualified']
                or status.get('fresh_identity') != fresh or status.get('method_source_sha256') != sources
                or status.get('execution') != execution or status.get('calibration_forward_calls') != 32
                or status.get('Native_control_qualified') is not False or status.get('grasp_qualified') is not False):
            raise ValueError('Same calibrated-checkpoint completed qualification status required')
        self.qualified = meta['model_qualified']
        if not self.qualified and not allow_unqualified_diagnostic:
            raise ValueError('Failed raw995 calibration cannot provide a centre or fallback threshold')
        self.model = TinyDepthInfluenceUNet().float().cpu()
        self.model.load_state_dict(payload['model_state_dict'], strict=True)
        self.model.eval()
        self.threshold = meta['selected_threshold']
        self.metadata, self.checkpoint_sha256 = meta, expected_sha256
        self.calibration_environment, self._sources = execution, sources

    def predict(self, inputs):
        assert_cpu_prediction_contract(self.model, self.calibration_environment)
        if qualification_source_map() != self._sources:
            raise ValueError('Qualified measurement source/profile changed')
        crops = make_near_goal_crops(inputs)
        probabilities = predict_probabilities(self.model, crops['features'])
        if not self.qualified:
            return dict(refiner_available=False, centre=None, probabilities=probabilities,
                reason='failed_raw995_fresh4_diagnostic_only_no_threshold_fallback', completion=False,
                checkpoint_sha256=self.checkpoint_sha256, Native_control_qualified=False, grasp_qualified=False)
        result = probability_to_observation(inputs, crops, probabilities, self.threshold)
        result.update(refiner_available=result['measurement']['refiner_available'],
            centre=result['measurement']['centre'], checkpoint_sha256=self.checkpoint_sha256,
            Native_control_qualified=False, grasp_qualified=False)
        return result
