"""Position-certificate runtime; generic/raw-mask qualification always false."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

from ..near_goal_depth_refiner_20261103.data import make_near_goal_crops, sha256
from ..near_goal_depth_refiner_20261103.model import TinyDepthInfluenceUNet, predict_probabilities
from .evaluation import predict_current_position, position_gate, _eligibility, THRESHOLD
from .qualification import (
    VERSION, PROTOCOL, PROTOCOL_SHA256, FIT_PROTOCOL_SHA256, CONSUMPTION_NAME,
    qualification_source_map, check_profile_bindings, check_execution_environment,
    validate_selected_checkpoint, pointer, checked_sha, state_digest, _local_evidence,
)
from .data import PARTITION_IDS, SHOT_IDS


class CurrentCubePositionMeasurementV7:
    def __init__(self, checkpoint, expected_sha256, *, allow_unqualified_diagnostic=False):
        if type(allow_unqualified_diagnostic) is not bool:
            raise ValueError('Explicit diagnostic bool required')
        path = Path(checkpoint).resolve(strict=True)
        if sha256(path) != checked_sha(expected_sha256):
            raise ValueError('Position checkpoint SHA mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        meta = payload.get('metadata', {})
        if (payload.get('version') != VERSION or meta.get('qualification_version') != VERSION
                or meta.get('stage') != 'position_calibrated'
                or type(meta.get('position_measurement_qualified')) is not bool
                or meta.get('qualified') is not False or meta.get('model_qualified') is not False
                or meta.get('raw_mask_model_qualified') is not False
                or meta.get('protocol_sha256') != FIT_PROTOCOL_SHA256
                or meta.get('position_profile_sha256') != PROTOCOL_SHA256
                or meta.get('position_profile') != pointer(PROTOCOL)
                or meta.get('same_V5_weights') is not True or meta.get('new_training') is not False
                or meta.get('position_measurement_threshold') != THRESHOLD
                or meta.get('position_calibration_parent_ids') != list(PARTITION_IDS)
                or meta.get('position_calibration_forward_calls') != 32
                or meta.get('completion') is not False or meta.get('orientation_observed') is not False
                or meta.get('known_axisaligned_50mm_Cube_prior') is not True
                or meta.get('raw_mask_failures_preserved') is not True
                or meta.get('Native_control_qualified') is not False or meta.get('grasp_qualified') is not False):
            raise ValueError('Only independent position certificate; raw/selected models cannot provide a centre')
        identity = meta.get('position_identity', {})
        sources = qualification_source_map()
        if identity.get('method_source_sha256') != sources or identity.get('profile') != pointer(PROTOCOL):
            raise ValueError('Exact current position method/profile identity required')
        original = identity.get('selected_evidence', {})
        ref = original['selected']
        check_profile_bindings(ref['path'], ref['sha256'], identity['manifest']['path'])
        selected, original_checked = validate_selected_checkpoint(ref['path'], ref['sha256'])
        if original_checked != original or meta.get('identity') != selected['metadata']['identity']:
            raise ValueError('Immutable original V5 fit evidence differs')
        for key, value in selected['metadata'].items():
            if key != 'stage' and meta.get(key) != value:
                raise ValueError('Position artifact must preserve all original selected-fit metadata')
        digest = state_digest(selected['model_state_dict'])
        if (meta.get('selected_checkpoint_sha256') != ref['sha256']
                or meta.get('selected_weights_sha256') != digest
                or state_digest(payload['model_state_dict']) != digest):
            raise ValueError('No retraining or weight changes permitted')
        evidence = _local_evidence(path.parent, meta.get('qualification_evidence', {}),
            ('position_identity', 'request', 'eligibility_before_predictions', 'prediction_freeze', 'qualification'))
        if evidence['position_identity'] != identity:
            raise ValueError('Independent position identity changed')
        for key, parent in (('request', Path(identity['capture_root']).resolve(strict=True)),
                            ('dataset_request', Path(identity['manifest']['path']).resolve().parent)):
            item = identity[key]; source = Path(item['path']).resolve(strict=True)
            if source.parent != parent or source.name != CONSUMPTION_NAME or sha256(source) != item['sha256']:
                raise ValueError('Original one-use position consumption request changed')
            if json.loads(source.read_text()) != evidence['request']:
                raise ValueError('Canonical request and dataset sibling must match')
        claimed = evidence['request']
        if (claimed.get('manifest') != identity['manifest'] or claimed.get('capture_root') != identity['capture_root']
                or claimed.get('selected_checkpoint_sha256') != ref['sha256']
                or claimed.get('qualification_version') != VERSION
                or claimed.get('position_profile_sha256') != PROTOCOL_SHA256
                or claimed.get('training_protocol_sha256') != FIT_PROTOCOL_SHA256
                or claimed.get('method_source_sha256') != sources
                or Path(claimed.get('output', '')).resolve() != path.parent
                or claimed.get('position_measurement_only') is not True
                or claimed.get('model_qualified') is not False or claimed.get('raw_mask_model_qualified') is not False
                or claimed.get('no_retry') is not True or claimed.get('request_consumes_cohort_even_if_failure') is not True):
            raise ValueError('Exact original position-only request required')
        eligibility = evidence['eligibility_before_predictions']; _eligibility(eligibility)
        freeze = evidence['prediction_freeze']
        expected_files = {'probabilities.npz', 'public_support_masks.npz', 'public_positions.json', 'eligibility_before_predictions.json'}
        if (freeze.get('complete') is not True or freeze.get('frozen_before_GT_scores') is not True
                or freeze.get('position_identity') != identity or freeze.get('selected') != ref
                or set(freeze.get('files', {})) != expected_files):
            raise ValueError('All public position/support/probability outputs must precede GT scores')
        for name, item in freeze['files'].items():
            source = Path(item['path']).resolve(strict=True)
            if source.parent != path.parent or source.name != name or sha256(source) != item['sha256']:
                raise ValueError('Frozen public prediction evidence changed')
        if (freeze['files']['eligibility_before_predictions.json']['sha256'] !=
                meta['qualification_evidence']['eligibility_before_predictions']['sha256']):
            raise ValueError('Original eligibility ledger differs')
        execution = freeze.get('execution', {})
        environment = meta.get('position_calibration_environment')
        if (execution.get('forward_calls') != 32 or execution.get('samples') != 32
                or execution.get('parent_count') != 4 or execution.get('fixed_threshold') != THRESHOLD
                or execution.get('threshold_selected_from_fresh') is not False
                or execution.get('environment') != environment):
            raise ValueError('Exactly32 fixed-scalar CPU2 forwards required')
        check_execution_environment()
        certificate = evidence['qualification']
        if (certificate.get('version') != VERSION or certificate.get('complete') is not True
                or certificate.get('position_identity') != identity or certificate.get('execution') != execution
                or certificate.get('predictions_frozen_before_GT_scores') is not True
                or certificate.get('source_before_after_verified') is not True
                or certificate.get('same_V5_weights') is not True or certificate.get('new_training') is not False
                or certificate.get('model_qualified') is not False or certificate.get('raw_mask_model_qualified') is not False
                or certificate.get('Native_control_qualified') is not False or certificate.get('grasp_qualified') is not False
                or certificate.get('TEST_read') is not False):
            raise ValueError('Completed independent position-only evaluation required')
        report = certificate['report']
        expected = [(p, p+'/'+s, 'calib') for p in PARTITION_IDS for s in SHOT_IDS]
        if ([(r['parent_id'], r['scene_id'], r['role']) for r in report['records']] != expected
                or [(r['scene_id'], r['oracle_geometry_eligible']) for r in report['records']] !=
                   [(r['scene_id'], r['eligible']) for r in eligibility['records']]
                or report.get('threshold_selected_from_fresh') is not False
                or report['positions']['oracle_geometry_eligible'] != eligibility['required_eligible']):
            raise ValueError('All32 fixed position scenes and legacy denominator required')
        with np.load(path.parent/'public_support_masks.npz', allow_pickle=False) as stored:
            if stored['scene_ids'].tolist() != [p+'/'+s for p in PARTITION_IDS for s in SHOT_IDS]:
                raise ValueError('Frozen support scene order differs')
            if len(report['per_scene_support']) != 32:
                raise ValueError('Every rejected and available support row required')
            for i, row in enumerate(report['per_scene_support']):
                for key in ('points', 'stencils'):
                    if row['mask_sha256'][key] != hashlib.sha256(stored[key][i].tobytes()).hexdigest():
                        raise ValueError('Evaluated supports must equal pre-score public masks')
        decision = position_gate(report['positions'], report['used_available_support']['points'],
                                 report['used_available_support']['stencils'])
        flag = decision['position_measurement_qualified']
        if (decision != report.get('decision') or decision != certificate.get('decision')
                or certificate.get('position_measurement_qualified') is not flag
                or meta['position_measurement_qualified'] is not flag
                or meta.get('position_calibration_state') != ('passed' if flag else 'failed')):
            raise ValueError('Only unchanged position/support gates determine position qualification')
        status = json.loads(path.with_name('status.json').read_text())
        if (status.get('version') != VERSION or status.get('complete') is not True
                or status.get('qualification_complete') is not True
                or status.get('calibrated_checkpoint_sha256') != expected_sha256
                or status.get('selected_checkpoint_sha256') != ref['sha256']
                or status.get('selected_weights_sha256') != digest or status.get('position_identity') != identity
                or status.get('execution') != execution or status.get('position_measurement_qualified') is not flag
                or status.get('model_qualified') is not False or status.get('raw_mask_model_qualified') is not False
                or status.get('Native_control_qualified') is not False or status.get('grasp_qualified') is not False):
            raise ValueError('Completed same-artifact position status required')
        if not flag and not allow_unqualified_diagnostic:
            raise ValueError('Failed position certificate cannot provide a centre')
        self.position_measurement_qualified = flag
        self.qualified = self.model_qualified = self.raw_mask_model_qualified = False
        self.model = TinyDepthInfluenceUNet().float().cpu()
        self.model.load_state_dict(payload['model_state_dict'], strict=True); self.model.eval()
        check_execution_environment(self.model, environment)
        self.metadata, self.checkpoint_sha256 = meta, expected_sha256
        self.threshold, self._sources, self.position_calibration_environment = THRESHOLD, sources, environment

    def predict(self, inputs):
        check_execution_environment(self.model, self.position_calibration_environment)
        if qualification_source_map() != self._sources:
            raise ValueError('Position source/profile changed after qualification')
        crops = make_near_goal_crops(inputs)
        probabilities = predict_probabilities(self.model, crops['features'])
        if not self.position_measurement_qualified:
            return dict(position_measurement_qualified=False, position_measurement_available=False,
                refiner_available=False, centre=None, probabilities=probabilities,
                reason='unqualified_position_diagnostic_only_no_centre_or_fallback',
                model_qualified=False, raw_mask_model_qualified=False, Native_control_qualified=False, grasp_qualified=False)
        result = predict_current_position(inputs, crops, probabilities)
        result.update(position_measurement_qualified=True,
            refiner_available=result['position_measurement_available'], checkpoint_sha256=self.checkpoint_sha256,
            probabilities=probabilities, model_qualified=False, raw_mask_model_qualified=False,
            Native_control_qualified=False, grasp_qualified=False)
        return result
