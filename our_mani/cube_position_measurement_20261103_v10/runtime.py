"""Position-certificate runtime; generic/raw-mask qualification always false."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from copy import deepcopy

from ..near_goal_depth_refiner_20261103.data import make_near_goal_crops, sha256
from ..near_goal_depth_refiner_20261103.model import TinyDepthInfluenceUNet, predict_probabilities
from .evaluation import position_gate, _eligibility, validate_prediction_route, validate_report_integrity
from .qualification import (
    VERSION, PROTOCOL, PROTOCOL_SHA256, FIT_PROTOCOL_SHA256, CONSUMPTION_NAME,
    qualification_source_map, check_profile_bindings, check_execution_environment,
    validate_selected_checkpoint, pointer, checked_sha, state_digest, _local_evidence,
)
from .data import PARTITION_IDS, SHOT_IDS
from .prediction import predict_current_position, validate_legacy_audit, PROPOSAL_THRESHOLDS, VERSION as PROPOSAL_VERSION


def _stat(path):
    s=Path(path).stat();return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def validate_scope_flags(value,decision):
    for key in ('position_measurement_qualified','scope12_EF_qualified','axis_general_qualified','composite_global_coverage_qualified'):
        if type(value.get(key))is not bool or value[key]is not decision[key]:raise ValueError('Independent scope-only certificate flags required')
    if value.get('active_certified_routes')!=decision['active_certified_routes']or value.get('axis_general_qualified')is not False:
        raise ValueError('Only independently certified legacy/scope12 routes, never axis')


class CurrentCubePositionMeasurementV10:
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
                or meta.get('force_or_contact_qualified') is not False
                or meta.get('protocol_sha256') != FIT_PROTOCOL_SHA256
                or meta.get('position_profile_sha256') != PROTOCOL_SHA256
                or meta.get('position_profile') != pointer(PROTOCOL)
                or meta.get('same_V5_weights') is not True or meta.get('new_training') is not False
                or tuple(meta.get('position_proposal_thresholds', ())) != PROPOSAL_THRESHOLDS
                or meta.get('position_prediction_version') != PROPOSAL_VERSION
                or meta.get('position_calibration_parent_ids') != list(PARTITION_IDS)
                or meta.get('position_calibration_forward_calls') != 80
                or meta.get('completion') is not False or meta.get('orientation_observed') is not False
                or meta.get('known_axisaligned_50mm_Cube_prior') is not True
                or meta.get('raw_mask_failures_preserved') is not True
                or meta.get('prior_position_failures_preserved') is not True
                or meta.get('Native_control_qualified') is not False or meta.get('grasp_qualified') is not False):
            raise ValueError('Only independent position certificate; raw/selected models cannot provide a centre')
        identity = meta.get('position_identity', {})
        sources = qualification_source_map()
        if identity.get('method_source_sha256') != sources or identity.get('profile') != pointer(PROTOCOL):
            raise ValueError('Exact current position method/profile identity required')
        if sha256(identity['manifest']['path']) != identity['manifest']['sha256']:
            raise ValueError('Actual manifest SHA must equal its pre-forward single-use binding')
        if not identity.get('bound_data_sha256') or any(sha256(p)!=digest for p,digest in identity['bound_data_sha256'].items()):
            raise ValueError('Original closed80 data/source evidence changed')
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
        if (execution.get('forward_calls') != 80 or execution.get('samples') != 80
                or execution.get('parent_count') != 8
                or tuple(execution.get('proposal_thresholds', ())) != PROPOSAL_THRESHOLDS
                or execution.get('proposal_version') != PROPOSAL_VERSION
                or execution.get('threshold_selected_from_fresh') is not False
                or execution.get('environment') != environment):
            raise ValueError('Exactly80 adaptive-public CPU2 forwards required')
        check_execution_environment()
        certificate = evidence['qualification']
        if (certificate.get('version') != VERSION or certificate.get('complete') is not True
                or certificate.get('position_identity') != identity or certificate.get('execution') != execution
                or certificate.get('predictions_frozen_before_GT_scores') is not True
                or certificate.get('source_before_after_verified') is not True
                or certificate.get('same_V5_weights') is not True or certificate.get('new_training') is not False
                or certificate.get('model_qualified') is not False or certificate.get('raw_mask_model_qualified') is not False
                or certificate.get('force_or_contact_qualified') is not False
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
            raise ValueError('All80 fixed position scenes and legacy denominator required')
        public_predictions = json.loads((path.parent/'public_positions.json').read_text())
        if [p['scene_id'] for p in public_predictions] != [p+'/'+s for p in PARTITION_IDS for s in SHOT_IDS]:
            raise ValueError('Every actual public adaptive route must remain in source order')
        for prediction in public_predictions: validate_prediction_route(prediction)
        with np.load(path.parent/'public_support_masks.npz', allow_pickle=False) as stored:
            if stored['scene_ids'].tolist() != [p+'/'+s for p in PARTITION_IDS for s in SHOT_IDS]:
                raise ValueError('Frozen support scene order differs')
            if len(report['per_scene_support']) != 80:
                raise ValueError('Every rejected and available support row required')
            for i, row in enumerate(report['per_scene_support']):
                prediction=public_predictions[i]
                prediction['raw_Cube_mask']=stored['raw_Cube_mask'][i]
                validate_legacy_audit(prediction,prediction.get('legacy_audit',{}))
                scored=report['records'][i]
                audit=prediction['legacy_audit']
                if (scored.get('legacy_baseline_available')is not audit['legacy_position_measurement_available']
                    or scored.get('legacy_pass_exact')is not audit['legacy_pass_returned_exact_object']):
                    raise ValueError('Frozen public legacy comparison must match scored control evidence')
                for key in ('points', 'stencils'):
                    if row['mask_sha256'][key] != hashlib.sha256(stored[key][i].tobytes()).hexdigest():
                        raise ValueError('Evaluated supports must equal pre-score public masks')
        decision = validate_report_integrity(report,eligibility)
        flag = decision['position_measurement_qualified']
        for value in (meta,certificate):validate_scope_flags(value,decision)
        if meta.get('scope_radius_m')!=.12:raise ValueError('Explicit independently qualified12cm scope required')
        if (decision != report.get('decision') or decision != certificate.get('decision')
                or certificate.get('position_measurement_qualified') is not flag
                or meta['position_measurement_qualified'] is not flag
                or meta.get('position_calibration_state') != ('passed' if flag else 'failed')):
            raise ValueError('Only unchanged position/support gates determine position qualification')
        status = json.loads(path.with_name('status.json').read_text())
        validate_scope_flags(status,decision)
        if (status.get('version') != VERSION or status.get('complete') is not True
                or status.get('qualification_complete') is not True
                or status.get('calibrated_checkpoint_sha256') != expected_sha256
                or status.get('selected_checkpoint_sha256') != ref['sha256']
                or status.get('selected_weights_sha256') != digest or status.get('position_identity') != identity
                or status.get('execution') != execution or status.get('position_measurement_qualified') is not flag
                or status.get('model_qualified') is not False or status.get('raw_mask_model_qualified') is not False
                or status.get('force_or_contact_qualified') is not False
                or status.get('Native_control_qualified') is not False or status.get('grasp_qualified') is not False):
            raise ValueError('Completed same-artifact position status required')
        if not flag and not allow_unqualified_diagnostic:
            raise ValueError('Failed position certificate cannot provide a centre')
        self.position_measurement_qualified = flag
        self.qualified = self.model_qualified = self.raw_mask_model_qualified = False
        self.force_or_contact_qualified=False
        self.model = TinyDepthInfluenceUNet().float().cpu()
        self.model.load_state_dict(payload['model_state_dict'], strict=True); self.model.eval()
        self.model.requires_grad_(False)
        check_execution_environment(self.model, environment)
        self.metadata, self.checkpoint_sha256 = deepcopy(meta), expected_sha256
        self.proposal_thresholds, self._sources, self.position_calibration_environment = PROPOSAL_THRESHOLDS, sources, environment
        self.scope12_EF_qualified=flag;self.axis_general_qualified=False
        self.composite_global_coverage_qualified=decision['composite_global_coverage_qualified']
        self.active_certified_routes=tuple(decision['active_certified_routes']);self.scope_radius_m=.12
        self._certificate_flag=flag;self._weights_digest=digest;self._model_object=self.model
        self._forward_function=type(self.model).forward;self._predict_function=type(self).predict
        self._metadata_digest=hashlib.sha256(json.dumps(self.metadata,sort_keys=True,allow_nan=False).encode()).hexdigest()
        guarded=set(sources)|{str(path),str(path.with_name('status.json'))}
        guarded.update(str((path.parent/r['path']).resolve())for r in meta['qualification_evidence'].values())
        guarded.update(item['path']for item in freeze['files'].values())
        guarded.update(item['path']for item in (original['selected'],original['fit_status'],*original['evidence'].values(),identity['request'],identity['dataset_request']))
        self._guarded_stats={name:_stat(name)for name in guarded}

    def _assert_live(self):
        if (type(self)is not CurrentCubePositionMeasurementV10 or self.model is not self._model_object
            or type(self.model)is not TinyDepthInfluenceUNet
            or getattr(self.model.forward,'__func__',None)is not self._forward_function
            or getattr(self.predict,'__func__',None)is not self._predict_function
            or self.position_measurement_qualified is not self._certificate_flag
            or self.scope12_EF_qualified is not self._certificate_flag or self.axis_general_qualified is not False
            or any(getattr(self,k)is not False for k in ('qualified','model_qualified','raw_mask_model_qualified','force_or_contact_qualified'))
            or self.scope_radius_m!=.12 or tuple(self.proposal_thresholds)!=(.995,.9)
            or self.composite_global_coverage_qualified is not self.metadata['composite_global_coverage_qualified']
            or self.active_certified_routes!=(('legacy_v8','scope12_EF')if self._certificate_flag else())):
            raise ValueError('Exact current scope-only runtime/model/flag identity required')
        if (hashlib.sha256(json.dumps(self.metadata,sort_keys=True,allow_nan=False).encode()).hexdigest()!=self._metadata_digest
            or state_digest(self.model.state_dict())!=self._weights_digest
            or any(_stat(p)!=value for p,value in self._guarded_stats.items())):
            raise ValueError('Immutable code/evidence/model/metadata changed after admission')
        if any(m._forward_hooks or m._forward_pre_hooks for m in self.model.modules()):
            raise ValueError('No forward hook may change the qualified model')
        module=torch.nn.modules.module
        if any(getattr(module,key,{})for key in ('_global_forward_hooks','_global_forward_pre_hooks')):
            raise ValueError('No global forward hook may alter qualified inference')

    def predict(self, inputs):
        self._assert_live()
        check_execution_environment(self.model, self.position_calibration_environment)
        crops = make_near_goal_crops(inputs)
        probabilities = predict_probabilities(self.model, crops['features'])
        if not self.position_measurement_qualified:
            return dict(position_measurement_qualified=False, position_measurement_available=False,
                refiner_available=False, centre=None, probabilities=probabilities,
                reason='unqualified_position_diagnostic_only_no_centre_or_fallback',
                scope12_EF_qualified=False,axis_general_qualified=False,active_certified_routes=[],
                model_qualified=False, raw_mask_model_qualified=False, force_or_contact_qualified=False,
                Native_control_qualified=False, grasp_qualified=False)
        result = predict_current_position(inputs, crops, probabilities)
        validate_prediction_route(result)
        result.update(position_measurement_qualified=True,
            scope12_EF_qualified=True,axis_general_qualified=False,
            active_certified_routes=list(self.active_certified_routes),
            composite_global_coverage_qualified=self.composite_global_coverage_qualified,
            refiner_available=result['position_measurement_available'], checkpoint_sha256=self.checkpoint_sha256,
            probabilities=probabilities, model_qualified=False, raw_mask_model_qualified=False,
            force_or_contact_qualified=False,
            Native_control_qualified=False, grasp_qualified=False)
        return result
