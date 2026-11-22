"""V70 position-only capability; generic/raw-model qualification stays false.

Only a genuinely passing independent V10 scope12 CPU2 position certificate is accepted.
The sole far geometry delta is private signed-K u+.5/v+.5 backprojection;
first Sphere fitting and all NN/public calibrations remain the originals. The
far branch uses the declared 6 mm component-consistency hypothesis, which has
known counterexamples; it is neither ownership nor physical certification.
"""
from collections import deque
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.nn.modules import module as nn_module

from ..benchmark_perception import BoxPrior, RedBoxTracker, box_surface
from ..contracts import ObjectState, Pose
from ..perception import CameraCalibration
from .pixel_centre_far_tracker_prototype_20261103 import PixelCentreFarCalibration
from ..near_goal_depth_refiner_20261103.model import TinyDepthInfluenceUNet
from ..cube_position_measurement_20261103_v10.runtime import CurrentCubePositionMeasurementV10
from ..cube_position_measurement_20261103_v10.qualification import (
    VERSION, PROTOCOL_SHA256, check_execution_environment)
from ..cube_position_measurement_20261103_v10.prediction import (
    VERSION as PREDICTION_VERSION, PROPOSAL_THRESHOLDS,
    prediction_route, validate_prediction_route as validate_certified_prediction_route)
from ..near_goal_cube_geometry_20261103_v1 import VERSION as LEGACY_GEOMETRY_VERSION
from ..near_goal_cube_geometry_scope12_20261103_v1 import VERSION as SCOPE_GEOMETRY_VERSION
from ..local_mpc_20261103.refiner_far_consistency_v60 import component_far_consistency
from .depth_refiner_public_inputs_v60 import pack_current_public_refiner_inputs
from .goal_aware_perception import GoalAwareRedBoxTracker, _validate_public, filter_observed_sphere_surface


_PIXEL_CENTRE_BACKPROJECT = PixelCentreFarCalibration.backproject

def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else None).encode()).hexdigest()


def _model_digest(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        a=tensor.detach().cpu().numpy()
        if not np.isfinite(a).all():raise ValueError('Finite frozen refiner tensors required')
        h.update((name+str(a.dtype)+str(a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()


def _packed_digest(inputs):
    h=hashlib.sha256()
    for name in sorted(inputs):
        a=np.asarray(inputs[name]);h.update((name+str(a.dtype)+str(a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()


def current_position_depth_cloud(measurement):
    """Use the SAME F accepted points, never a rebuilt integer-ray raw cloud."""
    if measurement.get('geometry_profile') not in (LEGACY_GEOMETRY_VERSION,SCOPE_GEOMETRY_VERSION):
        raise ValueError('Exact qualified geometryF measurement required')
    ledger=measurement.get('camera_point_ledger')
    if not isinstance(ledger,list) or [row.get('camera') for row in ledger]!=list(range(4)):
        raise ValueError('Complete current four-view D point ledger required')
    clouds=[];views=0
    for row in ledger:
        n=row['baseline_candidate_points'];keep=np.asarray(row['accepted'])
        if n==0 and keep.size==0:keep=np.empty(0,bool)
        xyz=np.asarray(row['point_world'],float).reshape(-1,3)
        if (type(n) is not int or n<0 or keep.dtype!=np.bool_ or keep.shape!=(n,)
            or xyz.shape!=(n,3) or int(keep.sum())!=row['accepted_points']
            or not np.isfinite(xyz[keep]).all()):
            raise ValueError('Exact actual accepted F support required')
        if keep.any():clouds.append(xyz[keep]);views+=1
    cloud=np.concatenate(clouds) if clouds else np.empty((0,3))
    actual=np.asarray(measurement.get('current_points_world'),float).reshape(-1,3)
    if (not np.array_equal(actual,cloud) or len(cloud)!=measurement.get('measured_points')
        or views!=measurement.get('visible_views')):
        raise ValueError('Current position/support/cloud identity differs')
    return cloud,views


def _file_stat(path):
    s=Path(path).stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def _evidence_pointers(value):
    """Only explicit evidence pointers; never bound raw-capture data."""
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:yield value
        else:
            for item in value.values():yield from _evidence_pointers(item)


def _position_flags(runtime):
    return (getattr(runtime,'position_measurement_qualified',None) is True
        and getattr(runtime,'scope12_EF_qualified',None) is True
        and getattr(runtime,'axis_general_qualified',None) is False
        and getattr(runtime,'active_certified_routes',None)==('legacy_v8','scope12_EF')
        and getattr(runtime,'scope_radius_m',None)==.12
        and type(getattr(runtime,'composite_global_coverage_qualified',None)) is bool
        and all(getattr(runtime,k,None) is False for k in ('qualified','model_qualified','raw_mask_model_qualified')))


PUBLIC_POSITION_EVIDENCE_KEYS=frozenset((
    'timestamp','producer_ready','measurement_available','model_checkpoint_sha256',
    'Native_inputs_used','position_measurement_qualified','model_qualified','raw_mask_model_qualified',
    'scope12_EF_qualified','axis_general_qualified','active_certified_routes','scope_radius_m','composite_global_coverage_qualified',
    'future_observations_used','observations_fabricated','packed_current_input_sha256',
    'accepted_source','accepted_history_reset_for_route_change','measured_cloud_points','measured_cloud_sha256',
    'template_points','template_dimensions_m','template_is_measured_cloud','orientation_observed',
    'velocity_observed','velocity_source','velocity_history_real_timestamps','reason','no_near_shell_fallback',
    'far_consistency_is_development_hypothesis','far_consistency_is_physical_or_ownership_certificate',
    'known_prior_counterexamples','existing_replay_near_false_far_zero_is_not_guarantee',
    'far_backprojection_pixel_centre_offset','far_calibration_is_private','raw_RGBD_calibration_rewritten'))
PRIVATE_POSITION_DIAGNOSTIC_KEYS=frozenset((
    'NN_current_measurement','far_legacy_surface_filter','far_consistency','far_raw_error'))
PUBLIC_PROPOSAL_KEYS=frozenset(('proposal_thresholds','threshold','selected_threshold','thresholds_tried',
    'proposal_route','adaptive_public_proposal_version','threshold_change_is_same_position_source',
    'measurement_route','scope_radius_m','legacy_thresholds_tried','legacy_proposal_route'))
PUBLIC_OPTIONAL_PROPOSAL_KEYS=frozenset(('legacy_proposal_trace',))
PUBLIC_PROPOSAL_ITEM_KEYS=frozenset(('threshold','geometry_available','reasons','measured_points',
    'visible_views','geometry_confidence'))


def project_public_position_evidence(metadata):
    """Explicit positive whitelist; complete F/source diagnostics stay private.

    This is not a forbidden-key scrubber or renamed provenance. Unregistered
    top-level or proposal keys fail rather than silently entering policy.
    """
    if metadata is None:return None
    if not isinstance(metadata,dict):raise ValueError('Actual private current tracker metadata required')
    known=PUBLIC_POSITION_EVIDENCE_KEYS|PRIVATE_POSITION_DIAGNOSTIC_KEYS|{'position_proposals'}
    if set(metadata)-known:raise ValueError('Unregistered tracker metadata cannot enter public evidence')
    public={k:deepcopy(metadata[k]) for k in PUBLIC_POSITION_EVIDENCE_KEYS if k in metadata}
    if 'position_proposals' in metadata:
        proposal=metadata['position_proposals']
        if (not isinstance(proposal,dict)
            or set(proposal)!=PUBLIC_PROPOSAL_KEYS|PUBLIC_OPTIONAL_PROPOSAL_KEYS|{'proposal_trace','route_uses_GT_or_future'}
            or proposal['route_uses_GT_or_future'] is not False):
            raise ValueError('Exact current public proposal metadata required')
        trace=proposal['proposal_trace'];legacy_trace=proposal['legacy_proposal_trace']
        if (not isinstance(trace,list) or not isinstance(legacy_trace,list)
            or any(not isinstance(row,dict) or set(row)!=PUBLIC_PROPOSAL_ITEM_KEYS for row in trace+legacy_trace)):
            raise ValueError('Only current public proposal counts/routes may enter policy')
        public['position_proposals']={k:deepcopy(proposal[k]) for k in PUBLIC_PROPOSAL_KEYS}
        public['position_proposals']['proposal_trace']=[{k:deepcopy(row[k]) for k in PUBLIC_PROPOSAL_ITEM_KEYS} for row in trace]
        public['position_proposals']['legacy_proposal_trace']=[{k:deepcopy(row[k]) for k in PUBLIC_PROPOSAL_ITEM_KEYS} for row in legacy_trace]
    return public


def validate_position_proposal_trace(prediction):
    """Validate both certified current routes, with one unchanged NN source ID."""
    if not isinstance(prediction,dict):raise ValueError('Actual dictionary public prediction required')
    validate_certified_prediction_route(prediction)
    route=prediction_route(prediction)
    if route not in ('legacy_v8','scope12_EF'):raise ValueError('No uncertified route')
    profile=LEGACY_GEOMETRY_VERSION if route=='legacy_v8' else SCOPE_GEOMETRY_VERSION
    measurement=prediction.get('measurement',{})
    available=prediction.get('position_measurement_available')
    trace=prediction.get('proposal_trace');tried=prediction.get('thresholds_tried')
    legacy_trace=[] if route=='legacy_v8' else prediction.get('legacy_proposal_trace')
    if (type(available)is not bool or not isinstance(trace,list)or not trace
        or not isinstance(legacy_trace,list)or not isinstance(measurement,dict)
        or measurement.get('geometry_profile')!=profile
        or prediction.get('geometry_version')!=profile
        or measurement.get('refiner_available')is not available
        or tuple(prediction.get('proposal_thresholds',()))!=PROPOSAL_THRESHOLDS
        or any(type(t)is not float for t in prediction['proposal_thresholds'])
        or any(type(t)is not float for t in tried)
        or any(not isinstance(row,dict)or set(row)!=PUBLIC_PROPOSAL_ITEM_KEYS for row in trace+legacy_trace)
        or any(type(row['threshold'])is not float or type(row['geometry_available'])is not bool for row in trace+legacy_trace)
        or any(trace[-1].get(key)!=measurement.get(key) for key in ('reasons','measured_points','visible_views','geometry_confidence'))
        or prediction.get('route_uses_GT_or_future')is not False
        or prediction.get('raw_mask_qualified')is not False or prediction.get('completion')is not False
        or (not available and prediction.get('centre')is not None)):
        raise ValueError('Exact certified selected/current rejected geometry and public trace required')
    if available:
        centre=np.asarray(prediction.get('centre'),float)
        actual=np.asarray(measurement.get('centre'),float)
        if centre.shape!=(3,)or not np.isfinite(centre).all()or not np.array_equal(centre,actual):
            raise ValueError('Actual same current measurement centre required')
    return dict(proposal_thresholds=list(PROPOSAL_THRESHOLDS),threshold=prediction['threshold'],
        selected_threshold=prediction['selected_threshold'],thresholds_tried=list(tried),
        proposal_route=prediction['proposal_route'],proposal_trace=deepcopy(trace),
        adaptive_public_proposal_version=prediction['adaptive_public_proposal_version'],route_uses_GT_or_future=False,
        threshold_change_is_same_position_source=True,measurement_route=route,
        scope_radius_m=.08 if route=='legacy_v8' else .12,
        legacy_thresholds_tried=[] if route=='legacy_v8' else list(prediction['legacy_thresholds_tried']),
        legacy_proposal_route=None if route=='legacy_v8' else prediction['legacy_proposal_route'],
        legacy_proposal_trace=deepcopy(legacy_trace))


class CurrentDepthPixelCentreTrackerV70(GoalAwareRedBoxTracker):
    """No runtime injection or diagnostic-model constructor exception exists."""
    def __init__(self, checkpoint, expected_sha256):
        if not isinstance(expected_sha256,str) or re.fullmatch('[0-9a-f]{64}',expected_sha256) is None:
            raise ValueError('Explicit trusted checkpoint SHA256 required')
        self.refiner_checkpoint=Path(checkpoint).resolve(strict=True)
        self.expected_refiner_sha256=expected_sha256
        self.refiner=CurrentCubePositionMeasurementV10(self.refiner_checkpoint,expected_sha256,
                                             allow_unqualified_diagnostic=False)
        if (type(self.refiner) is not CurrentCubePositionMeasurementV10
            or type(getattr(self.refiner,'model',None)) is not TinyDepthInfluenceUNet
            or not _position_flags(self.refiner)
            or not isinstance(getattr(self.refiner,'metadata',None),dict)):
            raise ValueError('Real passing position certificate required, never diagnostic/testdouble')
        self._runtime_object=self.refiner;self._model_object=self.refiner.model
        self._predict_function=CurrentCubePositionMeasurementV10.predict
        self._metadata_digest=_digest(self.refiner.metadata)
        self._tensor_digest=_model_digest(self.refiner.model)
        self._model_sources={str(self.refiner_checkpoint):expected_sha256}
        meta=self.refiner.metadata;identity=meta['position_identity']
        self._model_sources.update(identity['method_source_sha256'])
        status=self.refiner_checkpoint.with_name('status.json')
        self._model_sources[str(status)]=_sha(status)
        # Qualified sibling evidence and original selected-fit evidence have
        # DIFFERENT roots. Never reinterpret selected's six absolute pointers.
        for p in meta['qualification_evidence'].values():
            path=(self.refiner_checkpoint.parent/p['path']).resolve(strict=True)
            self._model_sources[str(path)]=p['sha256']
        for p in _evidence_pointers(identity['selected_evidence']):
            self._model_sources[str(Path(p['path']).resolve(strict=True))]=p['sha256']
        for key in ('profile','request','dataset_request','manifest'):
            p=identity[key];self._model_sources[str(Path(p['path']).resolve(strict=True))]=p['sha256']
        freeze_path=(self.refiner_checkpoint.parent/meta['qualification_evidence']['prediction_freeze']['path']).resolve(strict=True)
        for p in json.loads(freeze_path.read_text())['files'].values():
            self._model_sources[str(Path(p['path']).resolve(strict=True))]=p['sha256']
        for name,digest in self._model_sources.items():
            if _sha(name)!=digest:raise ValueError('Passing position source/evidence changed: '+name)
        self._source_stats={name:_file_stat(name) for name in self._model_sources}
        self.validate_refiner()
        super().__init__(tracker=RedBoxTracker(prior=BoxPrior(minimum_red=100),
            complete_geometry=False,history=5,points=256))
        self._accepted_source=None;self._accepted_source_before_reset=None
        self.last_measurement_metadata=None;self.last_measured_cloud=None
        self.last_current_target=None;self._last_observed_timestamp=None

    def validate_refiner(self):
        if any(not hasattr(self,name) for name in ('refiner','_runtime_object','_model_object',
                '_predict_function','_metadata_digest','_tensor_digest','_model_sources','_source_stats')):
            raise ValueError('Refiner has not completed genuine qualified construction')
        r=self.refiner
        if (type(r) is not CurrentCubePositionMeasurementV10 or r is not self._runtime_object
            or type(r.model) is not TinyDepthInfluenceUNet or r.model is not self._model_object
            or getattr(r.predict,'__func__',None) is not self._predict_function
            or self._predict_function is not CurrentCubePositionMeasurementV10.predict
            or not _position_flags(r)
            or r.metadata.get('position_measurement_qualified') is not True
            or r.metadata.get('scope12_EF_qualified') is not True or r.metadata.get('axis_general_qualified') is not False
            or r.metadata.get('active_certified_routes')!=['legacy_v8','scope12_EF']
            or r.metadata.get('scope_radius_m')!=.12
            or type(r.metadata.get('composite_global_coverage_qualified'))is not bool
            or r.composite_global_coverage_qualified is not r.metadata['composite_global_coverage_qualified']
            or any(r.metadata.get(k) is not False for k in ('qualified','model_qualified','raw_mask_model_qualified'))
            or r.metadata.get('scope')!='full32' or r.metadata.get('completion') is not False
            or r.metadata.get('position_profile_sha256')!=PROTOCOL_SHA256
            or PROTOCOL_SHA256!='09cfe386d68b0d262c7af0d5d42404e7b350eec5e5411d501a38c88037d06500'
            or VERSION!='current_Cube_position_only_V5weights_scope12_EF_cpu2_20261103_v10'
            or r.metadata.get('selected_checkpoint_sha256')!='dbf998242ad13fea26ede9c4b17a0e2bfd0daeb581331d21342af9a6b6d06bfa'
            or r.checkpoint_sha256!=self.expected_refiner_sha256
            or tuple(r.proposal_thresholds)!=PROPOSAL_THRESHOLDS or PROPOSAL_THRESHOLDS!=(.995,.9)
            or r.metadata.get('position_proposal_thresholds')!=list(PROPOSAL_THRESHOLDS)
            or r.metadata.get('position_prediction_version')!=PREDICTION_VERSION
            or _digest(r.metadata)!=self._metadata_digest):
            raise ValueError('Exact immutable passing V10 scope12-only CPU2 position refiner required')
        if nn_module._global_forward_hooks or nn_module._global_forward_pre_hooks:
            raise ValueError('No injected global NN forward hooks permitted')
        if any(m._forward_hooks or m._forward_pre_hooks or m.training for m in r.model.modules()):
            raise ValueError('Exact eval model without injected hooks required')
        check_execution_environment(r.model,r.position_calibration_environment)
        if _model_digest(r.model)!=self._tensor_digest:raise ValueError('Refiner weight tensors mutated')
        # Real runtime and this binding verified all SHA at construction. Each
        # frame checks immutable file stats and exact tensors/metadata, avoiding
        # rereading reports, prediction NPZ or raw-capture data.
        for name,expected in self._source_stats.items():
            if _file_stat(name)!=expected:raise ValueError('Position file identity changed: '+name)
        return True

    def model_sources(self):
        self.validate_refiner();return dict(self._model_sources)

    def public_position_evidence(self):
        return project_public_position_evidence(self.last_measurement_metadata)

    def reset(self):
        super().reset();self._accepted_source=None

    def reset_episode(self):
        super().reset_episode();self._accepted_source=None
        self.last_measurement_metadata=None;self.last_measured_cloud=None
        self.last_current_target=None;self._last_observed_timestamp=None

    def _association(self,center,now):
        if not self.history:return True
        dt=now-self.history[-1][0]
        if dt<=0:return False
        predicted=self.history[-1][1]+self.last_velocity*dt
        return bool(np.linalg.norm(center-predicted)<=.045+1.0*dt)

    def _accept(self,center,confidence,cloud,views,now,source):
        """Commit only one final measured centre, never a provisional raw centre."""
        center=np.asarray(center,float);cloud=np.asarray(cloud,float)
        if (center.shape!=(3,) or not np.isfinite(center).all() or cloud.ndim!=2 or cloud.shape[1]!=3
            or len(cloud)<12 or not np.isfinite(cloud).all() or not np.isfinite(confidence) or not .5<=confidence<=1):
            raise ValueError('Finite confidence-qualified current measurement required')
        if not self._association(center,now):raise ValueError('current_measurement_fails_causal_public_association')
        changed=self._accepted_source is not None and self._accepted_source!=source
        if changed:super().reset()
        samples=list(self.history)+[(now,center)];velocity=np.zeros(3)
        if len(samples)>1:
            times=np.array([p[0] for p in samples]);times-=times.mean()
            positions=np.array([p[1] for p in samples])
            velocity=(times[:,None]*(positions-positions.mean(0))).sum(0)/(times@times)
        self.history.append((now,center.copy()));self.last_velocity=velocity.copy()
        self._accepted_source=source
        pose=Pose(center);target=ObjectState('target',now,pose,
            box_surface(pose,self.prior.dimensions,256),velocity,np.zeros(3),confidence)
        self.last_measured_cloud=cloud.copy();self.last_current_target=target
        self.last_measurement_metadata.update(accepted_source=source,measurement_available=True,
            accepted_history_reset_for_route_change=changed,measured_cloud_points=len(cloud),
            measured_cloud_sha256=hashlib.sha256(cloud.tobytes()).hexdigest(),
            template_points=256,template_dimensions_m=list(self.prior.dimensions),
            template_is_measured_cloud=False,orientation_observed=False,
            velocity_observed=len(samples)>1,
            velocity_source='causal_real_final_centres_LS' if len(samples)>1 else 'unobserved_zero_initialization',
            velocity_history_real_timestamps=[float(sample[0]) for sample in samples])
        raw=dict(backend='v70_current_depth_refiner_or_far_R100_development',measured_points=len(cloud),
            visible_views=views,geometry_completion=True,simulator_pose_used=False,
            future_observations_used=False,known_dimensions_m=list(self.prior.dimensions))
        return target,raw

    def _unknown(self,reason):
        self.last_measurement_metadata.update(measurement_available=False,accepted_source='UNKNOWN',reason=reason,
            no_near_shell_fallback=True,accepted_history_reset_for_route_change=False)
        self.last_measured_cloud=None;self.last_current_target=None;self.reset()
        raise ValueError(reason)

    def _private_far_observation(self,observation):
        """Private half-pixel calibration for shell suppression and R100 only.

        This never replaces the original calibration objects used by the
        first Sphere fit, NN input packer, binding or P4 raw-frame cache.
        """
        _validate_public(observation)
        if (PixelCentreFarCalibration.__bases__!=(CameraCalibration,)
            or PixelCentreFarCalibration.backproject is not _PIXEL_CENTRE_BACKPROJECT):
            raise ValueError('Exact source-approved private pixel-centre backprojection required')
        private=dict(observation);calibrations={}
        for name,c in observation['observation/calibrations'].items():
            if type(c) is not CameraCalibration:
                raise ValueError('Only original public CameraCalibration may enter V70 far wrapping')
            calibrations[name]=PixelCentreFarCalibration(c.intrinsics,c.camera_to_world,
                                                        c.depth_scale,c.timestamp_offset)
        private['observation/calibrations']=calibrations
        return private

    def observe(self,observation):
        _validate_public(observation);now=float(observation['timestamp'])
        self.validate_refiner()
        if self.last_timestamp is not None and now<=self.last_timestamp:
            raise ValueError('one unique actual causal public RGBD frame required')
        gap=None if self.last_timestamp is None else now-self.last_timestamp
        self.last_timestamp=now;self._last_observed_timestamp=now
        self.last_current_target=None;self.last_measured_cloud=None
        self.last_measurement_metadata=dict(timestamp=now,producer_ready=False,measurement_available=False,
            model_checkpoint_sha256=self.expected_refiner_sha256,Native_inputs_used=False,
            position_measurement_qualified=True,model_qualified=False,raw_mask_model_qualified=False,
            scope12_EF_qualified=True,axis_general_qualified=False,active_certified_routes=['legacy_v8','scope12_EF'],
            scope_radius_m=.12,composite_global_coverage_qualified=self.refiner.composite_global_coverage_qualified,
            future_observations_used=False,observations_fabricated=False,
            far_consistency_is_development_hypothesis=True,far_consistency_is_physical_or_ownership_certificate=False,
            known_prior_counterexamples=73,existing_replay_near_false_far_zero_is_not_guarantee=True,
            accepted_history_reset_for_route_change=False,far_backprojection_pixel_centre_offset=.5,
            far_calibration_is_private=True,raw_RGBD_calibration_rewritten=False)
        if gap is not None and (gap>.075 or abs(gap-.05)>1e-6):
            return self._unknown('non50ms_or_stale_actual_frame_no_interpolation')
        if not self._initial_fit_attempted:self.initialize_from_public(observation)
        if self.sphere is None:return self._unknown('strict_first_public_sphere_unavailable_no_refit_or_fallback')
        self.last_measurement_metadata['producer_ready']=True
        inputs=pack_current_public_refiner_inputs(observation,self.sphere)
        self.last_measurement_metadata['packed_current_input_sha256']=_packed_digest(inputs)
        # Qualified NN is always evaluated first. Raw never appends a competing
        # same-timestamp centre to the final history used for NN association.
        prediction=self.refiner.predict(inputs)
        if prediction.get('checkpoint_sha256')!=self.expected_refiner_sha256:
            raise ValueError('Current result checkpoint identity mismatch')
        measurement=prediction.get('measurement',{})
        self.last_measurement_metadata['position_proposals']=validate_position_proposal_trace(prediction)
        self.last_measurement_metadata['NN_current_measurement']=deepcopy(measurement)
        if (prediction.get('position_measurement_qualified') is not True
            or prediction.get('model_qualified') is not False or prediction.get('raw_mask_model_qualified') is not False
            or prediction.get('scope12_EF_qualified')is not True or prediction.get('axis_general_qualified')is not False
            or prediction.get('active_certified_routes')!=['legacy_v8','scope12_EF']
            or prediction.get('composite_global_coverage_qualified')is not self.refiner.composite_global_coverage_qualified):
            raise ValueError('Prediction must retain separate position/raw qualification flags')
        if prediction.get('position_measurement_available') is True:
            if measurement.get('refiner_available') is not True or prediction.get('completion') is not False:
                raise ValueError('No completed/modelled centre may become a current NN measurement')
            cloud,views=current_position_depth_cloud(measurement)
            confidence=min(1.,views/3)*min(1.,len(cloud)/60)
            if (len(cloud)!=measurement.get('measured_points') or views!=measurement.get('visible_views')
                or confidence!=measurement.get('geometry_confidence')
                or not np.array_equal(np.asarray(prediction['centre']),np.asarray(measurement['centre']))):
                raise ValueError('NN current-depth cloud/confidence/centre identity mismatch')
            try:return self._accept(measurement['centre'],confidence,cloud,views,now,'NN_CURRENT_POSITION_F')
            except ValueError as exc:return self._unknown(str(exc))
        # Far-only candidate is a scratch observation using the actual final
        # preceding measured history. No provisional state escapes this scope.
        previous=list(self.history);velocity=self.last_velocity.copy()
        raw=None;metadata=None;raw_error=None
        try:
            private_far=self._private_far_observation(observation)
            filtered,trace=filter_observed_sphere_surface(private_far,self.sphere,self.prior)
            self.last_measurement_metadata['far_legacy_surface_filter']=trace
            raw,metadata=self.tracker.observe(filtered)
        except (ValueError,RuntimeError) as exc:raw_error=type(exc).__name__+': '+str(exc)
        finally:
            self.history.clear();self.history.extend(previous);self.last_velocity=velocity
        self.last_measurement_metadata['far_raw_error']=raw_error
        if raw is None:return self._unknown('NN_unavailable_and_no_current_far_component')
        if metadata.get('geometry_completion') is not False:
            raise ValueError('Far-consistency requires actual measured component, never template points')
        proof=component_far_consistency(raw.points_world,self.sphere.center)
        self.last_measurement_metadata['far_consistency']=proof
        if not proof['far_consistent'] or raw.confidence<.5:
            return self._unknown('NN_unavailable_without_far_consistency_no_near_shell_fallback')
        try:return self._accept(raw.pose.position,raw.confidence,raw.points_world,
                               metadata['visible_views'],now,'FAR_R100_DEVELOPMENT_CONSISTENCY')
        except ValueError as exc:return self._unknown(str(exc))
