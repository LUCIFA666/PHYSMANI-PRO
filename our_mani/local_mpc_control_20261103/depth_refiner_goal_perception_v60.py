"""V60 development perception candidate, not a deployed/qualified controller.

Only a genuinely qualified schedule160 CPU2 runtime can be constructed.  The
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
from ..near_goal_depth_refiner_20261103.data import erode_one_pixel, make_near_goal_crops
from ..near_goal_depth_refiner_20261103.model import TinyDepthInfluenceUNet
from ..near_goal_depth_refiner_20261103_v2.runtime import CurrentCubeDepthRefinerV2
from ..near_goal_depth_refiner_20261103_v2.training import VERSION
from ..near_goal_depth_refiner_20261103_v2.calibration import assert_cpu_prediction_contract
from ..local_mpc_20261103.refiner_far_consistency_v60 import component_far_consistency
from .depth_refiner_public_inputs_v60 import pack_current_public_refiner_inputs
from .goal_aware_perception import GoalAwareRedBoxTracker, _validate_public, filter_observed_sphere_surface


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


def current_mask_depth_cloud(inputs, raw_mask):
    """Exactly the current measured points used by the unchanged centre formula."""
    crops=make_near_goal_crops(inputs)
    mask=erode_one_pixel(np.asarray(raw_mask))
    if mask.shape!=(4,64,64):raise ValueError('One current4view raw mask required')
    yy,xx=np.indices((64,64));cloud=[];views=[]
    for cam in range(4):
        selected=mask[cam]&crops['valid_depth'][cam]
        if not selected.any():continue
        x=xx[selected]+crops['origins_xy'][cam,0];y=yy[selected]+crops['origins_xy'][cam,1]
        rays=np.c_[x,y,np.ones(len(x))]@np.linalg.inv(inputs['K'][cam]).T
        pc=rays*crops['depth_m'][cam,selected,None];T=inputs['T'][cam]
        cloud.append(pc@T[:3,:3].T+T[:3,3]);views.extend([cam]*len(x))
    return (np.concatenate(cloud) if cloud else np.empty((0,3))),len(set(views))


class CurrentDepthRefinerTrackerV60(GoalAwareRedBoxTracker):
    """No runtime injection or diagnostic-model constructor exception exists."""
    def __init__(self, checkpoint, expected_sha256):
        if not isinstance(expected_sha256,str) or re.fullmatch('[0-9a-f]{64}',expected_sha256) is None:
            raise ValueError('Explicit trusted checkpoint SHA256 required')
        self.refiner_checkpoint=Path(checkpoint).resolve(strict=True)
        self.expected_refiner_sha256=expected_sha256
        self.refiner=CurrentCubeDepthRefinerV2(self.refiner_checkpoint,expected_sha256,
                                             allow_unqualified_diagnostic=False)
        if (type(self.refiner) is not CurrentCubeDepthRefinerV2
            or type(getattr(self.refiner,'model',None)) is not TinyDepthInfluenceUNet
            or getattr(self.refiner,'qualified',False) is not True
            or not isinstance(getattr(self.refiner,'metadata',None),dict)):
            raise ValueError('Real fully loaded qualified refiner required, never diagnostic/testdouble')
        self._runtime_object=self.refiner;self._model_object=self.refiner.model
        self._predict_function=CurrentCubeDepthRefinerV2.predict
        self._metadata_digest=_digest(self.refiner.metadata)
        self._tensor_digest=_model_digest(self.refiner.model)
        self._model_sources={str(self.refiner_checkpoint):expected_sha256}
        self._model_sources.update(self.refiner.metadata['identity']['code_sha256'])
        status=self.refiner_checkpoint.with_name('status.json')
        self._model_sources[str(status)]=_sha(status)
        for p in self.refiner.metadata['evidence'].values():
            path=(self.refiner_checkpoint.parent/p['path']).resolve(strict=True)
            self._model_sources[str(path)]=p['sha256']
        self.validate_refiner()
        super().__init__(tracker=RedBoxTracker(prior=BoxPrior(minimum_red=100),
            complete_geometry=False,history=5,points=256))
        self._accepted_source=None;self._accepted_source_before_reset=None
        self.last_measurement_metadata=None;self.last_measured_cloud=None
        self.last_current_target=None;self._last_observed_timestamp=None

    def validate_refiner(self):
        if any(not hasattr(self,name) for name in ('refiner','_runtime_object','_model_object',
                '_predict_function','_metadata_digest','_tensor_digest','_model_sources')):
            raise ValueError('Refiner has not completed genuine qualified construction')
        r=self.refiner
        if (type(r) is not CurrentCubeDepthRefinerV2 or r is not self._runtime_object
            or type(r.model) is not TinyDepthInfluenceUNet or r.model is not self._model_object
            or getattr(r.predict,'__func__',None) is not self._predict_function
            or self._predict_function is not CurrentCubeDepthRefinerV2.predict
            or r.qualified is not True or r.metadata.get('model_qualified') is not True
            or r.metadata.get('scope')!='full32' or r.metadata.get('completion') is not False
            or r.metadata.get('calibration_device')!='cpu' or r.metadata.get('calibration_num_threads')!=2
            or VERSION!='near_goal_binary_current_depth_influence_schedule160_cpu2_20261103_v3'
            or r.checkpoint_sha256!=self.expected_refiner_sha256
            or r.threshold!=r.metadata.get('selected_threshold')
            or _digest(r.metadata)!=self._metadata_digest):
            raise ValueError('Exact immutable genuinely qualified full32 CPU2v3 refiner required; no diagnostic/testdouble')
        if nn_module._global_forward_hooks or nn_module._global_forward_pre_hooks:
            raise ValueError('No injected global NN forward hooks permitted')
        if any(m._forward_hooks or m._forward_pre_hooks or m.training for m in r.model.modules()):
            raise ValueError('Exact eval model without injected hooks required')
        assert_cpu_prediction_contract(r.model,r.calibration_environment)
        if _model_digest(r.model)!=self._tensor_digest:raise ValueError('Refiner weight tensors mutated')
        for name,digest in self._model_sources.items():
            if _sha(name)!=digest:raise ValueError('Qualified model/source/evidence changed: '+name)
        return True

    def model_sources(self):
        self.validate_refiner();return dict(self._model_sources)

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
        raw=dict(backend='v60_current_depth_refiner_or_far_R100_development',measured_points=len(cloud),
            visible_views=views,geometry_completion=True,simulator_pose_used=False,
            future_observations_used=False,known_dimensions_m=list(self.prior.dimensions))
        return target,raw

    def _unknown(self,reason):
        self.last_measurement_metadata.update(measurement_available=False,accepted_source='UNKNOWN',reason=reason,
            no_near_shell_fallback=True,accepted_history_reset_for_route_change=False)
        self.last_measured_cloud=None;self.last_current_target=None;self.reset()
        raise ValueError(reason)

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
            future_observations_used=False,observations_fabricated=False,
            far_consistency_is_development_hypothesis=True,far_consistency_is_physical_or_ownership_certificate=False,
            known_prior_counterexamples=73,existing_replay_near_false_far_zero_is_not_guarantee=True,
            accepted_history_reset_for_route_change=False)
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
        self.last_measurement_metadata['NN_current_measurement']=deepcopy(measurement)
        if prediction.get('refiner_available') is True:
            if measurement.get('refiner_available') is not True or prediction.get('completion') is not False:
                raise ValueError('No completed/modelled centre may become a current NN measurement')
            cloud,views=current_mask_depth_cloud(inputs,prediction['raw_Cube_mask'])
            confidence=min(1.,views/3)*min(1.,len(cloud)/60)
            if (len(cloud)!=measurement.get('measured_points') or views!=measurement.get('visible_views')
                or confidence!=measurement.get('geometry_confidence')
                or not np.array_equal(np.asarray(prediction['centre']),np.asarray(measurement['centre']))):
                raise ValueError('NN current-depth cloud/confidence/centre identity mismatch')
            try:return self._accept(measurement['centre'],confidence,cloud,views,now,'NN_CURRENT_DEPTH')
            except ValueError as exc:return self._unknown(str(exc))
        # Far-only candidate is a scratch observation using the actual final
        # preceding measured history. No provisional state escapes this scope.
        previous=list(self.history);velocity=self.last_velocity.copy()
        raw=None;metadata=None;raw_error=None
        try:
            filtered,trace=filter_observed_sphere_surface(observation,self.sphere,self.prior)
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
