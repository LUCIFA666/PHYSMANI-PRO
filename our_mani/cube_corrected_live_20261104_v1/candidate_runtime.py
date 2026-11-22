"""New causal cache and explicit oldpilot/newforecast development provider.

Old pilot weights are not represented as trained on the new motion context.
Only the predicted motion authority/feature family changes; fixed deadline,
positive-only unranked proposals and symmetry ledger methods are original.
"""
from collections import deque
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
import torch
from ..contracts import ObjectState,Forecast,Pose
from ..local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from ..local_mpc_p4_20261103.runtime import (PublicRowCache,FrozenPilotInference,LearnedCandidates,ProposalUnavailable,
    _walk_public,_time,_rgbd_reference,_PROVENANCE)
from ..local_mpc_p4_20261103.data import PUBLIC_ROW_KEYS,ROBOT_KEYS,unit_quaternion,CHECKPOINT_SHA256 as LEGACY_FORECAST_SHA
from ..local_mpc_p4_20261103.candidates import CandidateLedger
from ..local_mpc_p4_20261103.symmetry_runtime import SymmetricLearnedCandidates
from ..cube_candidate_matched_20261104_v2.data import public_features

CORRECTED_CACHE_SCHEMA='Cube_corrected_forecast_public_row_projection_real50ms_v1'

def _project_row(row,*,expected_forecast_sha256):
    if (type(expected_forecast_sha256)is not str or len(expected_forecast_sha256)!=64
            or expected_forecast_sha256==LEGACY_FORECAST_SHA):
        raise ValueError('Distinct explicit corrected forecast authority required')
    if not isinstance(row,dict) or set(row)!=PUBLIC_ROW_KEYS:raise ValueError('exact public observer row keys required')
    if type(row['index']) is not int or row['index']<0 or type(row['close_requested']) is not bool:
        raise ValueError('actual frame index and known close intent required')
    robot=row['public_robot'];r=row['result']
    if not isinstance(robot,dict) or set(robot)!=ROBOT_KEYS or not isinstance(r,dict):raise ValueError('exact public robot/result required')
    now=_time(robot['timestamp'])
    for key,size in (('q',7),('dq',7),('tcp_position',3),('finger_open_amounts',2)):
        value=np.asarray(robot[key],float)
        if value.shape!=(size,) or not np.isfinite(value).all():raise ValueError('finite public robot '+key+' required')
    unit_quaternion(robot['tcp_quaternion'],'public TCP quaternion')
    if np.any((np.asarray(robot['finger_open_amounts'])<0)|(np.asarray(robot['finger_open_amounts'])>1)):
        raise ValueError('public normalized fingers must be in[0,1]')
    if abs(_time(r.get('timestamp'))-now)>1e-8:raise ValueError('same-frame public result and robot required')
    for key in _PROVENANCE:
        if r.get(key) is not False:raise ValueError('disallowed public provenance: '+key)
    if r.get('frozen_checkpoint_sha256')!=expected_forecast_sha256:raise ValueError('unchanged frozen motion checkpoint required')
    if r.get('mode') not in ('FREE','UNRESOLVED','ATTACHED_PUBLIC','STALE'):raise ValueError('declared public motion mode required')
    for key in ('available','free_forecast_available','controlled_mode_available'):
        if type(r.get(key)) is not bool:raise ValueError('explicit public availability booleans required')
    obs=r.get('object_observation');obj=None
    if obs is not None:
        if not isinstance(obs,dict):raise ValueError('serialized public ObjectState or None required')
        if obs.get('object_id')!='target' or abs(_time(obs.get('timestamp'))-now)>1e-8:raise ValueError('current same-frame public target required')
        p=np.asarray(obs['pose']['position'],float);rot=np.asarray(obs['pose']['rotation'],float);points=np.asarray(obs['points_world'],float)
        if (p.shape!=(3,) or rot.shape!=(3,3) or points.ndim!=2 or points.shape[1]!=3 or len(points)<3
                or not all(np.isfinite(x).all() for x in (p,rot,points))):raise ValueError('finite current public geometry required')
        if not np.allclose(rot.T@rot,np.eye(3),atol=1e-5,rtol=0) or not np.isclose(np.linalg.det(rot),1.,atol=1e-5,rtol=0):
            raise ValueError('proper current public object rotation required')
        confidence=obs.get('confidence',1.);visible=obs.get('visible',True)
        if type(visible) is not bool or not 0<=confidence<=1:raise ValueError('public visibility/confidence required')
        obj=dict(object_id='target',timestamp=now,pose=dict(position=p.tolist(),rotation=rot.tolist()),
            points_world=points.tolist(),confidence=float(confidence),visible=visible)
    forecast=None;f=r.get('forecast')
    if f is not None:
        if not isinstance(f,dict):raise ValueError('serialized predicted Forecast required')
        m=f['metadata']
        if m.get('checkpoint_sha256')!=expected_forecast_sha256:raise ValueError('Forecast/result authority mismatch')
        seen=_time(m.get('observed_until'))
        if seen>now+1e-8:raise ValueError('future real observations forbidden')
        if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in m['history_frames'].values()):
            raise ValueError('integer actual motion-history counts required')
        metadata=dict(checkpoint_sha256=m['checkpoint_sha256'],observed_until=seen,
            history_frames={k:int(v) for k,v in m['history_frames'].items()},
            contexts={k:np.asarray(v,dtype=float).tolist() for k,v in m['contexts'].items()})
        forecast={key:deepcopy(f[key]) for key in ('object_ids','times','positions','rotations','linear_velocities','angular_velocities')}
        forecast['metadata']=metadata
    result={key:deepcopy(r[key]) for key in (*_PROVENANCE,'timestamp','mode','available','free_forecast_available','controlled_mode_available','frozen_checkpoint_sha256')}
    result.update(object_observation=obj,forecast=forecast)
    return dict(index=row['index'],result=result,public_robot=deepcopy(robot),close_requested=row['close_requested'],rgbd=_rgbd_reference(row['rgbd']))

def pack_public_row(result,public_robot,close_requested,index,rgbd,*,expected_forecast_sha256):
    """Pack only actual public typed inputs; no backend/state/labels argument."""
    if type(public_robot) is not PublicRobotSensors or not isinstance(result,dict):
        raise TypeError('typed PublicRobotSensors and public bridge dictionary required')
    _walk_public(result,typed=True)
    obj=result.get('object_observation');f=result.get('forecast')
    if obj is not None and type(obj) is not ObjectState:raise TypeError('real typed public ObjectState required')
    if f is not None and type(f) is not Forecast:raise TypeError('real typed frozen Forecast required')
    observation=None if obj is None else dict(object_id=obj.object_id,timestamp=float(obj.timestamp),
        pose=dict(position=obj.pose.position.tolist(),rotation=obj.pose.rotation.tolist()),points_world=obj.points_world.tolist(),
        confidence=float(obj.confidence),visible=obj.visible)
    predicted=None if f is None else dict(object_ids=list(f.object_ids),times=f.times.tolist(),positions=f.positions.tolist(),
        rotations=f.rotations.tolist(),linear_velocities=f.linear_velocities.tolist(),angular_velocities=f.angular_velocities.tolist(),
        metadata={key:deepcopy(f.metadata[key]) for key in ('checkpoint_sha256','observed_until','history_frames','contexts')})
    if predicted is not None:
        predicted['metadata']['contexts']={k:np.asarray(v).tolist() for k,v in predicted['metadata']['contexts'].items()}
    selected={key:result[key] for key in (*_PROVENANCE,'timestamp','mode','available','free_forecast_available','controlled_mode_available','frozen_checkpoint_sha256')}
    selected.update(object_observation=observation,forecast=predicted)
    row=dict(index=index,result=selected,public_robot=dict(timestamp=float(public_robot.timestamp),
        tcp_position=public_robot.tcp_position.tolist(),tcp_quaternion=public_robot.tcp_quaternion.tolist(),
        q=public_robot.q.tolist(),dq=public_robot.dq.tolist(),finger_open_amounts=public_robot.finger_open_amounts.tolist()),
        close_requested=close_requested,rgbd=rgbd)
    row=_project_row(row,expected_forecast_sha256=expected_forecast_sha256);_walk_public(row);return row

class CorrectedPublicRowCache(PublicRowCache):
    def __init__(self,*,expected_forecast_sha256,rgbd_root=None):
        if (type(expected_forecast_sha256)is not str or len(expected_forecast_sha256)!=64
                or expected_forecast_sha256==LEGACY_FORECAST_SHA):
            raise ValueError('Distinct explicit corrected forecast authority required')
        self.expected_forecast_sha256=expected_forecast_sha256
        super().__init__(rgbd_root=rgbd_root)
    def observe(self,row):
        digest=hashlib.sha256();_walk_public(row,hasher=digest);digest=digest.hexdigest()
        packed=_project_row(row,expected_forecast_sha256=self.expected_forecast_sha256);now=packed['public_robot']['timestamp']
        if self._last_time is not None:
            if abs(now-self._last_time)<=1e-8:
                if digest!=self._last_digest:raise ValueError('same-time public frame changed; no duplicate observations')
                return self.receipt()
            if now<self._last_time:raise ValueError('cannot insert past/future-reordered public frames')
            if packed['index']<=self._last_index:raise ValueError('actual public frame indices must increase')
            if abs(now-self._last_time-.05)>1e-6:self._rows.clear();self._checked_images.clear();self.gap_resets+=1
        if self.rgbd_root is not None:
            p=(self.rgbd_root/packed['rgbd']).resolve(strict=True)
            if self.rgbd_root not in p.parents:raise ValueError('RGBD reference escaped bound public directory')
            with np.load(p,allow_pickle=False) as image:
                timestamp=np.asarray(image['timestamp'])
                if timestamp.size!=1 or abs(_time(float(timestamp.reshape(-1)[0]))-now)>1e-8:raise ValueError('real RGBD NPZ timestamp differs from public frame')
            self._checked_images[now]=packed['rgbd']
        self._latest=packed;self._last_time=now;self._last_digest=digest;self._last_index=packed['index']
        r=packed['result'];o=r['object_observation']
        eligible=(r['mode']=='FREE' and packed['close_requested'] is False and o is not None and o['visible'] and o['confidence']>=.5)
        if eligible:self._rows.append(packed)
        else:self._rows.clear();self.unknown_resets+=1
        # Image provenance is bounded to the retained history, not the episode.
        self._checked_images={t:p for t,p in self._checked_images.items() if any(v['public_robot']['timestamp']==t for v in self._rows)}
        return self.receipt()

    def features(self,*,current_time):
        now=_time(current_time,'actual current time')
        if self._latest is None or abs(self._last_time-now)>1e-8:raise ProposalUnavailable('current public frame absent/stale/future')
        try:features,provenance=public_features(self._latest,list(self._rows),expected_forecast_sha256=self.expected_forecast_sha256)
        except (ValueError,KeyError,TypeError) as error:raise ProposalUnavailable('public feature contract unavailable: '+str(error)) from error
        provenance=dict(provenance,cache_schema=CORRECTED_CACHE_SCHEMA,cache_epoch=self._epoch,
            RGBD_file_timestamps_verified=self.rgbd_root is not None,
            observations_created_by_cache=0)
        return features,provenance

    def receipt(self):
        result=super().receipt()
        result.update(schema=CORRECTED_CACHE_SCHEMA,forecast_checkpoint_sha256=self.expected_forecast_sha256,
            legacy_ebec_authority_used=False,new_forecast_feature_distribution=True)
        return result

class CorrectedFrozenPilotInference(FrozenPilotInference):
    def __init__(self,checkpoint,*,expected_sha256,forecast_sha256):
        if type(forecast_sha256)is not str or len(forecast_sha256)!=64 or forecast_sha256==LEGACY_FORECAST_SHA:
            raise ValueError('Explicit new forecast identity required')
        self.forecast_sha256=forecast_sha256
        super().__init__(checkpoint,expected_sha256=expected_sha256)
    def predict(self,cache,*,current_time):
        if type(cache) is not CorrectedPublicRowCache:raise TypeError('exact causal public-row cache required')
        self._assert_frozen()
        if cache.expected_forecast_sha256!=self.forecast_sha256:raise ValueError('Candidate/cache corrected forecast identity differs')
        features,provenance=cache.features(current_time=current_time)
        if not np.array_equal(features['task_e'],np.full(16,1./8,dtype=np.float32)):
            raise ValueError('single-task pilot must retain fixed independent uniform task embedding')
        tensors={k:torch.from_numpy(v.copy())[None] for k,v in features.items()}
        with torch.inference_mode():output=self._model(**tensors)
        self.calls+=1
        return output,dict(provenance,model_checkpoint_sha256=self.expected_sha256,model_CPU_eval=True,
            calibrated_success_probability=False,task_classifier_trained=False)

    def receipt(self):
        value=super().receipt()
        value.update(candidate_feature_compatibility='legacy_pilot_with_corrected_forecast_development_only',
            forecast_checkpoint_sha256=self.forecast_sha256,trained_on_this_forecast=False,
            fixed_deadlines_and_unranked_logits_unchanged=True,closedloop_success_qualified=False)
        return value

class CorrectedLearnedCandidates(LearnedCandidates):
    def __init__(self,inference,cache):
        if type(inference)is not CorrectedFrozenPilotInference or type(cache)is not CorrectedPublicRowCache:
            raise TypeError('Exact corrected forecast/cache and explicit pilot adapter required')
        if inference.forecast_sha256!=cache.expected_forecast_sha256:
            raise ValueError('Provider/cache forecast identity differs')
        self.inference,self.cache=inference,cache
        self.ledger=CandidateLedger();self._epoch=cache.epoch
        self.last_snapshot=self.last_feature_provenance=None;self._archived_events=[]

class CorrectedSymmetricLearnedCandidates(SymmetricLearnedCandidates,CorrectedLearnedCandidates):
    """Original update/expiry/pi representative methods; only constructor changes."""
