"""CPU-only frozen P4 inference and causal public-row candidate integration.

No simulator, backend, teacher labels, checkpoint selection or fitting occurs
here. A producer must supply actual public sensor rows. Without ``rgbd_root``
the cache authenticates references/timing, not image bytes or physical origin.
"""
from collections import deque
from copy import deepcopy
import hashlib
import math
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from ..contracts import ObjectState, Forecast, Pose
from ..local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors, PublicMotionEvidence
from .data import PUBLIC_ROW_KEYS, ROBOT_KEYS, CHECKPOINT_SHA256, public_features, unit_quaternion
from .model import CandidateTransformerPilot
from .candidates import CandidateLedger


ROOT=Path(__file__).resolve().parents[2]
CACHE_SCHEMA='p4_feature_public_row_projection_real50ms_v1'
TRAIN_SOURCE_FILES=('scripts/train_p4_candidate_20261103.py','scripts/compile_p4_public_dataset_20261103.py',
    'our_mani/local_mpc_p4_20261103/__init__.py','our_mani/local_mpc_p4_20261103/model.py',
    'our_mani/local_mpc_p4_20261103/data.py')
ADAPTER_SHA256='24e7e6711874237706be0b346ccab27f6a9cfefcff3c01cc46d4ed099e61a701'
_PROVENANCE=('Native_inputs_used','future_observations_used','observations_fabricated','forecaster_is_injected_test_double')
_FORBIDDEN=frozenset(('gt','ground_truth','groundtruth','labels','evaluation_labels','native_state','native_result',
    'native_held','native_success','native_object_pose','native_object_position','task_success',
    'expert_stage','expert_action','instance_mask','instance_masks','instance_ids','native_handle','native_handles','handle'))


class ProposalUnavailable(ValueError):
    """No current causal public proposal; caller must choose an explicit hold."""


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _time(value,name='timestamp'):
    if isinstance(value,(bool,np.bool_)) or not np.isscalar(value) or not math.isfinite(value) or value<0:
        raise ValueError('finite nonnegative '+name+' required')
    return float(value)


def _walk_public(value,*,typed=False,path=(),hasher=None,state_held=False):
    """Reject opaque objects/privileged keys, without copying large arrays."""
    def mark(data):
        if hasher is not None:hasher.update(data)
    if value is None:mark(b'N');return
    if type(value) is bool:mark(b'T' if value else b'F');return
    if type(value) is int:mark(('I'+str(value)+';').encode());return
    if type(value) is float:
        if not math.isfinite(value):raise ValueError('nonfinite public scalar at '+'.'.join(path))
        mark(('D'+value.hex()+';').encode());return
    if type(value) is str:
        encoded=value.encode();mark(('S'+str(len(encoded))+':').encode()+encoded);return
    if isinstance(value,dict):
        if any(type(k) is not str for k in value):raise ValueError('public dictionary keys must be strings')
        mark(b'{')
        for key in sorted(value):
            lower=key.casefold()
            if lower in _FORBIDDEN or lower.startswith('gt_'):
                raise ValueError('privileged/label key forbidden in public input: '+key)
            if lower=='held':
                public_probe=(value[key] is False and path and path[-1] in ('probe','spec'))
                public_state=(state_held and not path and type(value[key]) is bool)
                if not(public_probe or public_state):raise ValueError('hidden held label forbidden')
            _walk_public(key,typed=typed,path=path,hasher=hasher,state_held=state_held)
            _walk_public(value[key],typed=typed,path=(*path,key),hasher=hasher,state_held=state_held)
        mark(b'}');return
    if type(value) in (list,tuple):
        if type(value) is tuple and not typed:raise TypeError('serialized public rows require JSON lists')
        mark(('L'+str(len(value))+':').encode())
        for index,item in enumerate(value):_walk_public(item,typed=typed,path=(*path,str(index)),hasher=hasher,state_held=state_held)
        return
    if typed and isinstance(value,np.ndarray):
        if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():raise ValueError('finite real public arrays required')
        return
    if typed and isinstance(value,np.generic):
        _walk_public(value.item(),typed=True,path=path,state_held=state_held);return
    if typed and type(value) is Pose:
        _walk_public(dict(position=value.position,rotation=value.rotation),typed=True,path=path);return
    if typed and type(value) is ObjectState:
        _walk_public(dict(object_id=value.object_id,timestamp=value.timestamp,pose=value.pose,points_world=value.points_world,
            linear_velocity=value.linear_velocity,angular_velocity=value.angular_velocity,confidence=value.confidence,visible=value.visible),typed=True,path=path);return
    if typed and type(value) is Forecast:
        _walk_public(dict(object_ids=value.object_ids,times=value.times,positions=value.positions,rotations=value.rotations,
            linear_velocities=value.linear_velocities,angular_velocities=value.angular_velocities,metadata=value.metadata),typed=True,path=path)
        return # points_world is unused and never copied or stored.
    if typed and type(value) is PublicMotionEvidence:
        _walk_public(dict(event=value.event,timestamp=value.timestamp,sources=value.sources,
            tcp_relative_pose=value.tcp_relative_pose,orientation_observable=value.orientation_observable),typed=True,path=path);return
    raise TypeError('opaque/non-public object forbidden at '+'.'.join(path))


def _rgbd_reference(value):
    if type(value) is not str or not value:
        raise ValueError('actual per-frame RGBD file reference required')
    p=PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or p.suffix!='.npz':
        raise ValueError('relative RGBD .npz reference without traversal required')
    return value


def _project_row(row):
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
    if r.get('frozen_checkpoint_sha256')!=CHECKPOINT_SHA256:raise ValueError('unchanged frozen motion checkpoint required')
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
        m=f['metadata'];seen=_time(m.get('observed_until'))
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


def pack_public_row(result,public_robot,close_requested,index,rgbd):
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
    row=_project_row(row);_walk_public(row);return row


class PublicRowCache:
    """At most20 measured50ms rows; gaps/unknown modes never create fake rows."""
    def __init__(self,*,rgbd_root=None):
        self.rgbd_root=None if rgbd_root is None else Path(rgbd_root).resolve(strict=True)
        self._epoch=0;self.reset()
    def reset(self):
        self._epoch+=1;self._rows=deque(maxlen=20);self._latest=None;self._last_time=None;self._last_digest=None
        self._last_index=None;self.gap_resets=0;self.unknown_resets=0;self._checked_images={}
    @property
    def epoch(self):return self._epoch
    @property
    def latest(self):return deepcopy(self._latest)
    @property
    def history(self):return deepcopy(list(self._rows))
    def observe(self,row):
        digest=hashlib.sha256();_walk_public(row,hasher=digest);digest=digest.hexdigest()
        packed=_project_row(row);now=packed['public_robot']['timestamp']
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
        try:features,provenance=public_features(self._latest,list(self._rows))
        except (ValueError,KeyError,TypeError) as error:raise ProposalUnavailable('public feature contract unavailable: '+str(error)) from error
        provenance=dict(provenance,cache_schema=CACHE_SCHEMA,cache_epoch=self._epoch,
            RGBD_file_timestamps_verified=self.rgbd_root is not None,
            observations_created_by_cache=0)
        return features,provenance
    def receipt(self):
        return dict(schema=CACHE_SCHEMA,epoch=self._epoch,last_timestamp=self._last_time,
            stored_history_frames=len(self._rows),history_timestamps=[r['public_robot']['timestamp'] for r in self._rows],
            history_RGBD_references=[r['rgbd'] for r in self._rows],maximum_frames=20,
            dropped_fields=['Forecast.points_world','binding_calibration_provenance','unused_public_diagnostics'],
            actual_RGBD_file_timestamps_verified=self.rgbd_root is not None,gap_resets=self.gap_resets,
            unknown_resets=self.unknown_resets,observations_fabricated=False,Native_handles_accepted=False)


def _state_digest(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        v=value.detach().cpu().contiguous();h.update((key+str(v.dtype)+str(tuple(v.shape))).encode());h.update(v.numpy().tobytes())
    return h.hexdigest()


class FrozenPilotInference:
    """Expected SHA is the caller's trusted final-weight identity, never guessed."""
    def __init__(self,checkpoint,*,expected_sha256):
        self.checkpoint=Path(checkpoint).resolve(strict=True)
        if type(expected_sha256) is not str or len(expected_sha256)!=64 or _sha(self.checkpoint)!=expected_sha256:
            raise ValueError('exact externally frozen P4 checkpoint SHA required')
        payload=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
        required={'schema','model_state','model_spec','train_parents','train_dataset_sha256','training_updates','seed','source_sha256','selection','task_classifier_trained'}
        if not isinstance(payload,dict) or set(payload)!=required:raise ValueError('exact final-pilot checkpoint schema required')
        if (payload['schema']!='P4_single_task_candidate_pilot_final_v1' or payload['training_updates']!=100
                or payload['selection']!='predeclared final update; no VAL/TEST checkpoint selection'
                or payload['task_classifier_trained'] is not False):raise ValueError('100-update final TRAIN pilot required, no VAL selection')
        parents=payload['train_parents']
        if (not isinstance(parents,list) or len(parents)<8 or len(set(parents))!=len(parents)
                or any(type(p) is not str or 'TRAIN' not in p or 'VAL' in p or 'TEST' in p for p in parents)):
            raise ValueError('independent predeclared TRAIN parent identities required')
        self._source_map=payload['source_sha256'];expected={str((ROOT/p).resolve()) for p in TRAIN_SOURCE_FILES}
        if not isinstance(self._source_map,dict) or set(self._source_map)!=expected:raise ValueError('exact training/model/data/compiler source map required')
        self._model=CandidateTransformerPilot()
        if payload['model_spec']!=self._model.spec():raise ValueError('frozen model architecture/units specification differs')
        state=payload['model_state'];reference=self._model.state_dict()
        if not isinstance(state,dict) or set(state)!=set(reference):raise ValueError('exact frozen parameter names required')
        for key,tensor in state.items():
            if (not isinstance(tensor,torch.Tensor) or tensor.device.type!='cpu' or tensor.dtype!=reference[key].dtype
                    or tensor.shape!=reference[key].shape or not bool(torch.isfinite(tensor).all())):raise ValueError('invalid frozen parameter: '+key)
            if key in ('future_input_scale','robot_input_scale','geometry_input_scale') and not torch.equal(tensor,reference[key]):
                raise ValueError('checkpoint fixed physical input scale differs from model specification: '+key)
        self._model.load_state_dict(state,strict=True);self._model.eval();self._model.requires_grad_(False)
        self._parameter_sha=_state_digest(self._model);self.expected_sha256=expected_sha256
        self.train_parents=tuple(parents);self.train_dataset_sha256=payload['train_dataset_sha256'];self.calls=0
        self._assert_frozen()
    def _assert_frozen(self):
        if any(_sha(p)!=h for p,h in self._source_map.items()):raise ValueError('frozen TRAIN/model/data/compiler source changed')
        if _sha(ROOT/'our_mani/local_mpc_p4_20261103/candidates.py')!=ADAPTER_SHA256:raise ValueError('candidate adapter source changed')
        if (self._model.training or any(p.requires_grad or p.device.type!='cpu' for p in self._model.parameters())
                or _state_digest(self._model)!=self._parameter_sha):raise ValueError('CPU frozen inference parameters or eval mode changed')
    def predict(self,cache,*,current_time):
        if type(cache) is not PublicRowCache:raise TypeError('exact causal public-row cache required')
        self._assert_frozen();features,provenance=cache.features(current_time=current_time)
        if not np.array_equal(features['task_e'],np.full(16,1./8,dtype=np.float32)):
            raise ValueError('single-task pilot must retain fixed independent uniform task embedding')
        tensors={k:torch.from_numpy(v.copy())[None] for k,v in features.items()}
        with torch.inference_mode():output=self._model(**tensors)
        self.calls+=1
        return output,dict(provenance,model_checkpoint_sha256=self.expected_sha256,model_CPU_eval=True,
            calibrated_success_probability=False,task_classifier_trained=False)
    def receipt(self):
        return dict(checkpoint_sha256=self.expected_sha256,source_sha256=deepcopy(self._source_map),
            parameter_sha256=self._parameter_sha,TRAIN_parent_count=len(self.train_parents),training_updates=100,
            source_dataset_sha256=self.train_dataset_sha256,inference_calls=self.calls,CPU_eval=True,
            task_embedding='fixed_uniform_single_task_SimNorm2x8_distinct_from_motion_context',
            arbitrary_rotation_or_execution_qualified=False,calibrated_success_probability=False)


class LearnedCandidates:
    """Legacy geometry.update(public_state) interface with a frozen ledger."""
    def __init__(self,inference,cache):
        if type(inference) is not FrozenPilotInference or type(cache) is not PublicRowCache:
            raise TypeError('exact frozen inference and external real-observer cache required')
        self.inference,self.cache=inference,cache;self.ledger=CandidateLedger();self._epoch=cache.epoch
        self.last_snapshot=None;self.last_feature_provenance=None;self._archived_events=[]
    @property
    def generation(self):return self.ledger.generation
    @property
    def events(self):
        return deepcopy(self._archived_events)+[dict(event,cache_epoch=self._epoch) for event in self.ledger.events]
    @property
    def candidates(self):return self.ledger.candidates
    def _validate_current(self,state):
        if not isinstance(state,dict):raise TypeError('public estimated state dictionary required')
        for key in ('Native_object_state_used','Native_held_used','Native_success_used'):
            if state.get(key) is not False:raise ValueError('Native/GT state forbidden: '+key)
        _walk_public(state,typed=True,state_held=True)
        now=_time(state.get('time'),'current public state time');row=self.cache.latest
        if row is None or abs(now-row['public_robot']['timestamp'])>1e-8:raise ProposalUnavailable('cache and current public state timestamps differ')
        for key in ('q','dq','tcp_position','tcp_quaternion','finger_open_amounts'):
            if not np.array_equal(np.asarray(state.get(key)),np.asarray(row['public_robot'][key])):
                raise ProposalUnavailable('current robot feedback differs from observed cache: '+key)
        obj=row['result']['object_observation']
        if obj is None or not obj['visible'] or obj['confidence']<.5:raise ProposalUnavailable('current public body is unknown')
        p=np.asarray(state.get('object_position'),float)
        if p.shape!=(3,) or not np.allclose(p,obj['pose']['position'],atol=1e-6,rtol=0):raise ProposalUnavailable('current body and public cache differ')
        rotation=unit_quaternion(state.get('object_quaternion'),'current public body quaternion').as_matrix()
        if not np.allclose(rotation,obj['pose']['rotation'],atol=1e-6,rtol=0):raise ProposalUnavailable('current body orientation and public cache differ')
        if type(state.get('held')) is not bool or type(state.get('free_motion_asserted')) is not bool:
            raise ValueError('explicit estimated held/free motion state required')
        if state['held'] and (row['result']['mode']!='ATTACHED_PUBLIC' or row['result']['available'] is not True
                or row['result']['controlled_mode_available'] is not True):
            raise ProposalUnavailable('held cannot be inferred from old candidate history')
        return now,row
    def update(self,state):
        if self._epoch!=self.cache.epoch:
            old_generation=self.ledger.generation
            self._archived_events.extend(dict(event,cache_epoch=self._epoch) for event in self.ledger.events)
            latest=self.cache.latest
            self._archived_events.append(dict(time=None if latest is None else latest['public_robot']['timestamp'],
                generation=old_generation,reason='cache_epoch_changed_retire_all_previous_candidates',
                previous_cache_epoch=self._epoch,cache_epoch=self.cache.epoch,
                retired=[dict(id=c['id'],deadline=c['deadline']) for c in self.ledger.candidates],
                warm_start_clear_required=True,old_deadlines_modified=False))
            # A fresh ledger permits the new episode's clock origin, while its
            # lifetime generation counter cannot collide with old warm IDs.
            self.ledger=CandidateLedger();self.ledger.generation=old_generation
            self._epoch=self.cache.epoch;self.last_snapshot=None;self.last_feature_provenance=None
        now,row=self._validate_current(state)
        free=(not state['held'] and state['free_motion_asserted'] and state.get('grasp_motion_mode','FREE_GRU')=='FREE_GRU')
        if not free:
            if not self.ledger.candidates:raise ProposalUnavailable('non-FREE state has no existing frozen proposal pool')
            self.last_snapshot=self.ledger.update_expiry(now)
            self.last_snapshot.update(phase_pool_frozen=True,NN_called=False,availability_is_reachability=False,events=self.events)
            return self.last_snapshot['candidates']
        _,provenance=self.cache.features(current_time=now) # validate fresh GRU even when reusing old poses/tau.
        forecast_end=float(row['result']['forecast']['times'][-1])
        status=self.ledger.update_expiry(now,forecast_end=forecast_end)
        call=status['new_generation_required']
        if call:
            output,provenance=self.inference.predict(self.cache,current_time=now)
            self.ledger.new_generation(now,output)
        self.last_feature_provenance=provenance
        self.last_snapshot=self.ledger.update_expiry(now,forecast_end=forecast_end)
        self.last_snapshot.update(phase_pool_frozen=False,NN_called=call,availability_is_reachability=False,events=self.events)
        return self.last_snapshot['candidates']
