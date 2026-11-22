"""Explicit new contact-aware admission; original pilot weights/K8 ledger math."""
from copy import deepcopy
import hashlib
import numpy as np
import torch
from ..cube_corrected_live_20261104_v1.candidate_runtime import (
    CorrectedPublicRowCache,CorrectedFrozenPilotInference,_project_row,_walk_public,_time)
from ..local_mpc_p4_20261103.runtime import LearnedCandidates,ProposalUnavailable
from ..local_mpc_p4_20261103.candidates import CandidateLedger
from ..local_mpc_p4_20261103.symmetry_runtime import SymmetricLearnedCandidates
from ..local_mpc_p4_20261103.data import unit_quaternion
from .features import public_network_features


class NativeContactPublicCacheV1(CorrectedPublicRowCache):
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
        eligible=(r['mode']=='FREE' and o is not None and o['visible'] and o['confidence']>=.5)
        if eligible:self._rows.append(packed)
        else:self._rows.clear();self.unknown_resets+=1
        # Image provenance is bounded to the retained history, not the episode.
        self._checked_images={t:p for t,p in self._checked_images.items() if any(v['public_robot']['timestamp']==t for v in self._rows)}
        return self.receipt()

    def features(self,*,current_time):
        if self.latest is None or abs(self.latest['public_robot']['timestamp']-current_time)>1e-8:
            raise ProposalUnavailable('Only the actual current public frame may make candidate tensors')
        try:
            features,provenance=public_network_features(self.latest,list(self._rows),expected_forecast_sha256=self.expected_forecast_sha256)
        except(ValueError,TypeError,KeyError)as error:raise ProposalUnavailable(str(error))from error
        provenance.update(cache_epoch=self.epoch,Native_pose_or_contact_in_numeric_NN_inputs=False,
            Native_membership_used_by_execution_admission=True,actual_close_intent_retained=self.latest['close_requested'])
        return features,provenance


class NativeContactPilotInferenceV1(CorrectedFrozenPilotInference):
    def predict(self,cache,*,current_time):
        if type(cache)is not NativeContactPublicCacheV1 or cache.expected_forecast_sha256!=self.forecast_sha256:
            raise TypeError('Exact contact-profile public numeric feature cache required')
        self._assert_frozen();features,provenance=cache.features(current_time=current_time)
        if not np.array_equal(features['task_e'],np.full(16,1./8,dtype=np.float32)):
            raise ValueError('Unchanged fixed independent single-task embedding required')
        tensors={k:torch.from_numpy(v.copy())[None]for k,v in features.items()}
        with torch.inference_mode():output=self._model(**tensors)
        self.calls+=1
        return output,dict(provenance,model_checkpoint_sha256=self.expected_sha256,model_CPU_eval=True,
            calibrated_success_probability=False,task_classifier_trained=False)


class NativeContactCandidatesV1(LearnedCandidates):
    def __init__(self,inference,cache):
        if type(inference)is not NativeContactPilotInferenceV1 or type(cache)is not NativeContactPublicCacheV1:
            raise TypeError('Exact explicit Native-contact profile, never old public-proof authority')
        self.inference,self.cache=inference,cache;self.ledger=CandidateLedger();self._epoch=cache.epoch
        self.last_snapshot=self.last_feature_provenance=None;self._archived_events=[]

    def _validate_current(self,state):
        if (state.get('Native_contact_feedback_used')is not True or state.get('Native_held_used')is not True
                or state.get('Native_object_state_used')is not False or state.get('Native_success_used')is not False):
            raise ValueError('Truthful Native contact provenance and public object geometry required')
        now=float(state['time']);row=self.cache.latest
        if row is None or abs(now-row['public_robot']['timestamp'])>1e-8:raise ProposalUnavailable('Current public cache required')
        for key in ('q','dq','tcp_position','tcp_quaternion','finger_open_amounts'):
            if not np.array_equal(np.asarray(state[key]),np.asarray(row['public_robot'][key])):
                raise ProposalUnavailable('Exact same actual public robot required: '+key)
        obj=row['result']['object_observation']
        if obj is None or obj.get('visible')is not True or obj['confidence']<.5:raise ProposalUnavailable('Current public object missing')
        if (not np.allclose(state['object_position'],obj['pose']['position'],atol=1e-6,rtol=0)
                or not np.allclose(unit_quaternion(state['object_quaternion'],'public object rotation').as_matrix(),obj['pose']['rotation'],atol=1e-6,rtol=0)):
            raise ProposalUnavailable('Candidate state cannot use Native object pose or controlled carry')
        if state['held']is not False or state['free_motion_asserted']is not True:
            raise ProposalUnavailable('Candidate NN is used only when actual Native membership is False')
        return now,row


class NativeContactSymmetricCandidatesV1(SymmetricLearnedCandidates,NativeContactCandidatesV1):
    """Same fixed deadlines/unranked slots/pi representative methods."""


from ..cube_corrected_live_20261104_v1.matched_candidate_runtime import MatchedCorrectedCandidateInference
from ..local_mpc_p4_20261103.candidates import _proposal_arrays


class NativeContactMatchedInferenceV1(MatchedCorrectedCandidateInference):
    """Actual trained matched weights; only exact new cache admission changes."""
    def predict(self,cache,*,current_time):
        if type(cache)is not NativeContactPublicCacheV1 or cache.expected_forecast_sha256!=self.forecast_sha256:
            raise TypeError('Exact new contact public cache and fitted forecast identity required')
        self._assert_frozen();features,provenance=cache.features(current_time=current_time)
        tensors={k:torch.from_numpy(v.copy())[None]for k,v in features.items()}
        with torch.inference_mode():output=self._model(**tensors)
        _proposal_arrays(output);self.calls+=1
        return output,dict(provenance,model_checkpoint_sha256=self.expected_sha256,
            model_CPU_eval=True,task_classifier_trained=False,calibrated_success_probability=False,
            matched_forecast_checkpoint_sha256=self.forecast_sha256)


class NativeContactMatchedCandidatesV1(NativeContactCandidatesV1):
    def __init__(self,inference,cache):
        if type(inference)is not NativeContactMatchedInferenceV1 or type(cache)is not NativeContactPublicCacheV1:
            raise TypeError('Actual matched checkpoint and exact contact public cache required')
        if inference.forecast_sha256!=cache.expected_forecast_sha256:raise ValueError('Matched model/cache forecast identity differs')
        self.inference,self.cache=inference,cache;self.ledger=CandidateLedger();self._epoch=cache.epoch
        self.last_snapshot=self.last_feature_provenance=None;self._archived_events=[]


class NativeContactMatchedSymmetricCandidatesV1(SymmetricLearnedCandidates,NativeContactMatchedCandidatesV1):
    """Same ledger/expiry/symmetry math with actual matched model identity."""
