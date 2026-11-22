"""Explicit newly trained candidate authority bound to the corrected forecast.

No old checkpoint is a fallback. Legacy ledger field names are retained only
as ABI names: coordinates use the public Forecast basis declared in model_spec.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
import torch

from ..cube_matched_candidate_20261104_v1 import CHECKPOINT_SCHEMA
from ..cube_matched_candidate_20261104_v1.model import model_from_spec,model_source_map,CandidateConfig
from ..cube_candidate_matched_20261104_v2 import FRAME
from ..local_mpc_p4_20261103.model import SCORE_SEMANTICS
from ..local_mpc_p4_20261103.runtime import LearnedCandidates,_state_digest
from ..local_mpc_p4_20261103.candidates import CandidateLedger,_proposal_arrays
from ..local_mpc_p4_20261103.symmetry_runtime import SymmetricLearnedCandidates
from .candidate_runtime import CorrectedPublicRowCache

ROOT=Path(__file__).resolve().parents[2]
TIME_SUPERVISION='actual_tau_masked_outside_0p05_0p8_no_clamp'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_epoch_metadata(payload):
    if (type(payload['training_updates'])is not int or payload['training_updates']<=0
            or type(payload['epochs_completed'])is not int or payload['epochs_completed']<=0
            or payload.get('selected_epoch_indexing')!='one_based'
            or type(payload['selected_epoch'])is not int
            or not 1<=payload['selected_epoch']<=payload['epochs_completed']):
        raise ValueError('Actual updates and one-based completed/selected epoch metadata required')


class MatchedCorrectedCandidateInference:
    def __init__(self,checkpoint,*,expected_sha256,forecast_sha256):
        if (type(expected_sha256)is not str or len(expected_sha256)!=64
                or type(forecast_sha256)is not str or len(forecast_sha256)!=64):
            raise ValueError('Explicit newly trained candidate and forecast identities required')
        self.checkpoint=Path(checkpoint).resolve(strict=True)
        if sha(self.checkpoint)!=expected_sha256:raise ValueError('Matched candidate checkpoint differs')
        payload=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
        required={'schema','model_spec','model_config','model_state','source_sha256','forecast_checkpoint_sha256',
            'target_frame','train_parents','selection_parents','heldout_parents','train_dataset_sha256',
            'selection_dataset_sha256','training_updates','epochs_completed','selected_epoch','seed',
            'task_classifier_trained','selection','time_supervision','proposal_score_semantics','selected_epoch_indexing'}
        if not isinstance(payload,dict) or not required<=set(payload):
            raise ValueError('Complete newly trained matched candidate payload required')
        if (payload['schema']!=CHECKPOINT_SCHEMA or payload['forecast_checkpoint_sha256']!=forecast_sha256
                or payload['target_frame']!=FRAME or payload['task_classifier_trained']is not False
                or payload['time_supervision']!=TIME_SUPERVISION or payload['proposal_score_semantics']!=SCORE_SEMANTICS):
            raise ValueError('Wrong candidate family/public frame/forecast/time/score semantics')
        _validate_epoch_metadata(payload)
        if type(payload['selection'])is not str or not payload['selection']:
            raise ValueError('Declared checkpoint selection rule required')
        parents=[payload[k] for k in ('train_parents','selection_parents','heldout_parents')]
        if any(not isinstance(v,list) or not v or len(set(v))!=len(v) or any(type(p)is not str for p in v) for v in parents):
            raise ValueError('Complete unique whole-parent role lists required')
        if any(set(parents[i])&set(parents[j]) for i in range(3) for j in range(i)):
            raise ValueError('Fit/selection/heldout source parents overlap')
        self._source_map={str((ROOT/p).resolve() if not Path(p).is_absolute() else Path(p).resolve()):h
            for p,h in payload['source_sha256'].items()}
        for p,h in model_source_map().items():
            if self._source_map.get(str(Path(p).resolve()))!=h:
                raise ValueError('Full new model/physical/math/source family missing from training provenance')
        if any(sha(p)!=h for p,h in self._source_map.items()):
            raise ValueError('Matched training sources changed')
        self._model=model_from_spec(payload['model_spec'])
        if self._model.config.to_dict()!=CandidateConfig.from_dict(payload['model_config']).to_dict():
            raise ValueError('Spec/config disagree')
        reference=self._model.state_dict();state=payload['model_state']
        if set(state)!=set(reference):raise ValueError('Exact matched-model parameter names required')
        for name,value in state.items():
            if (not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=reference[name].dtype
                    or value.shape!=reference[name].shape or not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid matched candidate parameter: '+name)
            if name.endswith('_input_scale') and not torch.equal(value,reference[name]):
                raise ValueError('Physical input scales changed')
        self._model.load_state_dict(state,strict=True);self._model.cpu().eval().requires_grad_(False)
        self._parameter_sha=_state_digest(self._model)
        self.expected_sha256=expected_sha256;self.forecast_sha256=forecast_sha256
        self._metadata={k:deepcopy(v)for k,v in payload.items()if k!='model_state'}
        self.calls=0;self._assert_frozen()

    def _assert_frozen(self):
        if (sha(self.checkpoint)!=self.expected_sha256 or any(sha(p)!=h for p,h in self._source_map.items())
                or self._model.training or any(p.requires_grad or p.device.type!='cpu' for p in self._model.parameters())
                or _state_digest(self._model)!=self._parameter_sha
                or any(m._forward_hooks or m._forward_pre_hooks for m in self._model.modules())):
            raise ValueError('Matched candidate model/checkpoint/source changed')

    def predict(self,cache,*,current_time):
        if type(cache)is not CorrectedPublicRowCache:raise TypeError('Exact corrected-forecast public cache required')
        if cache.expected_forecast_sha256!=self.forecast_sha256:
            raise ValueError('Training/runtime corrected forecast identity differs')
        self._assert_frozen();features,provenance=cache.features(current_time=current_time)
        tensors={k:torch.from_numpy(v.copy())[None]for k,v in features.items()}
        with torch.inference_mode():output=self._model(**tensors)
        _proposal_arrays(output);self.calls+=1
        return output,dict(provenance,model_checkpoint_sha256=self.expected_sha256,
            target_frame=FRAME,model_CPU_eval=True,task_classifier_trained=False,
            calibrated_success_probability=False,matched_forecast_checkpoint_sha256=self.forecast_sha256)

    def receipt(self):
        return dict(checkpoint_sha256=self.expected_sha256,source_sha256=deepcopy(self._source_map),
            parameter_sha256=self._parameter_sha,forecast_checkpoint_sha256=self.forecast_sha256,
            training_updates=self._metadata['training_updates'],epochs_completed=self._metadata['epochs_completed'],
            selected_epoch=self._metadata['selected_epoch'],selected_epoch_indexing='one_based',selection=self._metadata['selection'],
            TRAIN_parent_count=len(self._metadata['train_parents']),selection_parent_count=len(self._metadata['selection_parents']),
            heldout_parent_count=len(self._metadata['heldout_parents']),inference_calls=self.calls,
            target_frame=FRAME,time_supervision=TIME_SUPERVISION,trained_on_this_forecast=True,
            task_embedding='fixed_uniform_single_Cube_operation_not_task_classifier',
            legacy_ledger_field_names_have_public_reference_semantics=True,
            proposal_score_semantics=SCORE_SEMANTICS,calibrated_success_probability=False,
            CPU_eval=True,formal_pose_or_execution_qualified=False)


class MatchedCorrectedLearnedCandidates(LearnedCandidates):
    def __init__(self,inference,cache):
        if type(inference)is not MatchedCorrectedCandidateInference or type(cache)is not CorrectedPublicRowCache:
            raise TypeError('Exact matched model and corrected forecast cache required')
        if inference.forecast_sha256!=cache.expected_forecast_sha256:
            raise ValueError('Matched model/cache forecast differs')
        self.inference,self.cache=inference,cache
        self.ledger=CandidateLedger();self._epoch=cache.epoch
        self.last_snapshot=self.last_feature_provenance=None;self._archived_events=[]


class MatchedCorrectedSymmetricCandidates(SymmetricLearnedCandidates,MatchedCorrectedLearnedCandidates):
    """Same fixed-time ledger and original parallel-gripper representative."""
