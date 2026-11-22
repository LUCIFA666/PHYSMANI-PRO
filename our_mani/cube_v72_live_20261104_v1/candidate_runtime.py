"""Independent trained Cube provider, preserving the original P4 ledger ABI."""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
import torch

from ..local_mpc_p4_20261103.runtime import (
    PublicRowCache, FrozenPilotInference, LearnedCandidates, _state_digest)
from ..local_mpc_p4_20261103.candidates import CandidateLedger, _proposal_arrays
from ..local_mpc_p4_20261103.symmetry_runtime import SymmetricLearnedCandidates

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SCHEMA = 'Cube_candidate_transformer_20261104_v1'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenCubeCandidateInferenceV1:
    def __init__(self, checkpoint, *, expected_sha256):
        self.checkpoint = Path(checkpoint).resolve(strict=True)
        if type(expected_sha256) is not str or len(expected_sha256) != 64 or sha(self.checkpoint) != expected_sha256:
            raise ValueError('Exact externally frozen newly trained Cube checkpoint SHA required')
        from ..candidate_transformer_cube_20261104_v1.model import model_from_spec
        payload = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
        required = {'schema','model_state','model_spec','model_config','source_sha256',
            'train_parents','train_dataset_sha256','selection_parents','selection_dataset_sha256',
            'training_updates','seed','task_classifier_trained','selected_epoch','selection'}
        if not isinstance(payload,dict) or not required <= set(payload):
            raise ValueError('New Cube training payload required; old pilot/WorldTCP checkpoints cannot substitute')
        if (payload['schema'] != CHECKPOINT_SCHEMA or payload['training_updates'] != 1400
                or payload['task_classifier_trained'] is not False
                or payload['selection'] != 'minimum_parent_balanced_loss_on_existing_development_selection'):
            raise ValueError('Exact declared Cube fit/selection semantics required')
        train, selection = payload['train_parents'], payload['selection_parents']
        if (not isinstance(train,list) or not isinstance(selection,list) or not train or not selection
                or len(set(train)) != len(train) or len(set(selection)) != len(selection)
                or set(train)&set(selection) or any(type(p) is not str for p in train+selection)):
            raise ValueError('Disjoint complete fit/selection parent identities required')
        self._source_map = {str((ROOT/p).resolve() if not Path(p).is_absolute() else Path(p).resolve()):h
                            for p,h in payload['source_sha256'].items()}
        self._model = model_from_spec(payload['model_spec'])
        if self._model.spec() != payload['model_spec']:
            raise ValueError('Exact new candidate spec required')
        reference = self._model.state_dict(); state = payload['model_state']
        if not isinstance(state,dict) or set(state) != set(reference):
            raise ValueError('Exact new candidate parameter names required')
        for name, value in state.items():
            target = reference[name]
            if (not isinstance(value,torch.Tensor) or value.device.type != 'cpu'
                    or value.dtype != target.dtype or value.shape != target.shape
                    or not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid trained parameter: '+name)
            if name.endswith('_input_scale') and not torch.equal(value,target):
                raise ValueError('Unchanged physical P4 input scales required')
        self._model.load_state_dict(state,strict=True)
        self._model.eval(); self._model.requires_grad_(False)
        self._parameter_sha = _state_digest(self._model)
        self.expected_sha256 = expected_sha256; self.calls = 0
        self.train_parents = tuple(train); self.selection_parents = tuple(selection)
        self.train_dataset_sha256 = payload['train_dataset_sha256']
        self._metadata = {k:deepcopy(v) for k,v in payload.items() if k != 'model_state'}
        self._assert_frozen()

    def _assert_frozen(self):
        if any(sha(path)!=value for path,value in self._source_map.items()):
            raise ValueError('Trained Cube candidate source changed')
        if (self._model.training or any(p.requires_grad or p.device.type!='cpu' for p in self._model.parameters())
                or _state_digest(self._model)!=self._parameter_sha or sha(self.checkpoint)!=self.expected_sha256):
            raise ValueError('CPU frozen trained Cube weights/checkpoint changed')
        if any(m._forward_hooks or m._forward_pre_hooks for m in self._model.modules()):
            raise ValueError('No injected candidate forward hooks')

    def predict(self, cache, *, current_time):
        if type(cache) is not PublicRowCache: raise TypeError('Original actual causal public cache required')
        self._assert_frozen(); features, provenance = cache.features(current_time=current_time)
        if not np.array_equal(features['task_e'], np.full(16,1./8,dtype=np.float32)):
            raise ValueError('Fixed uniform single-operation task embedding required')
        tensors = {name:torch.from_numpy(value.copy())[None] for name,value in features.items()}
        with torch.inference_mode(): output = self._model(**tensors)
        _proposal_arrays(output)  # Preserve unranked body-local pose/tau ledger ABI.
        self.calls += 1
        return output, dict(provenance,model_checkpoint_sha256=self.expected_sha256,
            model_CPU_eval=True,calibrated_success_probability=False,task_classifier_trained=False,
            candidate_family='Cube_candidate_transformer_20261104_v1')

    def receipt(self):
        return dict(checkpoint_sha256=self.expected_sha256, source_sha256=deepcopy(self._source_map),
            parameter_sha256=self._parameter_sha, TRAIN_parent_count=len(self.train_parents),
            selection_parent_count=len(self.selection_parents), training_updates=self._metadata['training_updates'],
            selected_epoch=self._metadata['selected_epoch'], selection=self._metadata['selection'],
            source_dataset_sha256=self.train_dataset_sha256,inference_calls=self.calls,CPU_eval=True,
            task_embedding='fixed_uniform_single_operation_SimNorm2x8_distinct_from_motion_context',
            calibrated_success_probability=False, arbitrary_rotation_or_execution_qualified=False,
            high_speed_training_coverage_claimed=False, candidate_family=CHECKPOINT_SCHEMA)


class CubeLearnedCandidatesV1(LearnedCandidates):
    def __init__(self,inference,cache):
        if type(inference) is not FrozenCubeCandidateInferenceV1 or type(cache) is not PublicRowCache:
            raise TypeError('Exact new Cube provider and original public cache required')
        self.inference,self.cache = inference,cache
        self.ledger = CandidateLedger(); self._epoch = cache.epoch
        self.last_snapshot = self.last_feature_provenance = None; self._archived_events = []


class CubeSymmetricLearnedCandidatesV1(SymmetricLearnedCandidates,CubeLearnedCandidatesV1):
    """Only typed constructor changes; old expiry/symmetry/update functions reused."""


def candidate_loader(checkpoint,expected_sha256,mode):
    if mode == 'legacy_pilot': return FrozenPilotInference(checkpoint,expected_sha256=expected_sha256)
    if mode == 'cube_candidate_v1': return FrozenCubeCandidateInferenceV1(checkpoint,expected_sha256=expected_sha256)
    raise ValueError('Explicit legacy_pilot or cube_candidate_v1 profile required')
