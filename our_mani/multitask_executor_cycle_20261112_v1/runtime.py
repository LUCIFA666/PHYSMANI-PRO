"""Strict public-only existing multi-task visual encoder / world-command model.

This route is distinct from the Cube body-forecast candidate architecture.
No language-trained or multi-task GRU/contact-model claim is made here.
"""
from pathlib import Path
import time
import torch

from .common import sha, read, verify
from ..trained_world_reference_live_20261104_v1.runtime import (
    E2, E2_SHA256, WORLD, WORLD_SHA256, SPLIT, CACHE, source_map)
from ..trained_world_reference_live_20261104_v1.history import PublicHistoryV1
from ..world_tcp_command_warmstart_20261104_v3.features import FrozenFinePublicSceneMemoryV3
from ..world_tcp_command_warmstart_20261104_v3.evaluation import load_selected
from ..world_tcp_command_warmstart_20261104_v3.cache import cache_inputs, INPUT_KEYS
from ..world_tcp_command_warmstart_20261104_v3.model import WorldTCPCommandTransformerV3, CandidateConfig
from ..world_tcp_command_warmstart_20261104_v3.runtime_adapter import bank_from_v3_model_output, select_generic_reference


class PublicWorldCommandPolicy:
    def __init__(self, checkpoint=None, expected_sha256=None):
        torch.set_num_threads(2)
        self.sources = source_map()
        verify(self.sources)
        self.encoder = FrozenFinePublicSceneMemoryV3(E2, E2_SHA256, SPLIT, device='cpu')
        self.checkpoint = Path(checkpoint or WORLD).resolve(strict=True)
        self.checkpoint_sha256 = expected_sha256 or WORLD_SHA256
        if sha(self.checkpoint) != self.checkpoint_sha256:
            raise ValueError('Explicit unchanged world command checkpoint required')
        if checkpoint is None:
            self.model, self.metadata = load_selected(WORLD, WORLD_SHA256, CACHE, device='cpu')
        else:
            payload = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
            meta = payload['metadata']
            if (meta['schema'] != 'multitask_world_command_head_finetune_20261112_v1'
                    or meta['encoder_sha256'] != E2_SHA256 or meta['training_updates'] <= 0
                    or not meta['TRAIN_only'] or meta['VAL_read'] or meta['TEST_read']):
                raise ValueError('Actually trained new-family TRAIN-only world model required')
            verify(meta['source_sha256'])
            self.model = WorldTCPCommandTransformerV3(CandidateConfig(**meta['model_config']))
            if self.model.spec() != meta['model_spec']:
                raise ValueError('Model/physical-output specification differs')
            self.model.load_state_dict(payload['model_state_dict'], strict=True)
            self.model.cpu().eval().requires_grad_(False)
            self.metadata = meta
        self.history = PublicHistoryV1()
        self.calls = 0
        self.last_diagnostic = None
        self._stat = self.checkpoint.stat()

    def predict_reference(self, packet):
        from ..generic_reference_mpc_20261104_v1.public_observation import validate_policy_packet
        validate_policy_packet(packet)
        stat = self.checkpoint.stat()
        if (stat.st_ino, stat.st_mtime_ns, stat.st_size) != (self._stat.st_ino, self._stat.st_mtime_ns, self._stat.st_size):
            raise RuntimeError('Policy checkpoint changed during episode')
        self.history.append(packet, frame_index=self.calls)
        inputs = {k: torch.from_numpy(v) for k, v in self.history.arrays().items()}
        started = time.perf_counter()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            encoded = self.encoder.encode(**inputs)
        materialized = cache_inputs(encoded)
        tensors = {k: materialized[k].float() if materialized[k].is_floating_point() else materialized[k]
                   for k in INPUT_KEYS}
        with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
            output = self.model(**tensors)
        reference = select_generic_reference(bank_from_v3_model_output(output), timestamp=packet['timestamp'],
                                            policy_sha256=self.checkpoint_sha256, e_sha256=E2_SHA256,
                                            selection_rule='highest_demo_likelihood')
        self.calls += 1
        self.last_diagnostic = dict(calls=self.calls, wall_seconds=time.perf_counter()-started,
                                    history=self.history.trace(), source='current_past_RGBD_K_T_robot_only',
                                    task_ID_Native_pose_contact_or_success_in_NN=False,
                                    physical_motion_GRU_used=False, language_branch_used=False,
                                    proposal_logits_are_success_probabilities=False)
        return reference
