"""Same frozen weights; observations appended every actual50ms, not per action."""
import time
import torch
from ..multitask_executor_cycle_20261112_v1.runtime import PublicWorldCommandPolicy
from ..world_tcp_command_warmstart_20261104_v3.cache import cache_inputs,INPUT_KEYS
from ..world_tcp_command_warmstart_20261104_v3.runtime_adapter import bank_from_v3_model_output,select_generic_reference
from ..trained_world_reference_live_20261104_v1.runtime import E2_SHA256
from ..generic_reference_mpc_20261104_v1.public_observation import validate_policy_packet


class PublicContinuousHistoryPolicy(PublicWorldCommandPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Explicit GPU BF16 frozen inference required; no fitting occurs during Native')
        self.encoder.model.cuda().eval().requires_grad_(False)
        self.model.cuda().eval().requires_grad_(False)
        self.observed_frames=0
        self.last_observation_time=None

    def observe(self,packet,frame_index):
        validate_policy_packet(packet)
        if self.last_observation_time is not None and abs(packet['timestamp']-self.last_observation_time-.05)>2e-5:
            raise ValueError('Every actual50ms public frame required; no sparse action-endpoint history')
        if frame_index!=self.observed_frames:
            raise ValueError('Actual zero-origin physical frame sequence required')
        self.history.append(packet,frame_index=frame_index)
        self.observed_frames+=1
        self.last_observation_time=packet['timestamp']

    def predict_reference(self,packet):
        validate_policy_packet(packet)
        if self.last_observation_time is None or abs(packet['timestamp']-self.last_observation_time)>1e-8:
            raise ValueError('Actual current observed frame must already belong to the50ms history')
        stat=self.checkpoint.stat()
        if (stat.st_ino,stat.st_mtime_ns,stat.st_size)!=(self._stat.st_ino,self._stat.st_mtime_ns,self._stat.st_size):
            raise RuntimeError('Frozen policy checkpoint changed')
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('Inference-only frozen model required')
        arrays=self.history.arrays()
        inputs={k:torch.from_numpy(v).cuda()for k,v in arrays.items()}
        started=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            encoded=self.encoder.encode(**inputs)
        materialized=cache_inputs(encoded)
        tensors={k:materialized[k].to(device='cuda',dtype=torch.float32 if materialized[k].is_floating_point()else materialized[k].dtype)
                 for k in INPUT_KEYS}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=self.model(**tensors)
        reference=select_generic_reference(bank_from_v3_model_output(output),timestamp=packet['timestamp'],
                                           policy_sha256=self.checkpoint_sha256,e_sha256=E2_SHA256,
                                           selection_rule='highest_demo_likelihood')
        torch.cuda.synchronize()
        self.calls+=1
        self.last_diagnostic=dict(calls=self.calls,wall_seconds=time.perf_counter()-started,
                                  history=dict(self.history.trace(),source_physics_clock_independently_verified=True),
                                  actual50ms_contiguous_public_history=True,observed_frames=self.observed_frames,
                                  device='cuda_BF16_frozen_inference',GPU_optimizer_steps=0,
                                  historical_TRAIN_physical_dt_independently_verified=False,
                                  CPU_GPU_bitwise_equivalence_claim=False,
                                  task_ID_Native_pose_contact_success_or_future_in_NN=False,
                                  language_branch_used=False,physical_motion_GRU_used=False)
        return reference
