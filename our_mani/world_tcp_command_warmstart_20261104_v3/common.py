"""Bound feature/cache/BC pilot contracts; actual GPU starts only by Root."""
from pathlib import Path
import math

from ..world_tcp_command_warmstart_20261104_v2 import data as pilot
from .checks import source_map as core_sources
from .model import CandidateConfig,parameter_count
from .resources import ROOT,resources
from . import VERSION

CONFIG=ROOT/'config/framework/world_tcp_command_fine_20261104_v3/pilot.json'
SUBSET_SHA='eeb22774a62e548e45353b00fc17900bb56e6d438dc24e02c0fea4954ac06212'
COMPONENT='world_TCP_expert_command_warmstart_only'
read=pilot.full.read
sha=pilot.full.sha
pointer=pilot.full.pointer
write_new=pilot.full.write_new
verify_sources=pilot.full.verify_sources


def checked_pointer(value):
    if not isinstance(value,dict)or set(value)!={'path','sha256'}:raise ValueError('Exact path/SHA pointer required')
    path=(ROOT/value['path']).resolve(strict=True)
    if sha(path)!=value['sha256']:raise ValueError('Bound artifact changed: '+str(path))
    return path


def fixed_config():
    value=read(CONFIG)
    if (value.get('schema')!='fine_world_TCP_command_BC_pilot_20261104_v3'
            or value.get('exact_parameters')!=86772491 or value.get('epochs')!=5
            or value.get('effective_batch_size')!=16 or value.get('initial_microbatch')!=4
            or value.get('OOM_fallback_microbatch')!=2 or value.get('learning_rate')!=3e-5
            or value.get('weight_decay')!=.01 or value.get('gradient_clip_norm')!=1.
            or value.get('seed')!=20261104 or value.get('encoder_family')!='semantic_e207M_v2'
            or value.get('fit_windows')!=24592 or value.get('selection_windows')!=5168
            or value.get('current_network_end_to_end_ready')is not False
            or value.get('automatic_Native_evaluation')is not False
            or value.get('mixture_position_scale_m')!=.03 or value.get('mixture_rotation_scale_rad')!=.2
            or value.get('handoff_time_loss_weight')!=0.
            or value.get('model')!=dict(width=768,layers=12,heads=12,feedforward_multiplier=4,dropout=.1,memory_width=1024,memory_tokens=86,candidates=8)
            or value.get('precision')!='BF16_autocast_FP32_parameters_AdamW'
            or value.get('cache_dtypes')!=dict(visual_memory='bfloat16',fine_visual='bfloat16',fine_geometry='float32',other_public_float='float32',masks='bool')
            or parameter_count(CandidateConfig(**value['model']))!=86772491):
        raise ValueError('Root-fixed fine86M/BF16/5epoch/stage16 pilot changed')
    return value


def source_map():
    sources=core_sources()
    paths=[Path(__file__).with_name(name)for name in ('common.py','cache.py','training.py','evaluation.py')]
    paths.extend((CONFIG,ROOT/'scripts/cache_world_tcp_command_warmstart_20261104_v3.py',
                  ROOT/'scripts/train_world_tcp_command_warmstart_20261104_v3.py',
                  ROOT/'scripts/evaluate_world_tcp_command_warmstart_20261104_v3.py'))
    sources.update({str(path.resolve()):sha(path)for path in paths})
    return sources


def root_authorization(path,expected_sha,*,sources,subset_pointer,encoder_pointer):
    if sha(path)!=expected_sha:raise ValueError('External Root GPU authorization SHA required')
    value=read(path)
    if (value.get('schema')!='world_command_fine_v3_GPU_authorization_20261104_v1'
            or value.get('execution_authorized')is not True
            or value.get('source_sha256')!=sources or value.get('config')!=pointer(CONFIG)
            or value.get('subset')!=subset_pointer or value.get('encoder_checkpoint')!=encoder_pointer
            or value.get('encoder_family')!='semantic_e207M_v2'):
        raise ValueError('Actual Root/config/source/subset/selected-e2 GPU authorization required')
    checked_pointer(subset_pointer);checked_pointer(encoder_pointer)
    for name,allkey in (('native_lifecycle','all_native_children_reaped'),('prior_training_lifecycle','all_training_children_reaped')):
        part=value.get(name,{})
        if part.get(allkey)is not True or not isinstance(part.get('owners'),list)or not part['owners']:
            raise ValueError('All prior Native/training owners must actually exit and be reaped')
        for owner in part['owners']:
            actual=read(checked_pointer(owner))
            if type(actual.get('actual_process_returncode'))is not int or actual.get('actual_wait_completed')is not True:
                raise ValueError('Actual typed owner exits/waits required, not task-success selection')
    return pointer(path)


def checked_canary(path,expected_sha,*,sources,subset_pointer,encoder_pointer,cache_dtypes):
    if sha(path)!=expected_sha:raise ValueError('Actual Root GPU canary report+external SHA required')
    value=read(path)
    if (any(value.get(key)is not True for key in ('complete','passed','source_unchanged'))
            or value.get('schema')!='fine_world_v3_GPU_canary_20261104_v1'
            or value.get('source_sha256')!=sources or value.get('config')!=pointer(CONFIG)
            or value.get('subset')!=subset_pointer or value.get('encoder_checkpoint')!=encoder_pointer
            or value.get('cache_dtypes')!=cache_dtypes or type(value.get('microbatch'))is not int
            or value['microbatch']not in (2,4)or value.get('effective_batch_size')!=16
            or value.get('exact_parameters')!=86772491 or value.get('candidate_optimizer_steps')!=1
            or value.get('canary_weights_reused')is not False or value.get('full_history')is not True
            or value.get('actual_valid_frames')!=value['microbatch']*5
            or value.get('actual_fine_tokens_per_sample')!=512):
        raise ValueError('Actual same-source/e2/full-history fine GPU canary must pass')
    if (value.get('deterministic_algorithms')is not True or value.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'
            or value.get('cudnn_benchmark')is not False or value.get('candidate_training_mode')is not True
            or value.get('candidate_dropout')!=.1 or value.get('precision')!='BF16_autocast_FP32_parameters_AdamW'):
        raise ValueError('Canary must use the actual deterministic BF16/dropout training path')
    for key in ('encoder_cuda_batch_ms','candidate_cuda_forward_backward_step_ms','wall_seconds','peak_allocated_bytes','peak_reserved_bytes'):
        if isinstance(value.get(key),bool)or not isinstance(value.get(key),(int,float))or not math.isfinite(value[key])or value[key]<=0:
            raise ValueError('Actual finite CUDA memory/throughput measurements required')
    if value['microbatch']==2:
        previous=read(checked_pointer(value.get('OOM_predecessor')))
        if (previous.get('cuda_oom')is not True or previous.get('microbatch')!=4
                or previous.get('source_sha256')!=sources or previous.get('subset')!=subset_pointer
                or previous.get('encoder_checkpoint')!=encoder_pointer):
            raise ValueError('A microbatch2 report must retain its exact actualOOM4 predecessor')
    return value
