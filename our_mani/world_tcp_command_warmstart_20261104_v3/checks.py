"""Source-only/default preparation; no checkpoint or real CNN is loaded."""
from pathlib import Path

from ..world_tcp_command_warmstart_20261104_v2 import data as pilot
from ..semantic_task_e_large_20261104_v2.common import source_map as encoder_sources
from .model import parameter_count
from .resources import GLOBAL_LOCKS

ROOT=Path(__file__).resolve().parents[2]


def source_map():
    sources=pilot.source_map();sources.update(encoder_sources())
    for name in ('__init__.py','model.py','features.py','geometry.py','resources.py','runtime_adapter.py','checks.py'):
        path=Path(__file__).with_name(name);sources[str(path.resolve())]=pilot.full.sha(path)
    script=ROOT/'scripts/check_world_tcp_command_warmstart_20261104_v3.py'
    sources[str(script.resolve())]=pilot.full.sha(script)
    return sources


def check(subset):
    sources=source_map();pilot.full.verify_sources(sources)
    fit=pilot.PublicSubsetWindows(subset,'fit');selection=pilot.PublicSubsetWindows(subset,'selection')
    return dict(complete=True,core_only_ready=True,source_sha256=sources,
        subset=pilot.full.pointer(subset),fit_windows=len(fit),selection_windows=len(selection),
        exact_parameters=parameter_count(),coarse_memory_tokens=86,fine_memory_tokens=512,
        Transformer_sequence_including_condition_and_queries=612,
        ray_convention='K_inverse@[u+.5,v+.5,1]*raw_camera_Z',
        public_K_T_modified=False,geometry_cache_dtype='FP32',
        frozen_encoder_family='semantic_e207M_v2',trained_encoder_checkpoint_pending=True,
        actual_trained_NN_forwards=0,RGBD_getters=0,GPU=False,Native=False,TEST_reads=0,
        correct_global_lock_paths=[str(path)for path in GLOBAL_LOCKS],
        cache_or_fit_or_physicalhandoff_or_poseprecision_qualified=False)
