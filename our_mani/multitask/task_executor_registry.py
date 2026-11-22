"""Explicit task adapters for the shared continuous executor, TRAIN pilot first."""
from dataclasses import dataclass
from types import MappingProxyType, MethodType
from pathlib import Path
import importlib
from .receding_task_transit import install as install_transit
from .executor_trace import install_trace
from .reference_constraint_authority import install as install_authority
from .evaluation import file_sha
from .frozen_subset_batch_v132 import checked_subset


@dataclass(frozen=True)
class TaskAdapter:
    setup: str
    driver: str
    backend_module: str
    backend_class: str


ADAPTERS = MappingProxyType({
    **{task:TaskAdapter('joint16_buzz_smoke_v175_setup','run_task_executor_smoke_joint16_v174_r10',
        'joint16_native_v173','SharedJoint16NativeBackend')
       for task in ('beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed')},
    'insert_onto_rotating_peg':TaskAdapter('normal_ring_retarget_v384_setup',
        'run_task_executor_smoke_normal_ring_retarget_v383_r10','normal_ring_retarget_native_v382','SharedJoint16NativeBackend'),
    'insert_onto_rotating_peg_high_speed':TaskAdapter('high_peg_insert_v396_setup',
        'run_task_executor_smoke_high_peg_insert_v395_r10','high_peg_insert_native_v394','SharedJoint16NativeBackend'),
    **{task:TaskAdapter('joint16_bin_smoke_v219_setup','run_task_executor_smoke_joint16_v174_r10',
        'joint16_native_v173','SharedJoint16NativeBackend')
       for task in ('moving_basketball_in_hoop','moving_basketball_in_hoop_high_speed')},
    'place_cups_on_rotating_frame':TaskAdapter('cup_tracking_v380_setup',
        'run_task_executor_smoke_cup_tracking_v379_r10','cup_tracking_native_v378','NormalCupSlackBackend'),
    'place_cups_on_rotating_frame_high_speed':TaskAdapter('cup_tracking_v380_setup',
        'run_task_executor_smoke_cup_tracking_v379_r10','cup_tracking_native_v378','HighCupSlackBackend'),
    **{task:TaskAdapter('joint16_remove_v405_setup','run_task_executor_smoke_joint16_v174_r10',
        'joint16_native_v173','SharedJoint16NativeBackend')
       for task in ('remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed')},
    'put_rubbish_in_moving_bin_high_speed':TaskAdapter('high_bin_depth_stop_v290_setup',
        'run_task_executor_smoke_bin_stop_v276_r10','bin_stop_native_v275','SharedJoint16NativeBackend'),
})


def driver(task):
    return importlib.import_module('scripts.'+ADAPTERS[task].driver)


def binding(root, config, task):
    route = ADAPTERS[task]
    files = ('our_mani/multitask/task_executor_registry.py',
        'our_mani/multitask/receding_task_transit.py', 'our_mani/multitask/continuous_transit.py',
        'our_mani/multitask/executor_trace.py',
        'our_mani/multitask/reference_constraint_authority.py',
        'our_mani/multitask/'+route.backend_module+'.py', 'scripts/'+route.driver+'.py')
    return dict(schema='r10_task_registry_shared_executor_v1', task=task,
        original_backend_module=route.backend_module, original_backend_class=route.backend_class,
        original_driver=route.driver, replan_seconds=.2,
        source_geometry_and_phase_semantics_retained=True,
        context_checkpoint=config['context_checkpoint'],
        sources_sha256={p:file_sha(Path(root)/p) for p in files}, formal_release=False)


def verify(root, prepared):
    expected = binding(root, prepared['config'], prepared['args'].task)
    if prepared['config'].get('shared_task_executor') != expected:
        raise ValueError('freeze exact task adapter and common executor')
    checked_subset(root, prepared['files'],
        {str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()})
    return expected


class SharedTaskBackend:
    def __init__(self, task):
        route = ADAPTERS[task]
        module = importlib.import_module('our_mani.multitask.'+route.backend_module)
        self.original = getattr(module, route.backend_class)()

    def __getattr__(self, name):
        return getattr(self.original, name)

    def actor(self, prepared, env, descriptions):
        actor, identity = self.original.actor(prepared, env, descriptions)
        root = Path(__file__).resolve().parents[2]
        expected = binding(root, prepared['config'], prepared['args'].task)
        install_authority(actor)
        install_transit(actor, replan_seconds=expected['replan_seconds'])
        install_trace(actor)
        previous = actor.metrics_report
        def metrics(instance):
            return dict(previous(), shared_task_executor=expected)
        actor.metrics_report = MethodType(metrics, actor)
        return actor, identity
