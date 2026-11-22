"""Task geometry executor with one reference/actual constraint authority."""
from pathlib import Path
from types import MethodType
from .task_geometry_executor_native import TaskGeometryBackend, binding as task_binding
from .reference_constraint_authority import install
from .evaluation import file_sha
from .frozen_subset_batch_v132 import checked_subset

SOURCES = ('our_mani/multitask/task_constraint_executor_native.py',
    'our_mani/multitask/reference_constraint_authority.py')


def binding(root, config, task):
    return dict(schema='r10_original_task_geometry_consistent_constraint_authority_v1',
        task_geometry=task_binding(root, config, task),
        sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},
        actual_native_acceleration_limit=6., original_reference_reserve_unchanged=True,
        TRAIN_development_only=True, formal_release=False)


def verify(root, prepared):
    expected = binding(root, prepared['config'], prepared['args'].task)
    if prepared['config'].get('task_constraint_executor',{}).get(prepared['args'].task) != expected:
        raise ValueError('freeze the complete constraint authority overlay')
    checked_subset(root, prepared['files'],
        {str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()})
    return expected


class TaskConstraintBackend(TaskGeometryBackend):
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        install(actor)
        previous = actor.metrics_report
        expected = binding(Path(__file__).resolve().parents[2],prepared['config'],prepared['args'].task)
        def metrics(instance):
            return dict(previous(), task_constraint_executor=expected)
        actor.metrics_report = MethodType(metrics,actor)
        return actor, identity
