"""Source-bound common executor overlay on the already qualified task adapters."""
from pathlib import Path
from types import MethodType

from . import button_reference_native_v387 as button
from . import bin_physical_native_v235 as bin_native
from .continuous_transit import install_transit, SCHEMA
from .evaluation import file_sha
from .frozen_subset_batch_v132 import checked_subset

BUTTONS = ('push_moving_button', 'push_moving_button_high_speed')
BINS = ('put_rubbish_in_moving_bin',)
TASKS = BUTTONS + BINS


def binding(root, protocol, task):
    if task not in TASKS:
        raise ValueError('common executor pilot supports buttons and ordinary-bin regression')
    return dict(schema=SCHEMA, task=task, replan_seconds=.2,
        enabled_phases=['approach', 'press'] if task in BUTTONS else ['approach', 'acquire', 'lift', 'deposit'],
        sources_sha256={name: file_sha(Path(root)/name) for name in (
            'our_mani/multitask/continuous_executor_native.py',
            'our_mani/multitask/continuous_transit.py',
            'our_mani/multitask/bounded_task_transit.py')},
        actual_motor_and_task_success_unchanged=True, original_deadlines_preserved=True,
        full_actual_and_same_executor_reference_gates_required=True,
        physical_steps_owned_by_original_counted_dispatcher=True, formal_release=False)


def model_preflight(root, prepared):
    task = prepared['args'].task
    original = button if task in BUTTONS else bin_native
    result = original.model_preflight(root, prepared)
    expected = binding(root, prepared['config'], task)
    if prepared['config'].get('continuous_executor', {}).get(task) != expected:
        raise ValueError('freeze actual common executor source and configuration')
    checked_subset(root, prepared['files'],
        {str(Path(root)/name): sha for name, sha in expected['sources_sha256'].items()})
    return dict(result, continuous_executor=expected)


class ContinuousExecutorBackend:
    def __init__(self, task):
        if task not in TASKS:
            raise ValueError('unsupported common executor pilot task')
        original = button.ButtonIntentNativeBackend if task in BUTTONS else bin_native.SharedJoint16NativeBackend
        self.original = original()

    def __getattr__(self, name):
        return getattr(self.original, name)

    def actor(self, prepared, env, descriptions):
        actor, identity = self.original.actor(prepared, env, descriptions)
        expected = binding(Path(__file__).resolve().parents[2], prepared['config'], prepared['args'].task)
        install_transit(actor, phases=tuple(expected['enabled_phases']),
            replan_seconds=expected['replan_seconds'])
        previous = actor.metrics_report
        def metrics(instance):
            return dict(previous(), continuous_executor=expected)
        actor.metrics_report = MethodType(metrics, actor)
        return actor, identity
