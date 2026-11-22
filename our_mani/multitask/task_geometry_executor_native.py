"""Common renewal and diagnostics with the original task geometry retained."""
from pathlib import Path
from types import MethodType
import json
from . import button_reference_native_v387 as button
from . import bin_physical_native_v235 as bin_native
from .receding_task_transit import install as install_transit
from .moving_reference import install_moving_reference
from .button_standoff import install as install_standoff
from .executor_trace import install_trace
from .evaluation import file_sha
from .frozen_subset_batch_v132 import checked_subset

BUTTONS = ('push_moving_button', 'push_moving_button_high_speed')
TASKS = BUTTONS + ('put_rubbish_in_moving_bin',)
SOURCES = ('our_mani/multitask/task_geometry_executor_native.py',
    'our_mani/multitask/receding_task_transit.py', 'our_mani/multitask/continuous_transit.py',
    'our_mani/multitask/moving_reference.py', 'our_mani/multitask/button_standoff.py',
    'our_mani/multitask/executor_trace.py')


def binding(root, protocol, task):
    if task not in TASKS:
        raise ValueError('task geometry pilot supports buttons and ordinary bin')
    return dict(schema='r10_common_executor_original_task_geometry_v1', task=task,
        replan_seconds=.2, moving_reference_phases=['press'] if task in BUTTONS else [],
        source_geometry_prior=protocol['learned_artifacts'][task]['geometry'],
        button_standoff=task in BUTTONS, original_transit_geometry_preserved=True,
        sources_sha256={name:file_sha(Path(root)/name) for name in SOURCES},
        original_native_success_and_actual_motion_criteria_retained=True, formal_release=False)


def verify_overlay(root, prepared):
    task = prepared['args'].task
    expected = binding(root, prepared['config'], task)
    if prepared['config'].get('task_geometry_executor', {}).get(task) != expected:
        raise ValueError('source-bound task geometry executor required')
    checked_subset(root, prepared['files'],
        {str(Path(root)/name):sha for name,sha in expected['sources_sha256'].items()})
    return expected


class TaskGeometryBackend:
    def __init__(self, task):
        if task not in TASKS:
            raise ValueError('unsupported task geometry pilot')
        self.original = (button.ButtonIntentNativeBackend() if task in BUTTONS
            else bin_native.SharedJoint16NativeBackend())

    def __getattr__(self, name):
        return getattr(self.original, name)

    def actor(self, prepared, env, descriptions):
        actor, identity = self.original.actor(prepared, env, descriptions)
        root = Path(__file__).resolve().parents[2]
        expected = binding(root, prepared['config'], prepared['args'].task)
        install_transit(actor, replan_seconds=expected['replan_seconds'])
        if expected['moving_reference_phases']:
            install_moving_reference(actor, phases=tuple(expected['moving_reference_phases']),
                replan_seconds=expected['replan_seconds'])
            pointer = expected['source_geometry_prior']; path = root/pointer['path']
            if file_sha(path) != pointer['sha256']:
                raise ValueError('shared TRAIN contact geometry changed')
            install_standoff(actor, json.loads(path.read_text()),
                control_interval=expected['replan_seconds'])
        install_trace(actor)
        previous = actor.metrics_report
        def metrics(instance):
            return dict(previous(), task_geometry_executor=expected)
        actor.metrics_report = MethodType(metrics, actor)
        return actor, identity
