"""Common execution core plus explicit moving-frame button press control."""
from pathlib import Path
from types import MethodType
from .continuous_executor_native import ContinuousExecutorBackend, binding as common_binding, BUTTONS
from .moving_reference import install_moving_reference
from .executor_trace import install_trace
from .evaluation import file_sha
from .frozen_subset_batch_v132 import checked_subset


def binding(root, protocol, task):
    if task not in BUTTONS:
        raise ValueError('moving press pilot requires a button task')
    return dict(schema='r10_common_moving_button_control_v1', task=task,
        moving_phases=['press'], replan_seconds=.2,
        source_offset_model_unchanged=True, explicit_command_time_control_change=True,
        source_phase_guard_and_native_success_unchanged=True,
        sources_sha256={name: file_sha(Path(root)/name) for name in (
            'our_mani/multitask/moving_button_native.py',
            'our_mani/multitask/moving_reference.py',
            'our_mani/multitask/executor_trace.py')}, formal_release=False)


def verify_overlay(root, prepared):
    task, config = prepared['args'].task, prepared['config']
    common, expected = common_binding(root, config, task), binding(root, config, task)
    if config.get('continuous_executor', {}).get(task) != common or config.get('moving_button', {}).get(task) != expected:
        raise ValueError('source-bound common and moving button policies required')
    checks = {**common['sources_sha256'], **expected['sources_sha256']}
    checked_subset(root, prepared['files'], {str(Path(root)/name): sha for name, sha in checks.items()})
    return dict(common=common, moving_button=expected)


class MovingButtonBackend(ContinuousExecutorBackend):
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        expected = binding(Path(__file__).resolve().parents[2], prepared['config'], prepared['args'].task)
        install_moving_reference(actor, phases=tuple(expected['moving_phases']),
            replan_seconds=expected['replan_seconds'])
        install_trace(actor)
        previous = actor.metrics_report
        def metrics(instance):
            return dict(previous(), moving_button=expected)
        actor.metrics_report = MethodType(metrics, actor)
        return actor, identity
