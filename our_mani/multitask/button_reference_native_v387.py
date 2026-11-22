from copy import deepcopy
from pathlib import Path
from types import MethodType
from .button_author_native_v359 import model_preflight as original_preflight
from .button_author_native_v359 import ButtonIntentNativeBackend as OriginalBackend
from .button_author_transit_v358 import SCHEMA as OLD_SCHEMA
from .button_reference_transit_v386 import BoundedApproach, SCHEMA
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha


def binding(root, protocol, task):
    if task not in ('push_moving_button', 'push_moving_button_high_speed'):
        raise ValueError('button reference consistency policy only')
    return dict(schema='r10_button_prefix_and_stream_same_reference4_gate_v387', task=task,
        sources_sha256={name: file_sha(Path(root)/name) for name in (
            'our_mani/multitask/button_reference_native_v387.py',
            'our_mani/multitask/button_reference_transit_v386.py')},
        full_reference_remaining_and_offset_stop_use_same_original_reference4_gate=True,
        actual_state_Hermite_gate_and_motor_parameters6_unchanged=True,
        source_author_stage_command_fit_collision_native_success_and10s_unchanged=True,
        formal_VAL_TEST=False)


def model_preflight(root, prepared):
    result = original_preflight(root, prepared)
    task = prepared['args'].task
    expected = binding(root, prepared['config'], task)
    if prepared['config'].get('button_reference_v387', {}).get(task) != expected:
        raise ValueError('freeze actual button reference gate consistency')
    checked_subset(root, prepared['files'], {str(Path(root)/name): sha for name, sha in expected['sources_sha256'].items()})
    return dict(result, button_reference_v387=expected)


class ButtonIntentNativeBackend(OriginalBackend):
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        m = actor.continuous_reference_manager
        raw = deepcopy(m.approach_transit.normalized)
        if raw['version'] != OLD_SCHEMA:
            raise ValueError('expected frozen preceding button transit')
        raw['version'] = SCHEMA
        m.approach_transit = BoundedApproach(m, raw)
        before = actor.metrics_report
        expected = binding(Path(__file__).resolve().parents[2], prepared['config'], prepared['args'].task)
        def metrics(instance):
            return dict(before(), button_reference_v387=expected)
        actor.metrics_report = MethodType(metrics, actor)
        return actor, identity
