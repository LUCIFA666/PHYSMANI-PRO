"""Explicit physical-acquire and transport-completion cup development consumers."""
from pathlib import Path
from .cup_transport_native_v161 import NormalCupTransportBackend, HighCupTransportBackend
from .cup_transport_native_v161 import model_preflight as original_preflight
from . import cup_physical_acquire_v167 as policy
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha


def model_preflight(root, prepared):
    result = original_preflight(root, prepared)
    task = prepared['args'].task
    expected = policy.binding(root, prepared['config'], task)
    if prepared['config'].get('cup_physical_acquire_v167', {}).get(task) != expected:
        raise ValueError('actual cup physical acquire binding differs')
    files = {str(Path(root)/name): sha for name, sha in expected['sources_sha256'].items()}
    files[str(Path(__file__))] = file_sha(__file__)
    checked_subset(root, prepared['files'], files)
    return dict(result, cup_physical_acquire_v167=expected,
        extra_controller_intersection_relaxed=True, acquire_completion_policy_changed=True,
        native_physics_grasp_success_unchanged=True)


class NormalCupPhysicalBackend(NormalCupTransportBackend):
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        policy.install(actor, prepared['config'], Path(__file__).resolve().parents[2], prepared['args'].task)
        return actor, identity


class HighCupPhysicalBackend(HighCupTransportBackend):
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        policy.install(actor, prepared['config'], Path(__file__).resolve().parents[2], prepared['args'].task)
        return actor, identity
