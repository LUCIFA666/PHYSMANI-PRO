"""Shared16 bin consumer with explicit physical acquire, original other gates."""
from pathlib import Path
from .joint16_native_v173 import model_preflight as original_preflight
from .joint16_native_v173 import SharedJoint16NativeBackend as OriginalBackend
from . import bin_physical_acquire_v234 as policy
from .evaluation import file_sha
from .frozen_subset_batch_v132 import checked_subset


def model_preflight(root, prepared):
    result = original_preflight(root, prepared)
    expected = policy.binding(root, prepared['config'], prepared['args'].task)
    if prepared['config'].get('bin_physical_acquire_v234', {}).get(prepared['args'].task) != expected:
        raise ValueError('actual physical bin policy differs from frozen binding')
    files = {str(Path(root)/name): sha for name, sha in expected['sources_sha256'].items()}
    files[str(Path(__file__))] = file_sha(__file__)
    checked_subset(root, prepared['files'], files)
    return dict(result, bin_physical_acquire_v234=expected)


class SharedJoint16NativeBackend(OriginalBackend):
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        policy.install(actor, prepared['config'], Path(__file__).resolve().parents[2], prepared['args'].task)
        return actor, identity
