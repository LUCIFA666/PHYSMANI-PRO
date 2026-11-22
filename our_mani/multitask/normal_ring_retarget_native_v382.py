from pathlib import Path
from .normal_ring_native_v367 import model_preflight as original_preflight
from .normal_ring_native_v367 import SharedJoint16NativeBackend as OriginalBackend
from . import ring_causal_retarget_v371 as policy
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha


def model_preflight(root, prepared):
    result = original_preflight(root, prepared)
    task = prepared['args'].task
    expected = policy.binding(root, prepared['config'], task)
    if prepared['config'].get('ring_causal_retarget_v371', {}).get(task) != expected:
        raise ValueError('actual ring reissue policy differs')
    files = {str(Path(root)/name): sha for name, sha in expected['sources_sha256'].items()}
    files[str(Path(__file__))] = file_sha(__file__)
    checked_subset(root, prepared['files'], files)
    return dict(result, ring_causal_retarget_v371=expected)


class SharedJoint16NativeBackend(OriginalBackend):
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        policy.install(actor, prepared['config'], Path(__file__).resolve().parents[2], prepared['args'].task)
        return actor, identity
