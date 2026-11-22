"""Normal ring's own full TRAIN proof with checked support, align and identity."""
from pathlib import Path
from .joint16_native_v173 import model_preflight as original_preflight
from .joint16_native_v173 import SharedJoint16NativeBackend as OriginalBackend
from . import normal_ring_support_v366 as support
from . import reference_stop_guard_v274 as stop
from . import public_target_memory_v339 as memory
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha


def model_preflight(root, prepared):
    if prepared['args'].task != support.TASK or prepared['args'].split != 'train':
        raise ValueError('normal ring original fixed TRAIN development only')
    result = original_preflight(root, prepared)
    files = {str(Path(__file__)): file_sha(__file__)}
    proofs = {}
    for key, policy in [('normal_ring_support_v366', support),
                        ('reference_stop_guard_v274', stop), ('public_target_memory_v339', memory)]:
        expected = policy.binding(root, prepared['config'], support.TASK)
        if prepared['config'].get(key, {}).get(support.TASK) != expected:
            raise ValueError('actual normal ring policy binding differs: '+key)
        files.update({str(Path(root)/name): sha for name, sha in expected['sources_sha256'].items()})
        proofs[key] = expected
    checked_subset(root, prepared['files'], files)
    return dict(result, normal_ring_explicit_policy=proofs)


class SharedJoint16NativeBackend(OriginalBackend):
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        root = Path(__file__).resolve().parents[2]
        for policy in (stop, support, memory):
            policy.install(actor, prepared['config'], root, prepared['args'].task)
        return actor, identity
