from pathlib import Path
from .cup_horizon_native_v269 import model_preflight as original_preflight
from .cup_horizon_native_v269 import NormalCupSlackBackend as OriginalNormal, HighCupSlackBackend as OriginalHigh
from . import cup_current_tracking_v377 as policy
from . import reference_stop_guard_v274 as stop
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha


def model_preflight(root, prepared):
    result = original_preflight(root, prepared)
    task = prepared['args'].task
    files = {str(Path(__file__)): file_sha(__file__)}
    proofs = {}
    for key, module in [('cup_current_tracking_v377', policy), ('reference_stop_guard_v274', stop)]:
        expected = module.binding(root, prepared['config'], task)
        if prepared['config'].get(key, {}).get(task) != expected:
            raise ValueError('actual cup tracking policy binding differs: '+key)
        files.update({str(Path(root)/name): sha for name, sha in expected['sources_sha256'].items()})
        proofs[key] = expected
    checked_subset(root, prepared['files'], files)
    return dict(result, cup_tracking_policies_v378=proofs)


def install(actor, prepared):
    root = Path(__file__).resolve().parents[2]
    for module in (stop, policy):
        module.install(actor, prepared['config'], root, prepared['args'].task)


class NormalCupSlackBackend(OriginalNormal):
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        install(actor, prepared)
        return actor, identity


class HighCupSlackBackend(OriginalHigh):
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        install(actor, prepared)
        return actor, identity
