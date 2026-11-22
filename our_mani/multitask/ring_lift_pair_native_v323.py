"""Bounded metric-only collision witnesses around the frozen309 ring policy."""
from pathlib import Path
from types import MethodType
from .high_peg_support_native_v310 import model_preflight as original_preflight
from .high_peg_support_native_v310 import SharedJoint16NativeBackend as OriginalBackend
from .public_collision_diagnostic_v6 import CollisionObserver, binding
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha


def model_preflight(root, prepared):
    result = original_preflight(root, prepared)
    expected = binding(root, prepared['config']['baseline_repo'])
    if prepared['config'].get('ring_lift_pairs_v323') != expected:
        raise ValueError('freeze actual bounded read-only lift collision observer')
    files = {str(Path(root) / p['path']): p['sha256'] for p in
             (expected['source'], expected['arm_API'])}
    files[str(Path(__file__))] = file_sha(__file__)
    checked_subset(root, prepared['files'], files)
    return dict(result, ring_lift_pairs_v323=expected)


class SharedJoint16NativeBackend(OriginalBackend):
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        observer = CollisionObserver(actor, env)
        previous = actor.metrics_report
        def metrics(instance):
            return dict(previous(), ring_lift_pairs_v323=observer.receipt())
        actor.metrics_report = MethodType(metrics, actor)
        return actor, identity
