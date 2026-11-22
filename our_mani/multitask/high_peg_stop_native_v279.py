"""Original highpeg initial sequence plus additional per-reference stopping proofs."""
from pathlib import Path
from .high_peg_native_v245 import model_preflight as original_preflight
from .high_peg_native_v245 import SharedJoint16NativeBackend as OriginalBackend
from . import reference_stop_guard_v274 as policy
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha


def model_preflight(root,prepared):
    result=original_preflight(root,prepared);task=prepared['args'].task
    expected=policy.binding(root,prepared['config'],task)
    if prepared['config'].get('reference_stop_guard_v274',{}).get(task)!=expected:raise ValueError('actual next-stop policy differs')
    files={str(Path(root)/name):sha for name,sha in expected['sources_sha256'].items()}
    files[str(Path(__file__))]=file_sha(__file__);checked_subset(root,prepared['files'],files)
    return dict(result,reference_stop_guard_v274=expected)


class SharedJoint16NativeBackend(OriginalBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        policy.install(actor,prepared['config'],Path(__file__).resolve().parents[2],prepared['args'].task)
        return actor,identity
