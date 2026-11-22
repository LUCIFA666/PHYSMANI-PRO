"""Normal-ring common executor with explicit measured-grasp placement targets."""
from pathlib import Path
from .task_executor_registry import SharedTaskBackend
from .grasp_frame_compensation_native_compat import binding,install
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha
SOURCES=('our_mani/multitask/grasp_compensated_executor_compat.py','our_mani/multitask/grasp_frame_compensation_native_compat.py')


def verify(root,prepared):
    expected=binding(root,prepared['config'],prepared['args'].task)
    if prepared['config'].get('grasp_frame_compensation')!=expected:raise ValueError('freeze grasp correction policy')
    files={str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()}
    files.update({str(Path(root)/p):file_sha(Path(root)/p) for p in SOURCES})
    files[str(Path(root)/expected['prior']['path'])]=expected['prior']['sha256']
    checked_subset(root,prepared['files'],files)
    return expected


class GraspCompensatedBackend(SharedTaskBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        install(actor,prepared['config'],Path(__file__).resolve().parents[2],prepared['args'].task)
        return actor,identity
