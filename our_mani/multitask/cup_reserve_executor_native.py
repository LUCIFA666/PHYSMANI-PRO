"""Cup actor with reference4 reserve and its original moving-stream renewal."""
from pathlib import Path
from .task_executor_registry import SharedTaskBackend
from .cup_reference_reserve import binding,install,SOURCES as SERVICE_SOURCES,TASKS
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha
SOURCES=(*SERVICE_SOURCES,'our_mani/multitask/cup_reserve_executor_native.py')


def verify(root,prepared):
    expected=binding(root,prepared['config'],prepared['args'].task)
    if prepared['config'].get('cup_reference_reserve')!=expected:raise ValueError('freeze cup reserve')
    required={str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()}
    required[str(Path(__file__))]=file_sha(__file__);checked_subset(root,prepared['files'],required)
    return expected


class CupReserveBackend(SharedTaskBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        install(actor,prepared['config'],Path(__file__).resolve().parents[2],prepared['args'].task)
        return actor,identity
