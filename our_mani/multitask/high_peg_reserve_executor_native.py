"""High-peg actor with the same reference4 reserve used by the button executor."""
from pathlib import Path
from .task_executor_registry import SharedTaskBackend
from .reference_reserve_upgrade import binding,install,SOURCES as SERVICE_SOURCES,TASK
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha
TASKS=(TASK,)
SOURCES=(*SERVICE_SOURCES,'our_mani/multitask/high_peg_reserve_executor_native.py')


def verify(root,prepared):
    expected=binding(root,prepared['config'],prepared['args'].task)
    if prepared['config'].get('reference_reserve_upgrade')!=expected:raise ValueError('freeze high-peg reserve')
    required={str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()}
    required[str(Path(__file__))]=file_sha(__file__);checked_subset(root,prepared['files'],required)
    return expected


class HighPegReserveBackend(SharedTaskBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        install(actor,prepared['config'],Path(__file__).resolve().parents[2],prepared['args'].task)
        return actor,identity
