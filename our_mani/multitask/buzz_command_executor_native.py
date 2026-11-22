"""Common executor plus causal refresh of the original buzz command frames."""
from pathlib import Path
from .task_executor_registry import SharedTaskBackend
from .command_reference_service import install, binding, TASK_PHASES
from .evaluation import file_sha
from .frozen_subset_batch_v132 import checked_subset
SOURCES=('our_mani/multitask/buzz_command_executor_native.py',
    'our_mani/multitask/command_reference_service.py')


def verify(root,prepared):
    task=prepared['args'].task
    if task not in TASK_PHASES:raise ValueError('explicit buzz task adapter required')
    expected=binding(root,prepared['config'],task)
    if prepared['config'].get('command_reference_service')!=expected:
        raise ValueError('freeze declared command frame service')
    required={str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()}
    required.update({str(Path(root)/p):file_sha(Path(root)/p) for p in SOURCES})
    checked_subset(root,prepared['files'],required)
    return expected


class BuzzCommandBackend(SharedTaskBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        install(actor,prepared['config'],Path(__file__).resolve().parents[2],prepared['args'].task)
        return actor,identity
