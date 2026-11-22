"""Three-dimensional remove prefixes with acknowledged public grasp completion."""
from pathlib import Path
from .task_executor_registry import SharedTaskBackend,verify as verify_registry
from .remove_reference_reserve import binding as reserve_binding,install as install_reserve,SOURCES as RESERVE_SOURCES,TASKS
from .remove_command_service import binding as command_binding,install as install_command
from .remove_physical_capture import binding as capture_binding,install as install_capture
from .remove_approach_handoff import binding as handoff_binding,install as install_handoff
from .evaluation import file_sha
from .frozen_subset_batch_v132 import checked_subset

SOURCES=(*RESERVE_SOURCES,'our_mani/multitask/remove_command_service.py',
    'our_mani/multitask/remove_physical_capture.py','our_mani/multitask/remove_approach_handoff.py',
    'our_mani/multitask/remove_continuous_executor.py')


def binding(root,config,task):
    return dict(remove_reference_reserve=reserve_binding(root,config,task),
        remove_command_service=command_binding(root,config,task),
        remove_physical_capture={task:capture_binding(root,config,task)},
        remove_approach_handoff={task:handoff_binding(root,config,task)})


def verify(root,prepared):
    verify_registry(root,prepared)
    expected=binding(root,prepared['config'],prepared['args'].task)
    for key,value in expected.items():
        if prepared['config'].get(key)!=value:raise ValueError('freeze remove execution policy: '+key)
    checked_subset(root,prepared['files'],{str(Path(root)/p):file_sha(Path(root)/p) for p in SOURCES})
    return expected


class RemoveContinuousBackend(SharedTaskBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        root=Path(__file__).resolve().parents[2];task=prepared['args'].task
        install_reserve(actor,prepared['config'],root,task)
        install_command(actor,prepared['config'],root,task)
        install_capture(actor,prepared['config'],root,task)
        install_handoff(actor,prepared['config'],root,task)
        return actor,identity
