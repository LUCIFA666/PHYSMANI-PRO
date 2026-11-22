"""Cup reserve actor with gripper-only contact permissions."""
from pathlib import Path
from .cup_reserve_executor_native import CupReserveBackend,verify as verify_reserve
from .gripper_only_contact import binding as contact_binding,install
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha
SOURCES=('our_mani/multitask/gripper_only_contact.py','our_mani/multitask/cup_contact_executor.py')


def binding(root,config,task):
    return dict(task=task,contact=contact_binding(root),sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},formal_VAL_TEST=False)


def verify(root,prepared):
    verify_reserve(root,prepared);expected=binding(root,prepared['config'],prepared['args'].task)
    if prepared['config'].get('cup_contact_executor')!=expected:raise ValueError('freeze gripper-only target contact')
    checked_subset(root,prepared['files'],{str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()})
    return expected


class CupContactBackend(CupReserveBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions);install(actor,env)
        return actor,identity
