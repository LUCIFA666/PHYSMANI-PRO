"""Cup actor with original collision proofs and a low-speed acquire reserve."""
from pathlib import Path
from .cup_reserve_executor_native import CupReserveBackend
from .cup_contact_executor import verify as verify_contact
from .gripper_only_contact_native_compat import install as install_contact,binding as contact_binding
from .contact_speed_reserve import install as install_speed,binding as speed_binding
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha
from .cup_reference_reserve import TASKS
SOURCES=('our_mani/multitask/cup_contact_speed_executor.py',
    'our_mani/multitask/gripper_only_contact_native_compat.py','our_mani/multitask/contact_speed_reserve.py',
    'our_mani/multitask/lossless_case_archive.py')


def binding(root,config,task):
    return dict(task=task,contact=contact_binding(root),speed=speed_binding(root),
        sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},
        gripper_contact_policy_same_as439_with_static_names_captured_before_cleanup=True,
        complete_raw_evaluator_log_preserved_losslessly_after_cleanup=True,formal_VAL_TEST=False)


def verify(root,prepared):
    verify_contact(root,prepared);expected=binding(root,prepared['config'],prepared['args'].task)
    if prepared['config'].get('cup_contact_speed_executor')!=expected:raise ValueError('freeze contact speed reserve')
    checked_subset(root,prepared['files'],{str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()})
    return expected


class CupContactSpeedBackend(CupReserveBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        install_contact(actor,env);install_speed(actor)
        return actor,identity
