"""Buzz executor with author approach handoff and consistent acquire close gates."""
from pathlib import Path
from .buzz_command_executor_native import BuzzCommandBackend,verify as verify_original
from .grasp_completion_contract import binding as grasp_binding,install as install_grasp
from .author_approach_handoff import binding as handoff_binding,install as install_handoff
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha
SOURCES=('our_mani/multitask/buzz_handoff_executor_native.py',
    'our_mani/multitask/grasp_completion_contract.py','our_mani/multitask/author_approach_handoff.py')


def verify(root,prepared):
    verify_original(root,prepared);task=prepared['args'].task;config=prepared['config']
    expected_grasp=grasp_binding(root,config,task);expected_handoff=handoff_binding(root,config,task)
    if config.get('grasp_completion_contract',{}).get(task)!=expected_grasp:raise ValueError('freeze actual close intersection')
    if config.get('author_approach_handoff',{}).get(task)!=expected_handoff:raise ValueError('freeze actual author stage handoff')
    files={str(Path(root)/p):sha for p,sha in expected_grasp['sources_sha256'].items()}
    files.update({str(Path(root)/p):file_sha(Path(root)/p) for p in SOURCES})
    checked_subset(root,prepared['files'],files)
    return dict(grasp_completion_contract=expected_grasp,author_approach_handoff=expected_handoff)


class BuzzHandoffBackend(BuzzCommandBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        root=Path(__file__).resolve().parents[2];task=prepared['args'].task
        install_grasp(actor,prepared['config'],root,task)
        install_handoff(actor,prepared['config'],root,task)
        return actor,identity
