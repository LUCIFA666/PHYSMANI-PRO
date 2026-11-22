"""Restore the original public cup attachment input in the direct RML factory."""
from pathlib import Path
from .rml_component_factory_v643 import create_actor as _create_actor,cpu_preflight as _cpu_preflight
from .rml_cup_support_v707 import binding,install
from .evaluation import file_sha
from .frozen_subset_batch_v132 import checked_subset
CUPS=('place_cups_on_rotating_frame','place_cups_on_rotating_frame_high_speed')
SOURCES=('our_mani/multitask/rml_component_extensions_v708.py','our_mani/multitask/rml_cup_support_v707.py',
    'our_mani/multitask/cup_public_support_v148.py')


def verify(root,config,task):
    rule=binding(root,config,task)
    if config.get('RML_cup_public_support_v707',{}).get(task)!=rule:raise ValueError('freeze public held-cup measurement extension')
    return rule


def create_actor(root,env,config,method,descriptions,*,domain):
    actor,proof=_create_actor(root,env,config,method,descriptions,domain=domain)
    task=env.spec.task_name
    if task in CUPS:
        rule=verify(root,config,task);install(actor,config,root,task)
        proof=dict(proof,public_input_extensions=dict(cup_support_v707=rule))
    return actor,proof


def cpu_preflight(root,config,task,method,domain,files):
    proof=_cpu_preflight(root,config,task,method,domain,files)
    if task in CUPS:
        rule=verify(root,config,task)
        checked_subset(root,files,{str(Path(root)/n):file_sha(Path(root)/n) for n in SOURCES})
        proof=dict(proof,public_input_extensions=dict(cup_support_v707=rule))
    return proof
