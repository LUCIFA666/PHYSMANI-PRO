"""Allow source-qualified remove-cup command intentions to be reissued causally.

Uses the original retarget manager and all of its complete motion/collision
proofs. Candidate issue time, phase epoch/deadline, suffix budget, attempt and
learned template keep their original semantics. No task label is a controller
input and this adapter cannot mark a phase complete.
"""
from pathlib import Path
from types import MethodType
from copy import deepcopy
from .reference_time import COMMAND_GOAL
from .evaluation import file_sha
from .causal_retarget_interaction import CausalRetargetManager

TASKS=('remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed')
SCHEMA='r10_source_qualified_remove_causal_retarget_extension_v66'


def eligible(goal):
    return (goal.phase in ('approach','acquire') and goal.template.phase==goal.phase
        and goal.template.reference_time_basis==COMMAND_GOAL
        and goal.template.reference_kind=='observed_se3'
        and goal.template.reference_role in ('selected_cup','selected_spoke')
        and goal.template.reference_mode=='external')


def binding(root,protocol,task):
    root=Path(root).resolve()
    if task not in TASKS:raise ValueError('only the two source-qualified remove tasks')
    return dict(schema=SCHEMA,task=task,source=dict(path=str(Path(__file__).relative_to(root)),sha256=file_sha(__file__)),
        original_retarget_source=dict(path='our_mani/multitask/causal_retarget_interaction.py',sha256=file_sha(root/'our_mani/multitask/causal_retarget_interaction.py')),
        operation_qualification=protocol['remove_runtime_v48'][task]['qualification'],
        interaction=protocol['learned_artifacts'][task]['interaction'],
        extension='permit original manager to reissue external observed-SE3 approach/acquire COMMAND_GOAL intentions',
        original_template_phase_attempt_deadline_suffix_gates_preserved=True,phase_completion_claimed=False,
        source_canonical_public_completion_unchanged=True,native_dynamic_labels_used=False,formal_release=False)


def install(actor,protocol,root,task):
    expected=binding(root,protocol,task)
    if protocol.get('remove_retarget_v66',{}).get(task)!=expected:raise ValueError('freeze exact remove retarget extension')
    manager=actor.continuous_reference_manager.retarget
    if not isinstance(manager,CausalRetargetManager) or manager.epochs or manager.events:raise ValueError('install before first control step')
    original=manager._eligible
    def allow(instance,goal):return original(goal) or eligible(goal)
    manager._eligible=MethodType(allow,manager)
    previous=actor.metrics_report
    def metrics(instance):return dict(previous(),remove_retarget_v66=deepcopy(expected))
    actor.metrics_report=MethodType(metrics,actor)
