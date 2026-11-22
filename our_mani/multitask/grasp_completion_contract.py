"""A physical close must fit both grasp limits and the acquire completion gate."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import MethodType
from .interaction import InteractionLibrary,GripperTimingConfig
from .close_completion_gate_v68 import intersection
from .evaluation import file_sha
TASKS=('beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed')


def binding(root,config,task):
    if task not in TASKS:raise ValueError('explicit command-frame acquire tasks required')
    pointer=config['learned_artifacts'][task]['interaction']
    if file_sha(Path(root)/pointer['path'])!=pointer['sha256']:raise ValueError('TRAIN interaction changed')
    library=InteractionLibrary.load(Path(root)/pointer['path'])
    original=GripperTimingConfig(**config['gripper_timing_config'])
    effective=intersection(original,library.for_phase('acquire'))
    sources=('our_mani/multitask/grasp_completion_contract.py',
        'our_mani/multitask/close_completion_gate_v68.py','our_mani/multitask/interaction.py')
    return dict(schema='r10_fitted_command_acquire_and_original_grasp_intersection_v1',task=task,
        interaction=pointer,original_grasp_gate=asdict(original),effective_close_gate=asdict(effective),
        sources_sha256={p:file_sha(Path(root)/p) for p in sources},
        public_current_COMMAND_GOAL_canonical_used_by_existing_controller=True,
        original_grasp_gates_never_relaxed=True,completion_geometry_and_tolerances_unchanged=True,
        public_contact_still_required=True,native_future_or_success_values_used=False,formal_VAL_TEST=False)


def install(actor,config,root,task):
    expected=binding(root,config,task)
    if config.get('grasp_completion_contract',{}).get(task)!=expected:
        raise ValueError('source-bound actual acquire close contract required')
    c,m=actor.controller,actor.continuous_reference_manager
    if c.goal is not None or c.feedback_step_calls or m.confirmations:
        raise ValueError('install before first counted control step')
    if asdict(c.gripper_timing_config)!=expected['original_grasp_gate']:
        raise ValueError('unexpected preexisting grasp gate')
    actual=intersection(c.gripper_timing_config,c.library.for_phase('acquire'))
    if asdict(actual)!=expected['effective_close_gate']:
        raise ValueError('runtime templates differ from frozen TRAIN source')
    c.gripper_timing_config=actual
    previous=actor.metrics_report
    def metrics(instance):return dict(previous(),grasp_completion_contract=deepcopy(expected))
    actor.metrics_report=MethodType(metrics,actor)
    return actual
