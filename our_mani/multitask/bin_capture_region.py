"""Stop inside the original physical capture region without entering the table.

Control targets and physical capture geometry are separate: the original
unshifted current canonical pose still governs closing and acquire completion.
"""
from pathlib import Path
from dataclasses import replace,asdict
from types import MethodType
from copy import deepcopy
import numpy as np
from .interaction import CURRENT,COMMAND_GOAL,GripperTimingConfig
from .evaluation import file_sha
TASK='put_rubbish_in_moving_bin_high_speed'


def capture_template(template,shift,rule_sha):
    if (template.phase!='acquire' or template.reference_kind!='translation_only'
        or template.reference_mode!='external' or template.reference_time_basis!=CURRENT or template.lag_seconds!=0.):
        raise ValueError('current external translation-only acquire template required')
    positions=template.positions.copy();positions[:,2]+=shift
    canonical=template.at(1.)[0]
    return replace(template,positions=positions,reference_time_basis=COMMAND_GOAL,
        reference_time_rule_sha256=rule_sha,completion_geometry_xyzw=canonical.xyzw())


def binding(root,config,task):
    if task!=TASK:raise ValueError('explicit high-bin capture-region pilot')
    physical=asdict(GripperTimingConfig(.02,.1))
    if config['gripper_timing_config']!=physical:raise ValueError('original physical grasp gate required')
    return dict(schema='r10_bin_control_endpoint_inside_original_canonical_capture_region_v1',task=task,
        source_sha256=file_sha(Path(root)/'our_mani/multitask/bin_capture_region.py'),
        original_physical_grasp=physical,control_vertical_shift_m=.5*physical['close_position_tolerance_m'],
        control_offset_is_half_original_physical_position_tolerance=True,
        unshifted_current_canonical_pose_still_governs_close_and_completion=True,
        reference_control_semantics_explicitly_changed_to_committed_command_goal=True,
        stopped_actual_close_ack_and_unique_public_contact_still_required=True,
        original_fitting_tolerances_grasp_physics_collision_motion_and10s_unchanged=True,formal_VAL_TEST=False)


def install(actor,config,root,task):
    expected=binding(root,config,task)
    if config.get('bin_capture_region')!=expected:raise ValueError('freeze explicit capture region controller')
    c,m=actor.controller,actor.continuous_reference_manager
    if c.goal is not None or c.feedback_step_calls or m.confirmations:raise ValueError('install before execution')
    if asdict(c.gripper_timing_config)!=expected['original_physical_grasp']:raise ValueError('physical close gate changed')
    changed=[];templates=[]
    for t in c.library.templates:
        if t.phase=='acquire' and t.reference_mode=='external':
            converted=capture_template(t,expected['control_vertical_shift_m'],expected['source_sha256'])
            templates.append(converted)
            changed.append(dict(phase=t.phase,source_terminal_pose=t.at(1.)[0].xyzw().tolist(),
                actual_control_terminal_pose=converted.at(1.)[0].xyzw().tolist(),
                unchanged_physical_canonical_pose=converted.completion_geometry_xyzw.tolist()))
        else:templates.append(t)
    if len(changed)!=1:raise ValueError('one declared external physical acquire model required')
    c.library.templates=tuple(templates)
    previous=actor.metrics_report
    def metrics(instance):return dict(previous(),bin_capture_region=dict(binding=expected,templates=deepcopy(changed),
        measured_task_success_not_overridden=True))
    actor.metrics_report=MethodType(metrics,actor)
    return changed
