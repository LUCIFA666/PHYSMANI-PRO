"""Close only inside both the original grasp gate and fitted acquire gate.

Closing outside the stricter acquire geometry can make completion impossible:
with public contact the object is rigidly held at that residual. The stopped
capture and the physical close edge must use the same intersection of gates.
"""
from pathlib import Path
from dataclasses import asdict
from types import MethodType
from copy import deepcopy
from .evaluation import file_sha
from .interaction import GripperTimingConfig,InteractionLibrary

TASKS=('put_rubbish_in_moving_bin','put_rubbish_in_moving_bin_high_speed',
    'moving_basketball_in_hoop','moving_basketball_in_hoop_high_speed')
SCHEMA='r10_original_grasp_AND_fitted_acquire_completion_close_gate_v68'


def intersection(original,templates):
    if not templates or any(t.phase!='acquire' for t in templates):raise ValueError('actual fitted acquire templates required')
    return GripperTimingConfig(min(original.close_position_tolerance_m,*[t.position_tolerance_m for t in templates]),
        min(original.close_orientation_tolerance_rad,*[t.orientation_tolerance_rad for t in templates]))


def binding(root,protocol,task):
    root=Path(root).resolve()
    if task not in TASKS:raise ValueError('only current-frame bin/basketball acquire protocol')
    library=InteractionLibrary.load(root/protocol['learned_artifacts'][task]['interaction']['path'])
    templates=library.for_phase('acquire')
    if any(t.reference_time_basis!='current' for t in templates):raise ValueError('no command-intent/public-target conflation')
    original=GripperTimingConfig(**protocol['gripper_timing_config']);effective=intersection(original,templates)
    return dict(schema=SCHEMA,task=task,source=dict(path=str(Path(__file__).relative_to(root)),sha256=file_sha(__file__)),
        library=protocol['learned_artifacts'][task]['interaction'],original_grasp_gate=asdict(original),
        effective_close_AND_stopped_capture_gate=asdict(effective),
        position_orientation_grasp_gates_relaxed=False,fitted_phase_completion_changed=False,
        public_contact_still_required=True,original_stop_and_motion_gates_unchanged=True,
        native_labels_used=False,formal_release=False)


def install(actor,protocol,root,task):
    expected=binding(root,protocol,task)
    if protocol.get('close_completion_gate_v68',{}).get(task)!=expected:raise ValueError('freeze actual intersection gate')
    controller=actor.controller;manager=actor.continuous_reference_manager
    if manager.controller is not controller or controller.goal is not None or controller.feedback_step_calls:
        raise ValueError('install on final actual controller before first step')
    if asdict(controller.gripper_timing_config)!=expected['original_grasp_gate']:raise ValueError('unexpected preexisting close config')
    controller.gripper_timing_config=GripperTimingConfig(**expected['effective_close_AND_stopped_capture_gate'])
    # Both the actual controller close gate and manager.capture_at_current_grasp
    # read this final controller field. Do not replace completion or contact.
    previous=actor.metrics_report
    def metrics(instance):return dict(previous(),close_completion_gate_v68=deepcopy(expected))
    actor.metrics_report=MethodType(metrics,actor)
