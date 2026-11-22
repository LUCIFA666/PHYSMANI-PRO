"""Three-dimensional ball approach and acknowledged original physical capture."""
from pathlib import Path
from dataclasses import asdict
from types import MethodType
from copy import deepcopy
from .task_executor_registry import SharedTaskBackend
from .continuous_transit import install_transit
from .cup_physical_acquire_v167 import install_confirmation
from .bin_physical_acquire_v234 import install_completion
from .interaction import GripperTimingConfig
from .frozen_subset_batch_v132 import checked_subset
from .evaluation import file_sha
TASKS=('moving_basketball_in_hoop','moving_basketball_in_hoop_high_speed')
SOURCES=('our_mani/multitask/basketball_executor_native.py','our_mani/multitask/continuous_transit.py',
    'our_mani/multitask/bounded_task_transit.py','our_mani/multitask/cup_physical_acquire_v167.py',
    'our_mani/multitask/bin_physical_acquire_v234.py')


def binding(root,config,task):
    if task not in TASKS:raise ValueError('explicit ball task adapter')
    prior=config['close_completion_gate_v68'][task]
    if prior['original_grasp_gate']!=asdict(GripperTimingConfig(.02,.1)):raise ValueError('original physical grasp required')
    return dict(schema='r10_ball_3D_prefix_original_physical_capture_v1',task=task,
        replan_seconds=.2,three_dimensional_phases=['approach'],previous_extra_fitted_close_gate=prior['effective_close_AND_stopped_capture_gate'],
        original_physical_close_gate=prior['original_grasp_gate'],
        sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},
        approach_has_complete_actual_reference_cartesian_collision_and_offset_stop_proofs=True,
        physical_acquire='elapsed fitted duration; original SE3; actually dispatched stopped close; unique public held identity',
        learned_curves_original_fitting_caps_and_native_success_unchanged=True,
        total10s_and_original_actual_motion_limits_unchanged=True,formal_VAL_TEST=False)


def verify(root,prepared):
    expected=binding(root,prepared['config'],prepared['args'].task)
    if prepared['config'].get('basketball_executor')!=expected:raise ValueError('freeze actual ball executor')
    checked_subset(root,prepared['files'],{str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()})
    return expected


class BasketballBackend(SharedTaskBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        expected=binding(Path(__file__).resolve().parents[2],prepared['config'],prepared['args'].task)
        install_transit(actor,phases=tuple(expected['three_dimensional_phases']),replan_seconds=expected['replan_seconds'])
        c,m=actor.controller,actor.continuous_reference_manager
        if asdict(c.gripper_timing_config)!=expected['previous_extra_fitted_close_gate']:
            raise ValueError('unexpected previous capture gate')
        c.gripper_timing_config=GripperTimingConfig(**expected['original_physical_close_gate'])
        witnesses,events={},[];install_confirmation(m,witnesses,events);install_completion(c,witnesses,events)
        previous=actor.metrics_report
        def metrics(instance):return dict(previous(),basketball_executor=dict(binding=expected,
            actual_close_witnesses=deepcopy(witnesses),events=deepcopy(events),
            earlier_close_intersection_explicitly_superseded=True))
        actor.metrics_report=MethodType(metrics,actor)
        return actor,identity
