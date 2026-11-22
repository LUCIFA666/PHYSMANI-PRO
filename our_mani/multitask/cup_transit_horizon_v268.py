"""Longer original fully checked transit prefixes within the same causal horizon."""
from copy import deepcopy
from pathlib import Path
from types import MethodType
from .bounded_transit_v180 import BoundedApproach,SCHEMA
from .evaluation import file_sha

TASKS=('place_cups_on_rotating_frame','place_cups_on_rotating_frame_high_speed')


def binding(root,protocol,task):
    if task not in TASKS:raise ValueError('only the two cup placement tasks')
    previous=protocol['cup_transit_v181'][task]
    segment=.8;step=min(.4,.5*protocol['mpc_config']['maximum_linear_speed']*segment)
    if abs(previous['maximum_proposal_step_m']-.2)>1e-12 or abs(step-.32)>1e-12:raise ValueError('expected original .5s/.2m prefix policy')
    return dict(schema='r10_cup_point8_second_fully_checked_transit_prefix_v268',task=task,
        sources_sha256={'our_mani/multitask/cup_transit_horizon_v268.py':file_sha(__file__)},
        previous_segment_seconds=.5,actual_segment_seconds=segment,
        previous_maximum_proposal_step_m=.2,actual_maximum_proposal_step_m=step,
        maximum_step_rule='half original MPC linear speed times segment duration, capped by original.4 config limit',
        same_original_bounded3D_actual_reference_cartesian_collision_and_offset_brake_checks=True,
        forecast_horizon_and_absolute_phase_and_global10s_deadlines_unchanged=True,
        actual_motor_compliance_still_measured_not_guaranteed=True,
        model_fit_physical_capture_and_native_success_unchanged=True,formal_VAL_TEST=False)


def install(actor,protocol,root,task):
    expected=binding(root,protocol,task)
    if protocol.get('cup_transit_horizon_v268',{}).get(task)!=expected:raise ValueError('freeze the actual longer prefix policy')
    c,m=actor.controller,actor.continuous_reference_manager
    if c.goal is not None or c.feedback_step_calls or m.confirmations:raise ValueError('install before first control step')
    raw=deepcopy(m.approach_transit.normalized)
    if raw['segment_seconds']!=.5 or raw['maximum_TCP_step_m']!=.2:raise ValueError('unexpected prior active prefix config')
    raw.update(version=SCHEMA,segment_seconds=expected['actual_segment_seconds'],maximum_TCP_step_m=expected['actual_maximum_proposal_step_m'])
    m.approach_transit=BoundedApproach(m,raw)
    previous=actor.metrics_report
    def metrics(instance):return dict(previous(),cup_transit_horizon_v268=dict(binding=deepcopy(expected),
        actual_prefix_configuration=deepcopy(m.approach_transit.normalized),
        earlier_v181_point5_point2_heuristic_superseded=True))
    actor.metrics_report=MethodType(metrics,actor)
