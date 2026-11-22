"""Enter press only after reaching the original public approach action target.

Intersect the author's0.3m guard with action error plus one admitted target-motion step.
This is a conservative action-policy change, not a completed-skill certificate;
the original native benchmark still owns press success and the 10s clock.
"""
from pathlib import Path
from types import MethodType
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from .button_intent_native_v101 import ButtonIntentNativeBackend as OriginalBackend
from .button_intent_native_v101 import model_preflight as original_preflight
from .button_intent_library_v71 import TASKS
from .interaction import _world_target
from .evaluation import file_sha


def binding(root):
    root=Path(root).resolve()
    return dict(schema='r10_button_approach_action_capture_before_press_v152',
        prefix_source=dict(path='our_mani/multitask/button_intent_prefix_v155.py',sha256=file_sha(root/'our_mani/multitask/button_intent_prefix_v155.py')),
        prefix_forecast_time='current segment end plus fitted lag, within the same causal horizon',
        source=dict(path='our_mani/multitask/button_action_band_v152.py',
            sha256=file_sha(root/'our_mani/multitask/button_action_band_v152.py')),
        guard='original author distance guard AND fitted action error plus one counted sensor period of admitted target motion on current public approach target',
        native_press_success_unchanged=True,completed_skill_support_claimed=False,
        original_reference_reserve_and_actual_motion_collision_gates_unchanged=True,
        phase_budget_or_10s_clock_extended=False,TRAIN_only=True,formal_VAL_TEST=False)


def install(actor):
    c=actor.controller;events=[]
    from .button_intent_prefix_v155 import BoundedApproach,SCHEMA
    manager=actor.continuous_reference_manager
    raw=deepcopy(manager.approach_transit.normalized);raw['version']=SCHEMA
    manager.approach_transit=BoundedApproach(manager,raw)
    if c.goal is not None or c.feedback_step_calls:raise ValueError('install action guard before first step')
    original=c._phase_done
    def phase_done(instance,state,target,*,public_geometry=False):
        if not original(state,target,public_geometry=public_geometry):return False
        goal=instance.goal
        if goal is None or goal.phase!='approach':return False
        try:
            body=instance.generator.observed_reference(state,goal.reference_id,goal.template,instance.origin)
            current=_world_target(body.pose,np.zeros(3),np.zeros(3),goal.template,1.)[0]
            pe=float(np.linalg.norm(state.robot.pose.position-current.position))
            re=float(Rotation.from_matrix(state.robot.pose.rotation@current.rotation.T).magnitude())
            config=instance.generator.config
            position_cap=goal.template.position_tolerance_m+config.maximum_linear_speed*config.dt
            orientation_cap=goal.template.orientation_tolerance_rad+config.maximum_angular_speed*config.dt
            admitted=bool(pe<=position_cap and re<=orientation_cap)
            events.append(dict(time=float(state.robot.timestamp),approach_position_error_m=pe,
                approach_orientation_error_rad=re,position_cap_m=position_cap,
                orientation_cap_rad=orientation_cap,original_action_position_error_cap_m=goal.template.position_tolerance_m,
                original_action_orientation_error_cap_rad=goal.template.orientation_tolerance_rad,one_counted_period_seconds=config.dt,enter_press_admitted=admitted,
                original_author_distance_guard_passed=True,verified_completed_skill_claimed=False))
            return admitted
        except (KeyError,ValueError):return False
    c._phase_done=MethodType(phase_done,c);before=actor.metrics_report
    def metrics(instance):return dict(before(),button_approach_capture_v152=dict(events=deepcopy(events),
        native_press_completion_claimed=False,action_target_caps_are_not_actual_completion_supervision=True))
    actor.metrics_report=MethodType(metrics,actor)


def model_preflight(root,prepared):
    from scripts.evaluate_multitask_case_r10 import locked_digest
    result=original_preflight(root,prepared);expected=binding(root)
    if prepared['config'].get('button_approach_capture_v152')!=expected:raise ValueError('freeze approach-before-press guard')
    for key in ('source','prefix_source'):
        if locked_digest(Path(root)/expected[key]['path'],prepared['files'])!=expected[key]['sha256']:
            raise ValueError('action-band or segment-forecast source omitted from lock')
    return dict(result,button_approach_capture_v152=expected)


class ButtonIntentNativeBackend(OriginalBackend):
    def actor(self,prepared,env,descriptions):
        if prepared['args'].task not in TASKS:raise ValueError('only fixed original TRAIN button tasks')
        if prepared['config'].get('button_approach_capture_v152')!=binding(Path(__file__).resolve().parents[2]):
            raise ValueError('approach action guard binding differs')
        actor,identity=super().actor(prepared,env,descriptions);install(actor)
        return actor,identity
