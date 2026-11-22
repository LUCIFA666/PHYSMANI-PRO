"""Enter press only after reaching the original public approach action target.

Intersect the author's 0.3m state guard with the fitted approach action caps.
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
    return dict(schema='r10_button_approach_action_capture_before_press_v141',
        source=dict(path='our_mani/multitask/button_approach_capture_v141.py',
            sha256=file_sha(root/'our_mani/multitask/button_approach_capture_v141.py')),
        guard='original author distance guard AND fitted approach action position/orientation caps on current public target',
        native_press_success_unchanged=True,completed_skill_support_claimed=False,
        original_reference_reserve_and_actual_motion_collision_gates_unchanged=True,
        phase_budget_or_10s_clock_extended=False,TRAIN_only=True,formal_VAL_TEST=False)


def install(actor):
    c=actor.controller;events=[]
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
            admitted=bool(pe<=goal.template.position_tolerance_m and re<=goal.template.orientation_tolerance_rad)
            events.append(dict(time=float(state.robot.timestamp),approach_position_error_m=pe,
                approach_orientation_error_rad=re,position_cap_m=goal.template.position_tolerance_m,
                orientation_cap_rad=goal.template.orientation_tolerance_rad,enter_press_admitted=admitted,
                original_author_distance_guard_passed=True,verified_completed_skill_claimed=False))
            return admitted
        except (KeyError,ValueError):return False
    c._phase_done=MethodType(phase_done,c);before=actor.metrics_report
    def metrics(instance):return dict(before(),button_approach_capture_v141=dict(events=deepcopy(events),
        native_press_completion_claimed=False,action_target_caps_are_not_actual_completion_supervision=True))
    actor.metrics_report=MethodType(metrics,actor)


def model_preflight(root,prepared):
    from scripts.evaluate_multitask_case_r10 import locked_digest
    result=original_preflight(root,prepared);expected=binding(root)
    if prepared['config'].get('button_approach_capture_v141')!=expected:raise ValueError('freeze approach-before-press guard')
    if locked_digest(Path(root)/expected['source']['path'],prepared['files'])!=expected['source']['sha256']:
        raise ValueError('action-guard source omitted from lock')
    return dict(result,button_approach_capture_v141=expected)


class ButtonIntentNativeBackend(OriginalBackend):
    def actor(self,prepared,env,descriptions):
        if prepared['args'].task not in TASKS:raise ValueError('only fixed original TRAIN button tasks')
        if prepared['config'].get('button_approach_capture_v141')!=binding(Path(__file__).resolve().parents[2]):
            raise ValueError('approach action guard binding differs')
        actor,identity=super().actor(prepared,env,descriptions);install(actor)
        return actor,identity
