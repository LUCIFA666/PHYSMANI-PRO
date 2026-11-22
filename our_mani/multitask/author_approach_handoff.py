"""Use the author's public-distance stage guard, retaining physical grasp gates."""
from copy import deepcopy
from pathlib import Path
from types import MethodType
import numpy as np
from .reference_time import COMMAND_GOAL
from .specs import TASK_SPECS
from .evaluation import file_sha
TASKS=('beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed')


def binding(root,config,task):
    if task not in TASKS:raise ValueError('explicit author buzz stage adapter required')
    spec=TASK_SPECS[task]
    return dict(schema='r10_author_buzz_approach_public_canonical_distance_handoff_v1',
        task=task,approach_distance_m=.1,author_native_source_sha256=spec.native_source_sha256,
        source_sha256=file_sha(Path(root)/'our_mani/multitask/author_approach_handoff.py'),
        interaction=config['learned_artifacts'][task]['interaction'],
        public_current_TRAIN_canonical_waypoint_required=True,
        author_approach_guard_is_position_only=True,minimum_fitted_curve_duration_is_not_a_stage_guard=True,
        acquire_completion_grasp_collision_and_native_success_unchanged=True,
        extra_physics_camera_or_native_goal_reads=False,formal_VAL_TEST=False)


def install(actor,config,root,task):
    expected=binding(root,config,task)
    if config.get('author_approach_handoff',{}).get(task)!=expected:raise ValueError('freeze author stage adapter')
    c=actor.controller;m=actor.continuous_reference_manager
    if c.goal is not None or c.feedback_step_calls or m.confirmations:raise ValueError('install before execution')
    previous=c._phase_done;events=[]
    def done(instance,state,target,*,public_geometry=False):
        goal=instance.goal
        if goal is None or goal.phase!='approach':return previous(state,target,public_geometry=public_geometry)
        if goal.template.reference_time_basis!=COMMAND_GOAL or not public_geometry:return False
        # The caller has already reconstructed this target from fresh public
        # RGBD and the unchanged TRAIN canonical geometry.
        distance=float(np.linalg.norm(state.robot.pose.position-target.position))
        passed=distance<=expected['approach_distance_m']
        if passed:events.append(dict(time=float(state.robot.timestamp),distance_m=distance,
            original_phase_epoch=float(goal.selected_at),original_phase_deadline=float(goal.selected_at+goal.phase_budget),
            author_distance_guard_passed=True,grasp_or_task_success_claimed=False))
        return passed
    c._phase_done=MethodType(done,c)
    metrics_before=actor.metrics_report
    def metrics(instance):return dict(metrics_before(),author_approach_handoff=dict(binding=expected,events=deepcopy(events)))
    actor.metrics_report=MethodType(metrics,actor)
