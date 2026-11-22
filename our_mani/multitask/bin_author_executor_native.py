"""Author-command approach guard, 3D bin approach/acquire and physical capture."""
from pathlib import Path
from types import MethodType
from copy import deepcopy
import json
import numpy as np
from .task_executor_registry import SharedTaskBackend
from .continuous_transit import install_transit
from .evaluation import file_sha
from .frozen_subset_batch_v132 import checked_subset
TASK='put_rubbish_in_moving_bin_high_speed'
TASKS=(TASK,)
SOURCES=('our_mani/multitask/bin_author_executor_native.py','our_mani/multitask/continuous_transit.py',
    'our_mani/multitask/bounded_task_transit.py')


def binding(root,config,task):
    if task!=TASK:raise ValueError('high-bin author approach pilot first')
    p=Path(root)/'artifacts/framework/multitask_r10/bin_author_approach_v427'/task/'model.json'
    model=json.loads(p.read_text())
    if (model['parent_count']!=100 or not model['all_parents_used_equally']
        or model['native_success_or_completion_used_for_source_selection'] or model['VAL_used'] or model['TEST_used']):
        raise ValueError('all100 raw TRAIN author approach commands required')
    return dict(schema='r10_high_bin_original_author_approach_command_and_3D_capture_prefix_v1',task=task,
        model=dict(path=str(p.relative_to(root)),sha256=file_sha(p)),replan_seconds=.2,
        three_dimensional_phases=['approach','acquire'],
        sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},
        approach_target_is_actual_TRAIN_author_command_not_measured_completion_point=True,
        original_author_approach_distance_m=model['author_approach_distance_m'],
        original_physical_capture_acknowledgement_and_other_phase_completion_unchanged=True,
        original_full_reference_actual_cartesian_collision_stop_and10s_gates_unchanged=True,formal_VAL_TEST=False)


def verify(root,prepared):
    expected=binding(root,prepared['config'],prepared['args'].task)
    if prepared['config'].get('bin_author_executor')!=expected:raise ValueError('freeze actual author bin adapter')
    files={str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()}
    files[str(Path(root)/expected['model']['path'])]=expected['model']['sha256']
    checked_subset(root,prepared['files'],files)
    return expected


def install_guard(actor,model):
    c=actor.controller;previous=c._phase_done;events=[]
    offset=np.asarray(model['target_offset_world'],float)
    def done(instance,state,target,*,public_geometry=False):
        goal=instance.goal
        if goal is None or goal.phase!='approach':return previous(state,target,public_geometry=public_geometry)
        try:
            if state.perception_metadata.get('bodies',{}).get(goal.reference_id,{}).get('motion_mode')!='external':return False
            current=instance.generator.observed_reference(state,goal.reference_id,goal.template,instance.origin)
            point=current.pose.position+offset
            error=float(np.linalg.norm(state.robot.pose.position-point))
            passed=error<=model['author_approach_distance_m']
            if passed:events.append(dict(time=float(state.robot.timestamp),event='original_author_wp0_distance_guard_passed',
                public_waypoint_position=point.tolist(),distance_m=error,physical_grasp_or_success_claimed=False))
            return passed
        except (ValueError,KeyError,IndexError):return False
    c._phase_done=MethodType(done,c)
    return events


class BinAuthorBackend(SharedTaskBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        root=Path(__file__).resolve().parents[2];expected=binding(root,prepared['config'],prepared['args'].task)
        path=root/expected['model']['path']
        if file_sha(path)!=expected['model']['sha256']:raise ValueError('TRAIN author target changed')
        events=install_guard(actor,json.loads(path.read_text()))
        install_transit(actor,phases=tuple(expected['three_dimensional_phases']),replan_seconds=expected['replan_seconds'])
        previous=actor.metrics_report
        def metrics(instance):return dict(previous(),bin_author_executor=dict(binding=expected,events=deepcopy(events)))
        actor.metrics_report=MethodType(metrics,actor)
        return actor,identity
