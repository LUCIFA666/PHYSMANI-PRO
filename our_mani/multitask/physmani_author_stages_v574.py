"""Original author stage guards for bins/cup placement on public TRAIN geometry.

Regression fit caps remain training quality checks. They do not replace the
author's runtime position thresholds. Native task outcome remains parent-only.
"""
import ast
from copy import deepcopy
from pathlib import Path
from .evaluation import file_sha
from .specs import TASK_SPECS
from .physmani_skill_policy_v548 import SkillGeometryIntents,errors

RULES={
 'put_rubbish_in_moving_bin':dict(thresholds={'th_wp0':.1,'th_wp1':.01,'th_wp2':.05,'th_wp3':.05},
     phase_positions={'approach':.1,'lift':.05,'deposit':.05},close_distance=.01,
     one_decision_phases=[],release_phases=['deposit']),
 'place_cups_on_rotating_frame':dict(thresholds={'th_w0':.1,'th_w3':.1,'th_w4':.02,'th_w5':.03},
     phase_positions={'approach':.1,'transport':.1,'align':.02,'hang':.03},close_distance=.02,
     one_decision_phases=['lift'],release_phases=['hang'])}


def binding(root,native_repo,task):
    spec=TASK_SPECS[task];rule=RULES[spec.family];path=Path(native_repo)/spec.native_source_relative_path
    if file_sha(path)!=spec.native_source_sha256:raise ValueError('original author task source changed')
    tree=ast.parse(path.read_text());expert=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_expert_info')
    thresholds={n.targets[0].id:ast.literal_eval(n.value) for n in ast.walk(expert)
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
        and n.targets[0].id.startswith('th_')}
    if thresholds!=rule['thresholds']:raise ValueError('author waypoint thresholds differ')
    return dict(schema='r10_public_original_author_stage_guards_v574',task=task,
        native_source_sha256=spec.native_source_sha256,**deepcopy(rule),position_only_author_guards=True,
        source_sha256=file_sha(Path(root)/'our_mani/multitask/physmani_author_stages_v574.py'),
        minimum_dispatched_ticks_for_one_decision=10,physical_close_maximum_m=.02,
        physical_close_orientation_rad=.1,original_model_fit_caps_and_native_success_unchanged=True)


class AuthorStageIntents(SkillGeometryIntents):
    def __init__(self,frontend,rule):
        super().__init__(frontend)
        expected=RULES[TASK_SPECS[self.task].family]
        if (rule.get('schema')!='r10_public_original_author_stage_guards_v574'
            or rule.get('task')!=self.task or any(rule.get(k)!=v for k,v in expected.items())
            or rule.get('minimum_dispatched_ticks_for_one_decision')!=10):
            raise ValueError('exact original author stage guards required')
        self.author_rule=deepcopy(rule)

    def _done(self,goal):
        phase=goal['phase'];state=self.frontend.state
        if phase=='acquire':return super()._done(goal)
        if phase in self.author_rule['one_decision_phases']:
            ticks=[e for e in self.gripper_events if e['phase']==phase and e['time']>=self.phase_epoch-1e-8
                   and e.get('actually_dispatched') is True]
            return bool(state.robot.contact and len(ticks)>=10)
        target=self._completion_target(goal)
        if target is None:return False
        if phase in self.author_rule['release_phases']:
            if not state.robot.gripper_open or state.robot.contact:return False
        distance=errors(state.robot,target)[0];threshold=self.author_rule['phase_positions'][phase]
        return distance<threshold if phase=='hang' else distance<=threshold

    def _advance(self):
        goal=self.selected
        if goal is None or not self._done(goal):return
        timed=goal['phase'] in self.author_rule['one_decision_phases']
        self.events.append(dict(event='source_author_stage_guard_passed',phase=goal['phase'],
            time=float(self.frontend.state.robot.timestamp),native_success_claimed=False,
            guard='one_original_counted_decision' if timed else 'current_public_geometry_and_original_stage_guard',
            physical_endpoint_arrival_claimed=not timed))
        self.phase_index+=1;self.frontend.actor.controller.phase_index=self.phase_index
        self.selected=self.committed=None;self.origin=self.phase_epoch=None
        self.finished=self.phase_index==len(self.phase_order)

    def qualify_opening(self,requested,current_requested):
        goal=self.selected;state=self.frontend.state
        if goal is not None and goal['phase']=='acquire' and requested<.5 and self.close_witness is None:
            target=self._completion_target(goal)
            distance=None if target is None else errors(state.robot,target)[0]
            if distance is None or distance>self.author_rule['close_distance']:
                return 1.,dict(time=float(state.robot.timestamp),phase='acquire',desired_opening=float(requested),
                    current_requested_opening=float(current_requested),qualified_opening=1.,
                    reason='defer_close_until_original_author_grasp_waypoint',current_position_error_m=distance,
                    author_close_distance_m=self.author_rule['close_distance'],physical2cm_point1rad_gate_retained=True),None
        if (goal is not None and goal['phase'] in self.author_rule['release_phases']
            and requested>=.5 and current_requested<.5):
            target=self._completion_target(goal);distance=angle=None
            if target is not None:distance,angle=errors(state.robot,target)
            threshold=self.author_rule['phase_positions'][goal['phase']]
            accepted=distance is not None and (distance<threshold if goal['phase']=='hang' else distance<=threshold)
            return float(accepted),dict(time=float(state.robot.timestamp),phase=goal['phase'],desired_opening=1.,
                current_requested_opening=float(current_requested),qualified_opening=float(accepted),
                reason='original_author_current_public_release_guard' if accepted else 'defer_release_until_original_author_guard',
                current_position_error_m=distance,current_orientation_error_rad=angle,
                author_release_distance_m=threshold,fit_tolerance_is_not_a_stage_guard=True),None
        return super().qualify_opening(requested,current_requested)

    def receipt(self):return dict(super().receipt(),original_author_stage_rule=deepcopy(self.author_rule))
