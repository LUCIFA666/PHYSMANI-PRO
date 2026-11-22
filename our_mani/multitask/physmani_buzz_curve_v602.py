"""Retain the trained follow curve between the author's two Buzz waypoints.

A phase endpoint alone discards intermediate manipulation geometry. Advance
through the immutable TRAIN curve only from current public TCP distances, and
send each next pose through the same original RML/collision/gripper pipeline.
"""
import ast
from copy import deepcopy
from pathlib import Path
import numpy as np
from .specs import TASK_SPECS
from .evaluation import file_sha
from .physmani_skill_policy_v548 import SkillGeometryIntents,errors
from .physmani_geometry_goals_v593 import pose_goals
from .interaction import _reference_at,_world_target


def binding(root,native_repo,task):
    spec=TASK_SPECS[task]
    if spec.family!='beat_the_rotating_buzz':raise ValueError('explicit two Buzz tasks only')
    source=Path(native_repo)/spec.native_source_relative_path
    if file_sha(source)!=spec.native_source_sha256:raise ValueError('original Buzz task changed')
    expert=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='get_expert_info')
    values={n.targets[0].id:ast.literal_eval(n.value) for n in ast.walk(expert) if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('th_wp2','th_wp3_','th_wp3')}
    if values!={'th_wp2':.03,'th_wp3_':.03,'th_wp3':.03}:raise ValueError('unchanged author3cm follow/release guards required')
    return dict(schema='r10_public_trained_Buzz_curve_on_author_RML_v602',task=task,
        curve_phase='follow',original_source_waypoints=['wp2','wp3_'],current_public_progress_distance_m=.03,
        author_terminal_release_distance_m=.03,approach_lookahead_seconds=0.,other_lookahead_seconds=.5,
        immutable_TRAIN_curve_used=True,source_sha256=file_sha(Path(root)/'our_mani/multitask/physmani_buzz_curve_v602.py'),
        goal_helper_sha256=file_sha(Path(root)/'our_mani/multitask/physmani_geometry_goals_v593.py'),
        native_source_sha256=file_sha(source),physical_close_gate_unchanged=True,collision_checks_retained=True,
        measured_acceleration_is_diagnostic_only=True,native_success_and_failure_unchanged=True)


class BuzzCurveIntents(SkillGeometryIntents):
    def __init__(self,frontend,rule):
        super().__init__(frontend)
        if (rule.get('schema')!='r10_public_trained_Buzz_curve_on_author_RML_v602'
            or rule.get('task')!=self.task or rule.get('current_public_progress_distance_m')!=.03):
            raise ValueError('source-bound trained curve continuation required')
        self.curve_rule=deepcopy(rule);self.cursors={};self.curve_events=[]

    def candidates(self,phase):
        f=self.frontend;c=f.actor.controller;state=f.state
        rows=pose_goals(f,phase,origin=self.origin,lookahead=0. if phase=='approach' else .5,committed=self.committed)
        if phase!='follow':return rows
        result=[]
        for row in rows:
            t=row['template'];key=(t.anchor_id,row['reference_id'],row['manipulated_ids'])
            current=c.generator.observed_reference(state,row['reference_id'],t,self.origin)
            index=self.cursors.get(key,0)
            # Pass only a waypoint whose current public geometry was reached.
            # Predictions and elapsed time cannot stand in for this witness.
            while index<len(t.progress)-1:
                present=_world_target(current.pose,current.linear_velocity,current.angular_velocity,t,float(t.progress[index]))[0]
                if errors(state.robot,present)[0]>.03:break
                index+=1
            self.cursors[key]=index;progress=float(t.progress[index])
            ref,v,w=_reference_at(f.forecast,row['reference_id'],t,row['predicted_reference_time'],self.origin)
            target=_world_target(ref,v,w,t,progress)[0]
            present=_world_target(current.pose,current.linear_velocity,current.angular_velocity,t,progress)[0]
            result.append(dict(row,pose=target,present_pose=present,opening=0.,curve_progress=progress,
                curve_index=index,score=-float(np.linalg.norm(target.position-state.robot.pose.position))))
            self.curve_events.append(dict(time=float(state.robot.timestamp),reference_id=row['reference_id'],
                curve_index=index,curve_progress=progress,number_of_TRAIN_knots=len(t.progress),
                current_waypoint_distance_m=errors(state.robot,present)[0],
                current_public_TCP_xyzw=state.robot.pose.xyzw().tolist(),
                next_current_waypoint_xyzw=present.xyzw().tolist(),future_pose_intent_xyzw=target.xyzw().tolist(),
                progress_from_current_observation_only=True,TRAIN_curve_refitted=False))
        return sorted(result,key=lambda r:(-r['score'],r['reference_id'],r['manipulated_ids'],r['template'].anchor_id))

    def _done(self,goal):
        if goal['phase'] not in ('follow','release'):return super()._done(goal)
        if goal['phase']=='follow' and goal.get('curve_progress',0.)<1.:return False
        target=self._completion_target(goal);robot=self.frontend.state.robot
        return bool(target is not None and errors(robot,target)[0]<=.03
            and (goal['phase']!='release' or robot.gripper_open and not robot.contact))

    def qualify_opening(self,requested,current_requested):
        goal=self.selected
        if goal is None or goal['phase']!='release':return super().qualify_opening(requested,current_requested)
        target=self._completion_target(goal);distance=angle=None
        if target is not None:distance,angle=errors(self.frontend.state.robot,target)
        accepted=distance is not None and distance<=.03
        return float(accepted),dict(time=float(self.frontend.state.robot.timestamp),phase='release',
            desired_opening=1.,current_requested_opening=float(current_requested),qualified_opening=float(accepted),
            reason='original_author_Buzz_current_public3cm_release' if accepted else 'defer_Buzz_current_public3cm_release',
            current_position_error_m=distance,current_orientation_error_rad=angle,native_truth_read=False),None

    def receipt(self):return dict(super().receipt(),public_trained_follow_curve=dict(
        rule=deepcopy(self.curve_rule),events=deepcopy(self.curve_events),
        elapsed_time_alone_cannot_advance=True,global_budget_extended=False))
