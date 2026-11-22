"""Author remove stages and trained approach knots, with public RML intentions."""
import ast
from copy import deepcopy
from pathlib import Path
import numpy as np
from .evaluation import file_sha
from .specs import TASK_SPECS
from .physmani_skill_policy_v548 import SkillGeometryIntents,errors
from .physmani_geometry_goals_v593 import pose_goals
from .interaction import _reference_at,_world_target


def binding(root,native_repo,task):
    spec=TASK_SPECS[task]
    if spec.family!='remove_cups_from_rotating_frame':raise ValueError('two remove tasks only')
    path=Path(native_repo)/spec.native_source_relative_path
    if file_sha(path)!=spec.native_source_sha256:raise ValueError('original remove source changed')
    expert=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='get_expert_info')
    thresholds={n.targets[0].id:ast.literal_eval(n.value) for n in ast.walk(expert) if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id.startswith('th_')}
    if thresholds!={'th_w0':.1,'th_w1':.05,'th_w2':.05,'th_w3':.025,'th_w4':.05,'th_w5':.03}:
        raise ValueError('original remove stage distances required')
    return dict(schema='r10_public_remove_author_stages_TRAIN_approach_curve_v611',task=task,
        phase_positions=dict(approach=.05,detach=.025,transport=.05,place=.03),
        per_phase_delay=dict(approach=.5,acquire=0.,detach=.5,transport=0.,place=0.),
        first_approach_delay_seconds=1.,approach_curve_progress_distance_m=.05,
        native_source_sha256=file_sha(path),source_sha256=file_sha(Path(root)/'our_mani/multitask/physmani_remove_stages_v611.py'),
        physical_acquire_gate_position_m=.02,physical_acquire_gate_angle_rad=.1,
        original_TRAIN_templates_and_native_task_success_unchanged=True)


class RemoveAuthorIntents(SkillGeometryIntents):
    def __init__(self,frontend,rule):
        super().__init__(frontend)
        if rule.get('task')!=self.task or rule.get('schema')!='r10_public_remove_author_stages_TRAIN_approach_curve_v611':
            raise ValueError('source-bound original remove stages required')
        self.remove_rule=deepcopy(rule);self.cursors={};self.approach_first_issued=False;self.curve_events=[]

    def candidates(self,phase):
        f=self.frontend;c=f.actor.controller;s=f.state
        delay=self.remove_rule['per_phase_delay'][phase]
        if phase=='approach' and not self.approach_first_issued:delay=1.
        rows=pose_goals(f,phase,origin=self.origin,lookahead=delay,committed=self.committed)
        if phase=='place':
            result=[]
            for row in rows:
                target=self._completion_target(row)
                result.append(dict(row,opening=float(target is not None and errors(s.robot,target)[0]<=.03)))
            return result
        if phase!='approach':return rows
        result=[]
        for row in rows:
            t=row['template'];key=(t.anchor_id,row['reference_id'],row['manipulated_ids'])
            current=c.generator.observed_reference(s,row['reference_id'],t,self.origin)
            index=self.cursors.get(key,0)
            while index<len(t.progress)-1:
                present=_world_target(current.pose,current.linear_velocity,current.angular_velocity,t,float(t.progress[index]))[0]
                if errors(s.robot,present)[0]>.05:break
                index+=1
            self.cursors[key]=index;progress=float(t.progress[index])
            ref,v,w=_reference_at(f.forecast,row['reference_id'],t,row['predicted_reference_time'],self.origin)
            target=_world_target(ref,v,w,t,progress)[0]
            result.append(dict(row,pose=target,opening=1.,curve_progress=progress,
                score=-float(np.linalg.norm(target.position-s.robot.pose.position))))
            self.curve_events.append(dict(time=float(s.robot.timestamp),reference_id=row['reference_id'],progress=progress,
                current_public_progress_only=True,predicted_reference_time=row['predicted_reference_time'],
                current_TCP_xyzw=s.robot.pose.xyzw().tolist(),future_pose_intent=target.xyzw().tolist()))
        return sorted(result,key=lambda r:(-r['score'],r['reference_id'],r['manipulated_ids'],r['template'].anchor_id))

    def select(self):
        action=super().select()
        if action is not None and self.selected['phase']=='approach':self.approach_first_issued=True
        return action

    def _done(self,goal):
        if goal['phase']=='acquire':return super()._done(goal)
        target=self._completion_target(goal);robot=self.frontend.state.robot
        return bool(target is not None and errors(robot,target)[0]<=self.remove_rule['phase_positions'][goal['phase']]
            and (goal['phase']!='place' or robot.gripper_open and not robot.contact))

    def qualify_opening(self,requested,current_requested):
        goal=self.selected
        if goal is None or goal['phase']!='place':return super().qualify_opening(requested,current_requested)
        target=self._completion_target(goal);distance=angle=None
        if target is not None:distance,angle=errors(self.frontend.state.robot,target)
        accepted=distance is not None and distance<=.03
        return float(accepted),dict(time=float(self.frontend.state.robot.timestamp),phase='place',
            desired_opening=float(requested>=.5),current_requested_opening=float(current_requested),qualified_opening=float(accepted),
            reason='original_author_remove_current_public3cm_release' if accepted else 'defer_remove_current_public3cm_release',
            current_position_error_m=distance,current_orientation_error_rad=angle,native_truth_read=False),None

    def receipt(self):return dict(super().receipt(),original_remove_author_stages=deepcopy(self.remove_rule),
        public_approach_curve_events=deepcopy(self.curve_events))
