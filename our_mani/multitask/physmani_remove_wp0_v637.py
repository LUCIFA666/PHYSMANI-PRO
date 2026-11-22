"""Explicit author wp0 before wp1, using only TRAIN command geometry/public poses.

The existing completed motion curve's start is not an author command pose.
This adds the missing source stage and asks wp1 RML to reach its existing5cm
transition region, before acquisition's contact permissions are needed.
"""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from ..contracts import Pose
from .evaluation import file_sha
from .specs import TASK_SPECS
from .interaction import COMMAND_GOAL,_reference_at
from .physmani_geometry_goals_v593 import pose_goals
from .physmani_skill_policy_v548 import SkillGeometryIntents,errors
from .physmani_terminal_opening_v620 import RemoveAuthorIntents


def binding(root,native_repo,task):
    spec=TASK_SPECS[task]
    if spec.family!='remove_cups_from_rotating_frame':raise ValueError('original two remove tasks only')
    source=Path(native_repo)/spec.native_source_relative_path
    if file_sha(source)!=spec.native_source_sha256:raise ValueError('source author remove task changed')
    path=Path(root)/'artifacts/framework/multitask_r10/remove_wp0_command_v636'/task/'model.json'
    model=json.loads(path.read_text())
    if (model.get('schema')!='r10_all_original_TRAIN_remove_wp0_command_geometry_v636' or model['task']!=task
        or model['declared_parent_count']!=100 or model['independent_command_parent_support']<8
        or not model['all_available_wp0_command_parents_used_equally']
        or model['native_success_or_completion_used_for_selection'] or model['VAL_used'] or model['TEST_used']
        or model['maximum_parent_position_error_m']>.03 or model['maximum_parent_angle_error_rad']>.15):
        raise ValueError('independent original TRAIN wp0 command geometry required')
    return dict(schema='r10_author_remove_explicit_wp0_wp1_transition_region_v637',task=task,
        model=dict(path=str(path.relative_to(root)),sha256=file_sha(path)),
        native_source_sha256=file_sha(source),source_sha256=file_sha(Path(root)/'our_mani/multitask/physmani_remove_wp0_v637.py'),
        wp0_position_guard_m=.1,wp1_position_guard_m=.05,wp1_goal_standoff_m=.03,
        wp1_standoff_is_inside_original_transition_region=True,original_physical_close_and_collision_gates_unchanged=True,
        completed_wp0_skill_claimed=False)


class AuthorPreApproachRemoveIntents(RemoveAuthorIntents):
    def __init__(self,frontend,remove_rule,wp0_rule,model):
        super().__init__(frontend,remove_rule)
        if (wp0_rule.get('task')!=self.task or model.get('task')!=self.task
            or wp0_rule.get('wp0_position_guard_m')!=.1 or wp0_rule.get('wp1_goal_standoff_m')!=.03):
            raise ValueError('source-bound explicit wp0 and unchanged wp1 guard required')
        self.wp0_rule=deepcopy(wp0_rule);self.wp0_local=Pose.from_xyzw(model['local_pose_xyzw'])
        self.wp0_passed=False;self.wp1_first_issued=False;self.preapproach_events=[]

    def _completion_target(self,goal):
        if goal.get('author_substage')!='wp0':return super()._completion_target(goal)
        c=self.frontend.actor.controller
        try:
            ref=c.generator.observed_reference(self.frontend.state,goal['reference_id'],goal['template'],self.origin)
            return Pose(ref.pose.apply(self.wp0_local.position),ref.pose.rotation@self.wp0_local.rotation)
        except (KeyError,ValueError,IndexError):return None

    def _advance(self):
        goal=self.selected
        if goal is not None and goal.get('author_substage')=='wp0':
            target=self._completion_target(goal)
            if target is not None and errors(self.frontend.state.robot,target)[0]<=.1:
                self.wp0_passed=True;self.selected=None;self.committed=goal
                self.preapproach_events.append(dict(event='original_author_wp0_current10cm_passed',
                    time=float(self.frontend.state.robot.timestamp),phase_epoch=self.phase_epoch,
                    reference_id=goal['reference_id'],current_distance_m=errors(self.frontend.state.robot,target)[0],
                    phase_epoch_or_global_clock_reset=False,native_success_claimed=False))
            return
        return super()._advance()

    def candidates(self,phase):
        if phase!='approach':return super().candidates(phase)
        f=self.frontend;c=f.actor.controller;state=f.state
        delay=0. if not self.wp0_passed else .5 if self.wp1_first_issued else 1.
        rows=pose_goals(f,phase,origin=self.origin,lookahead=delay,committed=self.committed);result=[]
        for row in rows:
            if not self.wp0_passed:
                current=c.generator.observed_reference(state,row['reference_id'],row['template'],self.origin)
                pose=Pose(current.pose.apply(self.wp0_local.position),current.pose.rotation@self.wp0_local.rotation)
                result.append(dict(row,pose=pose,present_pose=pose,opening=1.,author_substage='wp0',
                    score=-float(np.linalg.norm(pose.position-state.robot.pose.position))))
                continue
            t=row['template']
            if t.reference_time_basis!=COMMAND_GOAL:raise ValueError('source wp1 current canonical geometry required')
            canonical=Pose.from_xyzw(t.completion_geometry_xyzw)
            ref,v,w=_reference_at(f.forecast,row['reference_id'],t,row['predicted_reference_time'],self.origin)
            future=Pose(ref.apply(canonical.position),ref.rotation@canonical.rotation)
            current=super()._completion_target(row)
            if current is None:continue
            direction=state.robot.pose.position-current.position;distance=float(np.linalg.norm(direction))
            if distance<1e-8:direction=np.zeros(3)
            else:direction/=distance
            pose=Pose(future.position+.03*direction,future.rotation)
            result.append(dict(row,pose=pose,present_pose=current,opening=1.,author_substage='wp1',
                source_current_canonical=current,source_predicted_canonical=future,wp1_standoff_m=.03,
                score=-float(np.linalg.norm(pose.position-state.robot.pose.position))))
        return sorted(result,key=lambda r:(-r['score'],r['reference_id'],r['manipulated_ids'],r['template'].anchor_id))

    def select(self):
        # The inherited remove.select marks the first wp1 intent. A wp0 intent
        # must not consume that source1s transition or advance a learned phase.
        action=SkillGeometryIntents.select(self)
        if action is not None and self.selected['phase']=='approach':
            goal=self.selected;target=self._completion_target(goal)
            if goal['author_substage']=='wp1':self.wp1_first_issued=True
            self.preapproach_events.append(dict(event='source_author_public_pregrasp_intent',
                time=float(self.frontend.state.robot.timestamp),source_stage=goal['author_substage'],
                reference_id=goal['reference_id'],lookahead_seconds=goal['lookahead_seconds'],
                current_TCP=self.frontend.state.robot.pose.xyzw().tolist(),action=action.tolist(),
                current_canonical=None if target is None else target.xyzw().tolist(),
                current_distance_m=None if target is None else errors(self.frontend.state.robot,target)[0],
                physical_close_or_native_success_claimed=False))
        return action

    def receipt(self):return dict(super().receipt(),author_explicit_wp0=dict(
        binding=deepcopy(self.wp0_rule),wp0_current_guard_passed=self.wp0_passed,
        events=deepcopy(self.preapproach_events),original_source_command_not_completed_curve_start=True))
