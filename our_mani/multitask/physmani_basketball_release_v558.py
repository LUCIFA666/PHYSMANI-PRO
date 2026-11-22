"""Author basketball release guard applied to current public canonical poses.

The TRAIN fit tolerance measures regression error. It is not the source
author's3cm release stage guard. Physical acquisition remains2cm/.1rad.
"""
import ast
from copy import deepcopy
from pathlib import Path
from .evaluation import file_sha
from .specs import TASK_SPECS
from .physmani_skill_policy_v548 import SkillGeometryIntents,errors


def binding(root,native_repo,task):
    if task not in ('moving_basketball_in_hoop','moving_basketball_in_hoop_high_speed'):
        raise ValueError('explicit basketball task only')
    spec=TASK_SPECS[task];path=Path(native_repo)/spec.native_source_relative_path
    if file_sha(path)!=spec.native_source_sha256:raise ValueError('original author basketball task source changed')
    tree=ast.parse(path.read_text());expert=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_expert_info')
    values=[ast.literal_eval(n.value) for n in ast.walk(expert) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='th_w3' for t in n.targets)]
    if values!=[.03]:raise ValueError('original author th_w3 release guard must remain3cm')
    return dict(schema='r10_author_basketball_public_current_release_guard_v558',task=task,
        native_source_sha256=spec.native_source_sha256,source_threshold='th_w3',release_distance_m=.03,
        author_guard_is_position_only=True,current_public_canonical_pose_required=True,
        regression_fit_caps_unchanged=True,physical_acquire_position_m=.02,physical_acquire_orientation_rad=.1,
        source_sha256=file_sha(Path(root)/'our_mani/multitask/physmani_basketball_release_v558.py'),
        native_success_and_RML_parameters_unchanged=True)


class AuthorBasketballReleaseIntents(SkillGeometryIntents):
    def __init__(self,frontend,rule):
        super().__init__(frontend)
        if (rule.get('schema')!='r10_author_basketball_public_current_release_guard_v558'
            or rule.get('task')!=self.task or rule.get('release_distance_m')!=.03
            or rule.get('author_guard_is_position_only') is not True):
            raise ValueError('frozen original author3cm basketball release guard required')
        self.release_rule=deepcopy(rule)

    def qualify_opening(self,requested,current_requested):
        goal=self.selected;state=self.frontend.state
        if goal is None or goal['phase']!='deposit' or requested<.5 or current_requested>=.5:
            return super().qualify_opening(requested,current_requested)
        target=self._completion_target(goal);distance=angle=None
        if target is not None:distance,angle=errors(state.robot,target)
        accepted=target is not None and distance<=self.release_rule['release_distance_m']
        return float(accepted),dict(time=float(state.robot.timestamp),phase='deposit',
            desired_opening=1.,current_requested_opening=float(current_requested),qualified_opening=float(accepted),
            reason='original_author_current_public3cm_release_guard' if accepted else 'defer_release_until_author_current_public3cm_guard',
            current_position_error_m=distance,current_orientation_error_rad=angle,
            author_position_only_threshold_m=.03,fit_tolerance_is_not_the_release_guard=True,native_truth_read=False),None

    def _done(self,goal):
        if goal['phase']!='deposit':return super()._done(goal)
        state=self.frontend.state;target=self._completion_target(goal)
        if target is None or not state.robot.gripper_open or state.robot.contact:return False
        return (state.robot.timestamp-self.phase_epoch>=goal['template'].duration_seconds-1e-8
                and errors(state.robot,target)[0]<=self.release_rule['release_distance_m'])

    def receipt(self):
        return dict(super().receipt(),author_basketball_release_rule=deepcopy(self.release_rule))
