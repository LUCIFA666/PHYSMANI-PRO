"""Use original author per-stage delays, instead of universal 0.5s targets.

This changes only public pose intentions and their source opening request.
Author physical, collision, phase and native outcome guards remain in force.
"""
import ast
from copy import deepcopy
from pathlib import Path
import numpy as np
from .specs import TASK_SPECS
from .evaluation import file_sha
from .physmani_author_stages_v574 import AuthorStageIntents,errors
from .physmani_geometry_goals_v593 import pose_goals

DELAYS={
    'put_rubbish_in_moving_bin':dict(approach=0.,acquire=0.,lift=0.,deposit_far=0.,deposit_near=.5),
    'place_cups_on_rotating_frame':dict(approach=0.,acquire=0.,lift=0.,transport=0.,align=.5,hang=.5)}


def binding(root,native_repo,task):
    spec=TASK_SPECS[task]
    if spec.family not in DELAYS:raise ValueError('explicit bin/cup author delay rules only')
    path=Path(native_repo)/spec.native_source_relative_path
    if file_sha(path)!=spec.native_source_sha256:raise ValueError('original author stage-delay source changed')
    expert=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='get_expert_info')
    # Retain branch expressions with their direct assignments as reviewable
    # evidence; do not infer delays from nested elif statements.
    branches=[]
    for node in ast.walk(expert):
        if not isinstance(node,ast.If) or 'stage' not in ast.unparse(node.test):continue
        values={n.targets[0].id:ast.literal_eval(n.value) for n in node.body if isinstance(n,ast.Assign)
            and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('t_delay','open')
            and isinstance(n.value,(ast.Constant,ast.UnaryOp))}
        if values:branches.append(dict(condition=ast.unparse(node.test),assignments=values))
    if not any(b['assignments'].get('t_delay')==0. for b in branches) or not any(b['assignments'].get('t_delay')==.5 for b in branches):
        raise ValueError('original zero-delay and half-second branches required')
    return dict(schema='r10_original_author_per_stage_public_lookahead_v594',task=task,
        delays=deepcopy(DELAYS[spec.family]),native_source_sha256=file_sha(path),source_branches=branches,
        bin_current_near_distance_m=.05,source_sha256=file_sha(Path(root)/'our_mani/multitask/physmani_stage_lookahead_v594.py'),
        goal_helper_sha256=file_sha(Path(root)/'our_mani/multitask/physmani_geometry_goals_v593.py'),
        physical_close_and_collision_gates_unchanged=True,TRAIN_templates_refitted=False,native_success_changed=False)


class AuthorLookaheadIntents(AuthorStageIntents):
    def __init__(self,frontend,stage_rule,delay_rule):
        super().__init__(frontend,stage_rule)
        family=TASK_SPECS[self.task].family
        if (delay_rule.get('schema')!='r10_original_author_per_stage_public_lookahead_v594'
            or delay_rule.get('task')!=self.task or delay_rule.get('delays')!=DELAYS[family]):
            raise ValueError('exact original per-stage delay binding required')
        self.delay_rule=deepcopy(delay_rule);self.intent_geometry=[]

    def candidates(self,phase):
        f=self.frontend;delays=self.delay_rule['delays']
        if phase=='deposit':
            rows=pose_goals(f,phase,origin=self.origin,lookahead=0.,committed=self.committed)
            result=[]
            for row in rows:
                target=self._completion_target(row)
                if target is None:continue
                near=errors(f.state.robot,target)[0]<=.05
                if near:
                    result.extend(dict(r,opening=1.) for r in pose_goals(f,phase,origin=self.origin,lookahead=.5,committed=row))
                else:result.append(dict(row,opening=0.))
            return sorted(result,key=lambda r:(-r['score'],r['reference_id'],r['manipulated_ids'],r['template'].anchor_id))
        rows=pose_goals(f,phase,origin=self.origin,lookahead=delays[phase],committed=self.committed)
        if phase=='hang':
            result=[]
            for row in rows:
                target=self._completion_target(row)
                near=target is not None and errors(f.state.robot,target)[0]<.03
                result.append(dict(row,opening=float(near)))
            return result
        return rows

    def select(self):
        action=super().select()
        if action is not None:
            goal=self.selected;target=self._completion_target(goal);state=self.frontend.state
            distance=angle=None
            if target is not None:distance,angle=errors(state.robot,target)
            self.intent_geometry.append(dict(time=float(state.robot.timestamp),phase=goal['phase'],
                phase_epoch=self.phase_epoch,public_TCP_xyzw=state.robot.pose.xyzw().tolist(),
                current_public_canonical_xyzw=None if target is None else target.xyzw().tolist(),
                intended_pose_action=action.tolist(),lookahead_seconds=goal['lookahead_seconds'],
                predicted_reference_time=goal['predicted_reference_time'],position_error_m=distance,
                orientation_error_rad=angle,reference_id=goal['reference_id'],
                native_body_pose_or_outcome_used=False))
        return action

    def receipt(self):return dict(super().receipt(),author_stage_lookahead=deepcopy(self.delay_rule),
        public_intent_geometry=deepcopy(self.intent_geometry))
