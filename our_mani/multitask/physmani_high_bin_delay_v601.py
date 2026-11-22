"""High-speed bin's distinct closed1s/open0.5s source branch.

The first deposit intent remains current, as in the wp2 transition. Subsequent
far closed deposit intents use1s; near release uses0.5s. No arm parameters change.
"""
import ast
from copy import deepcopy
from pathlib import Path
from .specs import TASK_SPECS
from .evaluation import file_sha
from .physmani_stage_lookahead_v594 import AuthorLookaheadIntents
from .physmani_geometry_goals_v593 import pose_goals
from .physmani_skill_policy_v548 import errors

TASK='put_rubbish_in_moving_bin_high_speed'


def binding(root,native_repo,task):
    if task!=TASK:raise ValueError('distinct high-speed bin rule only')
    spec=TASK_SPECS[task];source=Path(native_repo)/spec.native_source_relative_path
    if file_sha(source)!=spec.native_source_sha256:raise ValueError('original high-bin source changed')
    expert=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='get_expert_info')
    branch=next(n for n in ast.walk(expert) if isinstance(n,ast.If) and ast.unparse(n.test)=="stage in ['wp2', 'wp3'] and dist_to_wp3 > th_wp3")
    values={n.targets[0].id:ast.literal_eval(n.value) for n in branch.body if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('t_delay','open')}
    if values!={'t_delay':1.,'open':0}:raise ValueError('source closed1s high-bin branch required')
    return dict(schema='r10_high_bin_distinct_closed1s_open_half_source_delay_v601',task=task,
        first_deposit_delay_seconds=0.,closed_far_deposit_delay_seconds=1.,near_release_delay_seconds=.5,
        current_public_near_distance_m=.05,native_source_sha256=file_sha(source),
        source_sha256=file_sha(Path(root)/'our_mani/multitask/physmani_high_bin_delay_v601.py'),
        original_author_RML_and_native_success_unchanged=True)


class HighBinDelayIntents(AuthorLookaheadIntents):
    def __init__(self,frontend,stage_rule,delay_rule,high_rule):
        super().__init__(frontend,stage_rule,delay_rule)
        if high_rule.get('task')!=self.task or self.task!=TASK or high_rule.get('closed_far_deposit_delay_seconds')!=1.:
            raise ValueError('explicit high-speed source delay required')
        self.high_rule=deepcopy(high_rule);self.first_deposit_intent_issued=False

    def candidates(self,phase):
        if phase!='deposit':return super().candidates(phase)
        f=self.frontend;result=[]
        for row in pose_goals(f,phase,origin=self.origin,lookahead=0.,committed=self.committed):
            target=self._completion_target(row)
            if target is None:continue
            near=errors(f.state.robot,target)[0]<=.05
            delay=.5 if near else 1. if self.first_deposit_intent_issued else 0.
            result.extend(dict(r,opening=float(near)) for r in pose_goals(f,phase,origin=self.origin,lookahead=delay,committed=row))
        return sorted(result,key=lambda r:(-r['score'],r['reference_id'],r['manipulated_ids'],r['template'].anchor_id))

    def select(self):
        action=super().select()
        if action is not None and self.selected['phase']=='deposit':self.first_deposit_intent_issued=True
        return action

    def receipt(self):return dict(super().receipt(),effective_high_speed_bin_delay=deepcopy(self.high_rule),
        source_normal_bin_far_delay_overridden_for_high_speed_only=True)
