"""Author's optional wp4-open after the trained insert endpoint, on public inputs.

No release skill is claimed to have been trained. The existing compensated
insert canonical geometry supplies the target; source author rules supply only
its 0.5s continuation and open request. Native task outcome remains external.
"""
import ast
from copy import deepcopy
from pathlib import Path
from .specs import TASK_SPECS
from .evaluation import file_sha
from .physmani_ring_compensation_v580 import CorrectedRingIntents
from .physmani_skill_policy_v548 import errors


def binding(root,native_repo,task):
    spec=TASK_SPECS[task]
    if spec.family!='insert_onto_rotating_peg':raise ValueError('two original ring tasks only')
    path=Path(native_repo)/spec.native_source_relative_path
    if file_sha(path)!=spec.native_source_sha256:raise ValueError('author ring source changed')
    tree=ast.parse(path.read_text())
    expert=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_expert_info')
    branch=next(n for n in ast.walk(expert) if isinstance(n,ast.If) and ast.unparse(n.test)=="stage == 'wp4-open'")
    assigns={n.targets[0].id:ast.literal_eval(n.value) for n in branch.body if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('open','t_delay')}
    calls=[ast.unparse(n) for n in ast.walk(ast.Module(body=branch.body,type_ignores=[])) if isinstance(n,ast.Call)]
    if assigns!={'t_delay':.5,'open':1} or not any('task.w4' in n and n.startswith('compute_target_pose(') for n in calls):
        raise ValueError('original wp4-open target/delay/open rule required')
    return dict(schema='r10_author_optional_ring_terminal_release_v590',task=task,
        source_stage='wp4-open',reused_trained_endpoint_phase='insert',lookahead_seconds=.5,
        current_public_release_distance_m=.03,position_only_author_guard=True,
        native_source_sha256=file_sha(path),source_sha256=file_sha(Path(root)/'our_mani/multitask/physmani_ring_release_v590.py'),
        trained_release_skill_claimed=False,native_success_rule_changed=False)


class AuthorRingReleaseIntents(CorrectedRingIntents):
    def __init__(self,frontend,compensation_rule,prior,release_rule):
        super().__init__(frontend,compensation_rule,prior)
        if (release_rule.get('schema')!='r10_author_optional_ring_terminal_release_v590'
            or release_rule.get('task')!=self.task or release_rule.get('current_public_release_distance_m')!=.03
            or release_rule.get('lookahead_seconds')!=.5 or release_rule.get('trained_release_skill_claimed') is not False):
            raise ValueError('source-bound original terminal release required')
        if self.phase_order[-1]!='insert' or 'release' in self.phase_order:
            raise ValueError('only fill the missing optional terminal release')
        self.phase_order=(*self.phase_order,'release')
        self.release_rule=deepcopy(release_rule)

    def _advance(self):
        previous=self.selected
        super()._advance()
        if (previous is not None and previous['phase']=='insert' and self.selected is None
            and not self.finished and self.phase_order[self.phase_index]=='release'):
            # Keep the exact previously selected peg/ring identities. Do not
            # reselect a more convenient peg after insertion.
            self.committed=previous

    def candidates(self,phase):
        if phase!='release':return super().candidates(phase)
        return [dict(row,phase='release',opening=1.,source_author_optional_release=True,
            trained_release_skill_claimed=False) for row in super().candidates('insert')]

    def select(self):
        action=super().select()
        if action is not None and self.selected['phase']=='release':
            self.events[-1].update(event='source_author_terminal_release_pose_intent',
                source_stage='wp4-open',reused_trained_endpoint_phase='insert',trained_release_skill_claimed=False)
        return action

    def _done(self,goal):
        if goal['phase']!='release':return super()._done(goal)
        state=self.frontend.state;target=self._completion_target(goal)
        return bool(target is not None and state.robot.gripper_open and not state.robot.contact
            and errors(state.robot,target)[0]<=.03)

    def qualify_opening(self,requested,current_requested):
        goal=self.selected
        if goal is None or goal['phase']!='release':return super().qualify_opening(requested,current_requested)
        target=self._completion_target(goal);distance=angle=None
        if target is not None:distance,angle=errors(self.frontend.state.robot,target)
        accepted=distance is not None and distance<=.03
        return float(accepted),dict(time=float(self.frontend.state.robot.timestamp),phase='release',
            desired_opening=1.,current_requested_opening=float(current_requested),qualified_opening=float(accepted),
            reason='original_author_wp4_open_current_public3cm' if accepted else 'defer_author_wp4_open_current_public3cm',
            current_position_error_m=distance,current_orientation_error_rad=angle,
            source_author_rule=True,trained_release_skill_claimed=False,native_truth_read=False),None

    def receipt(self):return dict(super().receipt(),author_optional_terminal_release=deepcopy(self.release_rule))
