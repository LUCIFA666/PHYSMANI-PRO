"""Author nearest public cup/spoke selection until an actual own close edge.

The source expert recomputes its nearest cup on every query. A failed IK must
not permanently commit the initial rotating cup. After the actual close request
is dispatched, retain that exact manipulation identity and its existing proofs.
"""
from copy import deepcopy
from pathlib import Path
from .evaluation import file_sha
from .interaction import PolicyState
from .remove_runtime_v48 import nearest_roles
from .physmani_remove_wp0_v637 import AuthorPreApproachRemoveIntents


def binding(root,config,task):
    from .specs import TASK_SPECS
    spec=TASK_SPECS[task]
    if spec.family!='remove_cups_from_rotating_frame':raise ValueError('two source remove tasks only')
    source=Path(config['baseline_repo'])/spec.native_source_relative_path
    if file_sha(source)!=spec.native_source_sha256:raise ValueError('author source nearest-cup logic changed')
    return dict(schema='r10_author_public_remove_nearest_roles_until_own_close_v653',task=task,
        native_source_sha256=file_sha(source),source_sha256=file_sha(Path(root)/'our_mani/multitask/physmani_remove_source_roles_v653.py'),
        original_nearest_role_function_sha256=file_sha(Path(root)/'our_mani/multitask/remove_runtime_v48.py'),
        reselect_until_actual_own_close=True,after_own_close_identity_immutable=True,
        current_public_RGBD_TCP_only=True,phase_epoch_and_global_budget_unchanged=True,native_outcome_not_used=True)


class AuthorLiveRoleRemoveIntents(AuthorPreApproachRemoveIntents):
    def __init__(self,*a,source_role_rule,**kw):
        super().__init__(*a,**kw)
        if source_role_rule.get('task')!=self.task or source_role_rule.get('schema')!='r10_author_public_remove_nearest_roles_until_own_close_v653':
            raise ValueError('freeze original public nearest roles before own close')
        self.source_role_rule=deepcopy(source_role_rule);self.source_role_events=[]

    def select(self):
        f=self.frontend;c=f.actor.controller;s=f.state
        if s is not None and self.close_witness is None and not s.robot.contact:
            roles=nearest_roles(s,s.role_bindings,{},c.generator.config.maximum_observation_age)
            old={k:s.role_bindings.get(k,()) for k in ('selected_cup','selected_spoke')}
            for key in ('selected_cup','selected_spoke'):c.committed_roles.pop(key,None)
            f.state=PolicyState(s.robot,s.bodies,s.perception_metadata,roles)
            # Keep the current author substage while rebinding its source pose
            # to the current nearest public identities, exactly as the expert.
            if self.selected is not None:
                t=self.selected['template'];refs=roles.get(t.reference_role,())
                objects=roles.get(t.manipulated_role,())
                if len(refs)==len(objects)==1:
                    self.selected=dict(self.selected,reference_id=refs[0],manipulated_ids=tuple(objects))
                else:self.selected=None
            self.committed=None
            self.source_role_events.append(dict(time=float(s.robot.timestamp),before={k:list(v) for k,v in old.items()},
                after={k:list(roles.get(k,())) for k in old},own_close_dispatched=False,
                current_public_role_rule_only=True,phase_epoch=self.phase_epoch,phase_clock_reset=False))
        return super().select()

    def receipt(self):return dict(super().receipt(),source_author_nearest_role_lifecycle=dict(
        binding=deepcopy(self.source_role_rule),events=deepcopy(self.source_role_events),
        source_outcome_selection_or_episode_replacement=False))
