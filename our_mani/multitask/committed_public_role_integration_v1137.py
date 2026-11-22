"""Continue only a previously colour-confirmed public committed peg identity.

The original resolver still runs once per call. Only its precise current-colour
absence/ambiguity error can reuse a semantic identity after acquire. No stored
pose, forecast, collision permission or seating result is reused or invented.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ..contracts import ObjectState, Pose, RobotState
from .interaction import PolicyState, PhaseTemplate
from .perception import BodyTemplate, GeometryComponent
from .public_roles import PublicRoleResolver, PublicRoleResolutionError
from . import public_task_phase_integration_v1131 as parent

SCHEMA = 'r10_committed_public_role_continuation_integration_v1137'
SOURCE = 'our_mani/multitask/committed_public_role_integration_v1137.py'
TASKS = parent.RING_TASKS
ALLOWED_ERROR = 'public target colour is absent or ambiguous among current observed roles'
PARENT_LOCK = 'artifacts/framework/multitask_r10/TRAIN_public_phase_full16_v1132_setup/source_overlay_lock.json'
PARENT_LOCK_SHA = '99a04a6434f90d588656a9b5b8b0f708e5c54848a80868703349f5cc3509d290'
SOURCES = (SOURCE, *(('our_mani/multitask/'+name) for name in (
    'public_roles.py','colour_distribution.py','colour_component_domain_v642.py',
    'physmani_geometry_policy_v518.py','pipeline.py','physmani_skill_policy_v548.py','perception.py')))


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _finite(value):
    return type(value) in (float,int) and math.isfinite(value)


def _single(value):
    return type(value) in (tuple,list) and len(value)==1 and type(value[0]) is str


def binding(root, native_repo):
    root=Path(root)
    digest=lambda name:hashlib.sha256((root/name).read_bytes()).hexdigest()
    if digest(PARENT_LOCK)!=PARENT_LOCK_SHA:raise ValueError('unchanged1132 parent source lock required')
    return dict(schema=SCHEMA,sources_sha256={name:digest(name) for name in SOURCES},
        parent=parent.binding(root,native_repo),parent_source_overlay_lock=dict(path=PARENT_LOCK,sha256=PARENT_LOCK_SHA),
        changed_factor='post_acquire_confirmed_public_semantic_commitment_continuation',
        original_resolver_called_once_per_received_frame=True,only_allowed_original_error=ALLOWED_ERROR,
        first_identity_witness='original_resolver_success_return_and_public_colour_proof',
        current_pose_forecast_collision_and_ring_seating_permissions_granted=False,
        other_tasks='exact_original1131_objects',additional_native_observe_predict_resolver_or_success_calls=0,
        native_held_identity_verified=False,formal_VAL_TEST_qualification=False)


class PublicCommitmentContinuation:
    def __init__(self,source,actor,original,*,max_events=1000):
        if not isinstance(actor.role_resolver,PublicRoleResolver) or source.task not in TASKS:
            raise TypeError('owned original ring public colour resolver required')
        if type(max_events) is not int or not 1<=max_events<=10000:raise ValueError('bounded trace required')
        self.source,self.actor,self.original=source,actor,original
        self.max_events=max_events;self.events=[];self.dropped_events=0
        self.calls=self.successes=self.original_errors=self.uses=self.recording_errors=0
        self.other_errors=0;self.restored=None
        self.artifact_fingerprint=_digest(actor.role_resolver.artifact)
        self.witness=None;self.last_time=self.last_phase=self.context_key=self.invalidated_close_at=None
        self.rewound=self.scope_invalidated=False

    def record(self,event,**data):
        if len(self.events)<self.max_events:self.events.append(dict(event=event,**data))
        else:self.dropped_events+=1

    def reset(self,reason):
        self.witness=None
        self.record('semantic_witness_cleared',reason=reason)

    def context(self,descriptions,state,observation,templates):
        if isinstance(descriptions,str):descriptions=(descriptions,)
        if type(descriptions) not in (tuple,list) or not descriptions or any(type(x) is not str for x in descriptions):
            raise ValueError('unchanged_plain_descriptions_required')
        if type(state) is not PolicyState or type(state.robot) is not RobotState or type(observation) is not dict:
            raise ValueError('current_public_state_required')
        now=state.robot.timestamp
        if not _finite(now) or not _finite(observation.get('timestamp')) or abs(now-observation['timestamp'])>1e-8:
            raise ValueError('current_observation_clock_required')
        if type(templates) not in (tuple,list):raise ValueError('stored_template_sequence_required')
        topology=[]
        for template in templates:
            if type(template) is not BodyTemplate:raise ValueError('original_body_templates_required')
            components=[]
            for component in template.components:
                if type(component) is not GeometryComponent:raise ValueError('original_components_required')
                components.append((component.name,tuple(component.instance_ids)))
            topology.append((template.role_id,tuple(components)))
        if len({x[0] for x in topology})!=len(topology):raise ValueError('unique_template_roles_required')
        resolver=self.actor.role_resolver
        if resolver.spec.task_name!=self.source.task:raise ValueError('same_resolver_task_required')
        artifact=resolver.artifact
        if _digest(artifact)!=self.artifact_fingerprint:
            self.scope_invalidated=True
            self.reset('original_colour_artifact_changed');raise ValueError('original_colour_artifact_changed')
        key=_digest(dict(task=self.source.task,descriptions=list(descriptions),topology=sorted(topology),
            artifact=artifact,library_sha256=self.actor.controller.library.sha256))
        phase=self.source.phase_index;order=tuple(self.source.phase_order)
        if type(phase) is not int or not 0<=phase<=len(order) or 'acquire' not in order:
            raise ValueError('current_original_phase_required')
        if (self.last_time is not None and now<=self.last_time
                or self.last_phase is not None and phase<self.last_phase):
            self.rewound=self.scope_invalidated=True;self.reset('clock_or_phase_rewind_requires_fresh_owned_scope')
        elif self.context_key is not None and key!=self.context_key:
            self.scope_invalidated=True
            self.reset('public_language_topology_or_library_context_changed')
        self.last_time,self.last_phase,self.context_key=now,phase,key
        if phase>order.index('acquire') and (state.robot.contact is not True
                or not _finite(self.actor.requested_opening) or self.actor.requested_opening>=.5):
            self.invalidated_close_at=now
        return dict(now=now,key=key,descriptions=tuple(descriptions),topology=tuple(sorted(topology)),
                    artifact_sha256=artifact.get('artifact_sha256'),phase=phase,order=order,state=state,
                    inactive=phase==len(order))

    def remember(self,result,context):
        if context['inactive']:
            self.record('original_success_in_finished_phase_not_cached',time=context['now']);return
        if type(result) is not tuple or len(result)!=2:return self.reset('nonstandard_original_success')
        roles,proof=result
        if type(roles) is not dict or type(proof) is not dict:return self.reset('nonplain_original_success')
        target=proof.get('body_id');colour=proof.get('public_target_colour')
        evidence=proof.get('role_colour_evidence',{}).get(target,{})
        if (proof.get('selected_role')!='selected_peg' or not _single(roles.get('selected_peg'))
                or roles['selected_peg'][0]!=target or type(target) is not str
                or not target.startswith('peg') or not target[3:].isdigit()
                or proof.get('colour_artifact_sha256')!=context['artifact_sha256']
                or proof.get('native_target_or_variation_read') is not False
                or proof.get('teacher_labels_used_at_inference') is not False
                or evidence.get('valid') is not True or evidence.get('colour_name')!=colour
                or evidence.get('colour_artifact_sha256')!=context['artifact_sha256']):
            return self.reset('original_success_has_no_unique_public_colour_witness')
        committed=self.actor.controller.committed_roles.get('selected_peg')
        if committed is not None and tuple(committed)!=(target,):
            self.scope_invalidated=True
            return self.reset('new_success_conflicts_with_committed_target')
        self.witness=dict(body_id=target,public_target_colour=colour,key=context['key'],
            confirmed_at=context['now'],descriptions=list(context['descriptions']),
            template_topology=deepcopy(context['topology']),artifact_sha256=context['artifact_sha256'],
            original_proof=deepcopy(proof),original_proof_sha256=_digest(proof))
        self.record('original_public_identity_witness',body_id=target,time=context['now'],
                    original_proof_sha256=self.witness['original_proof_sha256'])

    def eligible(self,context):
        state=context['state'];now=context['now'];w=self.witness
        if self.scope_invalidated:return 'public_context_or_phase_rewind_requires_fresh_owned_scope'
        if context['inactive']:return 'finished_phase_does_not_use_semantic_continuation'
        if w is None or w['key']!=context['key']:return 'no_matching_original_success_witness'
        target=w['body_id'];c=self.actor.controller;s=self.source
        if tuple(c.committed_roles.get('selected_peg',()))!=(target,):return 'committed_target_does_not_match_witness'
        if context['phase']<=context['order'].index('acquire'):return 'post_acquire_phase_required'
        if state.robot.contact is not True or not _finite(self.actor.requested_opening) or self.actor.requested_opening>=.5:
            return 'current_closed_command_and_shared_held_required'
        close=s.close_witness
        if (type(close) is not dict or close.get('phase_index')!=context['order'].index('acquire')
                or tuple(close.get('manipulated_ids',()))!=('ring',)
                or not _finite(close.get('confirmed_after')) or close['confirmed_after']>now+1e-8
                or self.invalidated_close_at is not None and close['confirmed_after']<=self.invalidated_close_at):
            return 'valid_own_acquire_close_required'
        ids={role for role,_ in context['topology']}
        if set(state.bodies)!=ids or target not in ids or 'ring' not in ids:return 'current_public_ids_changed'
        if any(type(body) is not ObjectState or body.object_id!=identity or type(body.pose) is not Pose
               or not _finite(body.timestamp) or body.timestamp>now+1e-8 for identity,body in state.bodies.items()):
            return 'current_original_public_body_records_required'
        metadata=state.perception_metadata;info=metadata.get('bodies',{})
        if any(info.get(identity,{}).get('motion_mode') not in ('external','attached','released') for identity in ids):
            return 'current_public_modes_required'
        invalid=set(metadata.get('invalid_mode_ids',()))
        attached={identity for identity,item in info.items() if item.get('motion_mode')=='attached' and identity not in invalid}
        if attached!={'ring'} or invalid.intersection(('ring',target)) or info['ring'].get('mode_provenance')!='caller_controller_touch_action_proprioception':
            return 'unique_public_ring_attachment_required'
        if info[target].get('motion_mode')!='external':return 'current_external_selected_peg_required'
        measured=c.measured_attachments.get('ring')
        if (type(measured) is not dict or measured.get('source')!='current_RGBD_body_pose_and_robot_proprioception_at_public_contact'
                or not _finite(measured.get('observed_at')) or not close['confirmed_after']-1e-8<=measured['observed_at']<=now+1e-8
                or np.asarray(measured.get('local_pose_xyzw',()),float).shape!=(7,)
                or not np.isfinite(np.asarray(measured['local_pose_xyzw'],float)).all()):
            return 'existing_causal_measured_attachment_required'
        selected=s.selected
        if type(selected) is dict:
            template=selected.get('template');role=vars(template).get('reference_role') if isinstance(template,PhaseTemplate) else None
            if role=='selected_peg' and selected.get('reference_id')!=target:return 'selected_goal_conflicts_with_committed_target'
        if state.role_bindings.get('selected_peg') is not None and tuple(state.role_bindings['selected_peg'])!=(target,):
            return 'current_semantic_target_conflicts_with_witness'
        return None

    def resolve(self,descriptions,state,observation,templates):
        self.calls+=1
        try:result=self.original(descriptions,state,observation,templates)
        except PublicRoleResolutionError as error:
            self.original_errors+=1
            reason='original_error_not_colour_absence_or_ambiguity'
            try:
                context=self.context(descriptions,state,observation,templates)
                if error.args==(ALLOWED_ERROR,):reason=self.eligible(context)
                if reason is None:
                    w=self.witness;self.uses+=1
                    self.record('committed_semantic_identity_continued',time=context['now'],body_id=w['body_id'],original_error=ALLOWED_ERROR)
                    return {**state.role_bindings,'selected_peg':(w['body_id'],)},dict(
                        public_target_colour=w['public_target_colour'],selected_role='selected_peg',body_id=w['body_id'],
                        role_colour_evidence={},current_colour_revalidated=False,identity_qualification_at=w['confirmed_at'],
                        original_qualification_sha256=w['original_proof_sha256'],binding_provenance=SCHEMA,
                        original_resolution_error=ALLOWED_ERROR,current_pose_or_geometry_qualification_granted=False,
                        colour_artifact_sha256=w['artifact_sha256'],native_target_or_variation_read=False,
                        teacher_labels_used_at_inference=False,semantic_instance_bindings_are_extra_shared_prior=True)
            except Exception as diagnostic_error:
                reason='public_context_unavailable:'+type(diagnostic_error).__name__
            self.record('original_resolution_error_preserved',reason=reason,original_error=error.args[0] if error.args else None)
            raise
        except BaseException as error:
            self.other_errors+=1;self.record('other_original_error_preserved',type=type(error).__name__)
            raise
        else:
            self.successes+=1
            try:self.remember(result,self.context(descriptions,state,observation,templates))
            except ValueError as error:
                self.reset('original_success_not_cached:'+str(error))
            except Exception as error:
                self.recording_errors+=1;self.reset('unrecordable_public_success:'+type(error).__name__)
            return result

    def receipt(self):
        return dict(schema=SCHEMA,events=deepcopy(self.events),dropped_events=self.dropped_events,
            original_resolver_calls=self.calls,original_success_returns=self.successes,
            original_public_resolution_errors=self.original_errors,continuation_uses=self.uses,
            other_original_errors=self.other_errors,recording_errors=self.recording_errors,
            scope_rewind_disables_continuation=self.rewound,
            public_context_change_disables_continuation=self.scope_invalidated,
            owned_resolver_scope_restored=self.restored,public_success_witness=deepcopy(self.witness),
            original_resolver_called_once=True,cached_pose_or_forecast_returned=False,
            later_ready_collision_forecast_and1129_guard_unchanged=True,extra_native_calls=0)


@contextmanager
def integrated_components(root,config,task,actor,*,events=None,max_events=1000):
    with parent.integrated_components(root,config,task,actor) as (frontend,dispatched,executor):
        if task not in TASKS:
            yield frontend,dispatched,executor;return
        source=dispatched.source_policy;resolver=actor.role_resolver
        marker='_committed_public_role_continuation_v1137'
        if marker in vars(resolver):raise RuntimeError('owned resolver continuation scopes cannot nest')
        continuation=PublicCommitmentContinuation(source,actor,resolver.resolve,max_events=max_events)
        original_receipt=dispatched.receipt;original_resolve=resolver.resolve
        def receipt():return dict(original_receipt(),committed_public_role_continuation_v1137=continuation.receipt())
        original_error=None
        try:
            with patch.object(resolver,marker,continuation,create=True),patch.object(resolver,'resolve',continuation.resolve),patch.object(dispatched,'receipt',receipt):
                yield frontend,dispatched,executor
        except BaseException as error:
            original_error=error;raise
        finally:
            before=continuation.receipt()
            continuation.reset('owned_context_exit')
            continuation.restored=(resolver.resolve==original_resolve and dispatched.receipt==original_receipt and marker not in vars(resolver))
            if events is not None:events.update(schema=SCHEMA,restored=continuation.restored,
                witness_cleared=continuation.witness is None,continuation_receipt=before)
            if not continuation.restored and original_error is None:
                raise RuntimeError('owned semantic continuation methods did not restore')
