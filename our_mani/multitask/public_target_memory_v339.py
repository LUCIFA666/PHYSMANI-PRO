"""Remember a publicly qualified target identity within one fresh episode.

Colour identifies a persistent shared semantic component; it is not a new
pose measurement. Later lighting/occlusion cannot switch the task target.
Current pose, motion-mode, forecast and collision qualification stay with the
original pipeline. No cached colour record is presented as current RGBD.
"""
from copy import deepcopy
from pathlib import Path
from types import MethodType
import numpy as np
from .evaluation import file_sha, digest
from .public_roles import PublicRoleResolutionError

TASKS = ('insert_onto_rotating_peg', 'insert_onto_rotating_peg_high_speed')


def binding(root, protocol, task):
    if task not in TASKS:
        raise ValueError('explicit peg task target identity policy required')
    pointer = protocol['learned_artifacts'][task]['colour']
    if file_sha(Path(root) / pointer['path']) != pointer['sha256']:
        raise ValueError('original TRAIN colour model changed')
    names = ('public_target_memory_v339.py', 'public_roles.py', 'colour_distribution.py', 'colour_runtime_v18.py')
    return dict(schema='r10_public_colour_qualified_episode_target_identity_v339', task=task,
        sources_sha256={'our_mani/multitask/' + n: file_sha(Path(__file__).with_name(n)) for n in names},
        original_colour_model=pointer, first_binding_requires_original_unique_current_colour_qualification=True,
        identity_memory='same public requested colour, same shared semantic component IDs, same original TRAIN artifact, one fresh episode',
        current_colour_reconfirmation_required_after_identity_qualification=False,
        current_pose_orientation_motion_mode_forecast_collision_grasp_and_success_gates_unchanged=True,
        cached_history_does_not_grant_current_pose_or_geometry_permission=True,
        native_target_variation_or_teacher_used=False, classifier_refitted=False, formal_VAL_TEST=False)


def component_key(templates, body_id):
    selected = [t for t in templates if t.role_id == body_id]
    if len(selected) != 1:
        raise PublicRoleResolutionError('persistent target component topology missing or duplicated')
    return tuple((c.name, tuple(c.instance_ids)) for c in selected[0].components)


class EpisodeTargetIdentity:
    def __init__(self, original):
        if original.spec.task_name not in TASKS:
            raise ValueError('target identity memory only for explicitly scoped peg tasks')
        self.original = original
        self.artifact_fingerprint = digest(original.artifact)
        self.target = self.body_id = self.key = self.confirmed_at = self.first_report = None
        self.reuses = 0

    def resolve(self, descriptions, state, observation, templates):
        if isinstance(descriptions, str):
            descriptions = (descriptions,)
        if not descriptions or any(not isinstance(x, str) for x in descriptions):
            raise PublicRoleResolutionError('public task language is required')
        now = float(state.robot.timestamp)
        if not np.isfinite(now) or abs(float(observation['timestamp']) - now) > 1e-8:
            raise PublicRoleResolutionError('current public observation/proprioception required')
        if digest(self.original.artifact) != self.artifact_fingerprint:
            raise PublicRoleResolutionError('original TRAIN colour artifact mutated')
        target = self.original._requested_colour(tuple(descriptions))
        if self.body_id is None:
            roles, report = self.original.resolve(descriptions, state, observation, templates)
            body_id = report['body_id']
            if (report.get('selected_role') != 'selected_peg'
                    or tuple(roles.get('selected_peg', ())) != (body_id,)
                    or report['public_target_colour'] != target):
                raise PublicRoleResolutionError('original unique public peg binding required')
            self.key = component_key(templates, body_id)
            self.target, self.body_id, self.confirmed_at = target, body_id, now
            self.first_report = deepcopy(report)
            return roles, dict(report, identity_qualification_at=now, current_colour_revalidated=True)
        if target != self.target:
            raise PublicRoleResolutionError('public requested target changed inside an identity-memory episode')
        if component_key(templates, self.body_id) != self.key:
            raise PublicRoleResolutionError('persistent shared semantic target IDs changed')
        self.reuses += 1
        roles = dict(state.role_bindings, selected_peg=(self.body_id,))
        return roles, dict(public_target_colour=target, selected_role='selected_peg', body_id=self.body_id,
            role_colour_evidence={}, current_colour_revalidated=False,
            identity_qualification_at=self.confirmed_at,
            original_qualification_sha256=digest(self.first_report),
            binding_provenance='causal same-episode public colour qualification and persistent shared semantic IDs',
            current_pose_or_geometry_qualification_granted=False,
            semantic_instance_bindings_are_extra_shared_prior=True, pure_RGBD_detector_claimed=False,
            native_target_or_variation_read=False, teacher_labels_used_at_inference=False,
            colour_artifact_sha256=self.original.artifact['artifact_sha256'])

    def receipt(self):
        return dict(qualified_body_id=self.body_id, public_requested_colour=self.target,
            identity_qualified_at=self.confirmed_at, original_qualification=deepcopy(self.first_report),
            subsequent_identity_uses=self.reuses, current_pose_geometry_or_collision_permissions_granted=False)


def install(actor, protocol, root, task):
    expected = binding(root, protocol, task)
    if protocol.get('public_target_memory_v339', {}).get(task) != expected:
        raise ValueError('freeze actual public target identity memory')
    if actor.controller.feedback_step_calls or actor.role_resolver is None:
        raise ValueError('install on a fresh explicitly qualified colour actor')
    if actor.role_resolver.spec.task_name != task:
        raise ValueError('target resolver belongs to another native task')
    resolver = EpisodeTargetIdentity(actor.role_resolver)
    actor.role_resolver = resolver
    previous = actor.metrics_report
    def metrics(instance):
        return dict(previous(), public_target_memory_v339=dict(binding=expected, receipt=resolver.receipt()))
    actor.metrics_report = MethodType(metrics, actor)
