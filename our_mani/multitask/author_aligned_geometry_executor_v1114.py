"""Geometry pose intents on the actual shared PhysMani v7 actuator.

No public collision result vetoes a joint target here. Default execution does
not call that predicate or install a path wrapper. An explicitly enabled,
bounded legacy side diagnostic can report would_reject without changing a
target/return; its own existing probes are disclosed and errors are not scored
as task success. This new component grants no Native/VAL/TEST qualification.
"""
from copy import deepcopy
from contextlib import ExitStack
from pathlib import Path
import hashlib
import math
import numpy as np
from unittest.mock import patch

from .physmani_executor_v7 import PhysManiAdapter as OriginalExecutor
from ..sensor_alignment import physmani_executor_v6 as direct
from ..ablations.physmani_executor_v4 import ProtocolInvariantError, InfrastructureExecutionError

SCHEMA = 'r10_author_aligned_geometry_executor_v1114'
SOURCE = 'our_mani/multitask/author_aligned_geometry_executor_v1114.py'
INHERITED_SOURCES = (
    'our_mani/multitask/physmani_executor_v7.py',
    'our_mani/sensor_alignment/physmani_executor_v6.py',
    'our_mani/sensor_alignment/physmani_executor_v5.py',
    'our_mani/ablations/physmani_executor_v4.py',
    'our_mani/multitask/common_rml_protocol_v530.py',
)
NATIVE_SOURCES = (
    'third_party/RLBench/rlbench/environment.py',
    'third_party/PyRep/pyrep/robots/arms/arm.py',
    'third_party/PyRep/pyrep/robots/configuration_paths/arm_configuration_path.py',
    'third_party/PyRep/pyrep/backend/sim.py',
)


def binding(root, native_repo):
    checksum = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return dict(schema=SCHEMA, source_sha256=checksum(Path(root) / SOURCE),
        inherited_sources_sha256={p: checksum(Path(root) / p) for p in INHERITED_SOURCES},
        native_sources_sha256={p: checksum(Path(native_repo) / p) for p in NATIVE_SOURCES},
        inherited_executor='released_waypoints_counted_actuator_author_rgbd_terminal_v7',
        common_RML_parameters=dict(max_velocity=3., max_acceleration=12., max_jerk=1000.),
        counted_dt=.05, decision_ticks=10, budget_ticks=200,
        planner='original_v7_RRTConnect_ignore_collisions_True',
        public_RRT610_or701_installed=False, public_joint_collision_hard_veto=False,
        default_public_joint_diagnostic_calls=0,
        actual_joint_filter='native joint position bounds only; author direct RML targets',
        velocity_clipping='unchanged_v6_counterfactual_diagnostic_not_actual_qdot_clipping',
        native_task_outcome_and_incremental_gripper_unchanged=True,
        collision_free_or_motion_bound_claimed=False, qualification_granted=False)


def _plain(value, depth=0):
    if depth > 6:
        return dict(status='unknown', reason='bounded_depth')
    kind = type(value)
    if value is None or kind in (bool, int, str):
        return value
    if kind is float:
        return value if math.isfinite(value) else dict(status='unknown', reason='nonfinite')
    if kind in (list, tuple):
        if len(value) > 128:
            return dict(status='unknown', reason='bounded_sequence')
        return [_plain(v, depth + 1) for v in value]
    if kind is dict and len(value) <= 128 and all(type(k) is str for k in value):
        return {k: _plain(v, depth + 1) for k, v in value.items()}
    return dict(status='unknown', reason='value_not_on_plain_data_allowlist')


class DiagnosticPath:
    """Original step exactly once; never restore/veto an original target."""
    def __init__(self, path, diagnostic):
        object.__setattr__(self, '_source_path', path)
        object.__setattr__(self, '_diagnostic', diagnostic)

    def __getattr__(self, name):
        return getattr(self._source_path, name)

    def __setattr__(self, name, value):
        setattr(self._source_path, name, value)

    def step(self):
        result = self._source_path.step()
        generated = self._source_path._joint_position_action
        if generated is not None:
            self._diagnostic(np.asarray(generated, float).copy())
        return result


class AuthorAlignedGeometryExecutor(OriginalExecutor):
    def __init__(self, config, *, observe_public_frame, skill_policy=None,
                 admit_pose_action=None, admit_joint_target=None, collision_diagnostic_limit=0):
        if config.get('dt') != .05 or config.get('baseline_decision_ticks') != 10:
            raise ValueError('unchanged50ms/10tick shared v7 cadence required')
        if not callable(observe_public_frame):
            raise ValueError('existing public observation callback required')
        if admit_pose_action is not None and not callable(admit_pose_action):
            raise ValueError('pose intent integrity callback must be callable')
        if type(collision_diagnostic_limit) is not int or not 0 <= collision_diagnostic_limit <= 64:
            raise ValueError('explicit bounded0..64 joint side diagnostic calls per episode')
        if collision_diagnostic_limit and not callable(admit_joint_target):
            raise ValueError('explicit existing joint diagnostic callback required when enabled')
        if skill_policy is not None and not all(callable(getattr(skill_policy, name, None))
                                               for name in ('qualify_opening', 'confirm_opening')):
            raise ValueError('qualify/confirm witness callbacks required for skill policy')
        # Deliberately omit the inherited neural/WM constructor, like517.
        self.config = config
        self.observe_public_frame = observe_public_frame
        self.skill_policy = skill_policy
        self.admit_pose_action, self.admit_joint_target = admit_pose_action, admit_joint_target
        self.collision_diagnostic_limit = collision_diagnostic_limit
        self.geometry_events, self.execution_events = [], []
        self.diagnostic_calls = self.diagnostic_skipped = 0
        self._owned_env = None

    def observe(self, observation, robot):
        return self.observe_public_frame(observation, robot)

    def predict(self, *args, **kwargs):
        raise ProtocolInvariantError('geometry supplies pose actions; no neural policy prediction')

    def _diagnose(self, joints):
        if self.diagnostic_calls >= self.collision_diagnostic_limit:
            self.diagnostic_skipped += 1
            return
        self.diagnostic_calls += 1
        event = dict(time=float(self._owned_env.time), diagnostic_only=True,
                     target_was_not_vetoed=True, native_outcome_not_inferred=True)
        self.geometry_events.append(event)
        try:
            result = self.admit_joint_target(joints)
            if type(result) is not tuple or len(result) != 2 or type(result[0]) is not bool:
                raise ProtocolInvariantError('side diagnostic must return explicit(bool,proof)')
            accepted, proof = result
            event.update(accepted=accepted, would_reject=not accepted, proof=_plain(proof))
        except (ProtocolInvariantError, InfrastructureExecutionError):
            event['diagnostic_error'] = 'protocol_or_infrastructure'
            raise
        except Exception as error:
            event['diagnostic_error'] = type(error).__name__
            # A probe cleanup/error may have affected simulator state. Never
            # silently convert it to a task failure, an admitted target or success.
            raise InfrastructureExecutionError('bounded public side diagnostic failed') from error

    def execute(self, env, action, *, observer=None):
        if self._owned_env is not None and self._owned_env is not env:
            raise ProtocolInvariantError('one simulator instance per geometry adapter')
        self._owned_env = env
        action = np.asarray(action, float)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError('finite position/quaternion/gripper pose action required')
        proof = dict(reason='finite_explicit_pose_intent')
        if self.admit_pose_action is not None:
            integrity, proof = self.admit_pose_action(action.copy())
            if integrity is not True:
                raise ProtocolInvariantError('pose action differs from its explicit current public intent')
        before = int(env.ledger.ticks)
        service, original_plan = env.map_service, env.map_service.plan_path
        with direct._reference_binding_lock, ExitStack() as scopes:
            if self.collision_diagnostic_limit:
                def plan(*args, **kwargs):
                    # Forward the originalv7 ignore_collisions=True options exactly.
                    return DiagnosticPath(original_plan(*args, **kwargs), self._diagnose)
                scopes.enter_context(patch.object(service, 'plan_path', plan))
            original_apply = direct._apply_direct_target
            if self.skill_policy is not None:
                def apply(owned, joints, *, opening, phase):
                    if owned is not env:
                        raise ProtocolInvariantError('gripper witness escaped owned executor')
                    qualified, witness_proof, proposed = self.skill_policy.qualify_opening(opening, env._requested_opening)
                    result = original_apply(owned, joints, opening=qualified, phase=phase)
                    self.skill_policy.confirm_opening(qualified, witness_proof, proposed, confirmed_after=float(env.time))
                    return result
                scopes.enter_context(patch.object(direct, '_apply_direct_target', apply))
            # Completev7/v6/v5/terminal-arm MRO, including its RML cleanup.
            receipt = super().execute(env, action, observer=observer)
        after = int(env.ledger.ticks)
        if not before < after <= min(before + 10, 200):
            raise ProtocolInvariantError('shared original action must consume1..10ticks within200')
        self.execution_events.append(dict(time_after=float(env.time), ticks_before=before, ticks_after=after,
            action=action.tolist(), pose_admission=_plain(proof), original_executor_receipt=deepcopy(receipt)))
        return receipt

    def geometry_receipt(self):
        return dict(schema=SCHEMA, actions=deepcopy(self.execution_events),
            generated_target_admissions=deepcopy(self.geometry_events),
            public_joint_collision_hard_veto=False, public_RRT610_or701_installed=False,
            collision_diagnostic_limit=self.collision_diagnostic_limit,
            collision_diagnostic_calls=self.diagnostic_calls,
            collision_diagnostic_skipped=self.diagnostic_skipped,
            default_additional_collision_getter_or_probe_calls=0,
            enabled_legacy_diagnostic_may_use_its_original_probes=bool(self.collision_diagnostic_limit),
            native_policy_checkpoint_loaded=False, inherited_actual_v7_execution=True,
            native_task_success_owned_by_parent=True, original741_action_scope_owned_by_parent=True,
            collision_free_or_measured_motion_bound_claimed=False, formal_VAL_TEST_qualification_granted=False)


class AuthorRequestedOpeningWitness:
    """Optional new policy adapter: author opening plus command-edge witness.

The legacy qualifier runs once for public diagnostic/proof compatibility, but
its extra geometric defer does not replace the actual requested opening.
Witnesses certify an executed command edge, never which native object is held.
"""
    def __init__(self, policy, *, on_confirmed_opening=None):
        if not all(callable(getattr(policy, name, None)) for name in ('qualify_opening', 'confirm_opening')):
            raise ValueError('existing public skill witness callbacks required')
        if on_confirmed_opening is not None and not callable(on_confirmed_opening):
            raise ValueError('public confirmation hook must be callable')
        self.source_policy, self.on_confirmed_opening = policy, on_confirmed_opening
        self.opening_events = []

    def __getattr__(self, name):
        return getattr(self.source_policy, name)

    def qualify_opening(self, requested, current_requested):
        legacy, legacy_proof, _ = self.source_policy.qualify_opening(requested, current_requested)
        if type(legacy_proof) is not dict:
            raise ProtocolInvariantError('original public qualifier proof must be plain data')
        actual = float(requested >= .5)
        proof = dict(_plain(legacy_proof), reason='author_requested_opening_command_witness',
                     qualified_opening=actual, desired_opening=actual,
                     current_requested_opening=float(current_requested), legacy_qualified_opening=float(legacy),
                     legacy_qualification=dict(qualified_opening=float(legacy), proof=_plain(legacy_proof)),
                     legacy_geometric_qualification_is_diagnostic_only=True, native_capture_claimed=False)
        selected = self.source_policy.selected
        proposed = None
        if selected is not None and selected.get('phase') == 'acquire' and actual < .5 and current_requested >= .5:
            ids = selected.get('manipulated_ids')
            if type(ids) in (list, tuple) and len(ids) == 1 and type(ids[0]) is str:
                state = self.source_policy.frontend.state
                proposed = dict(phase_index=self.source_policy.phase_index, issued_at=float(state.robot.timestamp),
                    manipulated_ids=list(ids), distance_m=legacy_proof.get('current_position_error_m'),
                    angle_rad=legacy_proof.get('current_orientation_error_rad'),
                    witness_kind='own_executed_open_to_close_command_not_native_grasp_identity')
        return actual, proof, proposed

    def confirm_opening(self, opening, proof, proposed, *, confirmed_after):
        result = self.source_policy.confirm_opening(opening, proof, proposed, confirmed_after=confirmed_after)
        self.opening_events.append(dict(time=float(confirmed_after), opening=float(opening),
            proof=_plain(proof), proposed=_plain(proposed), actually_dispatched=True,
            native_capture_claimed=False))
        if self.on_confirmed_opening is not None:
            self.on_confirmed_opening(self.source_policy, opening, proof, proposed, confirmed_after=confirmed_after)
        return result

    def receipt(self):
        result = dict(self.source_policy.receipt())
        result['physical_grasp_gate'] = dict(mode='author_requested_opening_no_additional_SE3_defer',
                                             native_grasp_and_held_presence_unchanged=True)
        result['author_requested_opening_witness_v1114'] = dict(events=deepcopy(self.opening_events),
            legacy_qualify_and_confirm_calls_preserved=True, native_held_object_identity_read=False)
        return result
