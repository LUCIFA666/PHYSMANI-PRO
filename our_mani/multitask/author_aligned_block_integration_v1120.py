"""New block prototype on the original v538 public frontend and v7 actuator.

Keep the block's public tracker, first-contact support geometry and phase
rules. Its own opening qualifier runs once as a side diagnostic; its extra
SE3 defer does not veto the author's actual requested opening. Command-edge
witnesses do not certify a physical grasp. No Native/VAL/TEST qualification.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import numpy as np

from ..contracts import Pose, RobotState
from .joint16_tracks_v31 import BLOCKS
from .physmani_block_policy_v538 import (
    PublicBlockCollision, BlockPublicFrontend, BlockGeometryIntents,
)
from .author_aligned_geometry_executor_v1114 import (
    AuthorAlignedGeometryExecutor, ProtocolInvariantError, binding as actuator_binding,
    _plain,
)

SCHEMA = 'r10_author_aligned_block_integration_v1120'
SOURCE = 'our_mani/multitask/author_aligned_block_integration_v1120.py'
FROZEN_ACTUATOR = 'our_mani/multitask/author_aligned_geometry_executor_v1114.py'
FROZEN_ACTUATOR_SHA256 = '83199e573969d9d9bcd69cb1e4652edfab010fb815fcd19bfd0dc59d78c7d91e'
PUBLIC_SOURCES = (
    'our_mani/multitask/physmani_block_policy_v538.py',
    'our_mani/multitask/physmani_block_receipt_v541.py',
    'our_mani/multitask/physmani_block_record_v545.py',
    'our_mani/multitask/block_held_support_v298.py',
    'our_mani/multitask/joint16_block_context_v97.py',
    'our_mani/multitask/joint16_tracks_v31.py',
    'our_mani/benchmark_perception.py', 'our_mani/goal_perception.py', 'our_mani/contracts.py',
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(root, native_repo):
    root = Path(root)
    if _sha(root/FROZEN_ACTUATOR) != FROZEN_ACTUATOR_SHA256:
        raise ProtocolInvariantError('frozen author actuator changed before block integration')
    return dict(schema=SCHEMA, source_sha256=_sha(root/SOURCE), tasks=list(BLOCKS),
        public_sources_sha256={name: _sha(root/name) for name in PUBLIC_SOURCES},
        actuator=actuator_binding(root, native_repo),
        source_route='original744_explicit_v538_block_frontend_and_policy',
        public_perception_support_and_phase_rules_unchanged=True,
        actual_requested_opening_dispatched=True, legacy_SE3_opening_defer='diagnostic_only',
        public_joint_collision_hard_veto=False, collision_diagnostic_limit=0,
        original_native_incremental_gripper_mechanism=True,
        additional_native_getter_observation_or_success_calls=0,
        parent_owns_original_ReceiptBlockView_and_block_action_scope=True,
        parent_owns_530_attestation_reset_and_200tick_budget=True,
        actual_motion_or_collision_free_guarantee=False, formal_VAL_TEST_qualification=False)


def protocol_block_admission(policy):
    """Only existing immutable public DTOs and the issued block action record."""
    def admit(action):
        selected = policy.selected; robot = policy.frontend.robot
        if type(selected) is not dict or type(robot) is not RobotState or type(selected.get('pose')) is not Pose:
            return False, dict(reason='original_public_block_selected_pose_and_robot_required', protocol_only=True)
        when = selected.get('selected_at'); opening = selected.get('opening')
        if (type(when) not in (int, float) or not math.isfinite(when)
                or type(opening) not in (int, float) or not math.isfinite(opening)):
            return False, dict(reason='finite_block_intent_time_and_opening_required', protocol_only=True)
        recorded = selected.get('action')
        if type(recorded) is not np.ndarray or recorded.shape != (8,) or not np.isfinite(recorded).all():
            return False, dict(reason='original_explicit_block_action_record_required', protocol_only=True)
        action = np.asarray(action, float)
        expected = np.r_[selected['pose'].xyzw(), opening]
        same_time = abs(float(robot.timestamp)-when) < 1e-8
        same_action = action.shape == (8,) and np.isfinite(action).all() and np.array_equal(action, expected) and np.array_equal(action, recorded)
        accepted = bool(same_time and same_action)
        return accepted, dict(reason='same_current_public_block_pose_intent' if accepted else 'block_intent_mismatch',
            protocol_only=True, selected_at=float(when), public_robot_timestamp=float(robot.timestamp),
            same_time=bool(same_time), same_action=bool(same_action),
            legacy_pose_or_joint_admission_called=False, collision_predicate_called=False)
    return admit


class AuthorBlockRequestedOpeningWitness:
    """Block-specific command witness; no phase_index or manipulated IDs needed."""
    def __init__(self, policy, *, max_events=200):
        if not all(callable(getattr(policy, name, None)) for name in ('qualify_opening', 'confirm_opening')):
            raise ValueError('original block qualify/confirm callbacks required')
        if type(max_events) is not int or not 1 <= max_events <= 1000:
            raise ValueError('bounded block opening diagnostic trace required')
        self.source_policy = policy; self.max_events = max_events
        self.events = []; self.dropped_events = 0

    def __getattr__(self, name):
        return getattr(self.source_policy, name)

    def qualify_opening(self, requested, current_requested):
        legacy, legacy_proof, legacy_proposed = self.source_policy.qualify_opening(requested, current_requested)
        if type(legacy_proof) is not dict:
            raise ProtocolInvariantError('original block qualifier must return a plain proof')
        robot = self.source_policy.frontend.robot
        if type(robot) is not RobotState:
            raise ProtocolInvariantError('block witness requires the existing public robot DTO')
        actual = float(requested >= .5)
        proof = dict(_plain(legacy_proof), reason='author_requested_block_opening_command_witness',
            desired_opening=actual, qualified_opening=actual, current_requested_opening=float(current_requested),
            legacy_qualification=dict(qualified_opening=float(legacy), proof=_plain(legacy_proof),
                                      proposed=_plain(legacy_proposed)),
            legacy_SE3_defer_is_diagnostic_only=True, native_capture_claimed=False, native_truth_read=False)
        selected = self.source_policy.selected
        proposed = None
        if (self.source_policy.phase == 'acquire' and type(selected) is dict
                and selected.get('phase') == 'acquire' and actual < .5 <= current_requested):
            proposed = dict(issued_at=float(robot.timestamp), distance_m=legacy_proof.get('distance_m'),
                angle_rad=legacy_proof.get('angle_rad'),
                witness_kind='own_executed_block_open_to_close_command_not_native_capture')
        return actual, proof, proposed

    def confirm_opening(self, opening, proof, proposed, *, confirmed_after):
        result = self.source_policy.confirm_opening(opening, proof, proposed, confirmed_after=confirmed_after)
        event = dict(opening=float(opening), confirmed_after=float(confirmed_after), proof=_plain(proof),
                     proposed=_plain(proposed), actually_dispatched=True, native_capture_claimed=False)
        if len(self.events) < self.max_events:
            self.events.append(event)
        else:
            self.dropped_events += 1
        return result

    def receipt(self):
        result = dict(self.source_policy.receipt())
        held = deepcopy(result.get('held_geometry', {}))
        result['legacy_held_geometry_receipt'] = held
        # v538's legacy receipt sets this True unconditionally, including when
        # no admit_joint/check ran. Preserve it as legacy metadata and report
        # the new execution truth in the active receipt.
        result['held_geometry'] = dict(held,
            held_target_vs_arm_and_static_obstacles_checked=False,
            gripper_only_contact_exemption=False, public_joint_collision_predicate_executed=False,
            active_use='original_first_contact_public_support_geometry_only')
        result['author_aligned_block_integration_v1120'] = dict(schema=SCHEMA,
            requested_opening_dispatched=True, legacy_opening_qualifier_is_diagnostic_only=True,
            original_frontend_support_and_phase_rules_unchanged=True,
            actual_native_gripper_mechanism_unchanged=True, native_capture_claimed=False,
            public_joint_collision_hard_veto=False, collision_diagnostic_limit=0,
            opening_events=deepcopy(self.events), dropped_events=self.dropped_events,
            native_observation_getter_or_success_calls_added=0, formal_VAL_TEST_qualification=False)
        return result


def adapt(config, frontend, policy):
    """Adapt already constructed original538 objects, without extra observations."""
    if policy.frontend is not frontend or not callable(getattr(frontend, 'observe', None)):
        raise ProtocolInvariantError('one original public block frontend must own its intent policy')
    dispatched = AuthorBlockRequestedOpeningWitness(policy)
    executor = AuthorAlignedGeometryExecutor(config, observe_public_frame=frontend.observe,
        skill_policy=dispatched, admit_pose_action=protocol_block_admission(policy),
        admit_joint_target=None, collision_diagnostic_limit=0)
    return frontend, dispatched, executor


def construct(root, config, task, method, env):
    """Same explicit block constructors/order as frozen744, plus the new actuator."""
    if task not in BLOCKS or method not in ('cv_geometry', 'context_geometry'):
        raise ValueError('exactly the two original block tasks and geometry predictors')
    collision = PublicBlockCollision(env)
    frontend = BlockPublicFrontend(Path(root), config, task, method, env.robot_model, collision)
    policy = BlockGeometryIntents(frontend)
    return adapt(config, frontend, policy)


@contextmanager
def integrated_block_components(root, config, task, method, env):
    """Parent-owned scope API matching1118; env/reset/outcomes stay parent-owned."""
    yield construct(root, config, task, method, env)
