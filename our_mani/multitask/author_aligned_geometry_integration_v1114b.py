"""Fixed integration of the frozen v1114 actuator for a new experiment.

Only validate an issued public pose intent here. Never call a policy's legacy
admit_pose/admit_joint or a collision port. Native planning, physics and task
outcomes remain owned by the original v7 execution chain and its outer runner.
"""
import hashlib
import math
from pathlib import Path
import numpy as np

from ..contracts import Pose, RobotState
from .author_aligned_geometry_executor_v1114 import (
    AuthorAlignedGeometryExecutor, AuthorRequestedOpeningWitness,
    binding as executor_binding, ProtocolInvariantError,
)

SCHEMA = 'r10_author_aligned_geometry_integration_v1114b'
SOURCE = 'our_mani/multitask/author_aligned_geometry_integration_v1114b.py'
FROZEN_EXECUTOR_SOURCE = 'our_mani/multitask/author_aligned_geometry_executor_v1114.py'
FROZEN_EXECUTOR_SHA256 = '83199e573969d9d9bcd69cb1e4652edfab010fb815fcd19bfd0dc59d78c7d91e'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(root, native_repo):
    root = Path(root)
    if _sha(root/FROZEN_EXECUTOR_SOURCE) != FROZEN_EXECUTOR_SHA256:
        raise ProtocolInvariantError('frozen v1114 actuator source changed')
    return dict(schema=SCHEMA, source_sha256=_sha(root/SOURCE),
        frozen_executor_sha256=FROZEN_EXECUTOR_SHA256,
        executor=executor_binding(root, native_repo),
        public_contracts_sha256=_sha(root/'our_mani/contracts.py'),
        admission='same_current_public_selected_pose_and_opening_only',
        legacy_pose_or_joint_admission_called=False,
        collision_diagnostic_limit=0, public_collision_hard_veto=False,
        opening_dispatch='original_v7_requested_opening_with_public_command_edge_witness',
        legacy_geometric_opening_qualification='diagnostic_only_no_veto',
        additional_native_getter_or_collision_probe_calls=0,
        inherited_native_gripper_mechanism=True,
        old_policy_SE3_defer_behavior_preserved=False,
        parent_owns_741_action_scope_530_attestation_and_hold_budget=True,
        qualification_granted=False, collision_free_claimed=False)


def protocol_pose_admission(policy):
    """Return an integrity callback over already existing public records only.

    `selected` uses the same pose/opening/selected_at fields as v518/v548. Do
    not delegate to admit_pose: a new policy override must not reintroduce a
    collision gate through this callback. An invalid intent is a protocol error
    at the actuator boundary; it is not a physical/task failure or success.
    """
    def admit(action):
        selected = policy.selected
        state = policy.frontend.state
        if type(selected) is not dict or state is None:
            return False, dict(reason='current_public_selected_intent_required', protocol_only=True)
        robot = state.robot
        if type(robot) is not RobotState or type(selected.get('pose')) is not Pose:
            return False, dict(reason='original_public_robot_and_pose_DTOs_required', protocol_only=True)
        issued = selected.get('selected_at'); opening = selected.get('opening')
        if (type(issued) not in (int, float) or not math.isfinite(issued)
                or type(opening) not in (int, float) or not math.isfinite(opening)):
            return False, dict(reason='finite_public_intent_time_and_opening_required', protocol_only=True)
        action = np.asarray(action, float)
        current = float(robot.timestamp)
        if action.shape != (8,) or not np.isfinite(action).all():
            return False, dict(reason='finite_eight_component_pose_action_required', protocol_only=True)
        same_time = abs(current-issued) < 1e-8
        same_action = np.array_equal(action, np.r_[selected['pose'].xyzw(), opening])
        accepted = bool(same_time and same_action)
        return accepted, dict(reason='same_current_public_pose_intent' if accepted else 'public_intent_mismatch',
            protocol_only=True, selected_at=float(issued), public_robot_timestamp=current,
            same_time=bool(same_time), same_action=bool(same_action),
            collision_predicate_called=False, native_truth_read=False)
    return admit


def construct(config, frontend, policy, *, on_confirmed_opening=None):
    """Return `(frontend, dispatched_policy, executor)` without Native queries.

    Parent builds the source-bound frontend/new phase policy first. Its outer
    runner retains the existing observer, 741 action scope, 530 attestation,
    one-tick hold and 200-tick budget. The optional hook runs AFTER the original
    confirm succeeds, with `(source_policy, opening, proof, proposed,
    confirmed_after=...)`; publish to v1115 only when proposed is not None.
    """
    if policy.frontend is not frontend or not callable(getattr(frontend, 'observe', None)):
        raise ProtocolInvariantError('one existing public frontend must own the selected phase policy')
    dispatched = AuthorRequestedOpeningWitness(policy, on_confirmed_opening=on_confirmed_opening)
    executor = AuthorAlignedGeometryExecutor(config, observe_public_frame=frontend.observe,
        skill_policy=dispatched, admit_pose_action=protocol_pose_admission(policy),
        admit_joint_target=None, collision_diagnostic_limit=0)
    return frontend, dispatched, executor
