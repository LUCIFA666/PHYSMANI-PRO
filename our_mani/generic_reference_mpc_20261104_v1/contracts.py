"""World commands are expert-intent proposals, never observed object states."""
from dataclasses import dataclass
import numpy as np


def vector(value, size, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError('Finite ' + name + f'[{size}] required')
    return a.copy()


@dataclass(frozen=True)
class WorldTCPReference:
    timestamp: float
    position: object
    quaternion: object
    opening_intent: float
    selected_index: int
    selection_rule: str
    policy_checkpoint_sha256: str
    e_checkpoint_sha256: str
    opening_intent_condition: str = 'after_arm_sequence_unresolved_not_immediate_actuator_command'

    def __post_init__(self):
        if isinstance(self.timestamp, bool) or not np.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError('Real current observation timestamp required')
        p, q = vector(self.position, 3, 'world TCP position'), vector(self.quaternion, 4, 'world TCP quaternion')
        if not np.isclose(np.linalg.norm(q), 1., atol=1e-6, rtol=0):
            raise ValueError('Proper unit world TCP quaternion required')
        if not np.isfinite(self.opening_intent) or not 0 <= self.opening_intent <= 1:
            raise ValueError('Opening intent in[0,1] is a future desired intent, not an immediate actuator command')
        if self.opening_intent_condition != 'after_arm_sequence_unresolved_not_immediate_actuator_command':
            raise ValueError('World command labels do not supervise actual closing time or object alignment')
        if type(self.selected_index) is not int or not 0 <= self.selected_index < 8:
            raise ValueError('Explicit caller-selected K8 index required')
        if self.selection_rule not in ('caller_selected', 'highest_demo_likelihood'):
            raise ValueError('No implicit closest-TCP/task-success reference selection')
        for value in (self.policy_checkpoint_sha256, self.e_checkpoint_sha256):
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Actual policy and feature-extractor whole SHA identities required')
        p.setflags(write=False); q.setflags(write=False)
        object.__setattr__(self, 'position', p); object.__setattr__(self, 'quaternion', q)


def select_reference(proposals, *, timestamp, policy_sha256, e_sha256,
                     selected_index=None, selection_rule=None):
    """Optional explicit demo-likelihood ranking; no fake tau/probability head."""
    from scipy.spatial.transform import Rotation
    expected = {'world_tcp_positions', 'world_tcp_rotations', 'opening_intent', 'opening_logits',
        'proposal_logits', 'proposal_score_semantics', 'calibrated_success_probability',
        'output_semantics', 'handoff_tau_seconds', 'physical_handoff_qualified'}
    if set(proposals) != expected or proposals['output_semantics'] != 'expert_command_intent_not_observed_state':
        raise ValueError('Exact new world-command proposal schema required')
    if (proposals['handoff_tau_seconds'] is not None or proposals['physical_handoff_qualified'] is not False
            or proposals['calibrated_success_probability'] is not False
            or proposals['proposal_score_semantics'] != 'expert_command_demonstration_mixture_not_task_success'):
        raise ValueError('No handoff-time or physical-success claim may be fabricated')
    p, R, op, logits = [np.asarray(proposals[k], float) for k in
        ('world_tcp_positions', 'world_tcp_rotations', 'opening_intent', 'proposal_logits')]
    opening_logits = np.asarray(proposals['opening_logits'], float)
    if (p.shape != (8, 3) or R.shape != (8, 3, 3) or op.shape != (8,) or logits.shape != (8,)
            or opening_logits.shape != (8,)
            or any(not np.isfinite(a).all() for a in (p, R, op, logits, opening_logits))
            or np.any((op < 0) | (op > 1))
            or not np.allclose(R.transpose(0, 2, 1) @ R, np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(R), 1., atol=1e-5, rtol=0)):
        raise ValueError('Eight finite world SE3/open proposals and trained demo logits required')
    if selection_rule == 'highest_demo_likelihood':
        if selected_index is not None: raise ValueError('Do not mix explicit index with demo ranking')
        selected_index = int(np.argmax(logits))  # stable first-index tie, not success probability
    elif selection_rule != 'caller_selected' or type(selected_index) is not int:
        raise ValueError('Caller must explicitly declare reference selection')
    if type(selected_index) is not int or not 0 <= selected_index < 8:
        raise ValueError('Explicit K8 index must lie in0..7')
    return WorldTCPReference(timestamp, p[selected_index], Rotation.from_matrix(R[selected_index]).as_quat(),
        float(op[selected_index]), selected_index, selection_rule, policy_sha256, e_sha256)


def retained_gripper_command(state):
    """Actual persistent motor target; never threshold the BC opening intent.

    An executed100ms receipt only counts two physical ticks. It does not prove
    arm arrival, task-object alignment, or grant an arm-to-gripper transition.
    """
    h = state.get('actuator_state')
    if not isinstance(h, dict):
        raise ValueError('Actual persistent actuator state required')
    command = h.get('gripper_target')
    if (not np.isscalar(command) or isinstance(command, (bool, np.bool_))
            or command not in (0., 1.)):
        raise ValueError('Actual binary persistent gripper target required')
    return float(command)
