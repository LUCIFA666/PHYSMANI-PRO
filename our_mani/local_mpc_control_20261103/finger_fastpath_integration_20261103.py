"""Explicit per-core integration of the frozen finger-only computation view.

Call only after constructing a fresh coupled core and before its coordinator
or any phase-specific execution model is constructed. This changes computation
only: joint response, reference/FK callbacks, weights, masks, fingers and latent
semantics stay unchanged. No actual state or virtual hidden state is stored.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

from ..local_mpc_p2_20261103.execution_models import JointResidualFKExecutionModel
from ..local_mpc_p2_20261103_v2.servo_adapter import TorchServoTransition
from .coupled_response import CoupledWithFingerResponse, FrozenCoupledResponse
from .finger_response_fastpath_20261103 import FrozenFingerOnlyResponse


SCHEMA = 'per_core_frozen_finger_fastpath_integration_20261103_v1'
CHECKPOINT_SHA256 = '25a0e0cbd49e0e0c9e8a4671649f96a1bb565013011d728e88b2c2f3eaca0073'
PROTOTYPE_SHA256 = '77373c75adaca325008cb532e92bf2437cf250a7c8dc528c6f0658715c5443d6'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def enable_finger_fastpath(base_execution, provenance):
    """Mutate one fresh core's transition; return a new provenance dictionary.

    The original adapter/model/evidence and caller provenance are never changed.
    Unknown outer wrappers and repeated enabling are rejected, never flattened.
    All checks and replacement construction finish before the sole assignment
    to ``base_execution.transition_batch``. Callers must not run a core while
    enabling it; this is a construction-time operation, not a live toggle.
    """
    if type(base_execution) is not JointResidualFKExecutionModel:
        raise TypeError('exact unwrapped JointResidualFKExecutionModel core required')
    if type(provenance) is not dict:
        raise TypeError('actual factory provenance dictionary required')
    previous = provenance.get('finger_response_fastpath')
    if previous is not None:
        if not isinstance(previous, dict) or previous.get('runtime_enabled') is not False:
            raise ValueError('finger fastpath cannot be enabled twice or from ambiguous provenance')
    transition = base_execution.transition_batch
    if type(transition) is not CoupledWithFingerResponse:
        raise TypeError('exact CoupledWithFingerResponse required; unknown wrappers are not flattened')
    if type(transition.finger_response) is FrozenFingerOnlyResponse:
        raise ValueError('finger fastpath is already enabled on this core')
    if type(transition.joint_response) is not FrozenCoupledResponse:
        raise TypeError('exact original frozen global coupled joint response required')
    original = transition.finger_response
    if type(original) is not TorchServoTransition:
        raise TypeError('exact original v2 TorchServoTransition finger response required')
    evidence = original.evidence
    checkpoint = provenance.get('checkpoint', {})
    if (not isinstance(evidence, dict) or not isinstance(checkpoint, dict)
            or evidence.get('sha256') != CHECKPOINT_SHA256 or evidence.get('trained') is not True
            or evidence.get('training_complete') is not True or evidence.get('formal_TEST_used') is not False
            or checkpoint.get('sha256') != CHECKPOINT_SHA256 or checkpoint.get('training_complete') is not True
            or checkpoint.get('neural_response_scope') != 'two_finger_response_only_NN_joint_predictions_discarded'
            or provenance.get('controller') != 'coupled'):
        raise ValueError('exact completed frozen finger checkpoint and coupled-factory provenance required')
    original_path = Path(evidence.get('path', '')).resolve(strict=True)
    if Path(checkpoint.get('path', '')).resolve(strict=True) != original_path or _sha(original_path) != CHECKPOINT_SHA256:
        raise ValueError('finger checkpoint path/bytes differ from the frozen factory identity')
    prototype = Path(__file__).with_name('finger_response_fastpath_20261103.py')
    if _sha(prototype) != PROTOTYPE_SHA256:
        raise ValueError('finger computation prototype differs from the equivalence-reviewed source')

    # The view shares original.model; its evidence/metadata are deep copies.
    view = FrozenFingerOnlyResponse(original)
    view.evidence['finger_computation_view'].update(production_integrated=True, runtime_enabled=True,
        gripper_or_contact_requalified=False, original_evidence_modified=False)
    replacement = CoupledWithFingerResponse(transition.joint_response, view)
    source_hashes = {str(path.resolve()): _sha(path) for path in (Path(__file__), prototype)}
    updated = deepcopy(provenance)
    updated.setdefault('source_sha256', {}).update(source_hashes)
    updated['finger_response_fastpath'] = dict(schema=SCHEMA, runtime_enabled=True,
        computation_only=True, checkpoint_sha256=CHECKPOINT_SHA256, original_model_object_shared=True,
        original_weights_changed=False, original_adapter_evidence_modified=False,
        exact_joint_response_object_preserved=True, execution_model_object_preserved=True,
        reference_step_limits_FK_config_latent_initializer_preserved=True,
        actual_state_or_h_cached=False, cross_case_model_or_h_shared_by_integration=False,
        unused_TorchFK_skipped=True, discarded_joint_finite_checks_preserved=True,
        finger_computation_view=deepcopy(view.evidence['finger_computation_view']),
        source_sha256=source_hashes, finger_response_requalified=False,
        contact_or_grasp_requalified=False, precision_qualified=False,
        hard_realtime_qualified=False, Native_started=False, training_started=False)
    # Sole mutation, after all guards and independent provenance construction.
    base_execution.transition_batch = replacement
    return updated
