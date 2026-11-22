"""Prototype finger-only inference for the existing coupled response.

The original checkpoint, model and generic servo adapter remain unchanged.
This view shares the original CPU model object and computes exactly the same
finger outputs, without constructing the unconsumed Torch TCP/FK output.
It implements only CoupledWithFingerResponse's independent-finger contract:
it is intentionally not a generic joint transition and returns no q/dq.

No fitting, simulator, device selection, action selection or contact inference
occurs here.  Integration and numerical equivalence are separate obligations.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
import torch

from ..local_mpc_p2_20261103_v2.servo import JointServoResidual
from ..local_mpc_p2_20261103_v2.servo_adapter import TorchServoTransition, _matrix


class FrozenFingerOnlyResponse:
    """Share one exact old adapter's parameters, preserving its finger math.

    Discarded joint outputs still undergo their original float32 arithmetic
    and finite checks.  In particular, an infinite raw joint residual can
    saturate through tanh to a finite result; rejecting raw instead would
    change the original acceptance contract.
    """

    def __init__(self, original: TorchServoTransition):
        if type(original) is not TorchServoTransition or type(original.model) is not JointServoResidual:
            raise TypeError('exact original v2 TorchServoTransition/JointServoResidual required')
        model = original.model
        if model.training or any(layer.training for layer in model.modules()):
            raise ValueError('original CPU model must already be in eval mode')
        # Direct parameter inference cannot reproduce callbacks around the
        # skipped parent/FK forward. Checkpoint-loaded models have no hooks;
        # arbitrary hooked module behavior is outside this computation view.
        module_hooks = torch.nn.modules.module
        if (any(layer._forward_pre_hooks or layer._forward_hooks for layer in model.modules())
                or module_hooks._global_forward_pre_hooks or module_hooks._global_forward_hooks):
            raise ValueError('hook-free original frozen inference model required')
        if any(value.device.type != 'cpu' for value in (*model.parameters(), *model.buffers())):
            raise ValueError('shared original model must remain on CPU')
        if any(value.dtype != torch.float32 for value in model.parameters()):
            raise ValueError('original float32 network parameters required')
        self.original = original
        self.model = model  # No weight copy, replacement, .eval(), or mutation.
        self.evidence = deepcopy(original.evidence)
        self.evidence['finger_computation_view'] = dict(
            schema='shared_v2_finger_only_without_unused_TorchFK_20261103_v1',
            original_model_object_shared=True, original_weights_changed=False,
            original_joint_finite_checks_preserved=True, unused_TorchFK_skipped=True,
            original_forward_hooks_supported=False,
            generic_joint_transition=False, production_integrated=False,
            held_or_contact_inferred=False, equivalence_must_be_verified=True)
        self.metadata = deepcopy(original.metadata)
        self.metadata['output_scope'] = 'independent_fingers_feasibility_latent_only_for_CoupledWithFingerResponse'

    def predict_execution(self, x, references, latent, dt, *, held_anchor):
        if dt != .05:
            raise ValueError('servo network is trained for exactly 50ms transitions')
        q = np.asarray(x.get('q'), dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 7:
            raise ValueError('actual or predicted q[N,7] required')
        n = len(q)
        q = _matrix(q, (n, 7), 'q')
        dq = _matrix(x.get('dq'), (n, 7), 'dq')
        fingers = _matrix(x.get('finger_open_amounts'), (n, 2), 'actual two finger openings')
        if np.any((fingers < 0) | (fingers > 1)):
            raise ValueError('two actual finger openings must be in [0,1]')
        anchor = np.asarray(held_anchor)
        if anchor.shape != (n,) or anchor.dtype != np.bool_:
            raise ValueError('explicit bool held_anchor[N] required')
        reference = np.concatenate([_matrix(references.get(key), (n, 7), key)
                                    for key in ('q_ref', 'dq_ref', 'ddq_ref')], axis=-1)
        target = _matrix(references.get('gripper_target'), (n,), 'gripper_target')
        if not np.isin(target, (0., 1.)).all():
            raise ValueError('binary dispatched gripper target required')
        reference = np.c_[reference, target]
        state = np.c_[q, dq, fingers, anchor.astype(np.float64)]
        with torch.inference_mode():
            state = torch.as_tensor(state, dtype=torch.float32)
            reference = torch.as_tensor(reference, dtype=torch.float32)
            # These are the original forward's checks and ordered expressions.
            if (state.shape[-1] != 17 or reference.shape[:-1] != state.shape[:-1]
                    or reference.shape[-1] != 22 or not state.is_floating_point()
                    or not reference.is_floating_point() or not bool(torch.isfinite(state).all())
                    or not bool(torch.isfinite(reference).all())):
                raise ValueError('aligned finite actual state and known reference required')
            model = self.model
            raw = model.network((model.features(state, reference) - model.feature_mean) / model.feature_std)
            rq = raw[..., :7].tanh() * model.residual_q_bound
            rv = raw[..., 7:14].tanh() * model.residual_dq_bound
            next_q = reference[..., :7] + rq
            next_dq = reference[..., 7:14] + rv
            opening = state[..., 14:16]
            if not bool(((opening >= -1e-5) & (opening <= 1 + 1e-5)).all()):
                raise ValueError('actual finger openings outside [0,1]')
            rates = raw[..., 14:].reshape(*raw.shape[:-1], 2, 2).sigmoid()
            cap = opening.new_tensor(model.aperture_step_bound)
            opened = (opening - torch.minimum(opening, cap) * rates[..., 0]
                      + torch.minimum(1 - opening, cap) * rates[..., 1])
            # Old FK rejects nonfinite q before the adapter's finite checks.
            # The discarded q/dq checks stay, but no unused FK or q/dq copy.
            if not bool(torch.isfinite(next_q).all()):
                raise ValueError('finite floating joints[...,dof] required')
            if not bool(torch.isfinite(next_dq).all()):
                raise ValueError('nonfinite/out-of-range servo prediction')
            predicted_fingers = opened.detach().cpu().numpy().astype(np.float64, copy=True)
        if (not np.isfinite(predicted_fingers).all() or predicted_fingers.shape != (n, 2)
                or np.any((predicted_fingers < 0) | (predicted_fingers > 1))):
            raise ValueError('nonfinite/out-of-range servo prediction')
        return dict(finger_open_amounts=predicted_fingers, feasible=np.ones(n, dtype=bool),
                    latent=deepcopy(latent))
