"""Matched TRAIN variant: retain original qualification BEFORE a first close.

Only this decision differs from v1118: an acquire request cannot spend its
first actual close edge while the original public qualifier still defers it.
Native gripper semantics, actuator, public association and postcapture phase
repair remain identical. Release and other openings remain v1118 passthrough.
This is a policy comparison, not evidence of which object Native actually held.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import hashlib

from . import author_aligned_policy_integration_v1118 as parent
from .author_aligned_geometry_executor_v1114 import AuthorRequestedOpeningWitness

SCHEMA = 'r10_original_public_preclose_qualification_matched_variant_v1121'
SOURCE = 'our_mani/multitask/retained_preclose_policy_integration_v1121.py'


def binding(root, native_repo):
    return dict(schema=SCHEMA,
        source_sha256=hashlib.sha256((Path(root)/SOURCE).read_bytes()).hexdigest(),
        parent=parent.binding(root, native_repo),
        changed_factor='first_acquire_close_uses_original_policy_qualification',
        new_numeric_thresholds=False, Native_gripper_changed=False,
        release_and_other_opening_dispatch='unchanged_parent_author_requested_opening',
        public_joint_collision_hard_veto=False,
        action_conditioned_mode_and_remove_postcapture_transition_unchanged=True,
        additional_native_getters_or_success_calls=0,
        native_held_identity_verified=False, formal_VAL_TEST_qualification=False)


class RetainedPrecloseWitness(AuthorRequestedOpeningWitness):
    def __init__(self, policy, *, on_confirmed_opening=None, receipt_source=None):
        super().__init__(policy, on_confirmed_opening=on_confirmed_opening)
        self.receipt_source = receipt_source

    def qualify_opening(self, requested, current_requested):
        source = self.source_policy
        goal, witness = source.selected, source.close_witness
        own_close = witness is not None and witness.get('phase_index') == source.phase_index
        if (type(goal) is dict and goal.get('phase') == 'acquire'
                and requested < .5 <= current_requested and not own_close):
            # Call the actual original policy qualifier once. Its existing
            # public 2cm/0.1rad test (where applicable) and proposed witness
            # remain its responsibility. No new target or observation is read.
            qualified, proof, proposed = source.qualify_opening(requested, current_requested)
            if type(proof) is not dict:
                raise TypeError('original public opening qualification proof required')
            proof = dict(proof, dispatch_variant=SCHEMA,
                original_public_preclose_qualification_retained=True,
                legacy_geometric_qualification_is_diagnostic_only=False,
                native_capture_claimed=False)
            return qualified, proof, proposed
        return super().qualify_opening(requested, current_requested)

    def receipt(self):
        result = dict(self.receipt_source() if self.receipt_source else self.source_policy.receipt())
        # The parent facade was constructed, but this wrapper actually supplied
        # its opening calls. Do not retain the parent's empty dispatch trace or
        # a misleading blanket statement that preclose qualification was bypassed.
        result.pop('author_requested_opening_witness_v1114', None)
        result['physical_grasp_gate'] = dict(
            mode='original_public_policy_qualification_before_first_acquire_close',
            new_thresholds=False, native_gripper_unchanged=True,
            release_and_other_openings='parent_author_requested_opening')
        result['retained_preclose_dispatch_v1121'] = dict(schema=SCHEMA,
            events=deepcopy(self.opening_events),
            original_qualifier_and_confirm_called_once_per_actual_counted_command=True,
            witness_published_only_after_original_apply_and_confirm=True,
            Native_gripper_changed=False, extra_native_getters_or_success_calls=0,
            postcapture_spoke_geometry_gate_restored=False,
            task_success_or_native_held_identity_inferred=False)
        return result


@contextmanager
def integrated_components(root, config, task, actor):
    """Keep the complete parent lifetime; replace only its opening facade."""
    with parent.integrated_components(root, config, task, actor) as (frontend, dispatched, executor):
        if not isinstance(dispatched, AuthorRequestedOpeningWitness):
            # The original button route owns no acquire close witness.
            yield frontend, dispatched, executor
            return
        qualified = RetainedPrecloseWitness(dispatched.source_policy,
            on_confirmed_opening=dispatched.on_confirmed_opening,
            receipt_source=dispatched.receipt)
        with patch.object(executor, 'skill_policy', qualified):
            yield frontend, qualified, executor
