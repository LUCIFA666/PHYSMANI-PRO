"""Retain538's original qualification for the first block close edge only.

The complete1120 frontend, support geometry, phase controller and1114 actuator
remain in their parent scope. The parent's one raw qualifier call supplies the
original command/proposal; it is never called again to reconstruct evidence.
This public command rule neither identifies Native held bodies nor certifies
that2cm/.1rad is necessary or sufficient for a physical grasp.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from unittest.mock import patch

from . import author_aligned_block_integration_v1120 as parent
from .author_aligned_geometry_executor_v1114 import ProtocolInvariantError, _plain

SCHEMA='r10_retained_original_block_preclose_integration_v1153'
SOURCE='our_mani/multitask/retained_preclose_block_integration_v1153.py'
PARENT_SOURCE=parent.SOURCE
PARENT_SHA='aac6bddf10e9360e7606718d357b1acd9c5241bf92ab684d3a54a591759cd147'
QUALIFIER_SOURCE='our_mani/multitask/physmani_block_policy_v538.py'
QUALIFIER_SHA='7de2c7d4b8821c770cd8c646cdc924016b75b13f5ae5aa3514f4ae427be888fd'
SOURCES=tuple(dict.fromkeys((SOURCE,PARENT_SOURCE,QUALIFIER_SOURCE,*parent.PUBLIC_SOURCES,parent.FROZEN_ACTUATOR)))


def _sha(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def binding(root,native_repo):
    root=Path(root)
    if _sha(root/PARENT_SOURCE)!=PARENT_SHA or _sha(root/QUALIFIER_SOURCE)!=QUALIFIER_SHA:
        raise ProtocolInvariantError('frozen1120/538 block parent or qualifier changed')
    return dict(schema=SCHEMA,source_sha256=_sha(root/SOURCE),parent=parent.binding(root,native_repo),
        sources_sha256={name:_sha(root/name) for name in SOURCES},tasks=list(parent.BLOCKS),
        changed_factor='retained_original_block_first_preclose_witness',
        active_first_preclose=dict(source=QUALIFIER_SOURCE,source_sha256=QUALIFIER_SHA,
            callable='BlockGeometryIntents.qualify_opening',
            original_public_position_tolerance_m=.02,original_public_angle_tolerance_rad=.1,
            requires_original_current_grasp_and_open_public_gripper=True,new_numeric_thresholds=False),
        release_repeated_close_and_other_phases='exact_parent1120_requested_opening_dispatch',
        complete_parent_construction_and_lifetime_retained=True,
        frontend_support_phase_and1114_native_actuator_unchanged=True,
        qualifier_calls_per_original_counted_command=1,
        witness_published_only_after_original_apply_success_and_confirm=True,
        public_joint_collision_hard_veto=False,collision_diagnostic_limit=0,
        native_gripper_and_source_model_checkpoint_unchanged=True,
        additional_native_getters_observations_predictions_or_success_calls=0,
        public_threshold_is_native_grasp_necessity_or_sufficiency_claim=False,
        native_capture_or_held_identity_verified=False,formal_VAL_TEST_qualification=False)


class RetainedBlockPrecloseWitness:
    """Facade over the SAME1120 dispatch object and raw538 policy."""
    def __init__(self,dispatched):
        if not isinstance(dispatched,parent.AuthorBlockRequestedOpeningWitness):
            raise ProtocolInvariantError('original1120 block opening facade required')
        self.parent_policy=dispatched
        self.source_policy=dispatched.source_policy
        self.qualification_events=[];self.dropped_qualifications=0

    def __getattr__(self,name):return getattr(self.parent_policy,name)

    def qualify_opening(self,requested,current_requested):
        source=self.source_policy
        retain=(source.phase=='acquire' and requested<.5<=current_requested and source.close_witness is None)
        command,proof,proposed=self.parent_policy.qualify_opening(requested,current_requested)
        if retain:
            # Parent1120 already called the raw538 qualifier exactly once.
            # Use its saved ORIGINAL proposal, not1120's synthetic passthrough
            # command-edge proposal. No second current_grasp/observation/query.
            original=proof.get('legacy_qualification')
            if (type(original) is not dict or type(original.get('proof')) is not dict
                    or type(original.get('qualified_opening')) not in (int,float)
                    or not math.isfinite(original['qualified_opening'])
                    or original['qualified_opening'] not in (0.,1.)
                    or type(original.get('proposed')) not in (dict,type(None))):
                raise ProtocolInvariantError('one original538 command/proof/proposal required')
            command=original['qualified_opening'];proposed=deepcopy(original['proposed'])
            proof=dict(deepcopy(original['proof']),current_requested_opening=float(current_requested),
                legacy_qualification=deepcopy(original),dispatch_variant=SCHEMA,
                original_public_preclose_qualification_retained=True,
                legacy_SE3_defer_is_diagnostic_only=False,
                native_capture_claimed=False,native_truth_read=False)
        event=dict(requested_opening=float(requested),current_requested_opening=float(current_requested),
            qualified_opening=float(command),retained_original_first_preclose=bool(retain),
            proof=_plain(proof),proposed=_plain(proposed),
            evidence_kind='qualification_only_not_applied_or_confirmed',native_capture_claimed=False)
        if len(self.qualification_events)<self.parent_policy.max_events:
            self.qualification_events.append(event)
        else:self.dropped_qualifications+=1
        return command,proof,proposed

    def confirm_opening(self,opening,proof,proposed,*,confirmed_after):
        #1114 calls this only after its unchanged original Native apply returns.
        return self.parent_policy.confirm_opening(opening,proof,proposed,confirmed_after=confirmed_after)

    def receipt(self):
        result=dict(self.parent_policy.receipt())
        legacy=result.pop('author_aligned_block_integration_v1120',None)
        result['legacy_parent_block_integration_v1120']=dict(
            active_dispatch_description=False,metadata_is_parent_comparison_only=True,receipt=legacy)
        result['physical_grasp_gate']=dict(mode='original538_qualification_before_first_acquire_open_to_close_edge',
            original_numeric_thresholds_unchanged=True,native_gripper_unchanged=True,
            release_repeated_close_and_other_phases='parent1120_requested_opening_dispatch',
            native_capture_or_held_identity_verified=False)
        result['retained_preclose_block_integration_v1153']=dict(schema=SCHEMA,
            active_first_preclose_uses_original_qualifier=True,
            actual_parent_qualifier_calls_per_original_counted_command=1,
            original_proposal_from_same_qualifier_call=True,
            qualification_events=deepcopy(self.qualification_events),dropped_qualifications=self.dropped_qualifications,
            confirmed_opening_events=deepcopy(self.parent_policy.events),dropped_confirmations=self.parent_policy.dropped_events,
            confirmation_only_after_original_apply_success=True,
            original_frontend_support_phase_and_native_actuator_unchanged=True,
            no_extra_public_collision_hard_veto=True,additional_native_calls=0,
            native_capture_claimed=False,public_threshold_is_native_necessity_or_sufficiency_claim=False,
            formal_VAL_TEST_qualification=False)
        return result


@contextmanager
def integrated_block_components(root,config,task,method,env):
    """Keep the entire parent scope; replace only its final opening facade."""
    with parent.integrated_block_components(root,config,task,method,env) as (frontend,dispatched,executor):
        active=RetainedBlockPrecloseWitness(dispatched)
        with patch.object(executor,'skill_policy',active):
            yield frontend,active,executor
