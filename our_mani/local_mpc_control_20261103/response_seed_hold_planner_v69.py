"""V69 provenance-preserving boundary around the unchanged V68 planner.

The numeric CEM state remains the original validator's whitelist. A separate
original public-feedback snapshot supplies provenance to the frozen generator.
The complete numeric validator output must match before crossing either way;
no missing provenance field is fabricated in a numeric state.
"""
from copy import deepcopy
from pathlib import Path
import hashlib

from .response_seed_hold_planner_v68 import (
    ResponseSeedHoldPlannerV68,response_seed_planner_source_map as v68_sources,
    _complete_feedback_sha)
from .public_candidate_local_hold_v66 import _assert_current_clock
from .hold_orientation_guard_prototype_20261103 import _identity
from ..local_mpc_p2_20261103.execution_models import validate_feedback_state

PROVENANCE_KEYS=('Native_object_state_used','Native_held_used','Native_success_used')
VERSION='original_public_feedback_numeric_CEM_boundary_20261103_v69'


def response_seed_planner_source_map():
    sources=v68_sources()
    path=Path(__file__).resolve();sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def _authenticated_public_hold(state):
    if (not isinstance(state,dict) or state.get('grasp_motion_mode')!='ATTACHED_PUBLIC'
        or state.get('held')is not True or (state.get('phase_state')or{}).get('phase')!='hold'
        or any(key not in state or state[key]is not False for key in PROVENANCE_KEYS)):
        raise ValueError('Original explicitly public confirmed hold feedback required; no inferred provenance')
    _identity(state)


class ResponseSeedHoldPlannerV69(ResponseSeedHoldPlannerV68):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._boundary_sources=response_seed_planner_source_map()
        self._original_public_feedback=None
        self._canonical_numeric_feedback=None
        self._original_feedback_sha=None

    def _verified_original_feedback(self,numeric_state):
        raw=self._original_public_feedback
        if raw is None or self._canonical_numeric_feedback is None:
            raise ValueError('No original public feedback bound to this plan')
        _authenticated_public_hold(raw)
        _assert_current_clock(self.backend,raw['time'])
        if _complete_feedback_sha(raw)!=self._original_feedback_sha:
            raise ValueError('Original authenticated feedback changed during planning')
        # This includes every numeric key, not merely q/h. Quaternion unit
        # normalization is checked through the exact old validator itself.
        if _complete_feedback_sha(numeric_state)!=_complete_feedback_sha(self._canonical_numeric_feedback):
            raise ValueError('Numeric CEM state differs from validated original public feedback')
        if _identity(numeric_state)!=_identity(raw):
            raise ValueError('Actual q/dq/h/phase identity mismatch at public/numeric boundary')
        return raw

    def _compute_response_seed(self,numeric_state,candidates):
        raw=self._verified_original_feedback(numeric_state)
        return super()._compute_response_seed(raw,candidates)

    def _validate_fresh_rollout_binding(self,numeric_state,candidates,modes,proposal):
        raw=self._verified_original_feedback(numeric_state)
        return super()._validate_fresh_rollout_binding(raw,candidates,modes,proposal)

    def plan(self,state,candidates,*,warm_start=None):
        if self._original_public_feedback is not None:
            raise ValueError('No nested or stale original-feedback binding')
        if self._boundary_sources!=response_seed_planner_source_map():
            raise ValueError('Frozen V69 feedback boundary changed')
        _authenticated_public_hold(state)
        _assert_current_clock(self.backend,state['time'])
        caller_sha=_complete_feedback_sha(state)
        self._original_public_feedback=deepcopy(state)
        self._original_feedback_sha=caller_sha
        try:
            self._canonical_numeric_feedback=validate_feedback_state(self._original_public_feedback)
            numeric_sha=_complete_feedback_sha(self._canonical_numeric_feedback)
            result=super().plan(state,candidates,warm_start=warm_start)
            if _complete_feedback_sha(state)!=caller_sha:
                raise ValueError('Caller original feedback changed')
            if self._boundary_sources!=response_seed_planner_source_map():
                raise ValueError('Frozen V69 boundary changed during plan')
            result['diagnostics']['V69_original_public_feedback_boundary']=dict(
                version=VERSION,original_public_feedback_sha256=caller_sha,
                canonical_numeric_feedback_sha256=numeric_sha,
                complete_numeric_state_checked=True,original_provenance_keys_preserved=list(PROVENANCE_KEYS),
                provenance_fields_added_to_numeric_state=False,
                generator_received_original_public_feedback=True,
                CEM_received_original_numeric_validator_output=True,
                optimization_cost_domain_budget_and_dispatch_unchanged=True)
            return result
        finally:
            self._original_public_feedback=None
            self._canonical_numeric_feedback=None
            self._original_feedback_sha=None
