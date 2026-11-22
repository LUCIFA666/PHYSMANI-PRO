"""Public-only reobservation lifecycle and single-dispatch integration.

The original current measurement/proof/cache result is never modified. This
component gates a new close after a gap; it never resets an attempted grasp.
"""
from copy import deepcopy
import numpy as np
from ..cube_stale_reobserve_20261104_v1.recovery import (
    StaleReobserveSupervisorV1, execute_reobserve_cycle)


class ReobservePublicLifecycle:
    def __init__(self):
        self.supervisor=StaleReobserveSupervisorV1()
        self.fresh_frames=0;self.last_time=None;self.ever_confirmed=False
        self.no_attempt=False;self.stale_entry_serial=0;self.latest=None
        self._was_missing=False

    def observe(self,result,robot,*,active,effective_attachment,close_requested):
        now=float(robot.timestamp)
        consecutive=self.last_time is None or abs(now-self.last_time-.05)<=1e-6
        # Original owner id zero is only a sentinel before any actual attempt.
        self.no_attempt=(active.attempt_id==0 and active.attempt_started is None
            and not active.attempt_active and not active.failed and active.spec is None)
        self.ever_confirmed=self.ever_confirmed or active.confirmation is not None or (
            isinstance(effective_attachment,dict) and effective_attachment.get('mode')=='ATTACHED_PUBLIC')
        observed=self.supervisor.observe(result,robot,close_requested=close_requested,
            attempt_id=None if self.no_attempt else active.attempt_id,
            previously_confirmed=bool(self.ever_confirmed))
        measured=observed['current_measurement_available']
        self.fresh_frames=min(5,self.fresh_frames+1) if measured and consecutive else (1 if measured else 0)
        missing=not measured or not consecutive
        eligible=self.no_attempt and not close_requested and not self.ever_confirmed
        if missing and not self._was_missing and eligible:self.stale_entry_serial+=1
        self._was_missing=missing;self.last_time=now
        self.latest=dict(observed,fresh_real50ms_frames=self.fresh_frames,
            fresh_five_and_original_forecast_ready=bool(self.fresh_frames>=5 and observed['original_free_policy_ready']),
            no_physical_attempt_ever_started=self.no_attempt,ever_publicly_confirmed=self.ever_confirmed,
            preclose_reobserve_eligible=eligible,stale_entry_serial=self.stale_entry_serial,
            original_measurement_and_proof_result_mutated=False,original_proof_owner_reset=False)
        return deepcopy(self.latest)

    def receipt(self):
        return dict(latest=deepcopy(self.latest),maximum_reobserve_cycles=3,joint_retreat_cap_rad=.03,
            hint_horizon_seconds=.15,hint_applied_to_image_or_measurement=False,
            Sphere_or_crop_anchor_changed=False,five_real_frames_required_before_new_close=True,
            contact_or_visibility_guaranteed=False)


def clear_free_state_on_stale(coordinator,lifecycle):
    """Invalidate only pre-attempt FREE planning memory, never proof owners."""
    previous=getattr(coordinator,'_reobserve_seen_stale_entry',0)
    current=lifecycle.stale_entry_serial
    coordinator._reobserve_seen_stale_entry=current
    allowed=(current!=previous and lifecycle.no_attempt and not lifecycle.ever_confirmed)
    if allowed:
        coordinator.warm=None;coordinator.phase=None
    return dict(stale_entry_serial=current,free_warm_and_dwell_cleared=allowed,
        closed_or_failed_proof_owner_reset=False)


def guard_new_close(decision,coordinator,lifecycle,h):
    """A missing/rebuilding history cannot produce a new OPEN -> CLOSE edge."""
    action=decision.get('action')
    if (action is not None and h['gripper_target']==1. and float(action['opening'])==0.
            and not (lifecycle.latest or {}).get('fresh_five_and_original_forecast_ready',False)):
        original=deepcopy(decision)
        coordinator.warm=None;coordinator.phase=None
        return dict(mode='UNAVAILABLE',action=None,state=None,
            reason='reobserve_requires_five_current_real50ms_measurements_before_new_close',
            original_decision_not_dispatched=original,new_close_suppressed=True)
    return decision


def execute_selected_or_reobserve(backend,executor,decision,goal,opening,executed_action):
    """Exactly one two-tick call: accepted retreat OR original action/hold.

    A rejected/disabled retreat takes the original hold path. The original
    selected-action qualifier remains in the caller and is never bypassed.
    """
    lifecycle=backend.reobserve_lifecycle
    recovery=None
    if (decision.get('action') is None and decision.get('mode')=='UNAVAILABLE'
            and (lifecycle.latest or {}).get('preclose_reobserve_eligible',False)
            and executor.state()['gripper_target']==1. and backend.requested_opening==1.):
        proposal=lifecycle.supervisor.propose()
        recovery=dict(proposal=proposal,
            outcome=execute_reobserve_cycle(lifecycle.supervisor,backend,executor,proposal))
        if recovery['outcome']['executed']:
            executed_action=dict(joint_goal=np.asarray(proposal['joint_goal']),opening=1.,
                opening_argument=None,retained_requested_opening=1.,source='bounded_public_reobserve_joint_not_MPC_selected_action')
            return recovery['outcome']['executor'],executed_action,recovery
    receipt=executor.execute_joint_target(goal,opening=opening,
        phase='P4_candidate_transformer_v72_'+decision['mode'])
    return receipt,executed_action,recovery
