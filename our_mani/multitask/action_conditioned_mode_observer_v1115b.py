"""Keep a confirmed public close intent through delayed native attachment.

This NEW variant leaves v1115 byte-for-byte intact. Last-visible pose freshness
is checked only when a real close edge is published. A pending edge can span
many held=False ticks/actions while the public acquire phase, target identity
and closed command remain consistent. It is cancelled by an explicit open,
phase/target change, reset, or explicit caller cancellation, never by age alone.
"""
from copy import deepcopy

import numpy as np

from ..contracts import ObjectState, RobotState
from .mode_observer import PublicModeError
from .action_conditioned_mode_observer_v1115 import (
    ActionConditionedPublicModeObserver as Prototype,
    CLOCK_TOLERANCE, _finite, _ids, _pose, _stored,
)

SCHEMA = 'r10_delayed_close_action_conditioned_public_mode_observer_v1115b'


class ActionConditionedPublicModeObserver(Prototype):
    """Use sync at public select boundaries and hook after actual confirmation.

    Public select/hold integration must call synchronize_intent_context when its
    phase or selected identity can change. The actual-confirm hook below also
    synchronizes every dispatched opening command. No policy/native getter is
    called from this observer to discover those changes on its own.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_context = None
        self.open_command_active = False
        self.reset_epoch = 0

    def cancel_pending(self, reason='explicit_public_cancellation'):
        if type(reason) is not str or not reason:
            raise PublicModeError('explicit public cancellation reason required')
        old = self.pending_close
        self.pending_close = None
        self._record(event='pending_close_cancelled', reason=reason,
                     cancelled_intent=deepcopy(old), held_association_preserved=self.held,
                     native_grasp_identity_verified=False)
        return old is not None

    def synchronize_intent_context(self, *, phase=None, phase_index=None, manipulated_ids=None,
                                   requested_opening=None, selected_at=None):
        """Consume existing public context; None is unknown, not a new target.

        Replanning the same phase/target with a newer selected_at preserves the
        token. Supplying an explicit empty/non-single identity cancels it. This
        function never clears an already established held association merely
        because transport/detach begins; actual held=False releases that state.
        """
        if phase is not None and (type(phase) is not str or not phase):
            raise PublicModeError('plain public phase or unknown required')
        if phase_index is not None and (type(phase_index) is not int or phase_index < 0):
            raise PublicModeError('public phase index or unknown required')
        if requested_opening is not None and not _finite(requested_opening):
            raise PublicModeError('finite actual/requested opening or unknown required')
        if selected_at is not None and not _finite(selected_at):
            raise PublicModeError('finite public intent time or unknown required')
        if manipulated_ids is not None and (type(manipulated_ids) not in (tuple, list)
                or any(type(identity) is not str or not identity for identity in manipulated_ids)):
            raise PublicModeError('plain public identity sequence or unknown required')
        old = self.pending_close
        reason = None
        if requested_opening is not None and requested_opening >= .5:
            self.open_command_active = True
            reason = 'explicit_public_open_command'
        elif requested_opening is not None:
            self.open_command_active = False
        if old is not None and reason is None:
            if phase is not None and phase != 'acquire':
                reason = 'public_phase_changed'
            elif phase_index is not None and phase_index != old['phase_index']:
                reason = 'public_phase_index_changed'
            elif manipulated_ids is not None:
                if not _ids(manipulated_ids):
                    reason = 'selected_manipulated_identity_no_longer_single'
                elif manipulated_ids[0] != old['identity']:
                    reason = 'public_selected_manipulated_identity_changed'
        if old is not None and reason is not None:
            self.cancel_pending(reason)
        self.action_context = dict(phase=phase, phase_index=phase_index,
            manipulated_ids=None if manipulated_ids is None else list(manipulated_ids),
            requested_opening=requested_opening, selected_at=selected_at)
        return reason

    # Short alias for integration sites with a compact public selection hook.
    sync_context = synchronize_intent_context

    def record_confirmed_close(self, *, selected, proof, proposed, confirmed_after,
                               previous_states, confirmed_witness=None):
        if type(selected) is dict and type(proposed) is dict:
            self.synchronize_intent_context(phase=selected.get('phase'),
                phase_index=proposed.get('phase_index'), manipulated_ids=selected.get('manipulated_ids'),
                requested_opening=proof.get('qualified_opening') if type(proof) is dict else None,
                selected_at=selected.get('selected_at'))
        # The prototype rejects replay of an old edge without clearing the
        # existing token. Repeated actual closed commands use proposed=None in
        # on_confirmed_opening; they never republish a historical witness.
        return super().record_confirmed_close(selected=selected, proof=proof, proposed=proposed,
            confirmed_after=confirmed_after, previous_states=previous_states,
            confirmed_witness=confirmed_witness)

    def on_confirmed_opening(self, *, selected, opening, proof, proposed, confirmed_after,
                             previous_states, confirmed_witness=None, phase_index=None):
        """Exact adapter for v1114's AFTER-apply/AFTER-confirm callback.

        Call on every actual opening dispatch. New proposed!=None publishes a
        fresh edge; repeated closed commands with proposed=None retain it. The
        integration passes source_policy.selected, phase_index, already known
        previous_states and source_policy.close_witness; it makes no new sensor
        observation and never republishes an old witness as a new proposed edge.
        """
        if not _finite(opening) or not _finite(confirmed_after):
            raise PublicModeError('actual finite opening and confirmation time required')
        selection = selected if type(selected) is dict else {}
        self.synchronize_intent_context(phase=selection.get('phase'), phase_index=phase_index,
            manipulated_ids=selection.get('manipulated_ids'), requested_opening=opening,
            selected_at=selection.get('selected_at'))
        if opening >= .5:
            self._record(event='confirmed_public_open_dispatch', confirmed_after=float(confirmed_after),
                         pending_close_cleared=True, native_release_not_independently_verified=True)
            return False
        if proposed is None:
            self._record(event='closed_dispatch_without_new_edge', confirmed_after=float(confirmed_after),
                pending_identity=None if self.pending_close is None else self.pending_close['identity'])
            return self.pending_close is not None
        if (phase_index is not None and type(proposed) is dict
                and proposed.get('phase_index') != phase_index):
            self._record(event='close_intent_rejected',
                reason='actual_confirmed_public_phase_index_does_not_match_fresh_edge',
                native_grasp_identity_verified=False)
            return False
        return self.record_confirmed_close(selected=selected, proof=proof, proposed=proposed,
            confirmed_after=confirmed_after, previous_states=previous_states,
            confirmed_witness=confirmed_witness)

    def reset(self):
        """Explicit new-episode reset; never infer it from an unknown mode."""
        self.cancel_pending('explicit_episode_reset')
        self.held = self.held_provenance = self._last_close_key = self._last_timestamp = None
        self.released.clear()
        self.action_context = None
        self.open_command_active = False
        self.reset_epoch += 1
        self._record(event='public_mode_reset', reset_epoch=self.reset_epoch)

    def observe(self, previous_states, robot):
        r = _stored(robot, RobotState)
        now, contact = r['timestamp'], bool(r['contact'])
        if not _finite(now) or (self._last_timestamp is not None and now < self._last_timestamp - CLOCK_TOLERANCE):
            raise PublicModeError('public mode clock must be finite and nondecreasing')
        self._last_timestamp = now
        modes = {body: 'released' if body in self.released else 'external' for body in self.body_ids}
        invalid = set()
        pending, before = self.pending_close, self.held
        event = dict(event='public_mode_observe', timestamp=float(now), contact=contact,
            held_before=before, pending_identity=None if pending is None else pending['identity'],
            maximum_distance_m=self.maximum_distance, candidates_status='not_evaluated', candidates=None,
            nearest_candidate_evidence=None, native_grasp_identity_verified=False,
            action_context=deepcopy(self.action_context))
        if not contact:
            if self.held is not None:
                self.released.add(self.held)
                modes[self.held] = 'released'
            self.held = self.held_provenance = None
            event['decision'] = ('await_shared_held_bit_preserve_pending_close' if pending is not None
                                 else 'contact_false_release_held_association')
        elif self.held is not None:
            modes[self.held] = 'attached'
            event['decision'] = 'retain_existing_public_held_identity'
        elif self.open_command_active:
            invalid.update(self.graspable_ids)
            event['decision'] = 'explicit_open_context_blocks_new_attachment_association'
        else:
            if type(previous_states) not in (list, tuple):
                raise PublicModeError('existing public state sequence required')
            tcp_position, _ = _pose(r['pose'])
            candidates, seen = [], set()
            for state in previous_states:
                fields = _stored(state, ObjectState)
                identity = fields['object_id']
                if identity not in self.body_ids or identity in seen:
                    raise PublicModeError('unknown or duplicate public body identity')
                seen.add(identity)
                if identity not in self.graspable_ids or not bool(fields['visible']):
                    continue
                if not _finite(fields['timestamp']) or fields['timestamp'] > now + CLOCK_TOLERANCE:
                    raise PublicModeError('future or nonfinite public body pose is not causal input')
                position, _ = _pose(fields['pose'])
                distance = float(np.linalg.norm(position - tcp_position))
                if distance <= self.maximum_distance:
                    candidates.append(dict(identity=identity, distance_m=distance,
                                           observation_timestamp=float(fields['timestamp'])))
            event['candidates_status'], event['candidates'] = 'computed_once_from_existing_public_inputs', candidates
            ranked = sorted(candidates, key=lambda x: (x['distance_m'], x['identity']))
            event['nearest_candidate_evidence'] = ranked[0] if ranked else None
            event['nearest_is_diagnostic_not_an_ambiguity_override'] = True
            usable = None
            if pending is None:
                event['intent_status'] = 'no_confirmed_public_close_intent'
            elif pending['confirmed_after'] > now + CLOCK_TOLERANCE:
                event['intent_status'] = 'confirmation_not_yet_observed_on_public_clock'
            elif pending['identity'] not in {x['identity'] for x in candidates}:
                event['intent_status'] = 'intended_body_not_in_current_original_candidate_set'
            else:
                usable = pending['identity']
                event['intent_status'] = 'usable_owned_close_intent_in_original_candidate_set'
            if usable is not None:
                self.held = usable
                self.held_provenance = 'own_confirmed_close_intent_and_shared_contact_and_public_candidate'
                event['decision'] = 'associate_using_confirmed_public_action_intent'
            elif len(candidates) == 1:
                self.held = candidates[0]['identity']
                self.held_provenance = 'original_unique_public_candidate_and_shared_contact'
                event['decision'] = 'fallback_original_unique_candidate'
            else:
                invalid.update(self.graspable_ids)
                event['decision'] = 'ambiguity_or_no_public_candidate_remains_unknown'
            if self.held is not None:
                modes[self.held] = 'attached'
                self.pending_close = None
        event.update(held_after=self.held, held_provenance=self.held_provenance,
            pending_identity_after=None if self.pending_close is None else self.pending_close['identity'],
            invalid_ids=sorted(invalid), returned_modes=dict(modes))
        self._record(**event)
        return modes, invalid

    def receipt(self):
        result = super().receipt()
        result.update(schema=SCHEMA, action_context=deepcopy(self.action_context),
            reset_epoch=self.reset_epoch, open_command_active=self.open_command_active,
            pending_close_has_no_age_expiry=True,
            one_tick_limit_applies_only_to_close_publication_evidence=True,
            contact_false_releases_held_but_preserves_valid_pending_close=True,
            integration_requires_public_phase_target_open_notifications=True)
        return result
