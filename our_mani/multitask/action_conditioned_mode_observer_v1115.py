"""Public action-conditioned attachment association for a NEW experiment.

The old observer is unchanged. This observer consumes the same immutable public
ObjectState/RobotState inputs and an owned, actually confirmed close callback.
It never asks an environment for held IDs, body poses, future truth or outcomes.
An intended body must still belong to the original visible/graspable/distance
candidate set. Intent breaks an existing ambiguity; it does not certify which
object the native gripper holds. With no usable intent the old unique-candidate
rule remains in force. Nearest distances are diagnostic, never a new threshold.
"""
from copy import deepcopy
import math

import numpy as np

from ..contracts import ObjectState, Pose, RobotState
from .mode_observer import PublicModeError, PublicModeObserver

SCHEMA = 'r10_action_conditioned_public_mode_observer_v1115'
DT = .05
CLOCK_TOLERANCE = 1e-8


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _ids(value):
    return (type(value) in (tuple, list) and len(value) == 1
            and type(value[0]) is str and bool(value[0]))


def _stored(value, exact_type):
    if type(value) is not exact_type:
        raise PublicModeError('v1115 requires original immutable public DTOs')
    return object.__getattribute__(value, '__dict__')


def _pose(value):
    fields = _stored(value, Pose)
    return fields['position'], fields['rotation']


class ActionConditionedPublicModeObserver(PublicModeObserver):
    """Same observe signature, plus one publish point after a real close edge.

    Integration calls record_confirmed_close AFTER its existing physical close
    and existing confirm_opening have returned, using already available records.
    No additional perception/geometry getter or native query is needed.
    """
    def __init__(self, body_ids, graspable_ids, maximum_distance, *, max_events=1000):
        body_ids, graspable_ids = tuple(body_ids), tuple(graspable_ids)
        if (not body_ids or len(body_ids) != len(set(body_ids))
                or any(type(x) is not str or not x for x in body_ids)
                or len(graspable_ids) != len(set(graspable_ids))
                or not set(graspable_ids) <= set(body_ids)
                or not _finite(maximum_distance) or maximum_distance <= 0
                or type(max_events) is not int or not 1 <= max_events <= 10000):
            raise PublicModeError('invalid registered public body set/distance/trace bound')
        super().__init__(body_ids, graspable_ids, maximum_distance)
        self.max_events = max_events
        self.events = []
        self.dropped_events = 0
        self.pending_close = None
        self.held_provenance = None
        self._last_close_key = None
        self._last_timestamp = None

    def _record(self, **event):
        if len(self.events) < self.max_events:
            self.events.append(event)
        else:
            self.dropped_events += 1

    def record_confirmed_close(self, *, selected, proof, proposed, confirmed_after,
                               previous_states, confirmed_witness=None):
        """Publish only known public records from the OWN confirm callback.

        selected is the issued public pose intent; proposed is the existing fresh
        close-edge witness, not a persistent witness from an earlier command.
        Optional confirmed_witness lets the integration cross-check the witness
        its original confirm_opening already stored. No old 2cm/orientation gate
        is reintroduced here: any such diagnostics are retained as evidence only.
        Invalid publication returns False with an explicit reason, never attaches.
        """
        def reject(reason):
            self._record(event='close_intent_rejected', reason=reason,
                         native_grasp_identity_verified=False)
            return False
        if type(selected) is not dict or type(proof) is not dict or type(proposed) is not dict:
            return reject('fresh_plain_public_selected_proof_and_proposed_required')
        selected_ids, proposed_ids = selected.get('manipulated_ids'), proposed.get('manipulated_ids')
        if not _ids(selected_ids) or not _ids(proposed_ids) or tuple(selected_ids) != tuple(proposed_ids):
            return reject('single_matching_public_manipulated_identity_required')
        identity = selected_ids[0]
        if identity not in self.body_ids or identity not in self.graspable_ids:
            return reject('intended_identity_not_registered_graspable')
        if selected.get('phase') != 'acquire' or proof.get('phase') != 'acquire':
            return reject('public_acquire_close_context_required')
        phase_index = proposed.get('phase_index')
        if type(phase_index) is not int or phase_index < 0:
            return reject('public_phase_index_required')
        selected_at, issued_at = selected.get('selected_at'), proposed.get('issued_at')
        proof_time = proof.get('time')
        if not all(_finite(x) for x in (selected_at, issued_at, proof_time, confirmed_after)):
            return reject('finite_public_close_times_required')
        if (selected_at > issued_at + CLOCK_TOLERANCE
                or abs(proof_time - issued_at) > CLOCK_TOLERANCE
                or not 0 < confirmed_after - issued_at <= DT + CLOCK_TOLERANCE
                or self._last_timestamp is not None and confirmed_after < self._last_timestamp - CLOCK_TOLERANCE):
            return reject('inconsistent_or_noncausal_close_confirmation_times')
        opening = [proof.get(k) for k in
                   ('desired_opening', 'current_requested_opening', 'qualified_opening')]
        if not all(_finite(x) for x in opening) or not (opening[0] < .5 <= opening[1] and opening[2] < .5):
            return reject('fresh_qualified_open_to_closed_edge_required')
        if proof.get('native_truth_read') is True:
            return reject('native_truth_is_not_a_public_intent_source')
        if self.held is not None:
            return reject('existing_held_identity_must_release_before_new_close')
        key = (identity, phase_index, issued_at, confirmed_after)
        if key == self._last_close_key:
            return reject('duplicate_confirmed_close_edge')
        if confirmed_witness is not None:
            if (type(confirmed_witness) is not dict
                    or confirmed_witness.get('phase_index') != phase_index
                    or not _ids(confirmed_witness.get('manipulated_ids'))
                    or tuple(confirmed_witness['manipulated_ids']) != tuple(proposed_ids)
                    or confirmed_witness.get('issued_at') != issued_at
                    or confirmed_witness.get('confirmed_after') != confirmed_after):
                return reject('original_confirmed_witness_does_not_match_fresh_edge')
        if type(previous_states) not in (tuple, list):
            return reject('existing_public_previous_state_sequence_required')
        matches = []
        for state in previous_states:
            fields = _stored(state, ObjectState)
            if fields['object_id'] == identity:
                matches.append(fields)
        if len(matches) != 1:
            return reject('unique_last_public_state_for_intended_body_required')
        fields = matches[0]
        if not bool(fields['visible']):
            return reject('intended_body_not_last_publicly_visible')
        snapshot_time = fields['timestamp']
        if (not _finite(snapshot_time) or snapshot_time > issued_at + CLOCK_TOLERANCE
                or issued_at - snapshot_time > DT + CLOCK_TOLERANCE):
            return reject('last_public_pose_is_future_or_older_than_one_counted_tick')
        position, rotation = _pose(fields['pose'])
        if not np.isfinite(position).all() or not np.isfinite(rotation).all():
            return reject('nonfinite_last_public_pose')
        self.pending_close = dict(identity=identity, phase_index=phase_index,
            selected_at=float(selected_at), issued_at=float(issued_at), confirmed_after=float(confirmed_after),
            last_visible_pose=dict(timestamp=float(snapshot_time), position=position.tolist(), rotation=rotation.tolist()),
            source='owned_public_qualified_close_confirm_callback',
            diagnostic_position_error_m=proposed.get('distance_m') if _finite(proposed.get('distance_m')) else None,
            diagnostic_orientation_error_rad=proposed.get('angle_rad') if _finite(proposed.get('angle_rad')) else None,
            native_grasp_identity_verified=False)
        self._last_close_key = key
        self._record(event='close_intent_published', intent=deepcopy(self.pending_close))
        return True

    def observe(self, previous_states, robot):
        """Read each supplied public field once; perform one candidate pass."""
        r = _stored(robot, RobotState)
        now, contact = r['timestamp'], bool(r['contact'])
        if not _finite(now) or (self._last_timestamp is not None and now < self._last_timestamp - CLOCK_TOLERANCE):
            raise PublicModeError('public mode clock must be finite and nondecreasing')
        self._last_timestamp = now
        modes = {body: 'released' if body in self.released else 'external' for body in self.body_ids}
        invalid = set()
        before, pending = self.held, self.pending_close
        event = dict(event='public_mode_observe', timestamp=float(now), contact=contact,
            held_before=before, pending_identity=None if pending is None else pending['identity'],
            maximum_distance_m=self.maximum_distance, candidates_status='not_evaluated',
            candidates=None, nearest_candidate_evidence=None, native_grasp_identity_verified=False)
        if not contact:
            if self.held is not None:
                self.released.add(self.held)
                modes[self.held] = 'released'
            self.held = self.held_provenance = self.pending_close = None
            event['decision'] = 'contact_false_release_and_clear_close_intent'
        elif self.held is not None:
            modes[self.held] = 'attached'
            event['decision'] = 'retain_existing_public_held_identity'
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
            elif now - pending['confirmed_after'] > DT + CLOCK_TOLERANCE:
                event['intent_status'] = 'confirmed_intent_not_observed_within_one_counted_tick'
                self.pending_close = None
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
                     invalid_ids=sorted(invalid), returned_modes=dict(modes))
        self._record(**event)
        return modes, invalid

    def receipt(self):
        return dict(schema=SCHEMA, events=deepcopy(self.events), dropped_events=self.dropped_events,
            held_identity=self.held, held_provenance=self.held_provenance,
            pending_close=deepcopy(self.pending_close), original_maximum_distance_m=self.maximum_distance,
            fallback_policy='original_unique_candidate_only', nearest_distance_used_to_break_ambiguity=False,
            extra_native_getters_or_future_truth_or_task_outcomes_used=False,
            public_association_is_not_verified_native_grasp_identity=True)
