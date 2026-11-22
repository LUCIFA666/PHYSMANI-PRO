"""Formal consumers retain verified mode segments and censored raw sources."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from our_mani.multitask.release_preflight import (
    _completion_support,_semantic_source_proof,_semantic_library_metadata)
from tests.test_multitask_semantic_acquire_fitting import source


def template(value,mode,lag=0.):
    event=value.completion_events[0];binding=event.binding
    return SimpleNamespace(phase='acquire',anchor_id=binding.anchor_id,
        reference_role=binding.reference_role,reference_kind=binding.reference_kind,
        manipulated_role=binding.manipulated_role,reference_mode=mode,
        reference_time_basis=binding.reference_time_basis,lag_seconds=lag)


def rule(value):
    e=value.semantic_acquire_evidence
    return SimpleNamespace(sha256=e.derivation_rule_sha256,
        helper={'path':'verified_helper.py','sha256':e.derivation_module_sha256})


def metadata(values):
    r=rule(values[0])
    return {'semantic_acquire_derivation':dict(default_legacy_behavior_changed=False,
        original_events_command_ids_masks_and_censoring_preserved=True,
        rule_sha256_values=[r.sha256],helper_sha256_values=[r.helper['sha256']],
        per_source={v.source_id:_semantic_source_proof(v) for v in values})}


class SemanticAcquireReleaseTests(unittest.TestCase):
    def test_actual_multicommand_attached_segment_counts_once_with_semantic_attempt_identity(self):
        value=source(0)
        self.assertEqual(value.completion_events[0].eligible_command_ids,(2,))
        self.assertEqual(value.semantic_acquire_evidence.segments[1].raw_command_ids,(1,2))
        for mode in ('external','attached'):
            key,parents,attempts=_completion_support([value],template(value,mode))
            segment=next(s for s in value.semantic_acquire_evidence.segments if s.reference_mode==mode)
            self.assertEqual(key[-2:],(mode,'current'))
            self.assertEqual(parents,{value.grouping_source})
            self.assertEqual(attempts,{value.source_id+':semantic:'+segment.segment_id})

    def test_empty_enabled_evidence_cannot_regain_original_phase_transition_positive(self):
        value=source(0,physical=False)
        self.assertTrue(value.completion_events)
        for mode in ('external','attached'):
            _,parents,attempts=_completion_support([value],template(value,mode))
            self.assertFalse(parents);self.assertFalse(attempts)

    def test_external_positive_lag_cannot_cross_public_contact_boundary(self):
        value=source(0)
        _,parents,attempts=_completion_support([value],template(value,'external',.05))
        self.assertFalse(parents);self.assertFalse(attempts)

    def test_replicas_increase_attempts_without_inventing_parent_support(self):
        values=[source(0),source(1,parent='episode0')]
        _,parents,attempts=_completion_support(values,template(values[0],'attached'))
        self.assertEqual(len(parents),1);self.assertEqual(len(attempts),2)

    def test_metadata_keeps_empty_censored_sources_and_exact_raw_SHA(self):
        values=[source(0),source(1,physical=False)];proof=metadata(values)
        _semantic_library_metadata(proof,values,rule(values[0]))
        for change in ('omit_censored','raw_SHA','rule','helper','status','preserved'):
            bad=deepcopy(proof);body=bad['semantic_acquire_derivation']
            if change=='omit_censored':body['per_source'].pop(values[1].source_id)
            elif change=='raw_SHA':body['per_source'][values[0].source_id]['raw_bindings']['cache']['sha256']='c'*64
            elif change=='rule':body['rule_sha256_values']=['c'*64]
            elif change=='helper':body['helper_sha256_values']=['c'*64]
            elif change=='status':body['per_source'][values[1].source_id]['status']='completed'
            else:body['original_events_command_ids_masks_and_censoring_preserved']=False
            with self.subTest(change=change),self.assertRaises(ValueError):
                _semantic_library_metadata(bad,values,rule(values[0]))

    def test_undeclared_or_missing_semantic_metadata_rejected(self):
        value=source(0)
        with self.assertRaises(ValueError):_semantic_library_metadata({},[value],rule(value))
        with self.assertRaises(ValueError):_semantic_library_metadata(metadata([value]),[],None)


if __name__=='__main__':unittest.main()
