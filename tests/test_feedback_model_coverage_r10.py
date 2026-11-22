"""CPU negative boundaries; no recollected labels or production release."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

from our_mani.multitask import feedback_model_coverage as gate
from our_mani.multitask import task_release as legacy
from our_mani.multitask.candidate_success import (CONTINUATION_POLICY,FEATURE_SCHEMA_SHA256,LABEL_SCOPE)
from our_mani.multitask.evaluation import file_sha
from tests import test_multitask_task_release as fixtures


class FeedbackModelCoverageTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.TaskReleaseTests(methodName='runTest');self.f.setUp();self.f.fixture()
        self.root=self.f.root;self.task=self.f.task
    def tearDown(self):self.f.tearDown()
    def test_legacy_certificate_is_never_promoted_to_feedback_coverage(self):
        with self.assertRaisesRegex(ValueError,'missing, failed, foreign or stale'):
            gate.require_feedback_task_coverage_certificate(self.root,self.task,self.f.pointer(self.f.coverage),
                self.f.config_value,self.f.files)
    def test_critic_numerical_and_provenance_kernel_is_exact_legacy_AST(self):
        proof=gate._critic_kernel();self.assertTrue(proof['unchanged'])
        self.assertEqual(proof['normalized_AST_sha256'],'378ff51828436001a44a122b99e53d48e6170349eb327eb9d1c47b632c425c12')
    def test_old_critic_cannot_match_new_policy_even_with_trained_true(self):
        old_id='a'*64;new_id='b'*64
        metadata=dict(training_complete=True,classifier_fit_split='train',classifier_selection_split='train',
            normalization_fit_split='train',TEST_used_for_fitting=False,label_scope=LABEL_SCOPE,
            continuation_selection_policy=CONTINUATION_POLICY,feature_schema_sha256=FEATURE_SCHEMA_SHA256,
            continuation_sha256=old_id,interaction_library_sha256='c'*64,TRAIN_source_manifest_sha256='d'*64,synthetic=False)
        artifact=self.f.put('old_value.json',dict(metadata=metadata,heads={}))
        protocol=deepcopy(self.f.config_value);protocol['learned_artifacts'][self.task]['cv_value']=self.f.pointer(artifact)
        # Identity construction alone is a stub in this boundary. Actual model
        # loading/matching runs; it rejects before nonexistent fixture manifests.
        with patch.object(gate,'_feedback_identity',return_value={'sha256':new_id}):
            with self.assertRaisesRegex(ValueError,'completed matching TRAIN-only'):
                gate._feedback_critic(gate.models._Inputs(self.root),protocol,self.task,'cv',SimpleNamespace(sha256='c'*64),{})
    def test_missing_new_Value_is_pending_not_geometry_only_pass(self):
        protocol=deepcopy(self.f.config_value);protocol['learned_artifacts'][self.task]['cv_value']=None
        with patch.object(gate,'_feedback_identity',return_value={'sha256':'b'*64}):
            with self.assertRaisesRegex(ValueError,'explicit coverage input'):
                gate._feedback_critic(gate.models._Inputs(self.root),protocol,self.task,'cv',SimpleNamespace(sha256='c'*64),{})
    def test_old_coverage_source_cannot_claim_new_schema(self):
        report=deepcopy(self.f.coverage_value);report.update(schema=gate.COVERAGE,feedback_overlay_enabled=True,
            legacy_certificate_reused=False,legacy_model_provenance_source_sha256=file_sha(legacy.__file__))
        path=self.f.put('claimed_feedback_certificate.json',report);files=dict(self.f.files)
        files[str(path.relative_to(self.root))]=file_sha(path)
        with self.assertRaisesRegex(ValueError,'missing, failed, foreign or stale'):
            gate.require_feedback_task_coverage_certificate(self.root,self.task,self.f.pointer(path),self.f.config_value,files)
    def test_schema_only_without_extended_raw_proof_is_not_admission(self):
        report=deepcopy(self.f.coverage_value);report.update(schema=gate.COVERAGE,feedback_overlay_enabled=True,
            legacy_certificate_reused=False,legacy_model_provenance_source_sha256=file_sha(legacy.__file__),
            coverage_source_sha256=file_sha(gate.__file__))
        path=self.f.put('schema_only_feedback_certificate.json',report);files=dict(self.f.files)
        files[str(path.relative_to(self.root))]=file_sha(path)
        # The library input is also a legacy CPU fixture, so it must fail before
        # any unproved extended identity can be accepted.
        with self.assertRaises(Exception):
            gate.require_feedback_task_coverage_certificate(self.root,self.task,self.f.pointer(path),self.f.config_value,files)


if __name__=='__main__':unittest.main()
