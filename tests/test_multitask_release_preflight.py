"""Formal coverage boundaries in temporary CPU schema fixtures, no real models."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from our_mani.multitask import release_preflight as gate
from our_mani.multitask.evaluation import digest,file_sha,comparison_protocol_sha256
from our_mani.multitask.specs import TASK_SPECS
from tests.test_multitask_completed_skill_fitting import completed_source


def inventories():
    train={(task,'train'+str(i)) for task in TASK_SPECS for i in range(100)}
    counts=dict.fromkeys(TASK_SPECS,20)
    counts['insert_onto_rotating_peg']=18;counts['insert_onto_rotating_peg_high_speed']=19
    counts['put_rubbish_in_moving_bin_high_speed']=18
    val={(task,'val'+str(i)) for task,n in counts.items() for i in range(n)}
    return train,val,counts


class FormalCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.train,self.val,self.val_counts=inventories()
    def tearDown(self):self.temp.cleanup()
    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        return path
    def pointer(self,path):return dict(path=str(path.relative_to(self.root)),sha256=file_sha(path))

    def test_canonical_and_identity_gate_reject_pilot_counts_and_same_count_wrong_sources(self):
        path=self.write('TRAIN.json',dict(split='train',episodes=[dict(task=t,raw_episode_id=raw) for t,raw in self.train]))
        keys,proof=gate._canonical(gate._Inputs(self.root),path,'train')
        self.assertEqual(keys,self.train);self.assertEqual(proof['parents'],1400)
        data=json.loads(path.read_text());data['episodes']=data['episodes'][:140];path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'formal TRAIN|lacks a remaining task'):
            gate._canonical(gate._Inputs(self.root),path,'train')
        task,raw=next(iter(self.train));wrong=self.train-{(task,raw)}|{(task,'unregistered')}
        with self.assertRaisesRegex(ValueError,'missing=1, unexpected=1'):gate._exact(wrong,self.train,'TRAIN')

    def context_fixture(self,source_count=1400,optimizer_count=1400):
        from our_mani.multitask.motion_labels import SCHEMA
        entries=[dict(split='train',task=t,raw_episode_id=raw) for t,raw in sorted(self.train)[:source_count]]
        view=self.write('view.json',dict(entries=entries))
        source=self.write('training_sources.json',dict(entries=entries,source_episode_count=source_count,
            body_track_manifest_sha256=file_sha(view),normalization_fit_split='train'))
        checkpoint=self.root/'dynamics.pt';checkpoint.write_bytes(b'not a model; CPU loader mocked')
        metadata=dict(selection_split='validation',VAL_source_episodes=275,maximum_training_horizon_seconds=1.1,
            dynamics_target_label_schema=SCHEMA,motion_labels_sha256='a'*64,
            track_manifest_sha256=file_sha(view),training_source_manifest=dict(path_relative_to_checkpoint=source.name,
                sha256=file_sha(source),source_episode_count=source_count,cache_episode_count=len(entries)),
            optimizer_examples_per_parent_min=1,optimizer_examples_per_parent_max=1,TRAIN_windows=1400,VAL_windows=275)
        optimizer={f'train:{t}:{raw}':1 for t,raw in sorted(self.train)[:optimizer_count]}
        self.write('report.json',dict(metadata=metadata,checkpoint_sha256=file_sha(checkpoint),optimizer_coverage=optimizer))
        protocol=dict(context_checkpoint=self.pointer(checkpoint),dynamics_target_label_schema=SCHEMA)
        coverage=dict(train=dict(usable_windows=1400),validation=dict(usable_windows=275,per_task=self.val_counts))
        expected=dict(train=self.train,validation=self.val)
        return protocol,metadata,view,coverage,expected

    def test_self_consistent_140parent_checkpoint_is_still_a_formal_pilot(self):
        protocol,metadata,view,coverage,expected=self.context_fixture(source_count=140)
        with patch('our_mani.multitask.context_se3.MultiBodyContextDynamics',return_value=SimpleNamespace(metadata=metadata)),\
             patch.object(gate,'_context_view',return_value=(view,json.loads(view.read_text()),coverage)):
            with self.assertRaisesRegex(ValueError,'Context fitted TRAIN parent coverage differs'):
                gate._context(gate._Inputs(self.root),protocol,self.pointer(view),expected)

    def test_declared_1400_sources_cannot_hide_140_parent_optimizer_coverage(self):
        protocol,metadata,view,coverage,expected=self.context_fixture(optimizer_count=140)
        with patch('our_mani.multitask.context_se3.MultiBodyContextDynamics',return_value=SimpleNamespace(metadata=metadata)),\
             patch.object(gate,'_context_view',return_value=(view,json.loads(view.read_text()),coverage)):
            with self.assertRaisesRegex(ValueError,'actually reach an optimizer update'):
                gate._context(gate._Inputs(self.root),protocol,self.pointer(view),expected)

    def test_development_BC_cannot_pass_on_training_complete_alone(self):
        path=self.write('library.json',{})
        stub=SimpleNamespace(task_spec=SimpleNamespace(task_name='push_moving_button'),metadata=dict(synthetic=False,
            training_complete=True,development_support_override=True,fit_configuration=dict(development_only=True,
                minimum_completed_parent_sources=2,maximum_position_tolerance_m=.03,maximum_orientation_tolerance_rad=.15)))
        protocol=dict(learned_artifacts={'push_moving_button':dict(interaction=self.pointer(path))})
        with patch('our_mani.multitask.interaction.InteractionLibrary.load',return_value=stub):
            with self.assertRaisesRegex(ValueError,'development BC is forbidden'):
                gate._library(gate._Inputs(self.root),protocol,'push_moving_button',{}, {})

    def test_actual_completed_support_counts_parents_not_eighty_replicas(self):
        original=completed_source(0);event=original.completion_events[0]
        template=SimpleNamespace(phase=event.phase,anchor_id=event.binding.anchor_id,
            reference_role=event.binding.reference_role,reference_kind=event.binding.reference_kind,
            manipulated_role=event.binding.manipulated_role,reference_mode=event.reference_mode,
            reference_time_basis='current',lag_seconds=0.)
        sources=[replace(original,source_id='rep'+str(i)) for i in range(80)]
        key,parents,attempts=gate._completion_support(sources,template)
        self.assertEqual(len(parents),1);self.assertEqual(len(attempts),80)
        self.assertEqual(len(key),7);self.assertEqual(key[-1],'current')

    def certificate(self):
        protocol=dict(CPU_schema_fixture_only=True)
        libraries={};critics={}
        for task,spec in TASK_SPECS.items():
            support={phase.name+'|fixture':8 for phase in spec.phase_skills if not phase.optional_after_native_completion}
            libraries[task]=dict(passed=True,native_acquisition_parent_sources=100,original_TRAIN_priors=1400,
                completed_parent_sources_by_skill=support)
            for predictor in ('cv','context'):
                critics[task+'|'+predictor]=dict(passed=True,TRAIN_acquired_parent_sources=100,
                    VAL_acquired_parent_sources=self.val_counts[task],eligible_TRAIN_label_parents=2,
                    eligible_VAL_label_parents=0,ranking_enabled=False,geometry_fallback_is_legal=True)
        input_path=self.root/'temporary_evidence.txt';input_path.write_text('CPU schema fixture, no actual model or release')
        files={input_path.name:file_sha(input_path)}
        report=dict(schema=gate.SCHEMA,complete=True,formal_model_coverage_passed=True,errors=[],
            comparison_protocol_sha256=comparison_protocol_sha256(protocol),coverage_source_sha256=file_sha(Path(gate.__file__)),
            actual_TEST_read=False,model_fitting_performed=False,input_files_sha256=files,
            sections=dict(canonical_train=dict(per_task=dict.fromkeys(TASK_SPECS,100)),
                canonical_validation=dict(per_task=self.val_counts),context=dict(passed=True,TRAIN_parent_sources=1400,
                    VAL_parent_sources=275,optimizer_parent_sources=1400),libraries=libraries,critics=critics))
        return protocol,report,files

    def test_full_acquisition_with_disabled_unsupported_rank_can_remain_geometry(self):
        protocol,report,files=self.certificate();path=self.write('certificate.json',report);pointer=self.pointer(path)
        files[path.name]=pointer['sha256']
        receipt=gate.require_frozen_coverage_certificate(self.root,pointer,protocol,files)
        self.assertTrue(receipt['passed']);self.assertFalse(receipt['performance_or_SR_benefit_established'])
        # All twenty-eight tasks/predictors still need their real completed VAL
        # inventory; zero eligible classifier samples never excuse missing cases.
        report['sections']['critics']['insert_onto_rotating_peg|cv']['VAL_acquired_parent_sources']=17
        path.write_text(json.dumps(report));pointer=self.pointer(path);files[path.name]=pointer['sha256']
        with self.assertRaisesRegex(ValueError,'critic coverage/fallback proof differs'):
            gate.require_frozen_coverage_certificate(self.root,pointer,protocol,files)

    def test_certificate_rejects_two_parent_skill_and_incomplete_stronger_lock(self):
        protocol,report,files=self.certificate()
        report['sections']['libraries']['push_moving_button']['completed_parent_sources_by_skill']['press|fixture']=2
        path=self.write('certificate.json',report);pointer=self.pointer(path);files[path.name]=pointer['sha256']
        with self.assertRaisesRegex(ValueError,'pilot/unsupported interaction'):
            gate.require_frozen_coverage_certificate(self.root,pointer,protocol,files)
        protocol,report,files=self.certificate();path.write_text(json.dumps(report));pointer=self.pointer(path)
        with self.assertRaisesRegex(ValueError,'TEST lock must bind'):
            gate.require_frozen_coverage_certificate(self.root,pointer,protocol,files)

    def test_certificate_tolerates_only_release_registration_changes(self):
        protocol,report,files=self.certificate();path=self.write('certificate.json',report);pointer=self.pointer(path)
        files[path.name]=pointer['sha256']
        formal=dict(protocol,r10_validation_release={'path':'new_release.json','sha256':'a'*64},formal_comparison_ready=True)
        self.assertTrue(gate.require_frozen_coverage_certificate(self.root,pointer,formal,files)['passed'])
        with self.assertRaisesRegex(ValueError,'missing/stale/failed'):
            gate.require_frozen_coverage_certificate(self.root,pointer,dict(formal,seed=11),files)


if __name__=='__main__':unittest.main()
