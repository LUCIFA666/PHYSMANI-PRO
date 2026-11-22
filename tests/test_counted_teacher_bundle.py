"""Typed admission boundaries only; mocked auditors are not native evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from our_mani.multitask import counted_teacher_bundle as bundle
from our_mani.multitask import acquisition_manifest as consumer
from our_mani.multitask import acquisition_lineage as lineage
from our_mani.multitask import requery_acquisition as requery
from our_mani.multitask import teacher_supplement


class CountedBundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.primary_path=self.root/'primary3000.json'
        self.primary_path.write_text(json.dumps({'version':teacher_supplement.BUNDLE_VERSION}))
        self.aux_path=self.root/'counted100.json';self.aux_path.write_text(json.dumps({'version':requery.MANIFEST_VERSION}))
        self.path=self.root/'bundle3100.json'
        self.path.write_text(json.dumps(dict(version=bundle.BUNDLE_VERSION,
            primary3000_bundle=lineage.pointer(self.root,self.primary_path),
            counted_high_cup_manifest=lineage.pointer(self.root,self.aux_path),primary_Context_view_unchanged=True)))
        entries=[];receipts=[];mapping={};extra_entries=[];extra_receipts=[];extra_map={}
        tasks=[requery.TASK,*teacher_supplement.TASKS,*['fixture_'+str(i) for i in range(11)]]
        for task in tasks:
            for i in range(100):
                parent=f'train:{task}:episode{i}'
                for teacher in (True,False):
                    cid=f'{task}:{i}:teacher{teacher}:r0'
                    entry=dict(task=task,parent_source_group=parent,teacher_used=teacher,
                        context_primary_view=not teacher,context_training_eligible=not teacher,
                        augmentation_replica=0,behavior_replica=0,maximum_collection_ticks=200,
                        interaction_eligible=teacher)
                    entries.append(entry);mapping[cid]='a'*64
                    receipts.append(dict(case_id=cid,path=cid,sha256='c'*64,actual_plan_sha256='a'*64))
                    if teacher and task in teacher_supplement.TASKS:
                        aux=dict(entry,augmentation_replica=1,behavior_replica=1);key=f'{task}:{i}:teacherTrue:r1'
                        entries.append(aux);mapping[key]='b'*64
                        receipts.append(dict(case_id=key,path=key,sha256='d'*64,actual_plan_sha256='b'*64))
                    if teacher and task==requery.TASK:
                        aux=dict(entry,augmentation_replica=1,behavior_replica=1);key=f'{task}:{i}:teacherTrue:r1'
                        extra_entries.append(aux);extra_map[key]='e'*64
                        extra_receipts.append(dict(case_id=key,path=key,sha256='f'*64,actual_plan_sha256='e'*64))
        self.primary=dict(complete=True,expected=3000,completed=3000,entries=entries,case_receipts=receipts,
            TRAIN_parent_sources=1400,teacher_rollouts=1600,original_Context_K1_hold_rollouts=1400,
            primary_Context_view_unchanged=True,base_plan_sha256='0'*64,
            original_primary_augmentation_budget={'K1':True},auxiliary_augmentation_budget={'basketball':200})
        self.proof=dict(complete=True,verified_files_sha256={},per_case_actual_plan_sha256=mapping)
        self.extra=dict(audit_passed=True,complete=True,verified_files_sha256={},base_plan_sha256='0'*64,
            per_case_actual_plan_sha256=extra_map,acquisition_projection=dict(version=requery.MANIFEST_VERSION,
                complete=True,expected=100,completed=100,entries=extra_entries,case_receipts=extra_receipts,
                primary_K1_roster_unmodified=True,primary_Context_view_unmodified=True))

    def run_audit(self,primary=None,proof=None,extra=None):
        with patch.object(consumer,'load_acquisition_manifest',return_value=(self.primary if primary is None else primary,
                self.proof if proof is None else proof)), \
             patch.object(requery,'audit_manifest',return_value=self.extra if extra is None else extra,create=True):
            return bundle.audit_bundle(self.root,self.path)

    def test_full3100_preserves_original_raws_actual_plans_parent_folds_and_Context1400(self):
        actual=self.run_audit();value=actual['acquisition_projection']
        self.assertTrue(actual['complete']);self.assertFalse(actual['formal_model_or_policy_release'])
        self.assertEqual(value['completed'],3100);self.assertEqual(value['teacher_rollouts'],1700)
        self.assertEqual(value['TRAIN_parent_sources'],1400);self.assertEqual(value['original_Context_K1_hold_rollouts'],1400)
        self.assertEqual(value['entries'][:3000],self.primary['entries'])
        self.assertEqual(value['case_receipts'][:3000],self.primary['case_receipts'])
        self.assertIsNone(value['plan_sha256']);self.assertFalse(value['single_plan_SHA_claimed'])
        self.assertEqual(set(value['per_case_actual_plan_sha256'].values()),{'a'*64,'b'*64,'e'*64})

    def test_missing_primary_or_incomplete_new100_cannot_be_replaced_by_extra_data(self):
        for change in ('primary','primary_hold','new100','base','parent','teacher','context','replica','duplicate','raw_plan'):
            primary=deepcopy(self.primary);extra=deepcopy(self.extra)
            if change=='primary':primary['completed']=2999
            elif change=='primary_hold':primary['entries'][-1]['context_primary_view']=False
            elif change=='new100':extra['complete']=False
            elif change=='base':extra['base_plan_sha256']='9'*64
            elif change=='parent':extra['acquisition_projection']['entries'][0]['parent_source_group']='train:foreign:0'
            elif change=='teacher':extra['acquisition_projection']['entries'][0]['teacher_used']=False
            elif change=='context':extra['acquisition_projection']['entries'][0]['context_primary_view']=True
            elif change=='replica':extra['acquisition_projection']['entries'][0]['augmentation_replica']=True
            elif change=='raw_plan':extra['acquisition_projection']['case_receipts'][0]['actual_plan_sha256']='9'*64
            else:
                key=next(iter(extra['per_case_actual_plan_sha256']));sha=extra['per_case_actual_plan_sha256'].pop(key)
                extra['per_case_actual_plan_sha256'][next(iter(self.proof['per_case_actual_plan_sha256']))]=sha
            with self.subTest(change=change),self.assertRaises(ValueError):self.run_audit(primary=primary,extra=extra)

    def test_untyped_original_manifest_is_rejected_before_audit(self):
        self.primary_path.write_text(json.dumps({'version':'ordinary'}))
        value=json.loads(self.path.read_text());value['primary3000_bundle']=lineage.pointer(self.root,self.primary_path)
        self.path.write_text(json.dumps(value))
        with patch.object(consumer,'load_acquisition_manifest') as loader, \
             self.assertRaisesRegex(ValueError,'original primary3000'):bundle.audit_bundle(self.root,self.path)
        loader.assert_not_called()

    def test_consumer_demands_complete_per_case_map_and_rejects_mixed_schema(self):
        proof=self.run_audit()
        with patch.object(bundle,'audit_bundle',return_value=proof) as mocked:
            projection,_=consumer.load_acquisition_manifest(self.root,self.path)
        mocked.assert_called_once_with(self.root,self.path,require_complete=True)
        self.assertEqual(projection['version'],bundle.BUNDLE_VERSION)
        for change in ('complete','single','rewrite','map'):
            broken=deepcopy(proof)
            if change=='complete':broken['complete']=False
            elif change=='single':broken['acquisition_projection']['plan_sha256']='a'*64
            elif change=='rewrite':broken['acquisition_projection']['projection_rewrites_original_receipts']=True
            else:broken['per_case_actual_plan_sha256'].pop(next(iter(broken['per_case_actual_plan_sha256'])))
            with self.subTest(change=change),patch.object(bundle,'audit_bundle',return_value=broken),self.assertRaises(ValueError):
                consumer.load_acquisition_manifest(self.root,self.path)
        value=json.loads(self.path.read_text());value['schema']=teacher_supplement.BUNDLE_VERSION
        self.path.write_text(json.dumps(value))
        with patch.object(bundle,'audit_bundle') as mocked,self.assertRaisesRegex(ValueError,'cannot combine'):
            consumer.load_acquisition_manifest(self.root,self.path)
        mocked.assert_not_called()


from tests import test_multitask_composite_consumers as composite_fixture
from our_mani.multitask.training_view import load_acquisitions
from our_mani.multitask.artifact_fitting import load_train_cases


class CountedBundleConsumerTests(composite_fixture.CompositeConsumerTests):
    def setUp(self):
        super().setUp();self.projection['version']=bundle.BUNDLE_VERSION
        self.path.write_text(json.dumps({'version':bundle.BUNDLE_VERSION}))

    def audited(self,proof=None):
        return patch.object(bundle,'audit_bundle',return_value=self.proof() if proof is None else proof)

    def test_context_view_retains_both_actual_plans_but_only_hold_is_primary(self):
        with self.audited(),self.assertRaisesRegex(ValueError,'fixed-budget'):
            load_acquisitions(self.root,[self.path],'train',{})

    def test_incomplete_audit_cannot_enter_either_consumer(self):
        broken=self.proof();broken['complete']=False
        for function in (lambda:load_train_cases(self.root,self.path),lambda:load_acquisitions(self.root,[self.path],'train',{})):
            with self.audited(broken),self.assertRaisesRegex(ValueError,'complete verified'):function()

    def test_per_case_plan_mismatch_is_rejected_after_valid_raw_hashes(self):
        item=self.replace_record(1,lambda raw:raw.update(plan_sha256='c'*64))
        self.projection['case_receipts'][1]['sha256']=item['sha256']
        with self.audited(),self.assertRaisesRegex(ValueError,'incomplete'):load_train_cases(self.root,self.path)

    def test_native_clock_check_still_runs_for_a_composite(self):
        item=self.replace_record(1,lambda raw:raw['alignment'].update(actual_task_seconds=1.5))
        self.projection['case_receipts'][1]['sha256']=item['sha256']
        with self.audited(),self.assertRaisesRegex(ValueError,'clock'):load_train_cases(self.root,self.path)

    def test_changed_current_launcher_does_not_replace_its_verified_archive(self):
        archived=self.root/'archived_launch.sh';archived.write_bytes(self.fixture.launcher.read_bytes())
        proof=self.proof();proof['verified_files_sha256'][str(archived)]=lineage.file_sha(archived)
        self.fixture.launcher.write_text('# changed live; archived actual source is retained\n')
        with self.audited(proof):_,cases=load_train_cases(self.root,self.path)
        self.assertEqual(len(cases),2)


if __name__=='__main__':unittest.main()
