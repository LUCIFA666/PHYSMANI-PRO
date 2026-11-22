"""CPU boundary tests; mocked auditors never claim native aux data exist."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from our_mani.multitask import teacher_supplement as audit
from our_mani.multitask import acquisition_manifest as consumer
from our_mani.multitask import acquisition_lineage as lineage
from our_mani.multitask.training_view import load_acquisitions


class SupplementBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.primary_path=self.root/'primary.json';self.primary_path.write_text(json.dumps({'version':consumer.COMPOSITE_VERSION}))
        self.aux_path=self.root/'aux.json';self.aux_path.write_text(json.dumps({'version':audit.AUX_MANIFEST_VERSION}))
        self.path=self.root/'bundle.json';self.path.write_text(json.dumps(dict(version=audit.BUNDLE_VERSION,
            primary_manifest=lineage.pointer(self.root,self.primary_path),
            auxiliary_teacher_manifest=lineage.pointer(self.root,self.aux_path),primary_Context_view_unchanged=True)))
        tasks=list(audit.TASKS)+['other_task_'+str(i) for i in range(12)]
        entries=[];receipts=[];mapping={};aux_entries=[];aux_receipts=[];aux_mapping={}
        for task in tasks:
            for i in range(100):
                parent=f'train:{task}:episode{i}'
                for teacher in (True,False):
                    cid=f'{task}:{i}:{teacher}:r0'
                    entries.append(dict(task=task,parent_source_group=parent,teacher_used=teacher,
                        context_primary_view=not teacher,augmentation_replica=0,behavior_replica=0))
                    mapping[cid]='a'*64;receipts.append(dict(case_id=cid,path=cid,sha256='c'*64,actual_plan_sha256='a'*64))
                if task in audit.TASKS:
                    cid=f'{task}:{i}:True:r1'
                    aux_entries.append(dict(task=task,parent_source_group=parent,teacher_used=True,
                        context_primary_view=False,augmentation_replica=1,behavior_replica=1))
                    aux_mapping[cid]='b'*64;aux_receipts.append(dict(case_id=cid,path=cid,sha256='d'*64,actual_plan_sha256='b'*64))
        self.primary=dict(complete=True,expected=2800,completed=2800,entries=entries,case_receipts=receipts,
            base_plan_sha256='e'*64,augmentation_budget={'original_K1':True})
        self.primary_proof=dict(complete=True,per_case_actual_plan_sha256=mapping,verified_files_sha256={})
        self.aux=dict(complete=True,acquisition_projection=dict(expected=200,completed=200,entries=aux_entries,
            case_receipts=aux_receipts),base_plan_sha256='e'*64,per_case_actual_plan_sha256=aux_mapping,
            verified_files_sha256={})

    def run_audit(self):
        with patch.object(consumer,'load_acquisition_manifest',return_value=(self.primary,self.primary_proof)), \
             patch.object(audit,'audit_manifest',return_value=self.aux):
            return audit.audit_bundle(self.root,self.path)

    def test_original2800_and_aux200_preserve_1400_parents_and_distinct_actual_plan_shas(self):
        proof=self.run_audit();projection=proof['acquisition_projection']
        self.assertTrue(proof['complete']);self.assertEqual(len(projection['entries']),3000)
        self.assertEqual(projection['TRAIN_parent_sources'],1400)
        self.assertEqual(projection['teacher_rollouts'],1600)
        self.assertEqual(projection['original_Context_K1_hold_rollouts'],1400)
        self.assertIsNone(projection['plan_sha256']);self.assertFalse(projection['single_plan_SHA_claimed'])
        self.assertEqual(set(proof['per_case_actual_plan_sha256'].values()),{'a'*64,'b'*64})
        self.assertEqual(projection['entries'][:2800],self.primary['entries'])
        self.assertEqual(projection['case_receipts'][:2800],self.primary['case_receipts'])

    def test_actual_schema_only_historical_composite_is_accepted(self):
        self.primary_path.write_text(json.dumps({'schema':consumer.COMPOSITE_VERSION}))
        bundle=json.loads(self.path.read_text())
        bundle['primary_manifest']=lineage.pointer(self.root,self.primary_path)
        self.path.write_text(json.dumps(bundle))
        proof=self.run_audit()
        self.assertTrue(proof['complete'])
        self.assertEqual(proof['acquisition_projection']['completed'],3000)

    def test_untyped_primary_is_rejected_before_auditors(self):
        self.primary_path.write_text(json.dumps({'schema':'ordinary_manifest'}))
        bundle=json.loads(self.path.read_text())
        bundle['primary_manifest']=lineage.pointer(self.root,self.primary_path)
        self.path.write_text(json.dumps(bundle))
        with patch.object(consumer,'load_acquisition_manifest') as loader, \
             patch.object(audit,'audit_manifest') as aux, self.assertRaisesRegex(ValueError,'historical primary composite'):
            audit.audit_bundle(self.root,self.path)
        loader.assert_not_called();aux.assert_not_called()

    def test_aux_cannot_replace_missing_primary_hold_or_parent(self):
        for change in ('primary_incomplete','primary_missing_hold','aux_incomplete','aux_wrong_parent','wrong_base','duplicate_case'):
            with self.subTest(change=change):
                primary=deepcopy(self.primary);aux=deepcopy(self.aux)
                if change=='primary_incomplete':primary['completed']=2799
                elif change=='primary_missing_hold':primary['entries'][-1]['context_primary_view']=False
                elif change=='aux_incomplete':aux['complete']=False
                elif change=='aux_wrong_parent':aux['acquisition_projection']['entries'][0]['parent_source_group']='train:unknown:0'
                elif change=='wrong_base':aux['base_plan_sha256']='f'*64
                else:
                    old=next(iter(aux['per_case_actual_plan_sha256']));value=aux['per_case_actual_plan_sha256'].pop(old)
                    aux['per_case_actual_plan_sha256'][next(iter(self.primary_proof['per_case_actual_plan_sha256']))]=value
                with patch.object(consumer,'load_acquisition_manifest',return_value=(primary,self.primary_proof)), \
                     patch.object(audit,'audit_manifest',return_value=aux), self.assertRaises(ValueError):
                    audit.audit_bundle(self.root,self.path)

    def test_consumer_demands_complete_audit_exact_map_and_no_single_plan_forgery(self):
        proof=self.run_audit()
        with patch.object(audit,'audit_bundle',return_value=proof) as mocked:
            projection,actual=consumer.load_acquisition_manifest(self.root,self.path)
        mocked.assert_called_once_with(self.root,self.path,require_complete=True)
        self.assertEqual(projection['version'],audit.BUNDLE_VERSION)
        for change in ('incomplete','single_plan','bad_map','rewritten'):
            broken=deepcopy(proof)
            if change=='incomplete':broken['complete']=False
            elif change=='single_plan':broken['acquisition_projection']['plan_sha256']='a'*64
            elif change=='bad_map':broken['per_case_actual_plan_sha256'].pop(next(iter(broken['per_case_actual_plan_sha256'])))
            else:broken['acquisition_projection']['projection_rewrites_original_receipts']=True
            with self.subTest(change=change),patch.object(audit,'audit_bundle',return_value=broken),self.assertRaises(ValueError):
                consumer.load_acquisition_manifest(self.root,self.path)
        value=json.loads(self.path.read_text());value['shared_profile_execution_registration']={}
        self.path.write_text(json.dumps(value))
        with patch.object(audit,'audit_bundle') as mocked,self.assertRaisesRegex(ValueError,'cannot combine'):
            consumer.load_acquisition_manifest(self.root,self.path)
        mocked.assert_not_called()

    def test_context_view_rejects_aux_bundle_and_retains_original_K1_contract(self):
        proof=self.run_audit()
        with patch.object(audit,'audit_bundle',return_value=proof),self.assertRaisesRegex(ValueError,'fixed-budget'):
            load_acquisitions(self.root,[self.path],'train',{})




from tests import test_multitask_composite_consumers as composite_fixture
from our_mani.multitask.artifact_fitting import load_train_cases


class SupplementArtifactConsumerTests(composite_fixture.CompositeConsumerTests):
    """Reuse real array/clock fixtures at the separate strict auditor boundary."""
    def setUp(self):
        super().setUp()
        self.projection['version']=audit.BUNDLE_VERSION
        self.path.write_text(json.dumps({'version':audit.BUNDLE_VERSION}))

    def audited(self,proof=None):
        return patch.object(audit,'audit_bundle',return_value=self.proof() if proof is None else proof)

    def test_context_view_retains_both_actual_plans_but_only_hold_is_primary(self):
        with self.audited(),self.assertRaisesRegex(ValueError,'fixed-budget'):
            load_acquisitions(self.root,[self.path],'train',{})

    def test_per_case_plan_mismatch_is_rejected_after_valid_raw_hashes(self):
        item=self.replace_record(1,lambda record:record.update(plan_sha256='c'*64))
        self.projection['case_receipts'][1]['sha256']=item['sha256']
        with self.audited(),self.assertRaisesRegex(ValueError,'incomplete|wrongly supervised'):
            load_train_cases(self.root,self.path)

    def test_native_clock_check_still_runs_for_a_composite(self):
        item=self.replace_record(1,lambda record:record['alignment'].update(actual_task_seconds=1.5))
        self.projection['case_receipts'][1]['sha256']=item['sha256']
        with self.audited(),self.assertRaisesRegex(ValueError,'clock'):
            load_train_cases(self.root,self.path)

    def test_changed_current_launcher_does_not_replace_its_verified_archive(self):
        archived=self.root/'archived_launch.sh';archived.write_bytes(self.fixture.launcher.read_bytes())
        proof=self.proof();proof['verified_files_sha256'][str(archived)]=lineage.file_sha(archived)
        self.fixture.launcher.write_text('# newer live runtime; historical archive retained\n')
        with self.audited(proof):
            _,cases=load_train_cases(self.root,self.path)
        self.assertEqual(len(cases),2)
        ordinary=self.fixture.manifest('ordinary_teacher',[self.acquisitions[0]])
        with self.assertRaisesRegex(ValueError,'launcher source changed'):
            load_train_cases(self.root,ordinary)


if __name__=='__main__':unittest.main()
