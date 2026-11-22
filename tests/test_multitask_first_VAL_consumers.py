"""Consumer boundary checks; the independent VAL helper tests real275 chains."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

from our_mani.multitask.acquisition_manifest import COMPOSITE_VERSION,load_acquisition_manifest
from our_mani.multitask.artifact_fitting import load_train_cases
from our_mani.multitask.evaluation import file_sha
from our_mani.multitask.training_view import load_acquisitions
from tests import test_multitask_training_view as fixtures


class FirstVALConsumerTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.TrainingViewTests(methodName='runTest');self.fixture.setUp();self.root=self.fixture.root
        self.acquisition=self.fixture.acquisition('validation','VAL_source0',0,30,behavior='VAL_hold_motion_identification')
        self.path=self.fixture.manifest('first_VAL',[self.acquisition]);self.value=json.loads(self.path.read_text())
        self.value['shared_profile_execution_registration']=dict(path='first_execution_registry.json',sha256='e'*64)
        self.save()

    def tearDown(self):self.fixture.tearDown()
    def save(self):self.path.write_text(json.dumps(self.value))
    def proof(self):
        return dict(audit_passed=True,complete=True,first_execution=True,
            execution_kind='first_execution_VAL_shared_development_initialization',acquisition_projection=deepcopy(self.value),
            actual_plan_sha256_values=['a'*64],per_case_actual_plan_sha256={self.value['case_receipts'][0]['case_id']:'a'*64},
            verified_files_sha256={})
    def audited(self,proof=None):
        return patch('our_mani.multitask.validation_lineage.audit_validation_acquisition_lineage',
            return_value=self.proof() if proof is None else proof)
    def rewrite_record(self,mutate):
        receipt=self.value['case_receipts'][0];path=self.root/receipt['path'];record=json.loads(path.read_text());mutate(record)
        child=self.root/record['parent_process_receipt']['child_case_path']
        child.write_text(json.dumps({key:value for key,value in record.items() if key!='parent_process_receipt'}))
        record['parent_process_receipt']['child_case_sha256']=file_sha(child);path.write_text(json.dumps(record));receipt['sha256']=file_sha(path)
        self.value['entries'][0]=record['entry'];self.save()

    def test_first_VAL_retains_single_plan_and_separate_lineage_with_archived_launcher(self):
        original=[(self.root/self.value['case_receipts'][0]['path']).read_bytes(),self.acquisition[2].read_bytes()]
        archive=self.root/'archive/launch.sh';archive.parent.mkdir();archive.write_bytes(self.fixture.launcher.read_bytes())
        proof=self.proof();proof['verified_files_sha256'][str(archive)]=file_sha(archive)
        self.fixture.launcher.write_text('# later runtime version, never the historical launch\n')
        with self.audited(proof) as auditor:
            files={};selected,ledger,lineage=load_acquisitions(self.root,[self.path],'validation',files)
        auditor.assert_called_once_with(self.root,self.path,require_complete=True)
        self.assertEqual((len(selected),len(ledger)),(1,1));self.assertEqual(lineage[0]['plan_sha256'],'a'*64)
        self.assertEqual(lineage[0]['actual_plan_sha256_values'],['a'*64]);self.assertFalse(lineage[0]['historical_composite'])
        self.assertTrue(lineage[0]['first_execution_VAL_shared_profile'])
        self.assertEqual(lineage[0]['execution_kind'],'first_execution_VAL_shared_development_initialization')
        self.assertNotIn(str(self.fixture.launcher),files);self.assertEqual(files[str(archive)],file_sha(archive))
        self.assertEqual(original,[(self.root/self.value['case_receipts'][0]['path']).read_bytes(),self.acquisition[2].read_bytes()])
        with self.audited(proof),self.assertRaisesRegex(ValueError,'TRAIN teacher/hold'):
            load_train_cases(self.root,self.path)

    def test_incomplete_wrong_kind_or_rewritten_projection_cannot_enter(self):
        for kind in ('incomplete','kind','first_execution','projection','single_plan_list'):
            proof=self.proof()
            if kind=='incomplete':proof['complete']=False
            elif kind=='kind':proof['execution_kind']='historical_composite'
            elif kind=='first_execution':proof['first_execution']=False
            elif kind=='projection':proof['acquisition_projection']['plan_sha256']='b'*64
            else:proof['actual_plan_sha256_values']=['a'*64,'b'*64]
            with self.subTest(kind=kind),self.audited(proof),self.assertRaisesRegex(ValueError,'complete verified|original single-plan'):
                load_acquisition_manifest(self.root,self.path)

    def test_actual_plan_map_and_execution_markers_must_be_unambiguous(self):
        for kind in ('missing','extra','other_plan'):
            proof=self.proof();mapping=proof['per_case_actual_plan_sha256']
            if kind=='missing':mapping.clear()
            elif kind=='extra':mapping['unexecuted']='a'*64
            else:mapping[next(iter(mapping))]='b'*64
            with self.subTest(kind=kind),self.audited(proof),self.assertRaisesRegex(ValueError,'per-case plan|original single plan'):
                load_acquisition_manifest(self.root,self.path)
        self.value['schema']=COMPOSITE_VERSION;self.save()
        with self.audited() as auditor,self.assertRaisesRegex(ValueError,'cannot combine'):
            load_acquisition_manifest(self.root,self.path)
        auditor.assert_not_called()

    def test_valid_audit_does_not_bypass_existing_teacher_clock_and_array_checks(self):
        for kind in ('teacher','clock','array'):
            if kind=='teacher':
                self.rewrite_record(lambda record:record.update(teacher_actions_used=True));regex='wrongly supervised'
            elif kind=='clock':
                self.rewrite_record(lambda record:record.update(teacher_actions_used=False))
                self.rewrite_record(lambda record:record['alignment'].update(actual_task_seconds=2.));regex='clock'
            else:
                self.rewrite_record(lambda record:record['alignment'].update(actual_task_seconds=1.45))
                cache=self.acquisition[2]
                with np.load(cache,allow_pickle=False) as stored:arrays={name:stored[name].copy() for name in stored.files}
                arrays['rotations'][0,0]=np.zeros((3,3));np.savez_compressed(cache,**arrays)
                def bind(record):
                    entry=record['entry'];entry['cache_sha256']=file_sha(cache);record['files_sha256'][entry['cache']]=entry['cache_sha256']
                self.rewrite_record(bind);regex='SO3'
            with self.subTest(kind=kind),self.audited(),self.assertRaisesRegex(ValueError,regex):
                load_acquisitions(self.root,[self.path],'validation',{})


if __name__=='__main__':unittest.main()
