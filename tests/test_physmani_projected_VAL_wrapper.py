"""Actual run_case with projected CPU backend; no GPU/native process."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import evaluate_physmani_projected_VAL_r10 as w
from tests.test_evaluate_multitask_case_r10 import MockEnv,MockAdapter
from tests.test_multitask_baseline_component import config,camera_file,put
from our_mani.multitask.evaluation import case_identity,file_sha
b=w.component


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=w.BASE);self.root=Path(self.temp.name)
        self.config=config(); self.config['baseline_checkpoint']=str(self.root/'epoch_99999.pth')
        (self.root/'epoch_99999.pth').write_bytes(b'CPU weights only')
        self.row=camera_file(self.root);self.rows=[]
        for task in b.TASK_SPECS:
            n=18 if task in ('insert_onto_rotating_peg','put_rubbish_in_moving_bin_high_speed') else 19 if task=='insert_onto_rotating_peg_high_speed' else 20
            self.rows += [self.row if task==self.row['task'] and i==0 else
                dict(task=task,raw_episode_id='episode'+str(i),path='/readonly/'+task+'/'+str(i)) for i in range(n)]
        self.manifest=put(w.ROOT,str((self.root/'manifest.json').relative_to(w.ROOT)),dict(split='validation',episodes=self.rows))
        self.configptr=put(w.ROOT,str((self.root/'config.json').relative_to(w.ROOT)),self.config)
        ledger=put(w.ROOT,str((self.root/'calibration.json').relative_to(w.ROOT)),dict(entries=[b.calibration_dependency(self.row)]))
        entry= w.ptr(Path(w.__file__));launcher=w.ptr(w.ROOT/'scripts/run_aligned_gpu.sh')
        self.contract=put(w.ROOT,str((self.root/'contract.json').relative_to(w.ROOT)),dict(schema=b.CONTRACT,version=b.VERSION,
            approved_before_VAL_execution=True,projected_backend_required=True,
            projected_config_sha256=b.digest(b.project_baseline_config(self.config)),canonical_VAL_roster_sha256=b.digest(self.rows),
            calibration_ledger=ledger,execution_entrypoints=dict(parent_source=entry,child_source=entry,launcher_source=launcher)))
        self.files={str(p.resolve()):file_sha(p) for p in (self.root/'config.json',self.root/'manifest.json',self.root/'contract.json',
            self.root/'epoch_99999.pth',Path(w.__file__),Path(w.evaluator.__file__),Path(b.__file__),w.ROOT/'scripts/run_aligned_gpu.sh')}
        self.lock=put(w.ROOT,str((self.root/'lock.json').relative_to(w.ROOT)),dict(files=self.files))
    def tearDown(self):self.temp.cleanup()

    def prepared(self):
        args=Namespace(split='validation',method='physmani',task=self.row['task'],raw_episode_id=self.row['raw_episode_id'],replica=0,
            exploration_seed=None,record_demo=False,demo_only=False)
        identity=case_identity(split='validation',task=self.row['task'],raw_episode_id=self.row['raw_episode_id']+'__policy_aug_r0',
            method='physmani',source_row=self.row,config_sha256=self.configptr['sha256'],inputs_sha256=self.lock['sha256'],
            parent_raw_episode_id=self.row['raw_episode_id'],replica=0)
        return dict(args=args,config=deepcopy(self.config),config_path=self.root/'config.json',manifest_path=self.root/'manifest.json',
            manifest_sha256=self.manifest['sha256'],row=self.row,original_row=self.row,out=self.root/'case',identity=identity,
            files=self.files,lock_path=self.root/'lock.json',lock_sha256=self.lock['sha256'],r09_release={},r10_validation_release=None,record_demo=False)

    def backend_factory(self,**kwargs):
        test=self;calls=self.calls
        class Simulation(MockEnv):
            def __init__(self,*args,**kw):
                super().__init__();self.seed=0;calls.append(('env',kw['protocol']))
                self.reset_recovery_receipt=dict(complete=True,permission=test.config['development_reset_profile'])
                self.reset_receipt=dict(variation=0,reset_recovery_receipt=self.reset_recovery_receipt)
            def reset_development(self,**kw):return ['CPU public task'],None
        def baseline(prepared,*args):
            calls.append(('policy',prepared['config']));return MockAdapter(),object()
        inner=SimpleNamespace(initial_conditions=lambda *a:dict(metric_only=True,policy_use_allowed=False),baseline=baseline)
        backend=b.ProjectedBaselineBackend(**kwargs,inner=inner,simulation_class=Simulation)
        def observed():
            backend.witness['imported_dependency_files_sha256']={str(Path(w.__file__).resolve()):file_sha(w.__file__)}
            backend.witness['loaded_native_files_sha256']={str(Path(sys.executable).resolve()):file_sha(sys.executable)}
            backend.witness['actual_python_executable']=str(Path(sys.executable).resolve())
        backend._imports=observed
        self.backend=backend;return backend

    def test_real_original_run_case_gets_explicit_projected_backend_and_preserves_full_raw_identity(self):
        self.calls=[];prepared=self.prepared();before=deepcopy(prepared['config'])
        with patch.object(b,'verify_execution_contract',return_value={'CPU_fixture_only':True}), \
             patch.object(w.subprocess,'Popen',side_effect=AssertionError('native forbidden')):
            record=w.run_projected_case(prepared,self.contract,backend_factory=self.backend_factory)
        self.assertEqual(prepared['config'],before)
        for label,supplied in self.calls:
            self.assertTrue(all(set(b.OURS_LEAVES).isdisjoint(v) for v in supplied['learned_artifacts'].values()))
            self.assertNotIn('context_checkpoint',supplied)
        self.assertEqual(record['counted_clock']['ticks'],3);self.assertTrue(record['cleanup_completed'])
        base=w.read(prepared['out']/'evaluate_run_case_receipt.json')
        self.assertEqual(base,{k:v for k,v in record.items() if k!='baseline_projected_wrapper_execution'})
        self.assertEqual(record['entrypoint_sha256'],file_sha(w.evaluator.__file__))
        self.assertNotEqual(record['entrypoint_sha256'],record['baseline_projected_wrapper_execution']['actual_wrapper']['sha256'])
        self.assertEqual(record['identity'],prepared['identity']);self.assertEqual(record['config_sha256'],self.configptr['sha256'])
        self.assertEqual(w.read(prepared['out']/'inputs.json')['frozen_input_lock']['sha256'],self.lock['sha256'])

    def test_wrong_method_TEST_exploration_and_demo_rejected_before_backend(self):
        for field,value in (('method','cv_geometry'),('split','test'),('replica',1),('exploration_seed',5),('record_demo',True)):
            prepared=self.prepared();setattr(prepared['args'],field,value)
            with self.subTest(field=field),self.assertRaises(ValueError):
                w.run_projected_case(prepared,self.contract,backend_factory=lambda **kw:(_ for _ in ()).throw(AssertionError('not reached')))

    def test_unapproved_contract_blocks_before_native_and_never_self_approves(self):
        payload=w.read(self.root/'contract.json');payload['approved_before_VAL_execution']=False
        self.contract=put(w.ROOT,str((self.root/'contract.json').relative_to(w.ROOT)),payload)
        self.files[str(self.root/'contract.json')]=self.contract['sha256'];self.lock=put(w.ROOT,str((self.root/'lock.json').relative_to(w.ROOT)),dict(files=self.files))
        with self.assertRaisesRegex(ValueError,'approved'),patch.object(w.subprocess,'Popen',side_effect=AssertionError('native forbidden')):
            w.run_projected_case(self.prepared(),self.contract)
        self.assertFalse(w.read(self.root/'contract.json')['approved_before_VAL_execution'])

    def test_full275_plan_and_exact_own_child_command(self):
        plan=w.prepare_plan(self.root/'config.json',self.root/'manifest.json',self.root/'lock.json',self.root/'contract.json',
            plan_path=self.root/'plan.json',output=self.root/'output')
        self.assertEqual(len(plan['jobs']),275);self.assertEqual(plan['parent_source'],plan['child_source'])
        self.assertFalse(plan['official_TEST_used_for_fitting']);self.assertEqual(plan['methods'],['physmani'])
        command=w.worker_command(plan,plan['jobs'][0],self.root/'output/cases/first/attempt0000')
        self.assertEqual(command[:4],['bash','scripts/run_aligned_gpu.sh',plan['child_source']['path'],'--collect-case'])
        self.assertEqual(command[-2:],['--execution-contract',self.contract['path']])
        self.assertNotIn('--record-demo',command)
        with patch.object(b,'verify_execution_contract',return_value={}):
            self.assertEqual(w.verify_plan(self.root/'plan.json')['jobs'],plan['jobs'])

    def test_official_TEST_backend_keeps_original_reset_path_but_receives_projection(self):
        calls=[];backend=b.ProjectedBaselineBackend(root=w.ROOT,execution_contract=self.contract,
            inner=SimpleNamespace(environment=lambda p:(calls.append(p) or ('official','unchanged'))),
            simulation_class=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('no development reset for TEST')))
        prepared=self.prepared();prepared['args'].split='test'
        self.assertEqual(backend.environment(prepared),('official','unchanged'))
        self.assertTrue(all(set(b.OURS_LEAVES).isdisjoint(v) for v in calls[0]['config']['learned_artifacts'].values()))
        self.assertIn('context_checkpoint',prepared['config']);self.assertIsNone(backend.witness)


if __name__=='__main__':unittest.main()
