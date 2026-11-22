"""Task subset raw-chain boundaries; no native/GPU/optimizer is launched.

Only the separately tested runtime closure and HDF5 calibration reader are
mocked. Full275 plan identities, selected18/19/20 raw/parent/base/input chains,
physical source/clock witnesses, and shared original case kernel are real.
"""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from our_mani.multitask import baseline_component as original
from our_mani.multitask import baseline_task_component as gate
from our_mani.multitask.evaluation import file_sha,digest,case_identity,native_completion_record
from our_mani.multitask.simulation import development_seed
from our_mani.multitask.specs import TASK_SPECS
from tests.test_multitask_baseline_component import config


class BaselineTaskComponentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.task='moving_basketball_in_hoop'
        self.base=self.root/'artifacts/framework/multitask_r10';self.base.mkdir(parents=True)
    def tearDown(self):self.temp.cleanup()
    def put(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return self.ptr(path)
    def ptr(self,path):return dict(path=str(Path(path).relative_to(self.root)),sha256=file_sha(path))
    def read(self,path):return json.loads((self.root/path).read_text())
    def fixture(self,task=None):
        self.task=task or self.task;self.rows=[]
        for name in TASK_SPECS:
            n=18 if name in ('insert_onto_rotating_peg','put_rubbish_in_moving_bin_high_speed') else 19 if name=='insert_onto_rotating_peg_high_speed' else 20
            self.rows.extend(dict(task=name,raw_episode_id='episode'+str(i),path='/CPU/val_package_compressed/'+name+'/'+str(i)+'/data.h5') for i in range(n))
        self.roster=self.put('roster.json',dict(split='validation',episodes=self.rows))
        self.config=config();self.config.update(version='new_common_config',baseline_checkpoint=str(self.root/'epoch_99999.pth'),candidate_config={'frozen':True})
        permission=self.put('physical_permission.json',dict(CPU_fixture_only=True));self.config['development_reset_profile']=permission
        self.sources={}
        for name in ('scripts/evaluate_physmani_projected_VAL_r10.py','scripts/evaluate_multitask_case_r10.py',
                'scripts/run_aligned_gpu.sh','epoch_99999.pth','python-executable'):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# CPU source/runtime fixture, never executed\n')
            self.sources[str(path)]=file_sha(path)
        path=self.root/'our_mani/multitask/baseline_component.py';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(Path(original.__file__).read_bytes());self.sources[str(path)]=file_sha(path)
        wrapper=self.ptr(self.root/'scripts/evaluate_physmani_projected_VAL_r10.py')
        self.contract_value=dict(schema=original.CONTRACT,version=original.VERSION,frozen_utc='2026-01-01T00:00:00+00:00',
            approved_before_VAL_execution=True,projected_backend_required=True,actual_dependency_closure_reviewed=True,
            projected_config_sha256=digest(original.project_baseline_config(self.config)),canonical_VAL_roster_sha256=digest(self.rows),
            execution_entrypoints=dict(parent_source=wrapper,child_source=wrapper,launcher_source=self.ptr(self.root/'scripts/run_aligned_gpu.sh')))
        self.contract=self.put('execution_contract.json',self.contract_value);self.cfg=self.put('common_config.json',self.config)
        self.files=dict(self.sources)
        self.files.update({str(self.root/p['path']):p['sha256'] for p in (self.roster,permission,self.contract,self.cfg)})
        self.lock=self.put('baseline_actual_lock.json',dict(files=self.files))
        self.common_lock=self.put('common_union_lock.json',dict(files={**self.files,str(self.root/self.lock['path']):self.lock['sha256']}))
        self.plan_value=dict(version='r10_fresh_policy_queue_v1',created_utc='2026-01-02T00:00:00+00:00',split='validation',methods=['physmani'],
            replicas=1,source_episodes=275,expected_cases=275,official_TEST_used_for_fitting=False,case_isolation='fresh_simulator_process_per_episode',
            config=self.cfg,frozen_inputs=self.lock,manifest=self.roster,parent_source=wrapper,child_source=wrapper,
            baseline_execution_contract=self.contract,output=str((self.base/'baseline').relative_to(self.root)),jobs=[])
        for row in self.rows:
            identity=case_identity(split='validation',task=row['task'],raw_episode_id=row['raw_episode_id']+'__policy_aug_r0',method='physmani',
                source_row=row,config_sha256=self.cfg['sha256'],inputs_sha256=self.lock['sha256'],parent_raw_episode_id=row['raw_episode_id'],replica=0)
            self.plan_value['jobs'].append(dict(job_id=identity['sha256'][:20],identity=identity,row=row,source_raw_episode_id=row['raw_episode_id'],
                replica=0,exploration_seed=None,record_demo=False))
        self.plan_value['sha256']=digest(self.plan_value);self.plan=self.put('plan.json',self.plan_value)
        self.closure=dict(contract=self.contract_value,dependencies=self.sources,fingerprint=dict(CPU_closure_fixture=True),mutable_state_permission=None)
        self.canonical=[];self.raw=[];self.jobs=[job for job in self.plan_value['jobs'] if job['row']['task']==self.task]
        for index,job in enumerate(self.jobs):self.case(job,index)
        self.status=self.put(self.plan_value['output']+'/status.json',dict(complete=False,queue_running=True,plan_sha256=self.plan_value['sha256'],
            expected=275,completed=len(self.jobs),errors=[],stop_requested=False))
    def case(self,job,index):
        row=job['row'];seed,seed_identity=development_seed(row['task'],row,'validation',seed=self.config['seed'],replica=0)
        reset=dict(complete=True,permission=self.config['development_reset_profile'],source_resampled=False,official_TEST_allowed=False,
            task=row['task'],split='validation',replica=0,seed=seed,seed_identity=seed_identity,
            source=dict(task=row['task'],raw_episode_id=row['raw_episode_id'],path=row['path']),
            variation=TASK_SPECS[row['task']].supported_variations[seed % TASK_SPECS[row['task']].variation_count],
            recovery_applied=False,original_reset_succeeded=True)
        witness=dict(version=original.VERSION,execution_contract=self.contract,projected_config_sha256=digest(original.project_baseline_config(self.config)),
            full_config_identity_preserved=True,complete=True,projection_helper_sha256=self.sources[str(self.root/'our_mani/multitask/baseline_component.py')],
            calibration_binding_sha256=digest({'fixture_row':row}),split='validation',method='physmani',shared_reset_receipt=reset,
            actual_python_executable=str(self.root/'python-executable'),
            imported_dependency_files_sha256={str(self.root/self.plan_value['child_source']['path']):self.plan_value['child_source']['sha256']},
            loaded_native_files_sha256={str(self.root/'python-executable'):self.sources[str(self.root/'python-executable')]})
        identity=job['identity'];record=native_completion_record(identity,process_id=100+index,success=False,fatal=False,ticks=200,
            initial_conditions=dict(metric_only=True,policy_use_allowed=False,reset_receipt=dict(reset_recovery_receipt=reset),baseline_execution_dependency_witness=witness),
            alignment_receipt=dict(clock=dict(ticks=200,dt=.05,budget_seconds=10.,simulation_seconds=10.),actual_physics_seconds=10.,
                actual_task_seconds=10.,native_success_latched=False,native_terminal_failure_latched=False))
        record.update(video=None,cleanup_completed=True,baseline_executor=self.config['baseline_executor'],compute_seconds=1.,wall_seconds=2.,simulation_seconds=10.,
            evaluation_process_pid=100+index,source_row_sha256=identity['source_row_sha256'],checkpoint_sha256=self.sources[str(self.root/'epoch_99999.pth')],
            config_sha256=self.cfg['sha256'],manifest_sha256=self.roster['sha256'],frozen_input_lock=dict(path=str(self.root/self.lock['path']),sha256=self.lock['sha256']),
            entrypoint_sha256=self.sources[str(self.root/'scripts/evaluate_multitask_case_r10.py')])
        directory=self.root/self.plan_value['output']/'cases'/job['job_id']/'attempt0000';directory.mkdir(parents=True)
        command=['bash','scripts/run_aligned_gpu.sh',self.plan_value['child_source']['path'],'--collect-case','--config',self.cfg['path'],
            '--manifest',self.roster['path'],'--frozen-inputs',self.lock['path'],'--split','validation','--task',row['task'],
            '--raw-episode-id',row['raw_episode_id'],'--method','physmani','--replica','0','--out',str(directory.relative_to(self.root)),
            '--execution-contract',self.contract['path']]
        base=self.put(directory/'evaluate_run_case_receipt.json',record)
        inputs=self.put(directory/'inputs.json',dict(identity=identity,process_id=record['process_id'],frozen_input_lock=record['frozen_input_lock'],
            entrypoint_sha256=record['entrypoint_sha256'],manifest_sha256=self.roster['sha256'],original_manifest_row_sha256=digest(row),files=self.files,
            argv=[self.plan_value['child_source']['path'],*command[3:]]))
        record['baseline_projected_wrapper_execution']=dict(version='r10_actual_projected_PhysMani_VAL_wrapper_v1',execution_contract=self.contract,
            actual_wrapper=self.plan_value['child_source'],evaluator_source=self.ptr(self.root/'scripts/evaluate_multitask_case_r10.py'),
            full_config_identity_preserved=True,original_record_fields_rewritten=False,projected_config_sha256=digest(original.project_baseline_config(self.config)),
            base_run_case_receipt=base,base_run_case_inputs=inputs,dependency_witness_after_cleanup=deepcopy(witness),
            jit_artifact_execution=None,mutable_runtime_state_execution=None)
        raw=directory/'case.json';self.put(raw,record)
        log=directory/'native.log';log.write_text('CPU raw chain, native never launched\n')
        record['parent_case_process_receipt']=dict(child_path=str(raw.relative_to(self.root)),child_sha256=file_sha(raw),child_process_id=record['process_id'],
            launcher_process_id=9000+index,plan_sha256=self.plan_value['sha256'],parent_source_sha256=self.plan_value['parent_source']['sha256'],
            command=command,log_path=str(log.relative_to(self.root)),log_sha256=file_sha(log),returncode=0)
        canonical=self.root/self.plan_value['output']/'results'/(job['job_id']+'.json');self.put(canonical,record)
        self.canonical.append(canonical);self.raw.append(raw)
    def audit(self,*,config_pointer=None,lock_pointer=None):
        with patch.object(gate,'verify_execution_contract',return_value=self.closure),\
                patch.object(gate,'calibration_dependency',side_effect=lambda row:{'fixture_row':row}):
            return gate.validate_baseline_task_component(self.root,self.task,self.plan,self.contract,config_pointer or self.cfg,
                lock_pointer or self.common_lock,self.roster)
    def rewrite(self,index,record,*,base=None,inputs=None):
        wrapper=record['baseline_projected_wrapper_execution']
        if base is not None:wrapper['base_run_case_receipt']=self.put(wrapper['base_run_case_receipt']['path'],base)
        if inputs is not None:wrapper['base_run_case_inputs']=self.put(wrapper['base_run_case_inputs']['path'],inputs)
        raw={k:v for k,v in record.items() if k!='parent_case_process_receipt'};self.put(self.raw[index],raw)
        record['parent_case_process_receipt']['child_sha256']=file_sha(self.raw[index]);self.put(self.canonical[index],record)

    def test_same_config_full_task20_can_finish_before_other255(self):
        self.fixture();r=self.audit();self.assertEqual(r['sources'],20);self.assertTrue(r['baseline_task_component_reusable'])
        self.assertFalse(r['global_queue_complete_claimed']);self.assertFalse(r['formal_TEST_permitted'])
        self.assertEqual(r['full_plan_sources'],275);self.assertEqual(r['actual_other_task_completed_snapshot'],0)
        self.assertNotEqual(r['actual_lock'],r['common_lock']);self.assertFalse(r['single_lock_identity_claimed'])
    def test_eighteen_nineteen_actual_denominators_preserved(self):
        for task,n in [('insert_onto_rotating_peg',18),('insert_onto_rotating_peg_high_speed',19)]:
            with self.subTest(task=task):
                self.fixture(task);r=self.audit();self.assertEqual(r['sources'],n)
                # Use a new independent root for the next denominator fixture.
                self.temp.cleanup();self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
                self.base=self.root/'artifacts/framework/multitask_r10';self.base.mkdir(parents=True)
    def test_shared_case_AST_is_exact_and_no_global_mutation(self):
        proof=gate._case_kernel_proof();self.assertFalse(proof['dynamic_code_execution']);self.assertFalse(proof['original_module_patched'])
        self.assertFalse(proof['original_case_checks_weakened'])
    def test_partial_selected_task_and_foreign_canonical_rejected(self):
        self.fixture();data=self.canonical[0].read_bytes();self.canonical[0].unlink()
        with self.assertRaisesRegex(ValueError,'missing task'):self.audit()
        self.canonical[0].write_bytes(data);self.put(self.root/self.plan_value['output']/'results/foreign.json',{})
        with self.assertRaisesRegex(ValueError,'foreign'):self.audit()
    def test_old12_config_or_changed_unknown_field_not_projection_reused(self):
        self.fixture();changed=dict(self.config,unknown_new_policy_setting=True)
        pointer=self.put('other_config.json',changed)
        with self.assertRaisesRegex(ValueError,'historical projection'):self.audit(config_pointer=pointer)
    def test_union_lock_must_cover_actual_lock_and_all_dependencies(self):
        self.fixture();bad=self.read(self.common_lock['path']);bad['files'].pop(str(self.root/'python-executable'))
        pointer=self.put('wrong_union_lock.json',bad)
        with self.assertRaisesRegex(ValueError,'every actual baseline input'):self.audit(lock_pointer=pointer)
    def test_raw_byte_tamper_or_failed_process_cannot_be_scored(self):
        self.fixture();raw=self.raw[0].read_bytes();self.raw[0].write_bytes(raw+b' ')
        with self.assertRaisesRegex(ValueError,'raw child'):self.audit()
        self.raw[0].write_bytes(raw);record=self.read(self.canonical[0]);record['parent_case_process_receipt']['returncode']=1;self.put(self.canonical[0],record)
        with self.assertRaisesRegex(ValueError,'returncode'):self.audit()
    def test_original_base_and_inputs_mutation_cannot_be_hidden_by_new_SHA(self):
        self.fixture();record=self.read(self.canonical[0]);wrapper=record['baseline_projected_wrapper_execution'];inputs=self.read(wrapper['base_run_case_inputs']['path'])
        inputs['process_id']+=1;self.rewrite(0,record,inputs=inputs)
        with self.assertRaisesRegex(ValueError,'original evaluator inputs'):self.audit()
    def test_source_and_physics_reset_seed_guard(self):
        self.fixture();record=self.read(self.canonical[0]);record['recorded_initial_conditions']['reset_receipt']['reset_recovery_receipt']['seed']+=1
        self.rewrite(0,record)
        with self.assertRaisesRegex(ValueError,'source/seed'):self.audit()
    def test_clock_and_cleanup_guard(self):
        self.fixture();record=self.read(self.canonical[0]);record['alignment_receipt']['actual_physics_seconds']=9.
        self.rewrite(0,record)
        with self.assertRaisesRegex(Exception,'actual native clock'):self.audit()
    def test_duplicate_or_deleted_other_plan_task_not_silently_dropped(self):
        self.fixture();value=deepcopy(self.plan_value);value['jobs'].pop();value['sha256']=digest({k:v for k,v in value.items() if k!='sha256'})
        self.plan=self.put('changed_plan.json',value)
        with self.assertRaisesRegex(ValueError,'delete unready tasks'):self.audit()
    def test_prior_infrastructure_attempt_cannot_be_promoted(self):
        self.fixture();path=self.canonical[0];record=self.read(path);directory=(self.root/record['parent_case_process_receipt']['child_path']).parent.parent
        self.put(directory/'infrastructure_parent_receipt.json',dict(status='infrastructure_error'))
        with self.assertRaisesRegex(ValueError,'hidden retry'):self.audit()
    def test_other_task_error_does_not_block_clean_task_but_unknown_or_local_error_does(self):
        self.fixture();status=self.read(self.status['path']);other=next(job for job in self.plan_value['jobs'] if job['row']['task']!=self.task)
        status['errors']=[dict(job_id=other['job_id'],error='unrelated infrastructure error')];self.put(self.status['path'],status)
        self.assertTrue(self.audit()['complete'])
        status['errors']=[dict(job_id=self.jobs[0]['job_id'],error='actual task infrastructure error')];self.put(self.status['path'],status)
        with self.assertRaisesRegex(ValueError,'task infrastructure'):self.audit()
    def test_failed_test_or_demo_population_rejected(self):
        self.fixture();value=deepcopy(self.plan_value);value['official_TEST_used_for_fitting']=True
        value['sha256']=digest({k:v for k,v in value.items() if k!='sha256'});self.plan=self.put('TEST_plan.json',value)
        with self.assertRaisesRegex(ValueError,'full275'):self.audit()
    def test_changed_actual_source_rejected(self):
        self.fixture();(self.root/'scripts/evaluate_physmani_projected_VAL_r10.py').write_text('# mutated source\n')
        with self.assertRaisesRegex(ValueError,'execution closure changed'):self.audit()


if __name__=='__main__':unittest.main()
