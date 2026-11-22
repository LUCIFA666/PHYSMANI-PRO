"""Real wrapper + original run_case +275-case consumer/composer CPU proofs.

Only native simulation/policy/runtime observation, the separately exercised
execution closure/camera calibration, and model/parity/selection native proofs
are mocked. The wrapper, original evaluator, baseline component validator,
1100 ours raw receipts and 275 baseline raw/base/input/launcher chains are real.
"""
from argparse import Namespace
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from our_mani.multitask import baseline_component as component
from our_mani.multitask.evaluation import digest, file_sha, native_completion_record
from scripts import evaluate_physmani_projected_VAL_r10 as wrapper
from scripts import prepare_physmani_projected_VAL_r10 as preparation
from tests import test_multitask_composite_validation_release as composition_fixtures
from tests.test_evaluate_multitask_case_r10 import MockEnv, MockAdapter
from tests import test_physmani_projected_VAL_wrapper as wrapper_fixtures
from tests import test_multitask_baseline_dependencies as dependency_fixtures
from our_mani.multitask import baseline_dependencies


class ActualWrapperCompositeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = composition_fixtures.CompositeValidationReleaseTests(methodName='runTest')
        self.fixture.setUp(); self.f = self.fixture; self.root = self.f.root
        self.backend_calls = []; self.wrapper_calls = 0
        self.actual_wrapper = self.root/'scripts/evaluate_physmani_projected_VAL_r10.py'
        self.actual_evaluator = self.root/'scripts/evaluate_multitask_case_r10.py'
        self.actual_helper = self.root/'our_mani/multitask/baseline_component.py'
        for source, target in ((Path(wrapper.__file__),self.actual_wrapper),
                (Path(wrapper.evaluator.__file__),self.actual_evaluator),
                (Path(component.__file__),self.actual_helper)):
            target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(source.read_bytes())
            self.f.sources[str(target)] = file_sha(target)
        ledger = self.f.put('actual_calibration_ledger.json',dict(
            schema='r10_canonical_VAL_camera_dependency_ledger_v1',
            entries=[{'fixture_row':row} for row in self.f.rows]))
        entry = self.f.pointer(self.actual_wrapper)
        contract = dict(schema=component.CONTRACT,version=component.VERSION,
            approved_before_VAL_execution=True,projected_backend_required=True,
            actual_dependency_closure_reviewed=True,frozen_utc='2026-01-01T00:00:00+00:00',
            projected_config_sha256=digest(component.project_baseline_config(self.f.early)),
            canonical_VAL_roster_sha256=digest(self.f.rows),calibration_ledger=ledger,
            execution_entrypoints=dict(parent_source=entry,child_source=entry,
                launcher_source=self.f.pointer(self.root/'scripts/run_aligned_gpu.sh')))
        self.f.contract = self.f.put('execution_contract.json',contract)
        common = dict(self.f.sources)
        for pointer in (self.f.roster,self.f.asset,self.f.profile,self.f.experiment,
                self.f.early_config,self.f.contract,ledger,self.f.early['development_reset_profile']):
            common[str(self.root/pointer['path'])] = pointer['sha256']
        self.f.early_lock = self.f.put('early_lock.json',dict(files=common))
        self.f.final_lock = self.f.put('final_lock.json',dict(files={**common,
            str(self.root/self.f.final_config['path']):self.f.final_config['sha256']}))
        self.early_files = component.normalized(self.root,common)
        self.f.closure = dict(contract=contract,dependencies=self.f.sources,
            fingerprint=dict(CPU_native_closure_fixture_only=True),mutable_state_permission=None)
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(wrapper,'ROOT',self.root))
        self.stack.enter_context(patch.object(wrapper,'BASE',self.f.base))
        self.stack.enter_context(patch.object(wrapper,'STOP',self.root/'runtime/STOP'))
        self.stack.enter_context(patch.object(wrapper,'__file__',str(self.actual_wrapper)))
        self.stack.enter_context(patch.object(wrapper.evaluator,'ROOT',self.root))
        self.stack.enter_context(patch.object(wrapper.evaluator,'BASE',self.f.base))
        self.stack.enter_context(patch.object(wrapper.evaluator,'__file__',str(self.actual_evaluator)))
        self.stack.enter_context(patch.object(component,'__file__',str(self.actual_helper)))
        self.stack.enter_context(patch.object(component,'calibration_dependency',side_effect=lambda row:{'fixture_row':row}))
        self.stack.enter_context(patch.object(component,'verify_execution_contract',return_value=self.f.closure))
        self.stack.enter_context(patch.object(wrapper.subprocess,'Popen',side_effect=AssertionError('native forbidden')))
        self.f.early_plan_value = wrapper.prepare_plan(self.root/self.f.early_config['path'],
            self.root/self.f.roster['path'],self.root/self.f.early_lock['path'],self.root/self.f.contract['path'],
            plan_path=self.f.base/'actual_early_plan.json',output=self.f.base/'actual_early')
        self.f.early_plan = self.f.pointer(self.f.base/'actual_early_plan.json')
        self.f.baseline_paths = []
        for job in self.f.early_plan_value['jobs']:
            self.execute_actual_wrapper(job)
        self.f.put(self.f.early_plan_value['output']+'/status.json',dict(complete=True,
            plan_sha256=self.f.early_plan_value['sha256'],completed=275,expected=275,errors=[],stop_requested=False))
        # Our unchanged raw-schema branch also uses the new final lock identity.
        shutil.rmtree(self.root/self.f.ours_plan_value['output'])
        self.f.ours_plan_value = self.f.plan(list(composition_fixtures.composite.OURS),
            self.f.final_config,self.f.final_lock,'final_child.py','ours')
        self.f.ours_plan = self.f.put('ours_plan.json',self.f.ours_plan_value)
        self.f.ours_paths = self.f.raw_cases(self.f.ours_plan_value,self.f.final,baseline_only=False)
        self.f.composition_value.update(ours_queue_plan=self.f.ours_plan,
            baseline_early_queue_plan=self.f.early_plan,baseline_execution_contract=self.f.contract)
        self.f.composition_value['sha256'] = digest({k:v for k,v in self.f.composition_value.items() if k!='sha256'})
        self.f.composition = self.f.put('composition.json',self.f.composition_value)

    def tearDown(self):
        self.stack.close(); self.fixture.tearDown()

    def backend_factory(self, **kwargs):
        outer = self
        class Simulation(MockEnv):
            def __init__(self,*args,**values):
                super().__init__(); self.seed=0
                outer.backend_calls.append(deepcopy(values['protocol']))
                self.reset_recovery_receipt=dict(complete=True,
                    permission=outer.f.early['development_reset_profile'],
                    source_resampled=False,official_TEST_allowed=False)
                self.reset_receipt=dict(reset_recovery_receipt=self.reset_recovery_receipt)
            def reset_development(self,**values):return ['CPU public task'],None
        def baseline(prepared,*args):
            outer.backend_calls.append(deepcopy(prepared['config']))
            return MockAdapter(),object()
        inner=SimpleNamespace(baseline=baseline,initial_conditions=lambda env,*args:dict(
            metric_only=True,policy_use_allowed=False,reset_receipt=deepcopy(env.reset_receipt)))
        backend=component.ProjectedBaselineBackend(**kwargs,inner=inner,simulation_class=Simulation)
        def observe_runtime():
            backend.witness['imported_dependency_files_sha256']={str(self.actual_wrapper):file_sha(self.actual_wrapper)}
            exe=str(self.root/'python-executable')
            backend.witness['loaded_native_files_sha256']={exe:file_sha(exe)}
            backend.witness['actual_python_executable']=exe
        backend._imports=observe_runtime
        return backend

    def execute_actual_wrapper(self, job):
        plan=self.f.early_plan_value;row=job['row']
        child=self.root/plan['output']/'cases'/job['job_id']/'attempt0000'
        command=wrapper.worker_command(plan,job,child)
        args=Namespace(split='validation',method='physmani',task=row['task'],raw_episode_id=row['raw_episode_id'],
            replica=0,exploration_seed=None,record_demo=False,demo_only=False)
        prepared=dict(args=args,config=deepcopy(self.f.early),config_path=self.root/self.f.early_config['path'],
            manifest_path=self.root/self.f.roster['path'],manifest_sha256=self.f.roster['sha256'],
            row=row,original_row=row,out=child,identity=job['identity'],files=self.early_files,
            lock_path=self.root/self.f.early_lock['path'],lock_sha256=self.f.early_lock['sha256'],
            r09_release={},r10_validation_release=None,record_demo=False)
        with patch.object(sys,'argv',[plan['child_source']['path'],*command[3:]]):
            record=wrapper.run_projected_case(prepared,self.f.contract,backend_factory=self.backend_factory)
        self.wrapper_calls+=1
        log=child.parent/'attempt0000.log';log.write_text('CPU wrapper/original evaluator only, no native process\n')
        record['parent_case_process_receipt']=dict(child_path=str((child/'case.json').relative_to(self.root)),
            child_sha256=file_sha(child/'case.json'),child_process_id=record['process_id'],
            launcher_process_id=os.getpid()+100000,command=command,log_path=str(log.relative_to(self.root)),
            log_sha256=file_sha(log),plan_sha256=plan['sha256'],parent_source_sha256=plan['parent_source']['sha256'])
        canonical=self.root/plan['output']/'results'/(job['job_id']+'.json')
        self.f.put(canonical,record);self.f.baseline_paths.append(canonical)

    def first(self):
        path=self.f.baseline_paths[0];record=json.loads(path.read_text())
        return path,record,self.root/record['parent_case_process_receipt']['child_path']

    def rewrite_wrapped_chain(self, path, record, child, *, base=None, inputs=None):
        proof=record['baseline_projected_wrapper_execution']
        if base is not None:
            target=self.root/proof['base_run_case_receipt']['path'];self.f.put(target,base)
            proof['base_run_case_receipt']=self.f.pointer(target)
        if inputs is not None:
            target=self.root/proof['base_run_case_inputs']['path'];self.f.put(target,inputs)
            proof['base_run_case_inputs']=self.f.pointer(target)
        self.f.put(child,{k:v for k,v in record.items() if k!='parent_case_process_receipt'})
        record['parent_case_process_receipt']['child_sha256']=file_sha(child)
        self.f.put(path,record)

    def test_real275_wrappers_and1100_ours_compose_without_identity_rewriting(self):
        result=self.f.run_release()
        self.assertTrue(result['complete'],result.get('errors'))
        self.assertEqual(self.wrapper_calls,275);self.assertEqual(len(self.backend_calls),550)
        self.assertEqual(result['actual_baseline_queue_plan'],self.f.early_plan)
        self.assertEqual(result['actual_baseline_config'],self.f.early_config)
        self.assertEqual(result['validated_config'],self.f.final_config)
        self.assertNotEqual(result['actual_baseline_lock'],result['validated_input_lock'])
        self.assertEqual(result['completed_method_cases'],1375)
        self.assertFalse(result['single_full_config_identity_claimed'])
        for supplied in self.backend_calls:
            self.assertNotIn('context_checkpoint',supplied)
            self.assertTrue(all(set(component.OURS_LEAVES).isdisjoint(row)
                for row in supplied['learned_artifacts'].values()))
        _,record,child=self.first()
        self.assertEqual(record['entrypoint_sha256'],file_sha(self.actual_evaluator))
        self.assertNotEqual(record['entrypoint_sha256'],record['baseline_projected_wrapper_execution']['actual_wrapper']['sha256'])
        self.assertIn('--collect-case',record['parent_case_process_receipt']['command'])
        self.assertEqual(record['process_id'],json.loads((child.parent/'inputs.json').read_text())['process_id'])

    def test_base_receipt_sha_tamper_is_rejected_by_real_component_validator(self):
        _,record,_=self.first();path=self.root/record['baseline_projected_wrapper_execution']['base_run_case_receipt']['path']
        base=json.loads(path.read_text());base['compute_seconds']=999.;self.f.put(path,base)
        self.f.reject('fresh_baseline_component_validation')

    def test_missing_wrapper_or_original_entrypoint_rewrite_cannot_use_old_branch(self):
        path,record,child=self.first();record.pop('baseline_projected_wrapper_execution')
        self.f.put(child,{k:v for k,v in record.items() if k!='parent_case_process_receipt'})
        record['parent_case_process_receipt']['child_sha256']=file_sha(child);self.f.put(path,record)
        self.f.reject('fresh_baseline_component_validation')

    def test_original_input_PID_manifest_row_files_and_postcleanup_exe_are_bound(self):
        path,original,child=self.first();raw_child=child.read_text();raw_canonical=path.read_text()
        proof=original['baseline_projected_wrapper_execution'];input_path=self.root/proof['base_run_case_inputs']['path']
        original_inputs=input_path.read_text()
        for mutation in ('PID','manifest','row','files','postcleanup_exe'):
            record=deepcopy(original);inputs=json.loads(original_inputs)
            if mutation=='PID':inputs['process_id']+=1
            elif mutation=='manifest':inputs['manifest_sha256']='e'*64
            elif mutation=='row':inputs['original_manifest_row_sha256']='f'*64
            elif mutation=='files':inputs['files']={str(self.actual_evaluator):file_sha(self.actual_evaluator)}
            else:record['baseline_projected_wrapper_execution']['dependency_witness_after_cleanup']['actual_python_executable']='unbound-executable'
            self.rewrite_wrapped_chain(path,record,child,inputs=inputs if mutation!='postcleanup_exe' else None)
            with self.subTest(mutation=mutation):self.f.reject('fresh_baseline_component_validation')
            path.write_text(raw_canonical);child.write_text(raw_child);input_path.write_text(original_inputs)

    def test_parent_command_must_keep_wrapper_action_and_exact_execution_contract(self):
        path,original,_=self.first();saved=path.read_text()
        for mutation in ('action','contract'):
            record=deepcopy(original);command=record['parent_case_process_receipt']['command']
            if mutation=='action':command.remove('--collect-case')
            else:command[command.index('--execution-contract')+1]='other_contract.json'
            self.f.put(path,record)
            with self.subTest(mutation=mutation):self.f.reject('fresh_baseline_component_validation')
            path.write_text(saved)


class ActualWrapperQueueTests(unittest.TestCase):
    """Real bounded scheduler; child process and native payload are CPU fakes."""
    def setUp(self):
        self.fixture=wrapper_fixtures.WrapperTests(methodName='runTest');self.fixture.setUp()
        self.root=self.fixture.root;self.plan_path=self.root/'queue_plan.json'
        self.output=self.root/'queue_output'
        self.plan=wrapper.prepare_plan(self.root/'config.json',self.root/'manifest.json',
            self.root/'lock.json',self.root/'contract.json',plan_path=self.plan_path,output=self.output)
        self.jobs=self.plan['jobs'];self.calls=[];self.snapshots=[]
        self.original_atomic=wrapper.atomic_json

    def tearDown(self):self.fixture.tearDown()

    def emit_native_record(self,command):
        raw=command[command.index('--raw-episode-id')+1];task=command[command.index('--task')+1]
        job=next(row for row in self.jobs if row['row']['task']==task and row['row']['raw_episode_id']==raw)
        record=native_completion_record(job['identity'],process_id=os.getpid(),success=False,fatal=False,ticks=0,
            initial_conditions=dict(metric_only=True,policy_use_allowed=False),
            alignment_receipt=dict(clock=dict(ticks=0,dt=.05,budget_seconds=10.,simulation_seconds=0.),
                actual_physics_seconds=0.,actual_task_seconds=0.,native_success_latched=False,
                native_terminal_failure_latched=False))
        record.update(cleanup_completed=True)
        self.original_atomic(wrapper.ROOT/command[command.index('--out')+1]/'case.json',record)

    def observe_json(self,path,value,**kwargs):
        result=self.original_atomic(path,value,**kwargs)
        if Path(path).name=='status.json':self.snapshots.append(deepcopy(value))
        return result

    def success_process(self,command,**kwargs):
        self.calls.append(list(command));outer=self
        class Process:
            pid=os.getpid()+100000+len(outer.calls)
            returncode=None
            def wait(self,timeout=None):
                if self.returncode is None:outer.emit_native_record(command);self.returncode=0
                return self.returncode
            def poll(self):return self.returncode
        return Process()

    def test_two_worker_first_infra_stops_pending_but_retains_running_valid_case(self):
        both_started=threading.Event();failed_receipt_written=threading.Event();outer=self
        def process(command,**kwargs):
            outer.calls.append(list(command));number=len(outer.calls)
            if number==2:both_started.set()
            class Process:
                pid=os.getpid()+100000+number
                returncode=None
                def wait(self,timeout=None):
                    if self.returncode is not None:return self.returncode
                    if number==1:
                        if not both_started.wait(3):raise AssertionError('second allowed worker never started')
                        self.returncode=1
                    else:
                        if not failed_receipt_written.wait(3):raise AssertionError('first infra was not receipted')
                        outer.emit_native_record(command);self.returncode=0
                    return self.returncode
                def poll(self):return self.returncode
            return Process()
        def observe(path,value,**kwargs):
            result=self.observe_json(path,value,**kwargs)
            if Path(path).name=='infrastructure_parent_receipt.json':failed_receipt_written.set()
            return result
        with patch.object(wrapper,'verify_plan',return_value=deepcopy(self.plan)), \
             patch.object(wrapper.subprocess,'Popen',side_effect=process), \
             patch.object(wrapper,'atomic_json',side_effect=observe):
            result=wrapper.run_queue(self.plan_path,workers=2,job_ids=[row['job_id'] for row in self.jobs[:5]])
        self.assertEqual(len(self.calls),2)
        self.assertEqual(result['completed'],1);self.assertEqual(len(result['errors']),1)
        self.assertTrue(result['infrastructure_cancellation_requested']);self.assertFalse(result['complete'])
        self.assertFalse(result['queue_running']);self.assertEqual(result['expected'],275)
        self.assertTrue(any(row['queue_running'] and row['completed']==1 for row in self.snapshots))
        self.assertFalse(wrapper.STOP.exists())
        error=json.loads((wrapper.ROOT/result['errors'][0]['receipt']).read_text())
        self.assertFalse(error['counted_as_policy_failure']);self.assertFalse(error['automatic_retries'])
        self.assertEqual(len(list((self.output/'results').glob('*.json'))),1)
        self.assertEqual(len(list((self.output/'cases').iterdir())),2)

    def test_prelaunch_source_error_is_receipted_once_and_blocks_automatic_retry(self):
        count=0
        def verify(*args,**kwargs):
            nonlocal count
            count+=1
            if count>1:raise ValueError('frozen source changed before worker launch')
            return deepcopy(self.plan)
        ids=[row['job_id'] for row in self.jobs[:5]]
        with patch.object(wrapper,'verify_plan',side_effect=verify), \
             patch.object(wrapper.subprocess,'Popen',side_effect=AssertionError('native forbidden')) as process:
            result=wrapper.run_queue(self.plan_path,workers=1,job_ids=ids)
        process.assert_not_called();self.assertEqual(result['completed'],0)
        self.assertEqual(len(result['errors']),1)
        self.assertEqual(len(list((self.output/'infrastructure_prelaunch').glob('*.json'))),1)
        with patch.object(wrapper,'verify_plan',return_value=deepcopy(self.plan)), \
             patch.object(wrapper.subprocess,'Popen',side_effect=AssertionError('native forbidden')) as process, \
             self.assertRaisesRegex(ValueError,'no automatic retry'):
            wrapper.run_queue(self.plan_path,workers=1,job_ids=ids)
        process.assert_not_called()

    def test_realtime_subset_and_readonly_resume_keep_full275_denominator_without_doublecount(self):
        ids=[row['job_id'] for row in self.jobs[:2]]
        with patch.object(wrapper,'verify_plan',return_value=deepcopy(self.plan)), \
             patch.object(wrapper.subprocess,'Popen',side_effect=self.success_process), \
             patch.object(wrapper,'atomic_json',side_effect=self.observe_json):
            first=wrapper.run_queue(self.plan_path,workers=2,job_ids=ids)
        self.assertEqual(first['completed'],2);self.assertEqual(first['expected'],275)
        self.assertEqual(first['selected_complete'],2);self.assertFalse(first['complete'])
        self.assertTrue(any(row['queue_running'] and row['completed'] in (1,2) for row in self.snapshots))
        with patch.object(wrapper,'verify_plan',return_value=deepcopy(self.plan)), \
             patch.object(wrapper.subprocess,'Popen',side_effect=AssertionError('resume may not relaunch')) as process:
            resumed=wrapper.run_queue(self.plan_path,workers=2,job_ids=ids)
        process.assert_not_called();self.assertEqual(resumed['completed'],2)
        self.assertEqual(resumed['selected_complete'],2)
        self.assertTrue(all(row['status']=='resumed_verified' for row in resumed['outcomes']))
        self.assertEqual(len(self.calls),2)


class ActualDependencyPreparationTests(unittest.TestCase):
    """Real static inventory/archive, draft validator and full275 plan producer."""
    def setUp(self):
        self.fixture=dependency_fixtures.BaselineDependencyInventoryTests(methodName='runTest')
        self.fixture.setUp();self.root=self.fixture.root;self.stack=ExitStack()
        project=Path(wrapper.__file__).resolve().parents[1]
        local=set(component.REVIEW_SOURCE_FILES)
        local.update(str(path.relative_to(project)) for path in wrapper.evaluator.required_local_sources('physmani'))
        local.update(('scripts/evaluate_physmani_projected_VAL_r10.py',
            'scripts/prepare_physmani_projected_VAL_r10.py','scripts/run_multitask_fresh_queue_r10.py',
            'our_mani/multitask/baseline_dependencies.py','our_mani/multitask/baseline_runtime_state.py',
            'our_mani/multitask/baseline_import_scope.py'))
        local.update(preparation.import_scope.SCOPES)
        for name in local:
            source=project/name;target=self.root/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(source.read_bytes())
        self.stack.enter_context(patch.object(wrapper,'ROOT',self.root))
        self.stack.enter_context(patch.object(wrapper,'BASE',self.root/'artifacts/framework/multitask_r10'))
        self.stack.enter_context(patch.object(wrapper,'__file__',str(self.root/'scripts/evaluate_physmani_projected_VAL_r10.py')))
        self.stack.enter_context(patch.object(wrapper.evaluator,'ROOT',self.root))
        self.stack.enter_context(patch.object(wrapper.evaluator,'__file__',str(self.root/'scripts/evaluate_multitask_case_r10.py')))
        self.stack.enter_context(patch.object(preparation,'ROOT',self.root))
        self.stack.enter_context(patch.object(preparation,'__file__',str(self.root/'scripts/prepare_physmani_projected_VAL_r10.py')))
        state=preparation.runtime_state
        state_file=self.fixture.runtime/'system/settings.dat'
        payload=struct.pack('<iB',1,1)+struct.pack('<i',len(state.MARKER))+state.MARKER+struct.pack('<i',2)+b'No\0'
        self.fixture.file(state_file,payload)
        self.stack.enter_context(patch.object(state,'STATE_FILE',state_file))
        self.stack.enter_context(patch.object(state,'__file__',str(self.root/'our_mani/multitask/baseline_runtime_state.py')))
        scope=preparation.import_scope
        self.stack.enter_context(patch.object(scope,'__file__',str(self.root/'our_mani/multitask/baseline_import_scope.py')))
        for name in ('pyrep.py','backend/sim.py','backend/simConst.py'):
            target=self.fixture.repo/'third_party/PyRep/pyrep'/name
            self.fixture.file(target,(project.parent/'phsimani/PhysMani/third_party/PyRep/pyrep'/name).read_bytes())
        self.bookkeeping_evidence=self.fixture.file('bookkeeping_source_review.json',b'{"CPU_fixture_only":true}')
        self.import_edge_evidence=self.fixture.file('import_edge_source_review.json',json.dumps(dict(
            schema='r10_ROOT_PhysMani_VAL_ours_only_import_edge_review_v2',
            reviewed_for=dict(split='validation',method='physmani',replica=0,record_demo=False,demo_only=False),
            required_bound_source_sha256={name:file_sha(self.root/name)
                for name in (*scope.SCOPES,'scripts/evaluate_multitask_case_r10.py')})).encode())
        # CPU schema transfer only: this single helper-source chain is not a
        # claim that real conda/Xvfb/runtime dependencies have been reviewed.
        observer=dict(path=str(Path(state.__file__).resolve()),sha256=file_sha(state.__file__))
        self.launcher_evidence=self.fixture.file('launcher_source_review.json',json.dumps(dict(
            schema='r10_PhysMani_actual_launcher_source_environment_review_v5',
            preparation_environment_not_substituted_for_worker_evidence=True,
            actual_case_observer=dict(source=observer),source_chain=[observer],CPU_fixture_only=True)).encode())
        self.stack.enter_context(patch.object(component,'calibration_dependency',side_effect=lambda row:{'fixture_row':row}))
        self.stack.enter_context(patch.object(baseline_dependencies.shutil,'which',
            side_effect=lambda name:self.fixture.commands.get(name)))
        self.stack.enter_context(patch.object(wrapper.subprocess,'Popen',side_effect=AssertionError('native forbidden')))
        rows=[]
        for task in component.TASK_SPECS:
            n=18 if task in ('insert_onto_rotating_peg','put_rubbish_in_moving_bin_high_speed') else 19 if task=='insert_onto_rotating_peg_high_speed' else 20
            rows += [dict(task=task,raw_episode_id='episode'+str(i),path='/camera/'+task+'/'+str(i)) for i in range(n)]
        self.manifest=self.fixture.file('validation_manifest.json',json.dumps(dict(split='validation',episodes=rows)).encode())
        self.fixture.file('artifacts/framework/aligned_r09/final_report.json',b'{"CPU_fixture":true}')

    def tearDown(self):self.stack.close();self.fixture.tearDown()

    def builder(self,*args,**kwargs):
        # Location metadata is an explicit CPU fixture. File inventory hashing,
        # archival, contract validation and plan construction are not mocked.
        return baseline_dependencies.inventory_baseline_dependencies(*args,**kwargs,
            python_metadata=self.fixture.metadata)

    def test_real_preparer_inventory_archive_and_validator_make_only_unapproved275_draft(self):
        result=preparation.prepare(deepcopy(self.fixture.config),output='artifacts/framework/multitask_r10/draft',
            manifest_path=self.manifest,python_executable=self.fixture.python,
            runtime_roots=[self.fixture.runtime],inventory_builder=self.builder,
            bookkeeping_evidence=self.bookkeeping_evidence,import_edge_evidence=self.import_edge_evidence,
            launcher_evidence=self.launcher_evidence)
        self.assertFalse(result['approved']);self.assertFalse(result['native_started']);self.assertFalse(result['GPU_started'])
        self.assertEqual(result['prepared_VAL_cases'],275)
        plan=wrapper.verify_plan(self.root/result['plan']['path'],approved=False)
        self.assertEqual(len(plan['jobs']),275)
        self.assertEqual(plan['parent_source'],plan['child_source'])
        self.assertEqual(plan['baseline_execution_contract'],result['contract'])
        contract=json.loads((self.root/result['contract']['path']).read_text())
        self.assertFalse(contract['approved_before_VAL_execution'])
        self.assertFalse(contract['actual_dependency_closure_reviewed'])
        with self.assertRaisesRegex(ValueError,'approved'):
            wrapper.verify_plan(self.root/result['plan']['path'],approved=True)

    def test_derive_config_retains_fixed14_geometry_and_does_not_require_unfitted_ours_artifacts(self):
        asset=self.fixture.file('shared_CAD.json',b'{"CPU_fixture":true}')
        pointer=self.fixture.pointer(asset)
        shared=deepcopy(self.fixture.config)
        shared['learned_artifacts']={task:dict(geometry=pointer,
            interaction={'path':'not_fitted_yet.json','sha256':'f'*64}) for task in component.TASK_SPECS}
        manifest=dict(assets={task:pointer for task in component.TASK_SPECS})
        derived=preparation.derive_config(shared,{'baseline_checkpoint':shared['baseline_checkpoint']},manifest,root=self.root)
        for task in component.TASK_SPECS:self.assertEqual(derived['learned_artifacts'][task]['geometry'],pointer)
        retained=preparation.retained_pointers(derived,self.root)
        self.assertTrue(any(item['path']==str(asset) for item in retained))
        self.assertFalse(any('not_fitted' in item['path'] for item in retained))
        bad=deepcopy(shared);bad['learned_artifacts'][next(iter(component.TASK_SPECS))]['geometry']={'path':'changed','sha256':'e'*64}
        with self.assertRaisesRegex(ValueError,'geometry pointer changed'):
            preparation.derive_config(bad,{'baseline_checkpoint':shared['baseline_checkpoint']},manifest,root=self.root)


if __name__=='__main__':unittest.main()
