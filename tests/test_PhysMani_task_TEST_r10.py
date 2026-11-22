"""CPU authorization/source-boundary tests, never native TEST evidence."""
import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from our_mani.multitask import baseline_task_test as task
from our_mani.multitask import baseline_component as component
from our_mani.multitask.evaluation import digest,file_sha
from scripts import evaluate_PhysMani_task_TEST_r10 as wrapper

ROOT=Path(__file__).resolve().parents[1]


class TaskTESTBoundaries(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts/framework/multitask_r10',prefix='CPU_test_fixture_')
        self.work=Path(self.temp.name).resolve()
        self.manifest=ROOT/task.OFFICIAL_MANIFEST
        self.rows=task.official_task_rows(json.loads(self.manifest.read_text()),task.NORMAL_BASKETBALL)
        self.config=json.loads((ROOT/'artifacts/framework/multitask_r10/basketball_Context_geometry_TRAIN_development_demo_v1_setup/configuration.json').read_text())
        self.release=self.work/'release.json'
        self.release.write_text(json.dumps(dict(schema='r10_task_complete_three_method_geometry_validation_release_v1',
            task=task.NORMAL_BASKETBALL,complete=True,formal_test_permitted=True,protocol_acceptance_passed=True,
            split='validation',TEST_used=False,methods=list(task.METHODS),Value_required=False,Value_artifacts_loaded=False,expected_source_cases=20,expected_method_cases=60,completed_method_cases=60)))
        self.config['formal_comparison_ready']=True
        self.config['task_validation_release']=self.ptr(self.release)
        self.permission=self.work/'permission.json';self.permission.write_text('{}')
        self.config[task.RESOURCE_KEY]=dict(profile=self.ptr(self.permission),environment_contract=self.ptr(self.permission))
        self.configpath=self.work/'config.json';self.configpath.write_text(json.dumps(self.config))
        files={str(path.resolve()):file_sha(path) for path in (self.configpath,self.manifest,self.release,
            Path(task.__file__),Path(wrapper.__file__),ROOT/'scripts/evaluate_multitask_case_r10.py',
            ROOT/'our_mani/multitask/geometry_task_release.py',ROOT/'our_mani/multitask/gpu_budget.py')}
        self.lockpath=self.work/'lock.json';self.lockpath.write_text(json.dumps(dict(files=files)))
        self.args=SimpleNamespace(split='test',method='physmani',task=task.NORMAL_BASKETBALL,replica=0,
            exploration_seed=None,record_demo=False,demo_only=False,config=str(self.configpath),
            manifest=str(self.manifest),frozen_inputs=str(self.lockpath),raw_episode_id=self.rows[0]['raw_episode_id'],
            out=str(self.work/'output'))

    def tearDown(self):self.temp.cleanup()
    def ptr(self,path):return dict(path=str(Path(path).resolve().relative_to(ROOT)),sha256=file_sha(path))

    def metadata(self,args=None):
        # The separate oldR09 proof is not under test; no native backend call
        # is mocked as a completed source or validation score.
        with patch('scripts.evaluate_multitask_case_r10.r09_release',return_value=dict(CPU_fixture=True)):
            return task.metadata_preflight(args or self.args)

    def fixture_contract(self,prepared):
        oldconfig=self.work/'oldconfig.json';oldconfig.write_text(json.dumps(self.config))
        fake_kernel=self.work/'stock_descriptor.json';fake_kernel.write_text('{}')
        oldcontract=self.work/'original_contract.json'
        dummy=self.work/'dependency.txt';dummy.write_text('CPU fixture\n')
        groups={name:{str(dummy):file_sha(dummy)} for name in component.GROUPS}
        oldcontract.write_text(json.dumps(dict(jit_artifact_profile=self.ptr(fake_kernel),dependency_groups=groups,
            mutable_state_permission=self.ptr(self.permission),runtime_inventory=self.ptr(self.permission))))
        oldplan=self.work/'original_plan.json'
        plan=dict(config=self.ptr(oldconfig),baseline_execution_contract=self.ptr(oldcontract));oldplan.write_text(json.dumps(plan))
        review=self.work/'ROOT_review.json'
        review.write_text(json.dumps(dict(schema='r10_ROOT_PhysMani_normal_official_TEST_runtime_approval_v1',
            approved=True,config=self.ptr(self.configpath),manifest=self.ptr(self.manifest),
            task_validation_release=self.ptr(self.release),original_VAL_runtime_plan=self.ptr(oldplan),
            full560_task40_original_sources=True,VAL_import_edge_permission_used_for_TEST=False)))
        groups=copy.deepcopy(groups)
        sources=[Path(task.__file__),Path(wrapper.__file__),ROOT/'our_mani/multitask/geometry_task_release.py',ROOT/'our_mani/multitask/gpu_budget.py']
        for path in sources:groups['calibration_inputs'][str(path.resolve())]=file_sha(path)
        archive=self.work/'source_archive.json';entries=[]
        for i,path in enumerate(sources):
            stored=self.work/('source'+str(i)+'.py');stored.write_bytes(path.read_bytes())
            entries.append(dict(original_path=str(path.resolve()),path=str(stored.relative_to(ROOT)),sha256=file_sha(path)))
        archive.write_text(json.dumps(dict(schema=component.SOURCE_ARCHIVE,all_inputs_verified_before_and_after=True,entries=entries)))
        contract=dict(schema=task.CONTRACT,approved_before_TEST_execution=True,split='test',task=task.NORMAL_BASKETBALL,
            full_official_TEST_sources=560,selected_original_task_sources=40,official_TEST_manifest=self.ptr(self.manifest),
            original_task_rows_sha256=digest(self.rows),projected_config_sha256=digest(component.project_baseline_config(self.config)),
            task_validation_release=self.ptr(self.release),import_edge_review=None,VAL_source_edge_permission_used_for_TEST=False,
            development_reset_hook_used=False,original_verified_VAL_runtime_plan=self.ptr(oldplan),
            observer_stock_cache_descriptor=self.ptr(fake_kernel),old_VAL_scope_is_not_TEST_authorization=True,
            ROOT_review=self.ptr(review),TEST_GPU_execution_resource_budget=copy.deepcopy(self.config[task.RESOURCE_KEY]),
            dependency_groups=groups,source_archive=self.ptr(archive),mutable_state_permission=self.ptr(self.permission),
            runtime_inventory=self.ptr(self.permission))
        path=self.work/'TEST_contract.json';path.write_text(json.dumps(contract))
        files=prepared['files']
        for source in (review,self.permission):files[str(source)]=file_sha(source)
        for group in groups.values():files.update(group)
        files[str(path)]=file_sha(path)
        return path,contract,lambda ignored:plan

    def test_entire560_and_original40_are_preserved(self):
        self.assertEqual(len(self.rows),40)
        manifest=json.loads(self.manifest.read_text());manifest['episodes']=manifest['episodes'][:-1]
        with self.assertRaisesRegex(ValueError,'all560'):task.official_task_rows(manifest,task.NORMAL_BASKETBALL)

    def test_metadata_positive_has_no_native_or_complete_release_grant(self):
        prepared=self.metadata()
        self.assertEqual(prepared['identity']['split'],'test')
        self.assertIsNone(prepared['r10_validation_release'])
        self.assertEqual(prepared['row'],prepared['original_row'])

    def test_wrong_split_demo_and_source_cannot_enter_TEST(self):
        for field,value in [('split','validation'),('record_demo',True),('replica',1),('raw_episode_id','foreign')]:
            args=copy.deepcopy(self.args);setattr(args,field,value)
            with self.assertRaises(ValueError):self.metadata(args)

    def test_all_three_VAL_required_before_any_native(self):
        self.config['formal_comparison_ready']=False
        with self.assertRaisesRegex(ValueError,'three-method'):
            task.structural_release(self.config,task.NORMAL_BASKETBALL,json.loads(self.lockpath.read_text())['files'])
        prepared=self.metadata()
        with self.assertRaisesRegex(ValueError,'three-method'):
            task.authorize_task_TEST(prepared,checker=lambda *args,**kwargs:dict(passed=False))
        self.assertIsNone(prepared['r10_validation_release'])

    def test_new_contract_fixture_then_wrong_checkpoint_or_VAL_edge_rejected(self):
        prepared=self.metadata();path,contract,verify=self.fixture_contract(prepared)
        verified=task.load_TEST_contract(ROOT,self.ptr(path),prepared,verify_original=verify)
        self.assertTrue(verified['original_runtime_scope_verified'])
        self.assertTrue(verified['TEST_permission_still_requires_full_task_release'])
        bad=copy.deepcopy(prepared);bad['config']['baseline_checkpoint']='wrong_checkpoint.pth'
        changed=copy.deepcopy(contract);changed['projected_config_sha256']=digest(component.project_baseline_config(bad['config']))
        path.write_text(json.dumps(changed));bad['files'][str(path)]=file_sha(path)
        with self.assertRaisesRegex(ValueError,'checkpoint'):
            task.load_TEST_contract(ROOT,self.ptr(path),bad,verify_original=verify)
        changed=copy.deepcopy(contract);changed['import_edge_review']={'VAL-only':'not TEST'}
        path.write_text(json.dumps(changed));prepared['files'][str(path)]=file_sha(path)
        with self.assertRaisesRegex(ValueError,'TEST contract'):
            task.load_TEST_contract(ROOT,self.ptr(path),prepared,verify_original=verify)

    def test_authorization_calls_new_task_composer_full40_interface(self):
        prepared=self.metadata();seen=[]
        def checker(root,name,release,config,files,**kwargs):
            seen.append((name,release,kwargs))
            return dict(passed=True,VAL_source_cases=20,VAL_method_cases=60,TEST_source_cases=40)
        result=task.authorize_task_TEST(prepared,checker=checker)
        self.assertTrue(result['passed'])
        self.assertEqual(seen[0][2]['official_test_manifest'],self.manifest)

    def test_real_three_group_checker_rejects_header_only_fixture(self):
        prepared=self.metadata()
        # Real current consumer, not an injected mock: a plausible header
        # cannot substitute for actual three-way source/model/raw evidence.
        with self.assertRaises(ValueError):task.authorize_task_TEST(prepared)
        self.assertIsNone(prepared['r10_validation_release'])

    def test_legacy_five_group_release_or_changed_ROOT_review_is_rejected(self):
        prepared=self.metadata();path,contract,verify=self.fixture_contract(prepared)
        report=json.loads(self.release.read_text());report['schema']='r10_task_complete_five_method_validation_release_v2_subset_baseline'
        self.release.write_text(json.dumps(report));config=copy.deepcopy(self.config)
        config['task_validation_release']=self.ptr(self.release);files=copy.deepcopy(prepared['files'])
        files[str(self.release)]=file_sha(self.release)
        with self.assertRaisesRegex(ValueError,'three-method'):
            task.structural_release(config,task.NORMAL_BASKETBALL,files)
        bad=copy.deepcopy(contract);bad['ROOT_review']=None;path.write_text(json.dumps(bad))
        prepared['files'][str(path)]=file_sha(path)
        with self.assertRaisesRegex(ValueError,'pointer'):
            task.load_TEST_contract(ROOT,self.ptr(path),prepared,verify_original=verify)

    def test_runtime_metadata_preparer_keeps_full_closure_without_VAL_edge_grant(self):
        prepared=self.metadata();path,contract,verify=self.fixture_contract(prepared)
        planpath=ROOT/contract['original_verified_VAL_runtime_plan']['path'];oldplan=verify(None)
        original_path=ROOT/oldplan['baseline_execution_contract']['path'];original=json.loads(original_path.read_text())
        original['source_archive']=contract['source_archive'];original_path.write_text(json.dumps(original))
        oldplan['baseline_execution_contract']=self.ptr(original_path);planpath.write_text(json.dumps(oldplan))
        # Actual source files are copied/hashed by the preparer; only the old
        # standalone VAL verifier is injected in this small CPU fixture.
        lock=self.work/'prepare_union.json';lock.write_text(json.dumps(dict(files=prepared['files'])))
        with patch('scripts.evaluate_physmani_projected_VAL_r10.verify_plan',return_value=oldplan):
            result=task.prepare_TEST_runtime_contract(ROOT,self.configpath,self.manifest,lock,planpath,
                output=self.work/'prepared_TEST')
        self.assertFalse(result['approved']);self.assertFalse(result['native_GPU_started'])
        actual=json.loads((ROOT/result['contract']['path']).read_text())
        self.assertIsNone(actual['import_edge_review']);self.assertFalse(actual['VAL_source_edge_permission_used_for_TEST'])
        self.assertEqual(actual['selected_original_task_sources'],40)
        self.assertEqual(actual[task.RESOURCE_KEY],self.config[task.RESOURCE_KEY])
        frozen=json.loads((ROOT/result['lock']['path']).read_text())['files']
        self.assertIn(str(Path(task.__file__).resolve()),frozen)
        for name,checksum in frozen.items():self.assertEqual(file_sha(name),checksum)

    def test_official_backend_refuses_before_VAL_then_uses_original_TEST_dispatch(self):
        prepared=self.metadata();called=[]
        class Env:
            reset_receipt=dict(reset_kind='released_official_TEST_family_dispatch',original_source_replayed=True)
            def close(inner):called.append('close')
        class Inner:
            def environment(inner,value):
                called.append(value)
                return Env(),['official description']
        runtime=self.work/'runtime.json';runtime.write_text(json.dumps(dict(interpreter='/CPU/fixture/python')))
        verified=dict(contract=dict(mutable_state_permission=self.ptr(self.permission),runtime_inventory=self.ptr(runtime),
            observer_stock_cache_descriptor=self.ptr(self.permission)),dependencies={})
        backend=task.TestProjectedBaselineBackend(verified=verified,execution_contract=self.ptr(self.permission),inner=Inner())
        with self.assertRaisesRegex(ValueError,'three-method'):backend.environment(prepared)
        self.assertEqual(called,[])
        prepared['r10_validation_release']=dict(passed=True,TEST_source_cases=40)
        with (patch.object(task,'observe_TEST_resource_environment',return_value=dict(CPU_fixture=True)),
                patch.object(task.state,'load_permission',return_value=dict(resolved_path='CPU fixture')),
                patch.object(task.state,'snapshot',return_value=dict(CPU_fixture=True)),
                patch.object(task.state,'observe_headless',return_value=dict(value=True)),
                patch.object(task.state,'observe_launcher_environment',return_value=dict(CPU_fixture=True))):
            env,descriptions=backend.environment(prepared)
        self.assertEqual(called[0]['args'].split,'test')
        self.assertNotIn('context_checkpoint',called[0]['config'])
        self.assertFalse(backend.witness['development_reset_hook_used'])
        self.assertFalse(backend.witness['old_VAL_profile_used_as_TEST_permission'])

    def test_exact_TEST_resource_export_binding_without_MPS_or_GPU(self):
        folder=ROOT/'artifacts/framework/multitask_r10/PhysMani_normal_geometry_TEST_MPS_resource_v1'
        config=copy.deepcopy(self.config)
        config[task.RESOURCE_KEY]=dict(profile=self.ptr(folder/'profile.json'),
            environment_contract=self.ptr(folder/'execution_environment_contract.json'))
        profile=json.loads((folder/'profile.json').read_text())
        env=dict(CUDA_VISIBLE_DEVICES='0',CUDA_MPS_PIPE_DIRECTORY=profile['namespace']+'/p',
            CUDA_MPS_LOG_DIRECTORY=profile['namespace']+'/l',CUDA_MPS_SM_PARTITION=profile['GPU']['uuid']+'/Q1BV')
        with patch.dict(os.environ,env):
            result=task.observe_TEST_resource_environment(ROOT,config)
        self.assertEqual(result['actual_exports'],env)
        self.assertFalse(result['actual_SM_partition_qualification_claimed'])
        # Matching-looking exports alone never establish an actual SM budget.
        for key in ('CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY','CUDA_MPS_SM_PARTITION'):
            bad=dict(env);bad[key]='foreign'
            with patch.dict(os.environ,bad),self.assertRaises(ValueError):task.observe_TEST_resource_environment(ROOT,config)
        wrong=copy.deepcopy(config)
        wrong[task.RESOURCE_KEY]['environment_contract']=self.ptr(
            ROOT/'artifacts/framework/multitask_r10/PhysMani_normal_geometry_VAL_MPS_resource_v1/execution_environment_contract.json')
        with patch.dict(os.environ,env),self.assertRaisesRegex(ValueError,'contract/profile'):
            task.observe_TEST_resource_environment(ROOT,wrong)


class TaskTESTQueueBoundaries(unittest.TestCase):
    """Mock processes test dispatch; these are never native TEST cases."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts/framework/multitask_r10',prefix='CPU_TEST_queue_')
        self.work=Path(self.temp.name);self.output=self.work/'queue'
        self.jobs=[dict(job_id='CPU'+str(i),identity=dict(CPU_fixture=True,index=i),
            row=dict(task=task.NORMAL_BASKETBALL,raw_episode_id='CPU'+str(i))) for i in range(40)]
        self.plan=dict(sha256='CPU_plan',jobs=self.jobs,output=str(self.output.relative_to(ROOT)),
            config=dict(path='CPU/config'),manifest=dict(path='CPU/manifest'),
            frozen_inputs=dict(path='CPU/lock'),execution_contract=dict(path='CPU/contract'),
            child_source=dict(path=str(Path(wrapper.__file__).relative_to(ROOT))))

    def tearDown(self):self.temp.cleanup()

    def test_first_infrastructure_stops_pending_and_keeps_inflight_completion(self):
        started=[];guard=threading.Lock();both=threading.Event();outer=self
        class Process:
            def __init__(self,command,**kwargs):
                self.command=command;self.returncode=None
                with guard:
                    self.index=int(command[command.index('--raw-episode-id')+1][3:])
                    started.append(command);self.pid=1000+self.index
                    if len(started)==2:both.set()
            def wait(self,timeout=None):
                if self.index==0:
                    if not both.wait(5):raise RuntimeError('CPU fixture never launched second worker')
                    self.returncode=1;return 1
                deadline=time.monotonic()+5
                while not (outer.output/'queue_errors'/('CPU0.json')).exists():
                    if time.monotonic()>deadline:raise RuntimeError('CPU fixture no first infra receipt')
                    time.sleep(.005)
                out=ROOT/self.command[self.command.index('--out')+1];out.mkdir(parents=True)
                (out/'case.json').write_text(json.dumps(dict(process_id=9000,CPU_fixture=True)))
                self.returncode=0;return 0
        with (patch.object(wrapper,'verify_plan',return_value=self.plan),
                patch.object(wrapper,'validate_completed_case'),patch.object(wrapper.subprocess,'Popen',Process)):
            result=wrapper.run_queue(self.work/'unused',2)
        self.assertEqual(len(started),2);self.assertEqual(result['completed'],1)
        self.assertEqual(len(result['errors']),1);self.assertEqual(result['remaining_unattempted'],38)
        self.assertFalse(result['complete']);self.assertTrue(result['dispatch_stopped_on_infrastructure'])
        record=json.loads((self.output/'results'/'CPU1.json').read_text())
        self.assertEqual(record['process_id'],9000)
        self.assertEqual(record['parent_case_process_receipt']['launcher_process_id'],1001)
        self.assertFalse((self.output/'cases'/'CPU2').exists())

    def test_complete_readonly_resume_does_not_launch_or_recount(self):
        outer=self;launched=[]
        class Process:
            def __init__(self,command,**kwargs):
                self.command=command;self.pid=2000+len(launched);self.returncode=None;launched.append(command)
            def wait(self,timeout=None):
                out=ROOT/self.command[self.command.index('--out')+1];out.mkdir(parents=True)
                (out/'case.json').write_text(json.dumps(dict(process_id=9000+self.pid,CPU_fixture=True)))
                self.returncode=0;return 0
        with (patch.object(wrapper,'verify_plan',return_value=self.plan),patch.object(wrapper,'validate_completed_case'),
                patch.object(wrapper.subprocess,'Popen',Process)):
            first=wrapper.run_queue(self.work/'unused',2)
        self.assertTrue(first['complete']);self.assertEqual(first['completed'],40);self.assertEqual(len(launched),40)
        original={p.name:file_sha(p) for p in (self.output/'results').glob('*.json')}
        with (patch.object(wrapper,'verify_plan',return_value=self.plan),patch.object(wrapper,'validate_completed_case'),
                patch.object(wrapper.subprocess,'Popen',side_effect=AssertionError('resume launched a new child'))):
            second=wrapper.run_queue(self.work/'unused',2)
        self.assertTrue(second['complete']);self.assertEqual(second['completed'],40)
        self.assertEqual(original,{p.name:file_sha(p) for p in (self.output/'results').glob('*.json')})

    def test_existing_STOP_is_honored_without_writing_global_state(self):
        exists=Path.exists;stop=ROOT/'runtime/framework/multitask_r10_STOP'
        with (patch.object(wrapper,'verify_plan',return_value=self.plan),
                patch.object(Path,'exists',lambda path: True if path==stop else exists(path)),
                patch.object(wrapper.subprocess,'Popen',side_effect=AssertionError('STOP launched a child'))):
            result=wrapper.run_queue(self.work/'unused',2)
        self.assertFalse(result['complete']);self.assertTrue(result['stop_requested'])
        self.assertEqual(result['remaining_unattempted'],40);self.assertEqual(result['completed'],0)

    def test_wall_timeout_cleans_only_owned_process_group_and_stops_dispatch(self):
        signals=[];launched=[]
        class Process:
            def __init__(self,command,**kwargs):
                self.pid=2468;self.returncode=None;launched.append(kwargs)
            def wait(self,timeout=None):
                if timeout==3600:raise wrapper.subprocess.TimeoutExpired('CPU process',timeout)
                self.returncode=-15;return -15
        with (patch.object(wrapper,'verify_plan',return_value=self.plan),
                patch.object(wrapper.subprocess,'Popen',Process),
                patch.object(wrapper.os,'killpg',side_effect=lambda pid,sig:signals.append((pid,sig)))):
            result=wrapper.run_queue(self.work/'unused',1)
        self.assertEqual(len(launched),1);self.assertTrue(launched[0]['start_new_session'])
        self.assertEqual(signals[0][0],2468);self.assertEqual(len(signals),1)
        self.assertFalse(result['complete']);self.assertEqual(len(result['errors']),1)
        self.assertEqual(result['remaining_unattempted'],39)


if __name__=='__main__':unittest.main()
