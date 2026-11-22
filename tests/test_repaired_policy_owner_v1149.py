"""Synthetic CPU owner tests. No real graph preparation, VAL or simulation."""
from contextlib import contextmanager
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from scripts import run_repaired_policy_owner_v1149_r10 as owner
from tests.test_repaired_policy_graph_v1143 import graph_spec,put
from tests.test_repaired_readiness_v1144 import FreshFixture,write

PROJECT=Path(__file__).resolve().parents[1]


class Fixture:
    def __init__(self,root):
        self.root=Path(root);spec=graph_spec(root)
        def source(name):return put(root,name,(PROJECT/name).read_bytes())
        mine=source(owner.SOURCE)
        for key in ('train_owner','evaluation_owner'):
            spec['control']['components'][key]=mine;spec['control']['sources_map'][key]=mine
        auditor=source(owner.readiness.SOURCE)
        spec['control']['components']['train_auditor']=auditor;spec['control']['sources_map']['train_auditor']=auditor
        for key,name in (('environment_wrapper',owner.WRAPPER),('resource_guard',owner.GUARD),
                ('admission_guard',owner.GATE),('formal_source_reader',owner.GATE),('split_source_lock_writer',owner.WRITER)):
            value=source(name);spec['control']['components'][key]=value;spec['control']['sources_map'][key]=value
        for name in ('scripts/run_aligned_gpu.sh','scripts/run_local_sim.sh',owner.WORKER,owner.PREPARER):
            spec['control']['sources_map'][name]=source(name)
        consumer=source(owner.CONSUMER);spec['physical']['sources_map']['consumer']=consumer
        worker=spec['control']['sources_map'][owner.WORKER];preparer=spec['control']['sources_map'][owner.PREPARER]
        for row in spec['routes']:
            for split,key in (('train','train'),('validation','evaluation')):
                row[key].update(worker=worker,preparer=preparer,consumer=consumer,case_schema=owner.CASE_SCHEMAS[split])
        self.rows=[dict(task=t,raw_episode_id='episode'+str(i),path='never-opened/'+t+'/'+str(i)+'.h5')
            for t in owner.graphs.TASKS for i in range(100)]
        spec['train_source_manifest']=write(root,owner.TRAIN_MANIFEST,dict(split='train',episodes=self.rows))
        self.graph_ref=owner.graphs.write_policy_graph(root,spec,namespace='artifacts/framework/multitask_r10/owner_graph')
        self.graph=owner.graphs.read_policy_graph(root,self.graph_ref);self.stamp=owner.stamp(self.graph_ref,self.graph)
        self.environment_sha=self.graph['physical']['environment_manifest']['sha256']
        self.predecessor=dict(exec_completion=write(root,'metadata/prior_actual_exit.json',{'CPU_fixture':'exit only'}),
            process=dict(pid=2147483000,start_ticks=1))
        self.validation_rows=[dict(task=owner.graphs.TASKS[i%16],raw_episode_id='valepisode'+str(i),path='never-read-VAL/data.h5') for i in range(315)]
        bundle_dir=owner.BUNDLE_PREFIX+'synthetic_run/'
        self.admission=write(root,'artifacts/framework/multitask_r10/owner_admission/admission.json',{'CPU_fixture':'never an actual admission'})
        self.inputs=write(root,bundle_dir+'source_inputs.json',dict(split='validation',source_count=315,**self.stamp))
        self.suite=write(root,bundle_dir+'suite.json',dict(source_inputs=self.inputs,admission=self.admission,**self.stamp))
        self.lock=write(root,bundle_dir+'suite_lock.json',dict(admission=self.admission,graph=self.graph_ref,
            policy_sha256=self.graph['policy_sha256'],files={str(self.root/p['path']):p['sha256']
                for p in (self.graph_ref,self.admission,self.inputs,self.suite)}))
        self.bundle=dict(schema=owner.BUNDLE_SCHEMA,source_count=315,Native_started=False,TEST_read=False,
            admission=self.admission,source_inputs=self.inputs,suite=self.suite,suite_lock=self.lock,**self.stamp)
        self.bundle_ref=write(root,owner.BUNDLE_PREFIX+'synthetic_run/bundle.json',self.bundle)

    def review(self,split='train',namespace='fixture'):
        value=dict(schema=owner.REVIEW_SCHEMA,reviewer='root',review_complete=True,release_fixed_cohort=True,
            split=split,namespace=namespace,expected_cases=owner.split_count(split),automatic_retry=False,source_outcome_filtering=False,**self.stamp)
        if split=='train':value['predecessor_exec_completion']=self.predecessor['exec_completion']
        else:value['validation_bundle']=self.bundle_ref
        return write(self.root,'artifacts/framework/multitask_r10/root_review_'+split+'_'+namespace+'.json',value)

    @contextmanager
    def ready(self):
        with patch.object(owner,'ENV_SHA',self.environment_sha),patch.object(owner,'smoke_complete',return_value=self.predecessor):yield

    def prepare(self,split='train',namespace='fixture'):
        kwargs=dict(split=split,namespace=namespace,review_ref=self.review(split,namespace))
        if split=='validation':kwargs['bundle_ref']=self.bundle_ref
        return owner.prepare_owner(self.root,self.graph_ref,**kwargs)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temporary=TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.f=Fixture(self.root)

    def test_default_inert_and_TEST_rejected_before_any_graph_or_data_read(self):
        with patch.object(owner.subprocess,'Popen',side_effect=AssertionError('no launch')),patch('sys.stdout',new_callable=io.StringIO):
            result=owner.main([])
        self.assertEqual(result,dict(enabled=False,Native_started=False,GPU_started=False,VAL_read=False,TEST_read=False))
        with patch.object(owner,'graph_for_owner',side_effect=AssertionError('must refuse first')):
            with self.assertRaisesRegex(ValueError,'TEST refused'):
                owner.prepare_owner(self.root,self.f.graph_ref,split='test',namespace='never',review_ref={})

    def test_prepare_TRAIN_exact_original_first4_192_and_three_methods_fresh_only(self):
        with self.f.ready(),patch.object(owner.subprocess,'Popen',side_effect=AssertionError('no launch')):
            reference=self.f.prepare();plan,_=owner.read(self.root,reference)
            self.assertEqual(len(plan['sources']),64);self.assertEqual(len(plan['jobs']),192)
            self.assertEqual({j['raw_episode_id'] for j in plan['jobs']},{'episode0','episode1','episode2','episode3'})
            self.assertEqual({j['method'] for j in plan['jobs']},set(owner.graphs.METHODS))
            self.assertEqual(len({j['out'] for j in plan['jobs']}),192)
            self.assertTrue(all('/train/development_demo_only/' in j['out'] for j in plan['jobs']))
            self.assertTrue(all(not (self.root/j['out']).exists() and not (self.root/j['runtime']).exists() for j in plan['jobs']))
            with self.assertRaises(ValueError):self.f.prepare()

    def test_missing_smoke_or_Root_review_blocks_before_plan_write(self):
        with patch.object(owner,'ENV_SHA',self.f.environment_sha),patch.object(owner,'smoke_complete',side_effect=ValueError('smoke live')):
            with self.assertRaises(ValueError):self.f.prepare()
        self.assertFalse((self.root/owner.BASE/'train/fixture/plan.json').exists())
        review=self.f.review();value,_=owner.read(self.root,review);value['review_complete']=False
        review=write(self.root,review['path'],value)
        with self.f.ready(),self.assertRaises(ValueError):
            owner.prepare_owner(self.root,self.f.graph_ref,split='train',namespace='fixture',review_ref=review)

    def test_real_shape_smoke_requires_eight_fixed_sources_actual_tool_exit_and_dead_PID(self):
        put(self.root,owner.SMOKE_SOURCE,(PROJECT/owner.SMOKE_SOURCE).read_bytes())
        plan=dict(schema='r10_bounded_eight_TRAIN_author_release_regression_plan_v1145',automatic_retry=False,
            source_outcome_filtering=False,inputs_sha256={owner.SMOKE_SOURCE:owner.SMOKE_SOURCE_SHA},
            jobs=[dict(task=t,raw=r,method=m) for t,r,m in owner.SMOKE_CASES])
        planned=write(self.root,owner.SMOKE_PLAN,plan);process=dict(pid=2147483000,start_ticks=1)
        write(self.root,owner.SMOKE+'/launch.json',dict(pid=process['pid'],start_ticks=process['start_ticks'],plan=planned))
        status=dict(stage='complete',active=None,failed=[],completed=[])
        for i,(task,raw,method) in enumerate(owner.SMOKE_CASES):
            base=owner.SMOKE+'/dummy/'+str(i);out='artifacts/framework/multitask_r10/synthetic_smoke/'+str(i)
            case=write(self.root,out+'/case.json',dict(complete=True,cleanup_completed=True,infrastructure_error=None,
                official_task_success=False,native_terminal_failure=False))
            row=dict(task=task,raw=raw,method=method,case=case,owned_hooks=write(self.root,out+'/hooks.json',{}))
            for phase in ('CPU_check','native'):
                row[phase]={key:write(self.root,base+'/'+phase+'_'+key+'.json',{'CPU_fixture':True}) for key in ('launch','terminal','sidecar')}
            status['completed'].append(row)
        write(self.root,owner.SMOKE+'/status.json',status)
        terminal=write(self.root,owner.SMOKE+'/terminal.json',dict(complete=True,completed=8,infrastructure_failed=0,automatic_retry=False))
        actual=dict(schema='synthetic_root_observed_exit_v1145',actual_tool_exit_observed=True,natural_completion=True,
            returncode=0,owner_pid=process['pid'],owner_start_ticks=process['start_ticks'],owner_alive=False,
            native_cases_complete=8,completed_case_infrastructure_failures=0,plan=planned,terminal=terminal)
        reference=write(self.root,owner.SMOKE+'/exec_completion.json',actual)
        self.assertTrue(owner.smoke_complete(self.root)['natural_completion'])
        for field,value in (('actual_tool_exit_observed',False),('returncode',False),('returncode',1),
                ('owner_pid',42),('native_cases_complete',7),('natural_completion',False)):
            changed=dict(actual);changed[field]=value;write(self.root,reference['path'],changed)
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):owner.smoke_complete(self.root)
        write(self.root,reference['path'],actual)
        with patch.object(owner.readiness,'_not_live',return_value=False),self.assertRaises(ValueError):owner.smoke_complete(self.root)

    def test_owner_worker_environment_and_launcher_sources_must_be_graph_frozen(self):
        with self.f.ready():owner.graph_for_owner(self.root,self.f.graph_ref,'train')
        (self.root/owner.SOURCE).write_text('spoofed owner')
        with self.f.ready(),self.assertRaises(ValueError):owner.graph_for_owner(self.root,self.f.graph_ref,'train')

    def test_actual_command_method_specific_launcher_and_validation_bundle_args(self):
        with self.f.ready():
            reference=self.f.prepare();plan,_=owner.read(self.root,reference)
            for method in owner.graphs.METHODS:
                job=next(j for j in plan['jobs'] if j['method']==method)
                argv=owner.command(self.root,plan,self.f.graph,job,check=False)
                self.assertIn(owner.WORKER,argv);self.assertIn(owner.WRAPPER,argv)
                self.assertIn('scripts/run_aligned_gpu.sh' if method=='physmani' else 'scripts/run_local_sim.sh',argv)
                self.assertEqual(argv[argv.index('--manifest-sha256')+1],self.f.environment_sha)
                self.assertEqual(owner.command(self.root,plan,self.f.graph,job,check=True),argv+['--check'])
            validation=dict(plan,split='validation',validation_bundle=self.f.bundle_ref)
            argv=owner.command(self.root,validation,self.f.graph,plan['jobs'][0],check=False)
            for flag,key in (('--suite','suite'),('--suite-lock','suite_lock'),('--source-inputs','source_inputs'),('--admission','admission')):
                self.assertEqual(argv[argv.index(flag)+1],self.f.bundle[key]['path'])

    def admitted(self):
        activity=dict(lease=False,admission=False,calls=[])
        @contextmanager
        def lease(*args,**kwargs):
            self.assertFalse(activity['lease']);activity['lease']=True;activity['calls'].append('lease_enter')
            try:yield 'real_mock_permit'
            finally:activity['lease']=False;activity['calls'].append('lease_exit')
        @contextmanager
        def scope(*args):
            self.assertTrue(activity['lease']);activity['admission']=True
            try:yield {'synthetic':'admission'}
            finally:activity['admission']=False
        def rows(*args):
            self.assertTrue(activity['lease'] and activity['admission']);activity['calls'].append('315_read')
            return {},self.f.validation_rows,{}
        guard=NS(reserved_resources_after_TEST=lease,require_permit=lambda *a: self.assertTrue(activity['lease']))
        gate=NS(admission_scope=scope,read_formal_source=rows)
        def imports(name):
            if name==owner.GUARD[:-3].replace('/','.'):return guard
            if name==owner.GATE[:-3].replace('/','.'):return gate
            raise AssertionError('unexpected import '+name)
        return activity,imports

    def test_VAL315_read_only_with_parent_admission_short_lease_and_945_exact_jobs(self):
        activity,imports=self.admitted()
        with self.f.ready(),patch.object(owner.importlib,'import_module',side_effect=imports),patch.object(owner.subprocess,'Popen',side_effect=AssertionError('prepare cannot launch')):
            reference=self.f.prepare('validation');plan,_=owner.read(self.root,reference)
        self.assertEqual(len(plan['jobs']),945);self.assertEqual(len(plan['sources']),315)
        self.assertEqual(activity['calls'],['lease_enter','315_read','lease_exit'])
        self.assertFalse(activity['lease']);self.assertFalse(activity['admission'])
        self.assertTrue(all('/validation/' in j['out'] and 'demo_only' not in j['out'] for j in plan['jobs']))
        self.assertFalse(any((self.root/j['out']).exists() for j in plan['jobs']))

    def test_canonical_VAL_TEST_paths_cannot_be_smuggled_as_bundle_graph_or_plan(self):
        for name in ('artifacts/framework/protocol/validation_manifest.json','artifacts/framework/protocol/test_manifest.json'):
            reference=dict(root='project',path=name,sha256='a'*64)
            for value in (reference,name):
                with patch.object(owner,'read',side_effect=AssertionError('no metadata/source open')), \
                     patch.object(owner,'pointer',side_effect=AssertionError('no source hash')):
                    with self.assertRaises(ValueError):owner.validation_rows(self.root,self.f.graph_ref,self.f.graph,value)
                    with self.assertRaises(ValueError):owner.graph_for_owner(self.root,value,'train')
                    with self.assertRaises(ValueError):owner.execute_owner(self.root,value,'a'*64)

    def test_execute_requires_explicit_plan_SHA_and_rejects_mutated_grid_or_retry(self):
        with self.f.ready():
            reference=self.f.prepare()
            with patch.object(owner.subprocess,'Popen',side_effect=AssertionError('no launch')):
                with self.assertRaises(ValueError):owner.execute_owner(self.root,reference,'wrong')
            original,_=owner.read(self.root,reference)
            for kind in ('retry','duplicate','filtered','wrong_parent'):
                changed=deepcopy(original)
                if kind=='retry':changed['automatic_retry']=True
                elif kind=='duplicate':changed['jobs'][-1]=deepcopy(changed['jobs'][0])
                elif kind=='filtered':changed['source_outcome_filtering']=True
                else:changed['sources'][0]['raw_episode_id']='episode77'
                bad=write(self.root,reference['path'],changed)
                with self.subTest(kind=kind),self.assertRaises(ValueError):owner._load_plan(self.root,bad,admit_sources=True)

    def test_STOP_before_first_child_and_infra_failure_do_not_become_Native_false(self):
        with self.f.ready():
            reference=self.f.prepare();plan,_=owner.read(self.root,reference)
            stop=self.root/plan['owner_runtime']/'STOP';stop.write_text('stop')
            with patch.object(owner.subprocess,'Popen',side_effect=AssertionError('no launch')),patch('sys.stdout',new_callable=io.StringIO):
                with self.assertRaises(ValueError):owner.execute_owner(self.root,reference,reference['sha256'])
            status=json.loads((self.root/plan['owner_runtime']/'status.json').read_text())
            self.assertEqual(status['stage'],'failed');self.assertEqual(status['completed'],[])
            self.assertFalse(status['failed'][0]['native_failure_inferred'])
            self.assertFalse((self.root/plan['owner_runtime']/'terminal.json').exists())
            with self.assertRaises(ValueError):owner.assemble_cohort(self.root,reference,{'path':'missing.json','sha256':'a'*64})

    def test_new_fresh_owner_cannot_resume_a_launched_plan(self):
        with self.f.ready():
            reference=self.f.prepare();plan,_=owner.read(self.root,reference)
            write(self.root,plan['owner_runtime']+'/launch.json',{'prior':'attempt'})
            with patch.object(owner.subprocess,'Popen',side_effect=AssertionError('no repeat')):
                with self.assertRaises(ValueError):owner.execute_owner(self.root,reference,reference['sha256'])

    def test_VAL_parent_lease_is_gone_before_CPU_child_and_infra_stops_queue(self):
        activity,imports=self.admitted()
        def child(*args,**kwargs):
            self.assertFalse(activity['lease']);self.assertFalse(activity['admission'])
            raise RuntimeError('synthetic infrastructure child failure')
        with self.f.ready(),patch.object(owner.importlib,'import_module',side_effect=imports):
            reference=self.f.prepare('validation');plan,_=owner.read(self.root,reference)
            with patch.object(owner,'child_phase',side_effect=child) as launch,patch('sys.stdout',new_callable=io.StringIO):
                with self.assertRaises(RuntimeError):owner.execute_owner(self.root,reference,reference['sha256'])
            self.assertEqual(launch.call_count,1)
            status=json.loads((self.root/plan['owner_runtime']/'status.json').read_text())
            self.assertEqual(status['stage'],'failed');self.assertEqual(status['attempted'],1)
            with self.assertRaises(ValueError):owner.assemble_cohort(self.root,reference,{'path':'missing.json','sha256':'a'*64})

    def test_child_phase_records_the_exact_Popen_argv_pid_and_actual_exit(self):
        with self.f.ready():
            reference=self.f.prepare();plan,_=owner.read(self.root,reference);job=plan['jobs'][0]
            captured=[];process=dict(pid=400000,start_ticks=9000)
            class Child:
                pid=400000
                def wait(inner):return False  # bool cannot impersonate integer exit0
                def poll(inner):return False
            def launch(argv,**kwargs):captured.append(argv);return Child()
            with patch.object(owner.subprocess,'Popen',side_effect=launch),patch.object(owner,'process_identity',return_value=process):
                with self.assertRaises(ValueError):owner.child_phase(self.root,plan,self.f.graph,job,check=True)
            leaf=self.root/plan['owner_runtime']/'children'/job['task']/job['raw_episode_id']/job['method']/'CPU_check'
            value=json.loads((leaf/'launch.json').read_text());terminal=json.loads((leaf/'terminal.json').read_text())
            self.assertEqual(value['argv'],captured[0]);self.assertEqual(value['process'],process)
            self.assertIs(terminal['returncode'],False);self.assertFalse(value['automatic_retry'])

    def test_different_postexit_cohort_entrypoints_never_relabel_VAL_as_TRAIN(self):
        activity,imports=self.admitted()
        with self.f.ready(),patch.object(owner.importlib,'import_module',side_effect=imports):
            reference=self.f.prepare('validation')
            with self.assertRaisesRegex(ValueError,'distinct'):
                owner.assemble_cohort(self.root,reference,{'path':'missing.json','sha256':'a'*64})


class FullSyntheticExecution(unittest.TestCase):
    def test_serial_192_preserves_all_false_compact_receipts_and_only_root_can_assemble(self):
        with TemporaryDirectory() as directory,TemporaryDirectory() as template_directory:
            root=Path(directory);fixture=Fixture(root)
            original=FreshFixture(template_directory);template=json.loads((original.root/original.records[0]['case']['path']).read_text())
            # Copy the valid static530 profile and its byte-pinned dependency map
            # into the owner test graph, without reading a real model/dataset.
            spec=graph_spec(root)
            # Reuse already frozen owner declarations, changing only synthetic
            # physical profile inputs before this new graph is compiled.
            spec.update(physical=deepcopy(fixture.graph['physical']),control=deepcopy(fixture.graph['control']),
                routes=deepcopy(fixture.graph['routes']),train_source_manifest=fixture.graph['train_source_manifest'])
            for name in (*owner.readiness.RML_LOCAL_SOURCES,*owner.readiness.RML_NATIVE_SOURCES):
                value=put(root,name,(original.root/name).read_bytes());spec['physical']['sources_map'][name]=value
            profile=deepcopy(original.profile)
            spec['physical']['common_RML']['profile']=write(root,'metadata/actual_profile.json',profile)
            protocol=deepcopy(original.protocol)
            protocol['implementation']=spec['physical']['outcome_sampling']['source']
            protocol['runtime_consumer']=protocol['original_runtime']=spec['physical']['sources_map']['consumer']
            spec['physical']['outcome_sampling']['protocol']=write(root,'metadata/actual_protocol.json',protocol)
            fixture.graph_ref=owner.graphs.write_policy_graph(root,spec,namespace='artifacts/framework/multitask_r10/owner_full_graph')
            fixture.graph=owner.graphs.read_policy_graph(root,fixture.graph_ref);fixture.stamp=owner.stamp(fixture.graph_ref,fixture.graph)
            with fixture.ready():reference=fixture.prepare()
            plan,_=owner.read(root,reference);calls=[];counter=[0]
            def launch(argv,**kwargs):
                counter[0]+=1;outer=dict(pid=400000+counter[0],start_ticks=90000+counter[0]);check='--check' in argv
                task=argv[argv.index('--task')+1];raw=argv[argv.index('--raw-episode-id')+1];method=argv[argv.index('--method')+1]
                job=next(j for j in plan['jobs'] if (j['task'],j['raw_episode_id'],j['method'])==(task,raw,method))
                route=next(r for r in fixture.graph['routes'] if (r['task'],r['method'])==(task,method))
                source=next(r for r in plan['sources'] if (r['task'],r['raw_episode_id'])==(task,raw))
                witness=owner.readiness.make_graph_witness(plan['graph'],fixture.graph,route,source,split='train')
                inner=dict(pid=outer['pid']+100000,start_ticks=outer['start_ticks']+100000)
                case=deepcopy(template);case.update(task=task,method=method,source_raw_episode_id=raw,
                    raw_episode_id=raw+'__policy_aug_r0',parent_raw_episode_id=raw,process_id=inner['pid'],
                    version=owner.CASE_SCHEMAS['train'],graph_witness=witness,author_outcome_protocol=protocol,
                    source_overlay_lock=fixture.graph['physical']['source_overlay_lock'])
                block=task in owner.readiness.BLOCKS
                from tests.test_repaired_readiness_v1144 import sampled_case
                sampled=sampled_case(block=block,method=method)
                for key in ('native_outcome_sampling','alignment_receipt'):case[key]=sampled[key]
                case['alignment_receipt']['all_stage_counted_joint_motion_v191']=deepcopy(case['all_stage_counted_joint_motion'])
                case['common_author_RML_parameters']['profile_sha256']=owner.readiness.digest(profile)
                identity=dict(case['identity'],task=task,method=method,source_raw_episode_id=raw,parent_raw_episode_id=raw,
                    raw_episode_id=raw+'__policy_aug_r0',source_row_sha256=owner.readiness.digest(source),
                    config_sha256=route['configuration']['sha256'],frozen_inputs_sha256=route['frozen_inputs']['sha256'])
                case['component_domain'].update(task=task,method=method,raw_episode_id=raw)
                case['execution_protocol']=dict(schema='r10_graph_bound_repaired_policy_case_v1146',
                    actual_worker=route['train']['worker'],actual_preparer=route['train']['preparer'],
                    inherited_consumer=route['train']['consumer'],source_overlay_lock=fixture.graph['physical']['source_overlay_lock'],
                    selected_physical_factory=route['physical_factory'],factory_binding=witness['factory_binding'],
                    execution_branch=witness['model_branch'],actual_split='train',component_purpose='train_qualification',
                    automatic_retry=False,original_745_execution_acceptance_granted=False,qualification_granted_by_worker=False,
                    formal_score_contribution=False,additional_native_getter_observe_predict_or_success_calls=0,
                    unchanged_action_outcome_sampling=protocol,**fixture.stamp)
                identity.update(execution_protocol_schema=case['execution_protocol']['schema'],
                    execution_protocol_sha256=owner.readiness.digest(case['execution_protocol']))
                case['identity']=owner.readiness.bind_graph_identity(identity,witness)
                phase='CPU_check' if check else 'native';case_ref=None
                if not check:
                    case_ref=write(root,job['out']+'/case.json',case)
                    write(root,job['out']+'/owned_hook_receipt.json',dict(actual_split='train',reset=[] if block else [{'complete':True}],paths=[]))
                execution=dict(complete=True,actual_split='train',native_started=not check,GPU_started=False,
                    mode='CPU_check' if check else 'graph_bound_native_TRAIN',graph_witness=witness,identity=case['identity'],
                    component_domain=case['component_domain'],
                    execution_protocol=case['execution_protocol'],case=case_ref,native_success=False,native_failure=False,
                    native_false_preserved=True,runtime_scope_restoration={'complete':True,'restored':True})
                # CPU and native run in distinct processes; identity is process-free.
                value=dict(complete=True,process=inner,actual_worker=route['train']['worker'],actual_preparer=route['train']['preparer'],
                    graph_witness=witness,execution=execution,**fixture.stamp,
                    frozen_OS_environment=dict(complete=True,outer_wrapper_pid=outer['pid'],outer_wrapper_start_ticks=outer['start_ticks'],
                        observer_pid=inner['pid'],observer_start_ticks=inner['start_ticks']))
                write(root,job['runtime']+'/'+phase+'/worker_sidecar.json',value);calls.append((task,raw,method,check))
                return NS(pid=outer['pid'],wait=lambda:0,poll=lambda:0)
            def processes(pid=None):return dict(pid=2147483000,start_ticks=1) if pid is None else dict(pid=pid,start_ticks=90000+pid-400000)
            with fixture.ready(),patch.object(owner,'_refresh'),patch.object(owner.subprocess,'Popen',side_effect=launch), \
                    patch.object(owner,'process_identity',side_effect=processes),patch('sys.stdout',new_callable=io.StringIO):
                result=owner.execute_owner(root,reference,reference['sha256'])
                self.assertEqual(result['completed'],192);self.assertFalse(result['actual_tool_exit_observed_by_owner'])
                self.assertEqual(len(calls),384);self.assertTrue(all(calls[i][-1] is (i%2==0) for i in range(384)))
                status=json.loads((root/plan['owner_runtime']/'status.json').read_text())
                self.assertEqual(len(status['completed']),192);self.assertTrue(all(not r['native_success'] for r in status['completed']))
                self.assertNotIn('execution',json.dumps(status));self.assertNotIn('graph_witness',json.dumps(status))
                self.assertFalse((root/plan['owner_runtime']/'exec_completion.json').exists())
                missing={'path':'missing_actual_exec.json','sha256':'a'*64}
                with self.assertRaises(ValueError):owner.assemble_cohort(root,reference,missing)
                completion=dict(process=dict(pid=2147483000,start_ticks=1),actual_tool_exit_observed=True,natural_completion=True,
                    returncode=False,plan=reference,terminal=result['terminal'])
                actual=write(root,plan['owner_runtime']+'/exec_completion.json',completion)
                with self.assertRaises(ValueError):owner.assemble_cohort(root,reference,actual)
                completion['returncode']=0;actual=write(root,actual['path'],completion)
                cohort_ref=owner.assemble_cohort(root,reference,actual);cohort,_=owner.read(root,cohort_ref)
                self.assertEqual(cohort['schema'],owner.readiness.COHORT_SCHEMA);self.assertEqual(len(cohort['records']),192)
                self.assertFalse(cohort['readiness_granted_by_owner'])
                # Full frozen1144 audit verifies the exact owner-produced shape;
                # all source/data/launch records here are explicitly synthetic.
                audited=owner.readiness.audit_fresh_train192(root,fixture.graph_ref,cohort_ref)
                self.assertEqual(audited['completed'],192)
                with self.assertRaises(FileExistsError):owner.assemble_cohort(root,reference,actual)


if __name__=='__main__':unittest.main()
