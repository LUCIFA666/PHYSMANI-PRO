"""Bounded synthetic CPU tests; no real graph, lease, VAL or Native execution."""
from contextlib import contextmanager
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from scripts import run_parallel_repaired_owner_v1173_r10 as o


def put(root,name,value):
    p=Path(root)/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,sort_keys=True));return o.pointer(root,name)


def jobs(n=9):
    return [dict(job_id=f'j{i:04d}',task=o.graphs.TASKS[i%16],raw_episode_id='episode'+str(i//3),
        method=o.graphs.METHODS[i%3],split='train',configuration={'root':'project','path':'synthetic.json','sha256':'a'*64},
        out='synthetic/out/'+str(i),runtime='synthetic/runtime/'+str(i),source_row_sha256='b'*64) for i in range(n)]


def affinity_fixture(root,graph):
    data=dict(schema=o.AFFINITY_SCHEMA,allowed_logical_cpus=list(range(32)),physical_core_count=16,max_native_slots=12,
        core_groups=[dict(package=0,core=i,cpus=[i,i+16],L3_bytes=(96 if i<8 else 32)*2**20) for i in range(16)],
        native_slot_cpus=[[i,i+16] for i in range(12)],control_and_CPU_preflight_cpus=[12,13,14,15,28,29,30,31],
        bitwise_equivalence_claimed=False,taskset_root='/usr/bin',taskset_runs_outside820_as_control_tool=True,
        taskset_binary=dict(root='host_tools',path='taskset',sha256=o.hashlib.sha256(Path('/usr/bin/taskset').read_bytes()).hexdigest()))
    reference=put(root,o.AFFINITY_PATH,data)
    graph.setdefault('control',{}).setdefault('static_inputs',{}).update(parallel_CPU_affinity_profile=reference,parallel_taskset_binary=data['taskset_binary'])
    graph.setdefault('roots',{})['host_tools']='/usr/bin'
    return dict(data,profile_ref=reference)


class Clock:
    def __init__(self):self.t=0.
    def __call__(self):return self.t
    def sleep(self,dt):self.t+=dt


class MockDriver:
    """Only Python dictionaries and a synthetic clock, never processes."""
    def __init__(self,clock,*,fail_cpu=None):self.clock=clock;self.fail_cpu=fail_cpu;self.launched=[];self.joined=[];self.finished=[]
    def launch(self,a):
        self.launched.append(deepcopy(a));duration=.2 if a['phase']=='CPU_check' else (3 if a['job']['method']=='cv_geometry' else 2)
        return dict(a=a,start=self.clock(),end=self.clock()+duration)
    def poll(self,h):return 0 if self.clock()>=h['end'] else None
    def wait(self,h):self.joined.append(h['a']);self.clock.t=max(self.clock(),h['end']);return 0
    def collect(self,h,code):
        self.finished.append(h['a']);j=h['a']['job']
        if h['a']['phase']=='CPU_check':
            if j['job_id']==self.fail_cpu:raise ValueError('synthetic preflight failure')
            return {'synthetic_CPU_only':True}
        return dict(job_id=j['job_id'],task=j['task'],raw_episode_id=j['raw_episode_id'],method=j['method'],
            native_success=False,native_terminal_failure=j['method']=='physmani',motion_pass=False,
            native_interval=dict(start_monotonic=h['start'],end_monotonic=h['end']),synthetic_mock_not_Native_evidence=True)


def sample(clock,**extra):
    return dict(at=clock(),CPU_percent=20,RAM_percent=20,VRAM_percent=20,GPU_percent=20,**extra)


class SchedulerTests(unittest.TestCase):
    def test_CPU_before_Native_unique_slots_and_GPU_only_for_native_physmani(self):
        s=o.ParallelScheduler(jobs());first=s.dispatch()
        self.assertEqual(len(first),2);self.assertTrue(all(a['phase']=='CPU_check' and a['gpu_slot'] is None for a in first))
        self.assertEqual(s.dispatch(),[])
        for a in first:s.finish(a['job']['job_id'],'CPU_check',{})
        second=s.dispatch();native=[a for a in second if a['phase']=='native']
        self.assertEqual(len(native),2);self.assertEqual(len({a['slot'] for a in native}),2)
        self.assertTrue(all(a['gpu_slot'] is None for a in second if a['phase']=='CPU_check'))

    def test_all_false_fatal_motion_false_retained_and_final_order_original(self):
        c=Clock();d=MockDriver(c);result=o.run_scheduler({'jobs':jobs()},d,sampler=lambda p:sample(c),clock=c,sleep=c.sleep)
        self.assertTrue(result['complete']);self.assertEqual(len(d.launched),18)
        self.assertEqual([r['job_id'] for r in result['records']],[j['job_id'] for j in jobs()])
        self.assertTrue(all(not r['native_success'] and not r['motion_pass'] for r in result['records']))
        for j in jobs():self.assertEqual([a['phase'] for a in d.launched if a['job']['job_id']==j['job_id']],['CPU_check','native'])
        self.assertFalse(result['success_rate_used_for_scheduling']);self.assertFalse(result['automatic_retry'])

    def test_method_fairness_and_GPU_slots_bounded(self):
        c=Clock();d=MockDriver(c);r=o.run_scheduler({'jobs':jobs(18)},d,sampler=lambda p:sample(c),clock=c,sleep=c.sleep)
        self.assertTrue(r['complete']);cpu=[a['job']['method'] for a in d.launched if a['phase']=='CPU_check']
        self.assertEqual(cpu[:6],list(o.graphs.METHODS)*2)
        self.assertTrue(all(a['gpu_slot'] is None or a['gpu_slot']<5 for a in d.launched))

    def test_ready_queue_includes_inflight_CPU_and_stays_bounded(self):
        js=jobs(60)
        for j in js:j['method']='physmani'
        s=o.ParallelScheduler(js)
        for _ in range(40):
            dispatch=s.dispatch()
            for a in dispatch:
                if a['phase']=='CPU_check':s.finish(a['job']['job_id'],'CPU_check',{})
            self.assertLessEqual(sum(v in ('CPU_check','ready') for v in s.states.values()),24)
        self.assertEqual(sum(v=='native' for v in s.states.values()),1)

    def test_promotion_needs_actual_completed_physical_overlap_and_headroom_not_success(self):
        s=o.ParallelScheduler(jobs(2))
        for a in s.dispatch():s.finish(a['job']['job_id'],'CPU_check',{})
        running=s.dispatch()
        for a in running:
            if a['phase']=='native':s.finish(a['job']['job_id'],'native',dict(native_success=False,native_terminal_failure=True,motion_pass=False,
                native_interval=dict(start_monotonic=0.,end_monotonic=5.)))
        for t in (1.,2.,3.):s.sample(dict(at=t,CPU_percent=20,RAM_percent=20,VRAM_percent=20,GPU_percent=20))
        self.assertTrue(s.maybe_promote());self.assertEqual(s.limits,(4,2,2));self.assertFalse(s.maybe_promote())
        self.assertFalse(s.promotions[0]['success_rate_used'])

    def test_no_promotion_from_process_overlap_without_native_interval(self):
        s=o.ParallelScheduler(jobs(2))
        for t in (1.,2.,3.):s.sample(dict(at=t,CPU_percent=0,RAM_percent=0,VRAM_percent=0,GPU_percent=0))
        self.assertFalse(s.maybe_promote())

    def test_memory_above90_stops_new_admission_and_drains_without_kill(self):
        c=Clock();d=MockDriver(c)
        def resources(previous):
            v=sample(c)
            if c()>1:v['VRAM_percent']=91
            return v
        r=o.run_scheduler({'jobs':jobs(18)},d,sampler=resources,clock=c,sleep=c.sleep)
        self.assertFalse(r['complete']);self.assertEqual(r['drain_reason'],'memory_above90_drain')
        self.assertEqual(len(d.finished),len(d.launched));self.assertTrue(r['remaining_job_ids'])

    def test_administrative_STOP_only_at_admission_and_running_children_reaped(self):
        c=Clock();d=MockDriver(c)
        r=o.run_scheduler({'jobs':jobs()},d,sampler=lambda p:sample(c),clock=c,sleep=c.sleep,stop_requested=lambda:c()>.5)
        self.assertEqual(r['drain_reason'],'administrative_STOP_drain');self.assertEqual(len(d.finished),len(d.launched))
        self.assertFalse(r['complete'])

    def test_CPU_infra_failure_never_becomes_native_false_or_retries(self):
        c=Clock();d=MockDriver(c,fail_cpu='j0000')
        r=o.run_scheduler({'jobs':jobs()},d,sampler=lambda p:sample(c),clock=c,sleep=c.sleep)
        self.assertFalse(r['complete']);self.assertEqual(len(r['infrastructure_failures']),1)
        self.assertFalse(r['infrastructure_failures'][0]['Native_failure_inferred'])
        self.assertFalse(any(a['job']['job_id']=='j0000' and a['phase']=='native' for a in d.launched))

    def test_controller_exception_joins_children_before_leaving_parent_scope(self):
        c=Clock();d=MockDriver(c)
        def update(s):raise KeyboardInterrupt('synthetic only')
        r=o.run_scheduler({'jobs':jobs()},d,sampler=lambda p:sample(c),clock=c,sleep=c.sleep,on_update=update)
        self.assertEqual(len(d.joined),2);self.assertEqual(len(d.finished),len(d.launched));self.assertFalse(r['complete'])
        self.assertIn('KeyboardInterrupt',r['drain_reason'])

    def test_resource_monitor_failure_cannot_silently_run_unmeasured(self):
        c=Clock();d=MockDriver(c)
        def fail(p):raise OSError('synthetic')
        r=o.run_scheduler({'jobs':jobs()},d,sampler=fail,clock=c,sleep=c.sleep)
        self.assertEqual(d.launched,[]);self.assertFalse(r['complete'])

    def test_first_launch_failure_cancels_other_scheduled_but_unforked_jobs(self):
        c=Clock();d=MockDriver(c);calls=[]
        def fail(a):calls.append(a);raise OSError('synthetic fork failure')
        d.launch=fail
        r=o.run_scheduler({'jobs':jobs()},d,sampler=lambda p:sample(c),clock=c,sleep=c.sleep)
        self.assertEqual(len(calls),1);self.assertEqual(r['attempted'],1);self.assertEqual(len(r['infrastructure_failures']),1)

    def test_STOP_between_two_scheduled_launches_prevents_second_fork(self):
        c=Clock();d=MockDriver(c)
        r=o.run_scheduler({'jobs':jobs()},d,sampler=lambda p:sample(c),clock=c,sleep=c.sleep,
            stop_requested=lambda:len(d.launched)>=1)
        self.assertEqual(len(d.launched),1);self.assertEqual(len(d.finished),1)
        self.assertEqual(r['drain_reason'],'administrative_STOP_drain')

    def test_malformed_native_result_is_once_only_infra_and_other_children_drain(self):
        c=Clock();d=MockDriver(c);original=d.collect
        def collect(h,code):
            result=original(h,code)
            if h['a']['phase']=='native':result.pop('native_interval')
            return result
        d.collect=collect
        r=o.run_scheduler({'jobs':jobs()},d,sampler=lambda p:sample(c),clock=c,sleep=c.sleep)
        self.assertFalse(r['complete']);self.assertTrue(r['infrastructure_failures'])
        self.assertEqual(len(d.finished),len(d.launched))


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.graph=dict(physical_policy_sha256=o.PHYSICAL_SHA,control_graph_sha256='b'*64,policy_sha256='c'*64,
            routes=[dict(task=t,method=m,configuration=put(self.root,'synthetic/config/'+t+'_'+m+'.json',{}),GPU_required=m=='physmani')
                for t in o.graphs.TASKS for m in o.graphs.METHODS],physical={'environment_manifest':{'sha256':serial_env()}},control={'components':{}})
        self.affinity=affinity_fixture(self.root,self.graph)
        self.gr=put(self.root,'artifacts/framework/multitask_r10/synthetic_graph/policy_graph.json',self.graph)
        self.sources=[dict(task=t,raw_episode_id='episode'+str(i),path='never-read/'+t+'/'+str(i)+'.h5') for t in o.graphs.TASKS for i in range(4)]
        old=put(self.root,o.OLD_BASE+'/plan.json',dict(sources=self.sources))
        self.exit=put(self.root,o.TRANSITION_PATH,{'synthetic_fixture_not_actual_exit':True})
        self.predecessor=dict(original_plan=old,exit=self.exit,completed_prefix_count=5,records=[],old_graph={'root':'project','path':'old.json','sha256':'d'*64})

    def review(self,split='train',count=187,namespace='fixture',bundle=None):
        return put(self.root,'artifacts/framework/multitask_r10/synthetic_review/'+split+'.json',dict(schema=o.REVIEW_SCHEMA,
            reviewer='root',review_complete=True,release_fixed_cohort=True,split=split,namespace=namespace,expected_cases=count,
            automatic_retry=False,source_outcome_filtering=False,predecessor_exit=self.exit,validation_bundle=bundle,**o.stamp(self.gr,self.graph)))

    @contextmanager
    def fake_graph(self):
        with patch.object(o,'graph_for_owner',return_value=(self.graph,self.gr)),patch.object(o,'inspect_predecessor',return_value=self.predecessor), \
                patch.object(o,'_hardware_snapshot',return_value={k:self.affinity[k] for k in ('allowed_logical_cpus','core_groups')}), \
                patch.object(o,'physical_equivalence',return_value={'synthetic_fixture':True}):yield

    def test_default_inert_and_TEST_refused_before_read(self):
        with patch.object(o.subprocess,'Popen',side_effect=AssertionError('no process')),patch('sys.stdout',new_callable=io.StringIO):r=o.main([])
        self.assertFalse(r['enabled'])
        with patch.object(o.graphs,'read_policy_graph',side_effect=AssertionError('must refuse first')),self.assertRaises(ValueError):o.graph_for_owner(self.root,'never','test')

    def test_preserve64rows_global_ids_and_only_unstarted_suffix_fresh_outputs(self):
        with self.fake_graph():
            p=o.prepare_parallel_plan(self.root,self.gr,namespace='fixture',review_ref=self.review(),predecessor_exit_ref=self.exit)
            plan,_,_=o.load_plan(self.root,p)
        self.assertEqual(len(plan['sources']),64);self.assertEqual(plan['global_expected_cases'],192);self.assertEqual(plan['expected_cases'],187)
        self.assertEqual(plan['jobs'][0]['job_id'],'j0005');self.assertEqual(plan['jobs'][-1]['job_id'],'j0191')
        self.assertTrue(all(set(j)==o.JOB_KEYS for j in plan['jobs']));self.assertFalse(plan['case_graphs_relabelled'])
        self.assertTrue(all(not (self.root/j['out']).exists() for j in plan['jobs']))
        with self.fake_graph(),self.assertRaises(ValueError):o.prepare_parallel_plan(self.root,self.gr,namespace='fixture',review_ref=self.review(),predecessor_exit_ref=self.exit)

    def test_missing_actual_old_exit_blocks_plan_before_fork(self):
        with self.fake_graph(),patch.object(o,'inspect_predecessor',side_effect=ValueError('old live')),patch.object(o.subprocess,'Popen',side_effect=AssertionError('no fork')):
            with self.assertRaisesRegex(ValueError,'old live'):o.prepare_parallel_plan(self.root,self.gr,namespace='fixture',review_ref=self.review(),predecessor_exit_ref=self.exit)

    def test_load_rejects_retry_reordered_suffix_and_profile_changes(self):
        with self.fake_graph():
            p=o.prepare_parallel_plan(self.root,self.gr,namespace='fixture',review_ref=self.review(),predecessor_exit_ref=self.exit);plan,_=o.read(self.root,p)
            for kind in ('retry','order','profile'):
                v=deepcopy(plan)
                if kind=='retry':v['automatic_retry']=True
                elif kind=='order':v['jobs'].reverse()
                else:v['profile']['max_native_slots']=13
                bad=put(self.root,p['path'],v)
                with self.subTest(kind=kind),self.assertRaises(ValueError):o.load_plan(self.root,bad)

    def test_admitted_VAL315_by3_plan_without_prefix_or_TRAIN_relabel(self):
        rows=[dict(task=o.graphs.TASKS[i%16],raw_episode_id='syntheticval'+str(i),path='never-read-val.h5') for i in range(315)]
        bundle=put(self.root,o.BUNDLE_PREFIX+'synthetic/bundle.json',{'synthetic':True})
        with self.fake_graph(),patch.object(o,'validation_rows',return_value=(rows,bundle)) as gate:
            p=o.prepare_parallel_plan(self.root,self.gr,split='validation',namespace='fixture',review_ref=self.review('validation',945,bundle=bundle),bundle_ref=bundle)
            plan,_,_=o.load_plan(self.root,p,execute=True)
        self.assertEqual(gate.call_count,2);self.assertEqual(len(plan['jobs']),945);self.assertEqual(len(plan['sources']),315)
        self.assertEqual(plan['schema'],o.VAL_PLAN_SCHEMA);self.assertIsNone(plan['predecessor']);self.assertEqual(plan['preserved_prefix_count'],0)
        self.assertTrue(all('/validation/' in j['out'] and 'demo_only' not in j['out'] for j in plan['jobs']))

    def test_wrong_plan_pin_refuses_before_guard(self):
        with self.fake_graph():p=o.prepare_parallel_plan(self.root,self.gr,namespace='fixture',review_ref=self.review(),predecessor_exit_ref=self.exit)
        with patch.object(o,'_module',side_effect=AssertionError('no guard lease')),self.assertRaises(ValueError):o.execute_owner(self.root,p,'wrong')

    def test_load_plan_rechecks_Root_release_and_exact_thread_metadata(self):
        with self.fake_graph():
            p=o.prepare_parallel_plan(self.root,self.gr,namespace='fixture',review_ref=self.review(),predecessor_exit_ref=self.exit)
            plan,_=o.read(self.root,p);review,_=o.read(self.root,plan['review']);review['release_fixed_cohort']=False
            plan['review']=put(self.root,plan['review']['path'],review);changed=put(self.root,p['path'],plan)
            with self.assertRaises(ValueError):o.load_plan(self.root,changed)
        self.assertTrue(o.THREAD_CONTROL['launcher_defaults_preserved'])
        self.assertEqual(o.THREAD_CONTROL['effective_launcher_environment'],dict(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))

    def test_command_actual_new_worker_resource_job_slots_and_GPU_choice(self):
        j=jobs(3)[2];j['configuration']=self.graph['routes'][2]['configuration'];p=dict(split='train',graph=self.gr,CPU_affinity=self.affinity)
        session={'root':'project','path':'runtime/session.json','sha256':'a'*64}
        a=dict(slot=1,gpu_slot=0)
        argv=o.command(self.root,p,self.graph,j,check=False,assignment=a,session_ref=session)
        self.assertEqual(argv[:3],['/usr/bin/taskset','--cpu-list','1,17'])
        self.assertIn(o.WORKER,argv);self.assertIn('--resource-job-id',argv);self.assertIn('--resource-gpu-slot',argv)
        a['gpu_slot']=None;cpu=o.command(self.root,p,self.graph,j,check=True,assignment=a,session_ref=session)
        self.assertNotIn('--resource-gpu-slot',cpu);self.assertEqual(cpu[-1],'--check')
        self.assertEqual(cpu[2],'12,13,14,15,28,29,30,31')

    def test_real_validation_reader_signature_short_lease_and_scope_exit(self):
        directory=o.BUNDLE_PREFIX+'reader_fixture/';events=[];state=dict(lease=False,admission=False)
        gate_file=put(self.root,o.GATE,{'synthetic_source':True});writer_file=put(self.root,o.WRITER,{'synthetic_source':True})
        self.graph['control']['components'].update(admission_guard=gate_file,formal_source_reader=gate_file,split_source_lock_writer=writer_file)
        rows=[dict(task=o.graphs.TASKS[i%16],raw_episode_id='syntheticval'+str(i)) for i in range(315)]
        admission=put(self.root,directory+'admission.json',{'synthetic':True})
        suite=put(self.root,directory+'suite.json',{'synthetic':True});lock=put(self.root,directory+'suite_lock.json',{'files':{'synthetic':'a'*64}})
        inputs=put(self.root,directory+'source_inputs.json',{'synthetic':True})
        bundle=put(self.root,directory+'bundle.json',dict(schema=o.BUNDLE_SCHEMA,source_count=315,Native_started=False,TEST_read=False,
            admission=admission,suite=suite,suite_lock=lock,source_inputs=inputs,**o.stamp(self.gr,self.graph)))
        @contextmanager
        def metadata(root,*,enabled):
            self.assertTrue(enabled);state['lease']=True;events.append('lease_enter')
            try:yield 'synthetic_permit'
            finally:state['lease']=False;events.append('lease_exit')
        @contextmanager
        def admitted(root,graph_ref,admission_ref,permit):
            self.assertTrue(state['lease']);self.assertEqual(graph_ref,self.gr);self.assertEqual(admission_ref,admission)
            state['admission']=True
            try:yield
            finally:state['admission']=False
        def reader(root,suite,files,split,*,validation_release=None):
            self.assertIsNone(validation_release);self.assertTrue(state['lease'] and state['admission']);self.assertEqual(split,'validation')
            events.append('315_read');return {},rows,{}
        modules={o.GUARD:NS(metadata_scope=metadata),o.GATE:NS(admission_scope=admitted,read_formal_source=reader)}
        with patch.object(o,'_module',side_effect=lambda n:modules[n]):actual,_=o.validation_rows(self.root,self.gr,self.graph,bundle)
        self.assertEqual(actual,rows);self.assertEqual(events,['lease_enter','315_read','lease_exit']);self.assertFalse(state['lease'])


class PhaseClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.job=jobs(1)[0];self.job.update(out=o.ART+'/train/development_demo_only/cohorts/fixture/t/e/cv_geometry',runtime=o.RUNTIME+'/train/cohorts/fixture/t/e/cv_geometry')
        worker=put(self.root,o.WORKER,{'synthetic':True});preparer=put(self.root,o.PREPARER,{'synthetic':True})
        self.g=dict(physical_policy_sha256=o.PHYSICAL_SHA,control_graph_sha256='b'*64,policy_sha256='c'*64,
            physical={'environment_manifest':{'sha256':serial_env()}},routes=[dict(task=self.job['task'],method=self.job['method'],GPU_required=False,
                train=dict(worker=worker,preparer=preparer))])
        affinity=affinity_fixture(self.root,self.g)
        self.gr=put(self.root,'artifacts/framework/multitask_r10/synthetic_graph/policy_graph.json',self.g)
        base=o.BASE+'/train/fixture';self.plan=dict(split='train',owner_runtime=base,graph=self.gr,profile=o.PROFILE,jobs=[self.job],CPU_affinity=affinity)
        self.owner=dict(pid=2147482000,start_ticks=1);self.outer=dict(pid=2147482001,start_ticks=1);self.inner=dict(pid=2147482002,start_ticks=1)
        self.session=put(self.root,base+'/resources/session.json',dict(schema='r10_parent_owned_parallel_resource_session_v1171',owner=self.owner,
            graph=self.gr,profile=o.PROFILE,jobs=[self.job]))
        put(self.root,base+'/launch.json',dict(process=self.owner,resource_session=self.session))
        a=dict(job_id=self.job['job_id'],phase='native',slot=0,gpu_slot=None)
        self.claim=dict(schema='r10_once_only_parallel_resource_claim_v1171',session=self.session,phase='native',process=self.inner,
            slot=0,gpu_slot=None,automatic_retry=False,test_only=False,job=self.job)
        self.claim_path=base+'/resources/claims/j0000_native/claim.json';cr=put(self.root,self.claim_path,self.claim)
        self.resource=dict(schema='r10_parallel_child_resource_receipt_v1171',body_completed=True,restored=True,leases_removed=True,
            errors=[],test_only=False,session=self.session,process=self.inner,actual_parent=self.owner,phase='native',job_id='j0000',
            slot=0,gpu_slot=None,parent_global_locks_verified=True,claim=cr)
        put(self.root,str(Path(self.claim_path).parent/'terminal.json'),self.resource)
        leaf=base+'/children/'+self.job['task']+'/'+self.job['raw_episode_id']+'/'+self.job['method']+'/native'
        self.sidecar=dict(schema='r10_parallel_graph_bound_repaired_policy_case_v1172',complete=True,process=self.inner,
            actual_worker=worker,actual_preparer=preparer,resource_scope_receipt=self.resource,
            execution=dict(complete=True,native_started=True,actual_split='train',mode='graph_bound_native_TRAIN',native_false_preserved=True,
                actual_native_interval=dict(start_monotonic=1.,end_monotonic=2.)),**o.stamp(self.gr,self.g),
            frozen_OS_environment=dict(complete=True,outer_wrapper_pid=self.outer['pid'],outer_wrapper_start_ticks=1,
                observer_pid=self.inner['pid'],observer_start_ticks=1))
        mask=dict(profile=affinity['profile_ref'],expected_cpus=[0,16],actual_main_cpus=[0,16],all_threads_checked=True,
            thread_cpus=[dict(tid=self.inner['pid'],cpus=[0,16])],vanished_thread_ids=[],additional_native_calls=0)
        self.sidecar['CPU_affinity_receipt']=dict(before=deepcopy(mask),after=deepcopy(mask))
        self.refs=dict(launch=put(self.root,leaf+'/launch.json',dict(process=self.outer,resource_assignment=a,resource_session=self.session,
            expected_CPU_cpus=[0,16],
            check_only=False,automatic_retry=False,actual_worker=worker,actual_preparer=preparer,started_monotonic=0.,
            argv=o.command(self.root,self.plan,self.g,self.job,check=False,assignment=a,session_ref=self.session),**o.stamp(self.gr,self.g))),
            terminal=put(self.root,leaf+'/terminal.json',dict(process=self.outer,returncode=0,automatic_retry=False,finished_monotonic=3.)),
            sidecar=put(self.root,self.job['runtime']+'/native/worker_sidecar.json',self.sidecar))

    def test_complete_synthetic_phase_has_actual_session_claim_and_strict_closure(self):
        o.validate_phase(self.root,self.plan,self.g,self.job,self.refs,check=False)

    def test_claim_wrong_path_is_rejected_before_parsing_foreign_contents(self):
        self.sidecar['resource_scope_receipt']['claim']=dict(root='project',path='artifacts/framework/protocol/test_manifest.json',sha256='a'*64)
        self.refs['sidecar']=put(self.root,self.refs['sidecar']['path'],self.sidecar)
        with self.assertRaises(ValueError):o.validate_phase(self.root,self.plan,self.g,self.job,self.refs,check=False)

    def test_wrong_claim_phase_session_slot_or_process_cannot_hide_behind_valid_sidecar(self):
        original=deepcopy(self.resource)
        for key,value in (('phase','CPU_check'),('session',dict(self.session,sha256='d'*64)),('slot',1),('process',self.outer)):
            changed=deepcopy(self.claim);changed[key]=value
            resource=deepcopy(original);resource['claim']=put(self.root,self.claim_path,changed)
            put(self.root,str(Path(self.claim_path).parent/'terminal.json'),resource)
            self.sidecar['resource_scope_receipt']=resource;self.refs['sidecar']=put(self.root,self.refs['sidecar']['path'],self.sidecar)
            with self.subTest(key=key),self.assertRaises(ValueError):o.validate_phase(self.root,self.plan,self.g,self.job,self.refs,check=False)

    def test_after_scope_thread_affinity_escape_is_not_a_clean_phase(self):
        self.sidecar['CPU_affinity_receipt']['after']['thread_cpus'][0]['cpus']=[1,17]
        self.refs['sidecar']=put(self.root,self.refs['sidecar']['path'],self.sidecar)
        with self.assertRaises(ValueError):o.validate_phase(self.root,self.plan,self.g,self.job,self.refs,check=False)

    def test_affinity_profile_drift_or_extra_native_calls_refused(self):
        original=deepcopy(self.sidecar)
        for key,value in (('profile',dict(self.plan['CPU_affinity']['profile_ref'],sha256='e'*64)),('additional_native_calls',1),('all_threads_checked',False)):
            changed=deepcopy(original);changed['CPU_affinity_receipt']['before'][key]=value
            self.refs['sidecar']=put(self.root,self.refs['sidecar']['path'],changed)
            with self.subTest(key=key),self.assertRaises(ValueError):o.validate_phase(self.root,self.plan,self.g,self.job,self.refs,check=False)


class AffinityTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.graph={};self.a=affinity_fixture(self.root,self.graph)

    def actual(self):return {k:self.a[k] for k in ('allowed_logical_cpus','core_groups')}

    def test_graph_profile_actual_topology_binary_and_full_sibling_mapping_bind(self):
        with patch.object(o,'_hardware_snapshot',return_value=self.actual()):value=o.read_CPU_affinity(self.root,self.graph)
        self.assertEqual(value,self.a)

    def test_topology_drift_duplicate_core_partial_SMT_or_tool_drift_refused(self):
        original=deepcopy(self.a)
        for kind in ('allowed_drift','duplicate_native','partial_SMT','overlap','taskset_drift'):
            data=deepcopy(original);data.pop('profile_ref');actual=self.actual()
            if kind=='allowed_drift':actual=dict(actual,allowed_logical_cpus=list(range(31)))
            elif kind=='duplicate_native':data['native_slot_cpus'][1]=data['native_slot_cpus'][0]
            elif kind=='partial_SMT':data['native_slot_cpus'][0]=[0,1]
            elif kind=='overlap':data['control_and_CPU_preflight_cpus'][0]=0
            else:data['taskset_binary']['sha256']='a'*64
            self.graph['control']['static_inputs']['parallel_CPU_affinity_profile']=put(self.root,o.AFFINITY_PATH,data)
            self.graph['control']['static_inputs']['parallel_taskset_binary']=data['taskset_binary']
            with patch.object(o,'_hardware_snapshot',return_value=actual),self.subTest(kind=kind),self.assertRaises(ValueError):o.read_CPU_affinity(self.root,self.graph)

    def test_parent_pins_all_existing_threads_and_restores_after_exception(self):
        original={10:set(range(32)),11:{0,16}};states=deepcopy(original)
        def get(tid):return set(range(32)) if tid==0 else states[tid]
        def setmask(tid,cpus):states[tid]=set(cpus)
        with patch.object(o,'_self_thread_ids',return_value=[10,11]),patch.object(o.os,'sched_getaffinity',side_effect=get),patch.object(o.os,'sched_setaffinity',side_effect=setmask):
            with self.assertRaisesRegex(RuntimeError,'synthetic'):
                with o.parent_CPU_affinity([12,28]):
                    self.assertEqual(states,{10:{12,28},11:{12,28}});raise RuntimeError('synthetic')
        self.assertEqual(states,original)

    def test_control_only_extra_root_allowed_but_physical_or_nonworker_route_changes_refused(self):
        ptr=lambda name:dict(root='project',path=name,sha256='a'*64)
        old=dict(physical_policy_sha256=o.PHYSICAL_SHA,physical={'shared':ptr('original.py')},roots={'project':'/synthetic','native':'/synthetic_native'},
            routes=[dict(task=t,method=m,configuration=ptr(t+'_'+m+'.json'),train=dict(worker=ptr('old.py'),preparer=ptr('oldprep.py'),case_schema='old',consumer=ptr('consumer.py'),purpose='train_qualification'),
                evaluation=dict(worker=ptr('old.py'),preparer=ptr('oldprep.py'),case_schema='old',consumer=ptr('consumer.py'),purpose='evaluation'))
                for t in o.graphs.TASKS for m in o.graphs.METHODS])
        new=deepcopy(old);new['roots']['host_tools']='/usr/bin'
        for route in new['routes']:
            for split in ('train','evaluation'):route[split].update(worker=ptr('new1172.py'),preparer=ptr('newprep1172.py'),case_schema='new1172')
        with patch.object(o.graphs,'read_policy_graph',return_value=old):
            self.assertEqual(o.physical_equivalence(self.root,{},new)['added_control_only_roots'],['host_tools'])
            for kind in ('old_root','physical','route','purpose'):
                changed=deepcopy(new)
                if kind=='old_root':changed['roots']['native']='/changed'
                elif kind=='physical':changed['physical']['extra']={'root':'host_tools','path':'taskset'}
                elif kind=='route':changed['routes'][0]['configuration']={'root':'host_tools','path':'taskset'}
                else:changed['routes'][0]['train']['purpose']='evaluation'
                with self.subTest(kind=kind),self.assertRaises(ValueError):o.physical_equivalence(self.root,{},changed)


class PredecessorTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        graph=dict(physical_policy_sha256=o.PHYSICAL_SHA)
        self.gr=put(self.root,'artifacts/framework/multitask_r10/synthetic_old/policy_graph.json',graph)
        js=jobs(192)
        for j in js:j.pop('job_id');j.pop('split')
        self.plan=dict(split='train',physical_policy_sha256=o.PHYSICAL_SHA,graph=self.gr,expected_cases=192,jobs=js)
        self.pr=put(self.root,o.OLD_BASE+'/plan.json',self.plan)
        self.lr=put(self.root,o.OLD_BASE+'/launch.json',dict(process=o.OLD_PROCESS,plan=self.pr))
        self.rows=[]
        for i,j in enumerate(js[:2]):
            row={k:j[k] for k in ('task','raw_episode_id','method')}
            for phase in ('CPU_check','native'):
                folder=o.OLD_BASE+'/synthetic_children/'+str(i)+'/'+phase
                row[phase]=dict(launch=put(self.root,folder+'/launch.json',{'process':dict(pid=2147482000+i,start_ticks=1)}),
                    sidecar=put(self.root,folder+'/worker_sidecar.json',{'process':dict(pid=2147483000+i,start_ticks=1)}))
            self.rows.append(row)
        message='owner1169: STOP before next fixed source'
        self.status=dict(stage='failed',active=None,completed=self.rows,
            failed=[dict(native_failure_inferred=False,message=message,active=None,exception_type='ValueError',automatic_retry=False)],error={'message':message})
        self.sr=put(self.root,o.OLD_BASE+'/status.json',self.status);self.fr=put(self.root,o.OLD_BASE+'/failure.json',self.status)
        self.tool=put(self.root,o.OLD_BASE+'/synthetic_actual_tool_output.json',{'synthetic_not_actual_root_exit':True})
        self.e=dict(schema=o.TRANSITION_SCHEMA,actual_tool_exit_observed=True,controlled_drain=True,natural_completion=False,
            returncode=1,native_children_all_dead=True,Native_killed=False,old_plan=self.pr,old_launch=self.lr,
            old_status=self.sr,terminal_or_failure=self.fr,process=o.OLD_PROCESS,completed_prefix_count=2,actual_tool_output=self.tool)
        self.er=put(self.root,o.TRANSITION_PATH,self.e)

    @contextmanager
    def dependencies(self):
        with patch.object(o,'OLD_PLAN_SHA',self.pr['sha256']),patch.object(o,'OLD_GRAPH_SHA',self.gr['sha256']), \
                patch.object(o.graphs,'read_policy_graph',return_value={}), \
                patch.object(o.readiness,'_not_live',return_value=True), \
                patch.object(o.serial,'completed_case',side_effect=lambda root,plan,g,j,c,n:next(r for r in self.rows if o.readiness._key(r)==o.readiness._key(j))):yield

    def test_valid_synthetic_administrative_exit_keeps_old_graph_prefix_identity(self):
        with self.dependencies():result=o.inspect_predecessor(self.root,self.er)
        self.assertEqual(result['completed_prefix_count'],2);self.assertEqual(result['records'],self.rows)
        self.assertEqual(result['old_graph'],self.gr);self.assertFalse(result['case_graphs_relabelled'])

    def test_natural_exit_bool_rc_killed_or_missing_tool_proof_cannot_fake_drain(self):
        for key,value in (('returncode',True),('returncode',0),('natural_completion',True),('Native_killed',True),('actual_tool_output',None)):
            e=dict(self.e);e[key]=value;ref=put(self.root,o.TRANSITION_PATH,e)
            with self.dependencies(),self.subTest(key=key,value=value),self.assertRaises(ValueError):o.inspect_predecessor(self.root,ref)

    def test_old_live_owner_or_child_and_non_STOP_error_refuse_suffix(self):
        with self.dependencies(),patch.object(o.readiness,'_not_live',return_value=False),self.assertRaises(ValueError):o.inspect_predecessor(self.root,self.er)
        status=deepcopy(self.status);status['failed'][0]['message']='unrelated infrastructure failure'
        sr=put(self.root,o.OLD_BASE+'/status.json',status);fr=put(self.root,o.OLD_BASE+'/failure.json',status)
        e=dict(self.e,old_status=sr,terminal_or_failure=fr);ref=put(self.root,o.TRANSITION_PATH,e)
        with self.dependencies(),self.assertRaises(ValueError):o.inspect_predecessor(self.root,ref)


class CollectorTests(unittest.TestCase):
    def test_closed_owner_requires_Root_actual_strict_exit0_and_dead_identity(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);base=o.BASE+'/train/fixture';graph=dict(physical_policy_sha256=o.PHYSICAL_SHA,control_graph_sha256='b'*64,policy_sha256='c'*64)
            gr=put(root,'artifacts/framework/multitask_r10/g/policy_graph.json',graph);process=dict(pid=2147483000,start_ticks=1)
            source=put(root,o.SOURCE,{'synthetic_source_only':True});job=jobs(1)[0]
            plan=dict(split='train',owner_runtime=base,graph=gr,expected_cases=1,jobs=[job]);pr=put(root,base+'/plan.json',plan)
            row=dict(job_id=job['job_id'],CPU_check={},native={})
            put(root,base+'/launch.json',dict(source=source,plan=pr,process=process,**o.stamp(gr,graph)))
            put(root,base+'/status.json',dict(stage='complete',failed=[],active=None,attempted=1,completed=[row]))
            terminal=put(root,base+'/terminal.json',dict(complete=True,completed=1,attempted=1,infrastructure_failed=0,active=None,
                automatic_retry=False,drain_reason=None,remaining_job_ids=[],process=process,**o.stamp(gr,graph)))
            actual=dict(actual_tool_exit_observed=True,natural_completion=True,returncode=0,process=process,plan=pr,terminal=terminal)
            ar=put(root,base+'/exec_completion.json',actual)
            with patch.object(o,'load_plan',return_value=(plan,pr,graph)),patch.object(o,'completed_case',return_value=row):
                observed=o._closed_owner(root,pr,ar);self.assertEqual(observed[3],[row])
                for field,value in (('returncode',False),('natural_completion',False),('actual_tool_exit_observed',False)):
                    changed=dict(actual);changed[field]=value;ref=put(root,ar['path'],changed)
                    with self.subTest(field=field),self.assertRaises(ValueError):o._closed_owner(root,pr,ref)
                ar=put(root,ar['path'],actual)
                with patch.object(o.readiness,'_not_live',return_value=False),self.assertRaises(ValueError):o._closed_owner(root,pr,ar)

    def test_lineage_and_VAL_cohort_have_distinct_schemas_after_Root_exit_only(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);graph=dict(physical_policy_sha256=o.PHYSICAL_SHA,control_graph_sha256='b'*64,policy_sha256='c'*64)
            gr=put(root,'artifacts/framework/multitask_r10/g/policy_graph.json',graph)
            owner={k:put(root,'synthetic/'+k+'.json',{'synthetic':True}) for k in ('launch','terminal','exec_completion')}
            predecessor={'exit':put(root,'synthetic/transition.json',{'synthetic':True})}
            for split in ('train','validation'):
                plan=dict(split=split,owner_runtime=o.BASE+'/'+split+'/fixture',graph=gr,predecessor=predecessor,
                    preserved_prefix_count=191 if split=='train' else 0,validation_bundle=None)
                pr=put(root,plan['owner_runtime']+'/plan.json',plan)
                with patch.object(o,'_closed_owner',return_value=(plan,pr,graph,[],owner)):
                    fn=o.assemble_cohort if split=='train' else o.assemble_validation_cohort
                    reference=fn(root,pr,owner['exec_completion']);value,_=o.read(root,reference)
                self.assertEqual(value['schema'],'r10_mixed_control_TRAIN192_lineage_v1174' if split=='train' else o.VAL_COHORT_SCHEMA)
                self.assertTrue(value['complete']);self.assertEqual(value['graph'],gr)
                if split=='train':self.assertFalse(value['case_graphs_relabelled']);self.assertEqual(value['preserved_prefix_count'],191)
                else:self.assertEqual(value['expected_cases'],945)


def serial_env():return o.serial.ENV_SHA


if __name__=='__main__':unittest.main()
