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

from scripts import run_parallel_repaired_owner_v1197_r10 as o


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
        self.assertEqual(len(first),6);self.assertTrue(all(a['phase']=='CPU_check' and a['gpu_slot'] is None for a in first))
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
        self.assertTrue(s.maybe_promote());self.assertEqual(s.limits,(4,2,6));self.assertFalse(s.maybe_promote())
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
        self.assertEqual(len(d.joined),6);self.assertEqual(len(d.finished),len(d.launched));self.assertFalse(r['complete'])
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


class ParallelHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.base=o.HANDOFF_BASE;self.owner=o.HANDOFF_PROCESS
        source=put(self.root,'scripts/run_parallel_repaired_owner_v1189_r10.py',{'synthetic_frozen_previous_owner':True})
        self.graph=dict(physical_policy_sha256=o.PHYSICAL_SHA,control_graph_sha256='b'*64,policy_sha256='c'*64,
            control=dict(components=dict(train_owner=source)))
        self.gr=put(self.root,'artifacts/framework/multitask_r10/synthetic_previous/policy_graph.json',self.graph)
        self.js=jobs(118)
        for i,j in enumerate(self.js,74):j['job_id']=f'j{i:04d}'
        self.plan=dict(schema='r10_parallel_fixed_TRAIN_suffix_plan_v1189',split='train',owner_runtime=self.base,graph=self.gr,
            global_expected_cases=192,preserved_prefix_count=74,expected_cases=118,jobs=self.js,**{k:self.graph[k] for k in o.readiness.HASHES})
        self.pr=put(self.root,self.base+'/plan.json',self.plan)
        self.session=put(self.root,self.base+'/resources/session.json',{'synthetic_session':True})
        self.lr=put(self.root,self.base+'/launch.json',dict(process=self.owner,plan=self.pr,resource_session=self.session,**o.stamp(self.gr,self.graph)))
        self.records=[dict(job_id=j['job_id'],task=j['task'],raw_episode_id=j['raw_episode_id'],method=j['method'],CPU_check={},native={})
            for j in (self.js[1],self.js[4])]
        expected=[('j0074','CPU_check'),('j0075','CPU_check'),('j0075','native'),('j0078','CPU_check'),('j0078','native')]
        for index,(jid,phase) in enumerate(expected):
            j=next(j for j in self.js if j['job_id']==jid)
            base=self.base+'/children/'+j['task']+'/'+j['raw_episode_id']+'/'+j['method']+'/'+phase+'/'
            outer=dict(pid=2147481000+index,start_ticks=1);inner=dict(pid=2147482000+index,start_ticks=1)
            put(self.root,base+'launch.json',dict(process=outer,resource_assignment=dict(job_id=jid,phase=phase)))
            put(self.root,base+'terminal.json',dict(process=outer,returncode=0))
            put(self.root,j['runtime']+'/'+phase+'/worker_sidecar.json',dict(process=inner,complete=True))
        self.status=dict(stage='drained',completed=self.records,failed=[],active=None,attempted=3,plan=self.pr,
            automatic_retry=False,drain_reason='administrative_STOP_drain',resource_session=self.session)
        self.sr=put(self.root,self.base+'/status.json',self.status)
        self.terminal=dict(complete=False,completed=2,infrastructure_failed=0,active=None,attempted=3,plan=self.pr,
            process=self.owner,automatic_retry=False,drain_reason='administrative_STOP_drain',resource_session=self.session,
            remaining_job_ids=[j['job_id'] for j in self.js if j['job_id'] not in ('j0075','j0078')],**o.stamp(self.gr,self.graph))
        self.tr=put(self.root,self.base+'/terminal.json',self.terminal)
        self.actual=dict(schema='r10_actual_controlled_parallel_owner_exit_v1197',actual_tool_exit_observed=True,
            controlled_drain=True,natural_completion=False,returncode=1,actual_exec_session=99802,Native_killed=False,
            all_child_processes_dead=True,completed_count=2,process=self.owner,plan=self.pr,status=self.sr,terminal=self.tr)
        self.ar=put(self.root,self.base+'/actual_exec_completion_handoff_v1197.json',self.actual)
        self.e=dict(schema=o.HANDOFF_SCHEMA,actual_tool_exit_observed=True,controlled_drain=True,natural_completion=False,
            returncode=1,native_children_all_dead=True,Native_killed=False,completed_count=2,process=self.owner,
            plan=self.pr,launch=self.lr,status=self.sr,terminal=self.tr,actual_tool_output=self.ar)
        self.er=put(self.root,o.HANDOFF_PATH,self.e)

    @contextmanager
    def fixture(self):
        def previous_case(root,plan,graph,job,cpu,native):return next(r for r in self.records if r['job_id']==job['job_id'])
        def previous_phase(root,plan,graph,job,refs,*,check):return o.read(root,refs['sidecar'])[0]
        previous=NS(completed_case=previous_case,validate_phase=previous_phase)
        with patch.object(o,'HANDOFF_PLAN_SHA',self.pr['sha256']),patch.object(o,'HANDOFF_GRAPH_SHA',self.gr['sha256']), \
                patch.object(o.graphs,'read_policy_graph',return_value=self.graph),patch.object(o,'_module',return_value=previous), \
                patch.object(o.readiness,'_not_live',return_value=True):yield

    def inspect(self):
        with self.fixture():return o.inspect_parallel_handoff(self.root,self.er)

    def rewrite_status(self):
        self.sr=put(self.root,self.base+'/status.json',self.status);self.e['status']=self.sr;self.actual['status']=self.sr
        self.e['actual_tool_output']=put(self.root,self.ar['path'],self.actual);self.er=put(self.root,o.HANDOFF_PATH,self.e)

    def test_noncontiguous_completed_set_preserves_original_graph_and_CPU_only_diagnostics(self):
        r=self.inspect()
        self.assertEqual(r['completed_job_ids'],['j0075','j0078']);self.assertEqual(r['preserved_count'],2)
        self.assertEqual(len(r['CPU_only_records']),1);self.assertEqual(r['old1194_graph'],self.gr)
        self.assertFalse(r['case_graphs_relabelled']);self.assertTrue(r['completed_records_reused_without_reexecution'])

    def test_live_previous_owner_blocks_before_any_scientific_case_read(self):
        with self.fixture(),patch.object(o.readiness,'_not_live',return_value=False), \
                patch.object(o,'_module',side_effect=AssertionError('old case reader must not run')):
            with self.assertRaises(ValueError):o.inspect_parallel_handoff(self.root,self.er)

    def test_planned_or_bool_exit_cannot_fake_real_controlled_handoff(self):
        original=deepcopy(self.e)
        for key,value in (('actual_tool_exit_observed',False),('returncode',True),('natural_completion',True),
                ('controlled_drain',False),('Native_killed',True),('completed_count',True)):
            changed=deepcopy(original);changed[key]=value;self.er=put(self.root,o.HANDOFF_PATH,changed)
            with self.subTest(key=key),self.assertRaises(ValueError):self.inspect()

    def test_duplicate_or_infrastructure_failure_cannot_be_relabelled_clean_segment(self):
        self.status['completed']=[self.records[0],self.records[0]];self.rewrite_status()
        with self.assertRaises(ValueError):self.inspect()
        self.status['completed']=self.records;self.status['failed']=[{'job_id':'j0075','phase':'native'}];self.rewrite_status()
        with self.assertRaises(ValueError):self.inspect()

    def test_wrong_source_or_TEST_role_rejected_before_read(self):
        with patch.object(o,'read',side_effect=AssertionError('role must be checked first')):
            with self.assertRaises(ValueError):o.inspect_parallel_handoff(self.root,dict(root='project',path='artifacts/framework/protocol/test_manifest.json',sha256='a'*64))

    def add_stop_abort(self):
        j=self.js[3];base=self.base+'/children/'+j['task']+'/'+j['raw_episode_id']+'/'+j['method']+'/CPU_check/'
        outer=dict(pid=2147481500,start_ticks=1)
        put(self.root,base+'launch.json',dict(process=outer,check_only=True,graph=self.gr,
            resource_assignment=dict(job_id=j['job_id'],phase='CPU_check')))
        put(self.root,base+'terminal.json',dict(process=outer,returncode=1))
        (self.root/base/'stderr.log').write_text('ValueError: parallel1187: parallel owner STOP blocks a new capability')
        (self.root/self.base/'STOP').write_text('synthetic administrative boundary')
        self.status['failed']=[dict(job_id=j['job_id'],phase='CPU_check',Native_failure_inferred=False)]
        self.status['attempted']=4;self.terminal['attempted']=4;self.terminal['infrastructure_failed']=1
        self.e['terminal']=put(self.root,self.base+'/terminal.json',self.terminal);self.actual['terminal']=self.e['terminal']
        self.rewrite_status();return j

    def test_pre_admission_STOP_abort_is_separate_from_accepted_science(self):
        self.add_stop_abort();r=self.inspect()
        self.assertEqual(r['preserved_count'],2);self.assertEqual(len(r['pre_admission_aborts']),1)
        self.assertFalse(r['pre_admission_aborts'][0]['Native_scientific_execution_started'])

    def test_STOP_cannot_excuse_failure_after_claim_or_actual_phase_entry(self):
        j=self.add_stop_abort();claim=self.base+'/resources/claims/'+j['job_id']+'_CPU_check/claim.json'
        put(self.root,claim,{'synthetic_actual_claim':True})
        with self.assertRaises(ValueError):self.inspect()
        (self.root/claim).unlink();put(self.root,j['runtime']+'/CPU_check/launch.json',{'synthetic_entry':True})
        with self.assertRaises(ValueError):self.inspect()


class QualificationWaveTests(unittest.TestCase):
    def test_all_stages_have_six_CPU_slots_and_waiting_wave_does_not_block_prefill(self):
        self.assertTrue(all(stage[2]==6 for stage in o.LEVELS))
        s=o.ParallelScheduler(jobs(30));s.level=1;s.states['j0000']='ready'
        a=s.dispatch();self.assertEqual(sum(x['phase']=='CPU_check' for x in a),6)
        self.assertFalse(any(x['phase']=='native' for x in a))

    def test_full_wave_requires_GPU_feasibility_and_drains_previous_stage_first(self):
        s=o.ParallelScheduler(jobs(30));s.level=1
        for j in s.jobs[:8]:s.states[j['job_id']]='ready'
        s.running['j0008']=dict(job=s.jobs[8],phase='native',slot=0,gpu_slot=0);s.states['j0008']='native'
        self.assertFalse(any(a['phase']=='native' for a in s.dispatch()))
        s.finish('j0008','native',dict(native_success=False,native_terminal_failure=True,motion_pass=False,native_interval=dict(start_monotonic=0,end_monotonic=1)))
        a=[x for x in s.dispatch() if x['phase']=='native']
        self.assertEqual(len(a),4);self.assertEqual(len(s.qualification_wave_jobs[1]),4)
        self.assertLessEqual(sum(x['gpu_slot'] is not None for x in a),2)

    def test_ready24_all_Phys_does_not_starve_queued_geometry_or_claim_qualification(self):
        js=jobs(30)
        for j in js[:24]:j['method']='physmani'
        for j in js[24:]:j['method']='cv_geometry'
        s=o.ParallelScheduler(js);s.level=2
        for j in js[:24]:s.states[j['job_id']]='ready'
        a=s.dispatch();native=[x for x in a if x['phase']=='native'];cpu=[x for x in a if x['phase']=='CPU_check']
        self.assertEqual(len(native),3);self.assertEqual(len(cpu),3)
        self.assertNotIn(2,s.qualification_wave_jobs)
        self.assertEqual(s.wave_events[-1]['reason'],'ready_backpressure_relief')
        self.assertFalse(s.maybe_promote())

    def test_tail_with_only_Phys_eventually_drains_without_waiting_for_nonexistent_geometry(self):
        js=jobs(23)
        for j in js:j['method']='physmani'
        s=o.ParallelScheduler(js);s.level=2
        for j in js:s.states[j['job_id']]='ready'
        while len(s.completed)<len(js):
            a=s.dispatch();self.assertTrue(a)
            for item in a:
                self.assertEqual(item['phase'],'native')
                s.finish(item['job']['job_id'],'native',dict(native_success=False,native_terminal_failure=False,motion_pass=False,
                    native_interval=dict(start_monotonic=0,end_monotonic=1)))
        self.assertNotIn(2,s.qualification_wave_jobs);self.assertFalse(s.maybe_promote())

    def test_slow_CPU90_short_physical45_can_qualify_full2_4_8_waves_without_success_gate(self):
        c=Clock();d=MockDriver(c)
        def launch(a):
            d.launched.append(deepcopy(a));duration=90 if a['phase']=='CPU_check' else 137
            return dict(a=a,start=c(),end=c()+duration)
        original=d.collect
        def collect(h,code):
            record=original(h,code)
            if h['a']['phase']=='native':record['native_interval']=dict(start_monotonic=h['start']+83,end_monotonic=h['end']-9)
            return record
        d.launch=launch;d.collect=collect
        r=o.run_scheduler({'jobs':jobs(60)},d,sampler=lambda p:sample(c),clock=c,sleep=c.sleep)
        self.assertTrue(r['complete']);self.assertEqual([p['to_level'] for p in r['promotions']],[1,2,3])
        self.assertEqual([len(r['qualification_wave_jobs'][i]) for i in range(3)],[2,4,8])
        self.assertTrue(all(not row['native_success'] for row in r['records']))

    def test_later_fresh_completed_overlap_can_qualify_after_first_wave_missed_overlap(self):
        s=o.ParallelScheduler(jobs(8));s.level=1
        for j in s.jobs:s.states[j['job_id']]='ready'
        first=[a for a in s.dispatch() if a['phase']=='native'];self.assertEqual(len(first),4)
        for index,a in enumerate(first):
            s.finish(a['job']['job_id'],'native',dict(native_success=False,native_terminal_failure=False,motion_pass=False,
                native_interval=dict(start_monotonic=index*2,end_monotonic=index*2+1)))
        for t in (2.,3.,4.):s.sample(dict(at=t,CPU_percent=20,RAM_percent=20,VRAM_percent=20,GPU_percent=20))
        self.assertFalse(s.maybe_promote())
        later=[a for a in s.dispatch() if a['phase']=='native'];self.assertEqual(len(later),4)
        self.assertFalse({a['job']['job_id'] for a in first}&{a['job']['job_id'] for a in later})
        for a in later:
            s.finish(a['job']['job_id'],'native',dict(native_success=False,native_terminal_failure=False,motion_pass=False,
                native_interval=dict(start_monotonic=10,end_monotonic=20)))
        for t in (11.,12.,13.):s.sample(dict(at=t,CPU_percent=20,RAM_percent=20,VRAM_percent=20,GPU_percent=20))
        self.assertTrue(s.maybe_promote());self.assertEqual(s.level,2)


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
        self.superseded_exit=put(self.root,o.SUPERSEDED_EXIT_PATH,{'synthetic_fixture_not_actual_exit':True})
        self.superseded=dict(exit=self.superseded_exit,completed_native_cases=0)
        self.retired_exit=put(self.root,o.RETIRED_EXIT_PATH,{'synthetic_retired_partial_exit':True})
        self.retired=dict(exit=self.retired_exit,attempted_Native_cases=2)
        self.handoff_exit=put(self.root,o.HANDOFF_PATH,{'synthetic_actual_handoff':True})
        self.handoff=dict(exit=self.handoff_exit,old1194_graph={'root':'project','path':'old1194.json','sha256':'e'*64},
            completed_job_ids=['j0007','j0011'],completed_records=[{'job_id':'j0007'},{'job_id':'j0011'}],preserved_count=2)

    def review(self,split='train',count=185,namespace='fixture',bundle=None):
        return put(self.root,'artifacts/framework/multitask_r10/synthetic_review/'+split+'.json',dict(schema=o.REVIEW_SCHEMA,
            reviewer='root',review_complete=True,release_fixed_cohort=True,split=split,namespace=namespace,expected_cases=count,
            automatic_retry=False,source_outcome_filtering=False,predecessor_exit=self.exit,validation_bundle=bundle,
            superseded_CPU_batch_exit=self.superseded_exit,retired_partial_batch_exit=self.retired_exit,
            parallel_handoff=self.handoff_exit,preserved_parallel_count=2,accepted_parallel_cases_reexecuted=False,
            historical_Native_reexecution_disclosed=True,repeated_development_Native_cases=2,
            source_outcomes_used_for_rerun_or_version_selection=False,**o.stamp(self.gr,self.graph)))

    @contextmanager
    def fake_graph(self):
        with patch.object(o,'graph_for_owner',return_value=(self.graph,self.gr)),patch.object(o,'inspect_predecessor',return_value=self.predecessor), \
                patch.object(o,'inspect_superseded_CPU_batch',return_value=self.superseded), \
                patch.object(o,'inspect_retired_partial_batch',return_value=self.retired), \
                patch.object(o,'inspect_parallel_handoff',return_value=self.handoff), \
                patch.object(o,'_hardware_snapshot',return_value={k:self.affinity[k] for k in ('allowed_logical_cpus','core_groups')}), \
                patch.object(o,'physical_equivalence',return_value={'synthetic_fixture':True}):yield

    def test_default_inert_and_TEST_refused_before_read(self):
        with patch.object(o.subprocess,'Popen',side_effect=AssertionError('no process')),patch('sys.stdout',new_callable=io.StringIO):r=o.main([])
        self.assertFalse(r['enabled'])
        with patch.object(o.graphs,'read_policy_graph',side_effect=AssertionError('must refuse first')),self.assertRaises(ValueError):o.graph_for_owner(self.root,'never','test')

    def test_preserve64rows_global_ids_and_only_unstarted_suffix_fresh_outputs(self):
        with self.fake_graph():
            p=o.prepare_parallel_plan(self.root,self.gr,superseded_CPU_exit_ref=self.superseded_exit,retired_partial_batch_exit_ref=self.retired_exit,parallel_handoff_ref=self.handoff_exit,namespace='fixture',review_ref=self.review(),predecessor_exit_ref=self.exit)
            plan,_,_=o.load_plan(self.root,p)
        self.assertEqual(len(plan['sources']),64);self.assertEqual(plan['global_expected_cases'],192);self.assertEqual(plan['expected_cases'],185)
        self.assertEqual(plan['jobs'][0]['job_id'],'j0005');self.assertEqual(plan['jobs'][-1]['job_id'],'j0191')
        self.assertTrue(all(set(j)==o.JOB_KEYS for j in plan['jobs']));self.assertFalse(plan['case_graphs_relabelled'])
        self.assertTrue(all(not (self.root/j['out']).exists() for j in plan['jobs']))
        self.assertEqual(plan['preserved_parallel_count'],2)
        self.assertFalse({'j0007','j0011'}&{j['job_id'] for j in plan['jobs']})
        self.assertEqual(len(plan['jobs'])+plan['preserved_prefix_count']+plan['preserved_parallel_count'],192)
        with self.fake_graph(),self.assertRaises(ValueError):o.prepare_parallel_plan(self.root,self.gr,superseded_CPU_exit_ref=self.superseded_exit,retired_partial_batch_exit_ref=self.retired_exit,parallel_handoff_ref=self.handoff_exit,namespace='fixture',review_ref=self.review(),predecessor_exit_ref=self.exit)

    def test_missing_actual_old_exit_blocks_plan_before_fork(self):
        with self.fake_graph(),patch.object(o,'inspect_predecessor',side_effect=ValueError('old live')),patch.object(o.subprocess,'Popen',side_effect=AssertionError('no fork')):
            with self.assertRaisesRegex(ValueError,'old live'):o.prepare_parallel_plan(self.root,self.gr,superseded_CPU_exit_ref=self.superseded_exit,retired_partial_batch_exit_ref=self.retired_exit,parallel_handoff_ref=self.handoff_exit,namespace='fixture',review_ref=self.review(),predecessor_exit_ref=self.exit)

    def test_load_rejects_retry_reordered_suffix_and_profile_changes(self):
        with self.fake_graph():
            p=o.prepare_parallel_plan(self.root,self.gr,superseded_CPU_exit_ref=self.superseded_exit,retired_partial_batch_exit_ref=self.retired_exit,parallel_handoff_ref=self.handoff_exit,namespace='fixture',review_ref=self.review(),predecessor_exit_ref=self.exit);plan,_=o.read(self.root,p)
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
            p=o.prepare_parallel_plan(self.root,self.gr,superseded_CPU_exit_ref=self.superseded_exit,retired_partial_batch_exit_ref=self.retired_exit,parallel_handoff_ref=self.handoff_exit,split='validation',namespace='fixture',review_ref=self.review('validation',945,bundle=bundle),bundle_ref=bundle)
            plan,_,_=o.load_plan(self.root,p,execute=True)
        self.assertEqual(gate.call_count,2);self.assertEqual(len(plan['jobs']),945);self.assertEqual(len(plan['sources']),315)
        self.assertEqual(plan['schema'],o.VAL_PLAN_SCHEMA);self.assertIsNone(plan['predecessor']);self.assertEqual(plan['preserved_prefix_count'],0)
        self.assertTrue(all('/validation/' in j['out'] and 'demo_only' not in j['out'] for j in plan['jobs']))

    def test_wrong_plan_pin_refuses_before_guard(self):
        with self.fake_graph():p=o.prepare_parallel_plan(self.root,self.gr,superseded_CPU_exit_ref=self.superseded_exit,retired_partial_batch_exit_ref=self.retired_exit,parallel_handoff_ref=self.handoff_exit,namespace='fixture',review_ref=self.review(),predecessor_exit_ref=self.exit)
        with patch.object(o,'_module',side_effect=AssertionError('no guard lease')),self.assertRaises(ValueError):o.execute_owner(self.root,p,'wrong')

    def test_load_plan_rechecks_Root_release_and_exact_thread_metadata(self):
        with self.fake_graph():
            p=o.prepare_parallel_plan(self.root,self.gr,superseded_CPU_exit_ref=self.superseded_exit,retired_partial_batch_exit_ref=self.retired_exit,parallel_handoff_ref=self.handoff_exit,namespace='fixture',review_ref=self.review(),predecessor_exit_ref=self.exit)
            plan,_=o.read(self.root,p);review,_=o.read(self.root,plan['review']);review['release_fixed_cohort']=False
            plan['review']=put(self.root,plan['review']['path'],review);changed=put(self.root,p['path'],plan)
            with self.assertRaises(ValueError):o.load_plan(self.root,changed)
        self.assertTrue(o.THREAD_CONTROL['launcher_defaults_preserved'])
        self.assertEqual(o.THREAD_CONTROL['effective_launcher_environment'],dict(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))

    def test_superseded_CPU_failure_is_mandatory_at_prepare_load_and_review(self):
        with self.fake_graph(),patch.object(o,'inspect_superseded_CPU_batch',side_effect=ValueError('missing actual CPU exit')):
            with self.assertRaisesRegex(ValueError,'missing actual CPU exit'):
                o.prepare_parallel_plan(self.root,self.gr,superseded_CPU_exit_ref=self.superseded_exit,retired_partial_batch_exit_ref=self.retired_exit,parallel_handoff_ref=self.handoff_exit,
                    namespace='fixture',review_ref=self.review(),predecessor_exit_ref=self.exit)
        with self.fake_graph():
            p=o.prepare_parallel_plan(self.root,self.gr,superseded_CPU_exit_ref=self.superseded_exit,retired_partial_batch_exit_ref=self.retired_exit,parallel_handoff_ref=self.handoff_exit,
                namespace='fixture',review_ref=self.review(),predecessor_exit_ref=self.exit)
            with patch.object(o,'inspect_superseded_CPU_batch',side_effect=ValueError('changed CPU exit')):
                with self.assertRaisesRegex(ValueError,'changed CPU exit'):o.load_plan(self.root,p)
            plan,_=o.read(self.root,p);review,_=o.read(self.root,plan['review'])
            review['superseded_CPU_batch_exit']=dict(self.superseded_exit,sha256='f'*64)
            plan['review']=put(self.root,plan['review']['path'],review);changed=put(self.root,p['path'],plan)
            with self.assertRaises(ValueError):o.load_plan(self.root,changed)

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

    def test_retired_partial_exit_is_mandatory_and_replay_disclosure_cannot_be_removed(self):
        with self.fake_graph(),patch.object(o,'inspect_retired_partial_batch',side_effect=ValueError('missing retired batch')):
            with self.assertRaisesRegex(ValueError,'missing retired batch'):
                o.prepare_parallel_plan(self.root,self.gr,superseded_CPU_exit_ref=self.superseded_exit,
                    retired_partial_batch_exit_ref=self.retired_exit,parallel_handoff_ref=self.handoff_exit,namespace='fixture',review_ref=self.review(),predecessor_exit_ref=self.exit)
        with self.fake_graph():
            p=o.prepare_parallel_plan(self.root,self.gr,superseded_CPU_exit_ref=self.superseded_exit,
                retired_partial_batch_exit_ref=self.retired_exit,parallel_handoff_ref=self.handoff_exit,namespace='fixture',review_ref=self.review(),predecessor_exit_ref=self.exit)
            with patch.object(o,'inspect_retired_partial_batch',side_effect=ValueError('retired evidence changed')):
                with self.assertRaisesRegex(ValueError,'retired evidence changed'):o.load_plan(self.root,p)
            plan,_=o.read(self.root,p);original=deepcopy(plan)
            for field,value in (('historical_Native_reexecution_disclosed',False),('repeated_development_Native_cases',False),
                    ('source_outcomes_used_for_rerun_or_version_selection',True),('automatic_retry_scope','all_history_no_repeats')):
                changed=deepcopy(original);changed[field]=value
                with self.subTest(field=field),self.assertRaises(ValueError):o.load_plan(self.root,put(self.root,p['path'],changed))

    def test_actual_parallel_handoff_is_mandatory_and_cannot_change_after_plan_review(self):
        with self.fake_graph(),patch.object(o,'inspect_parallel_handoff',side_effect=ValueError('previous owner still running')):
            with self.assertRaisesRegex(ValueError,'previous owner still running'):
                o.prepare_parallel_plan(self.root,self.gr,superseded_CPU_exit_ref=self.superseded_exit,
                    retired_partial_batch_exit_ref=self.retired_exit,parallel_handoff_ref=self.handoff_exit,
                    namespace='fixture',review_ref=self.review(),predecessor_exit_ref=self.exit)
        with self.fake_graph():
            p=o.prepare_parallel_plan(self.root,self.gr,superseded_CPU_exit_ref=self.superseded_exit,
                retired_partial_batch_exit_ref=self.retired_exit,parallel_handoff_ref=self.handoff_exit,
                namespace='fixture',review_ref=self.review(),predecessor_exit_ref=self.exit)
            plan,_=o.read(self.root,p);original=deepcopy(plan)
            changed=deepcopy(original);changed['preserved_parallel_segment']['completed_job_ids']=['j0007']
            with self.assertRaises(ValueError):o.load_plan(self.root,put(self.root,p['path'],changed))
            changed=deepcopy(original);review,_=o.read(self.root,changed['review']);review['parallel_handoff']=dict(self.handoff_exit,sha256='f'*64)
            changed['review']=put(self.root,changed['review']['path'],review)
            with self.assertRaises(ValueError):o.load_plan(self.root,put(self.root,p['path'],changed))

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
        guard=put(self.root,o.GUARD,{'synthetic_guard':True})
        parent=put(self.root,'our_mani/multitask/post_completed_TEST_resource_guard_v1106.py',{'synthetic_parent_guard':True})
        auditor=put(self.root,'our_mani/multitask/parallel_repaired_readiness_v1198.py',{'synthetic_auditor':True})
        self.g['control']=dict(components=dict(resource_guard=guard,train_auditor=auditor),sources_map=dict(parent=parent,guard=guard))
        affinity=affinity_fixture(self.root,self.g)
        self.gr=put(self.root,'artifacts/framework/multitask_r10/synthetic_graph/policy_graph.json',self.g)
        base=o.BASE+'/train/fixture';self.plan=dict(split='train',owner_runtime=base,graph=self.gr,profile=o.PROFILE,jobs=[self.job],CPU_affinity=affinity)
        self.owner=dict(pid=2147482000,start_ticks=1);self.outer=dict(pid=2147482001,start_ticks=1);self.inner=dict(pid=2147482002,start_ticks=1)
        from tests.test_parallel_repaired_readiness_v1198 import make_session
        self.session=make_session(self.root,base,self.gr,self.g,owner=self.owner,jobs=[self.job])
        put(self.root,base+'/launch.json',dict(process=self.owner,resource_session=self.session))
        a=dict(job_id=self.job['job_id'],phase='native',slot=0,gpu_slot=None)
        self.claim=dict(schema='r10_once_only_parallel_resource_claim_v1195',session=self.session,phase='native',process=self.inner,
            slot=0,gpu_slot=None,automatic_retry=False,test_only=False,job=self.job)
        self.claim_path=base+'/resources/claims/j0000_native/claim.json';cr=put(self.root,self.claim_path,self.claim)
        self.resource=dict(schema='r10_parallel_child_resource_receipt_v1195',body_completed=True,restored=True,leases_removed=True,
            errors=[],test_only=False,session=self.session,process=self.inner,actual_parent=self.owner,phase='native',job_id='j0000',
            slot=0,gpu_slot=None,parent_global_locks_verified=True,claim=cr,purpose='child_slot',Native_authorized=True,
            child_empty_inventory_claimed=False)
        saved_session=json.loads((self.root/self.session['path']).read_text())
        from tests.test_parallel_repaired_readiness_v1198 import make_kernel_lock_evidence
        for label,nonce in (('admission','a'*64),('final','b'*64)):
            self.resource['parent_broker_'+label]=dict(schema='r10_parent_lock_broker_verification_v1195',
                session=self.session,claim=cr,owner=self.owner,client=self.inner,boot_id=saved_session['boot_id'],nonce=nonce,
                checked_utc='2026-10-24T00:00:00+00:00',parent_FD_verified=True,kernel_FLOCK_verified=True,
                client_direct_parent_FD_read=False,server_authenticated_by_SO_PEERCRED=True,client_authenticated_by_SO_PEERCRED=True,
                server_uid=1000,server_gid=1000,client_uid=1000,client_gid=1000,lock_proof=saved_session['parent_global_locks'],
                guard=saved_session['guard'],parent_guard=saved_session['parent_guard'],job_id=self.job['job_id'],phase='native',slot=0,gpu_slot=None)
            self.resource['parent_broker_'+label].update(make_kernel_lock_evidence(saved_session))
        put(self.root,str(Path(self.claim_path).parent/'terminal.json'),self.resource)
        leaf=base+'/children/'+self.job['task']+'/'+self.job['raw_episode_id']+'/'+self.job['method']+'/native'
        self.sidecar=dict(schema='r10_parallel_graph_bound_repaired_policy_case_v1196',complete=True,process=self.inner,
            actual_worker=worker,actual_preparer=preparer,resource_scope_receipt=self.resource,
            execution=dict(complete=True,native_started=True,actual_split='train',mode='graph_bound_native_TRAIN',native_false_preserved=True,
                actual_native_interval=dict(start_monotonic=1.,end_monotonic=2.)),**o.stamp(self.gr,self.g),
            frozen_OS_environment=dict(complete=True,outer_wrapper_pid=self.outer['pid'],outer_wrapper_start_ticks=1,
                observer_pid=self.inner['pid'],observer_start_ticks=1))
        self.sidecar['resource_proof']=deepcopy(self.resource)
        self.sidecar['resource_proof'].update(body_completed=False,restored=False)
        self.sidecar['resource_proof'].pop('parent_broker_final')
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

    def test_both_live_parent_proofs_bind_credentials_claim_guard_and_actual_slot(self):
        original=deepcopy(self.sidecar)
        cases=(('parent_FD_verified',False),('kernel_FLOCK_verified',False),('client_direct_parent_FD_read',True),
            ('server_authenticated_by_SO_PEERCRED',False),('client_authenticated_by_SO_PEERCRED',False),
            ('client',self.outer),('owner',self.inner),('boot_id','wrong-boot'),('nonce','a'),
            ('checked_utc','2026-10-24'),('server_uid',1001),('client_gid',1001),
            ('guard',dict(root='project',path='foreign.py',sha256='f'*64)),
            ('claim',dict(root='project',path='foreign.json',sha256='f'*64)),
            ('job_id','j0001'),('phase','CPU_check'),('slot',True),('lock_proof',[]))
        for label in ('parent_broker_admission','parent_broker_final'):
            for field,value in cases:
                changed=deepcopy(original);resource=changed['resource_scope_receipt'];resource[label][field]=value
                if label=='parent_broker_admission':changed['resource_proof'][label]=deepcopy(resource[label])
                put(self.root,str(Path(self.claim_path).parent/'terminal.json'),resource)
                self.refs['sidecar']=put(self.root,self.refs['sidecar']['path'],changed)
                with self.subTest(label=label,field=field),self.assertRaises(ValueError):
                    o.validate_phase(self.root,self.plan,self.g,self.job,self.refs,check=False)

    def test_no_replayed_nonce_missing_final_or_final_at_admission(self):
        original=deepcopy(self.sidecar)
        for kind in ('nonce_reuse','missing_final','premature_final','changed_initial'):
            changed=deepcopy(original);resource=changed['resource_scope_receipt']
            if kind=='nonce_reuse':resource['parent_broker_final']['nonce']=resource['parent_broker_admission']['nonce']
            elif kind=='missing_final':resource.pop('parent_broker_final')
            elif kind=='premature_final':changed['resource_proof']['parent_broker_final']=deepcopy(resource['parent_broker_final'])
            else:changed['resource_proof']['parent_broker_admission']['nonce']='c'*64
            put(self.root,str(Path(self.claim_path).parent/'terminal.json'),resource)
            self.refs['sidecar']=put(self.root,self.refs['sidecar']['path'],changed)
            with self.subTest(kind=kind),self.assertRaises(ValueError):o.validate_phase(self.root,self.plan,self.g,self.job,self.refs,check=False)

    def test_per_FD_kernel_and_independent_conflict_evidence_are_required_in_both_checks(self):
        original=deepcopy(self.sidecar)
        for label in ('parent_broker_admission','parent_broker_final'):
            for kind in ('missing_FD','raw_owner','stat_false','inode','missing_conflicts','not_conflict','bad_errno'):
                changed=deepcopy(original);proof=changed['resource_scope_receipt'][label]
                if kind=='missing_FD':proof['per_FD_lock_evidence']=[]
                elif kind=='raw_owner':proof['per_FD_lock_evidence'][0]['raw']=proof['per_FD_lock_evidence'][0]['raw'].replace(str(self.owner['pid']),'1')
                elif kind=='stat_false':proof['per_FD_lock_evidence'][0]['FD_stat_before_after_verified']=False
                elif kind=='inode':proof['per_FD_lock_evidence'][0]['inode']+=1
                elif kind=='missing_conflicts':proof['independent_conflict_evidence']=[]
                elif kind=='not_conflict':proof['independent_conflict_evidence'][0]['confirmed']=False
                else:proof['independent_conflict_evidence'][0]['conflict_errno']=0
                if label=='parent_broker_admission':changed['resource_proof'][label]=deepcopy(proof)
                put(self.root,str(Path(self.claim_path).parent/'terminal.json'),changed['resource_scope_receipt'])
                self.refs['sidecar']=put(self.root,self.refs['sidecar']['path'],changed)
                with self.subTest(label=label,kind=kind),self.assertRaises(ValueError):
                    o.validate_phase(self.root,self.plan,self.g,self.job,self.refs,check=False)


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
            for split in ('train','evaluation'):route[split].update(worker=ptr('new1196.py'),preparer=ptr('newprep1196.py'),case_schema='new1196')
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


class SupersededCPUFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.base=o.SUPERSEDED_BASE;self.process=o.SUPERSEDED_PROCESS
        self.gr=put(self.root,'artifacts/framework/multitask_r10/synthetic_failed/policy_graph.json',{'synthetic':True})
        self.jobs=jobs(118)
        for i,j in enumerate(self.jobs,74):j['job_id']=f'j{i:04d}'
        self.plan=dict(schema='r10_parallel_fixed_TRAIN_suffix_plan_v1173',graph=self.gr,split='train',owner_runtime=self.base,
            preserved_prefix_count=74,global_expected_cases=192,expected_cases=118,jobs=self.jobs,automatic_retry=False,
            source_outcome_filtering=False,physical_policy_sha256=o.PHYSICAL_SHA)
        self.pr=put(self.root,self.base+'/plan.json',self.plan)
        self.lr=put(self.root,self.base+'/launch.json',dict(process=self.process,plan=self.pr))
        self.status=dict(stage='drained',active=None,attempted=2,automatic_retry=False,drain_reason='infrastructure_failure_drain',
            plan=self.pr,completed=[],failed=[dict(job_id=j,phase='CPU_check',Native_failure_inferred=False) for j in ('j0075','j0076')])
        self.sr=put(self.root,self.base+'/status.json',self.status)
        self.terminal=dict(active=None,attempted=2,automatic_retry=False,drain_reason='infrastructure_failure_drain',plan=self.pr,
            complete=False,completed=0,infrastructure_failed=2,process=self.process,remaining_job_ids=[j['job_id'] for j in self.jobs])
        self.tr=put(self.root,self.base+'/terminal.json',self.terminal)
        phases=[]
        for i,job in enumerate(self.jobs[1:3]):
            base=self.base+'/children/'+job['task']+'/'+job['raw_episode_id']+'/'+job['method']+'/CPU_check/'
            child=dict(pid=2147483000+i,start_ticks=1)
            launch=put(self.root,base+'launch.json',dict(process=child,check_only=True,graph=self.gr,
                resource_assignment=dict(job_id=job['job_id'],phase='CPU_check'),**{k:job[k] for k in ('task','raw_episode_id','method')}))
            terminal=put(self.root,base+'terminal.json',dict(process=child,returncode=1,automatic_retry=False))
            stderr=self.root/base/'stderr.log';stderr.write_text('PermissionError: denied /proc/32008/fd/4')
            phases.append(dict(launch=launch,terminal=terminal,process=child,stderr=o.pointer(self.root,base+'stderr.log')))
        self.actual=dict(schema='r10_actual_unqualified_CPU_parallel_owner_exit_v1178',actual_tool_exit_observed=True,
            natural_completion=False,returncode=1,actual_exec_session=28165,Native_started=False,Native_killed=False,
            owner_dead=True,all_children_dead=True,source_outcomes_created=False,scientific_qualification_granted=False,
            completed_native_cases=0,attempted_CPU_cases=2,automatic_retry=False,graph=self.gr,preserved_original_prefix_count=74,
            process=self.process,plan=self.pr,launch=self.lr,status=self.sr,terminal=self.tr,CPU_phases=phases)
        self.ar=put(self.root,self.base+'/failed_CPU_exec_completion_v1181.json',self.actual)
        self.e=dict(schema=o.SUPERSEDED_EXIT_SCHEMA,actual_tool_exit_observed=True,natural_completion=False,returncode=1,
            Native_started=False,all_child_processes_dead=True,completed=0,attempted=2,process=self.process,
            plan=self.pr,launch=self.lr,status=self.sr,terminal=self.tr,actual_tool_output=self.ar)
        self.er=put(self.root,o.SUPERSEDED_EXIT_PATH,self.e)

    @contextmanager
    def pins(self):
        with patch.object(o,'SUPERSEDED_PLAN_SHA',self.pr['sha256']),patch.object(o,'SUPERSEDED_GRAPH_SHA',self.gr['sha256']):yield

    def inspect(self):
        with self.pins():return o.inspect_superseded_CPU_batch(self.root,self.er)

    def rewrite_actual(self):
        self.e['actual_tool_output']=put(self.root,self.ar['path'],self.actual)
        self.er=put(self.root,o.SUPERSEDED_EXIT_PATH,self.e)

    def test_exact_CPU_only_failed_control_exit_preserves74_and_grants_no_qualification(self):
        result=self.inspect()
        self.assertEqual(result['preserved_original_prefix_count'],74)
        self.assertEqual(result['completed_native_cases'],0);self.assertFalse(result['Native_retried'])
        self.assertFalse(result['scientific_qualification_granted'])

    def test_foreign_VAL_TEST_or_old_transition_role_refused_before_read(self):
        for name in ('artifacts/framework/protocol/validation_manifest.json','artifacts/framework/protocol/test_manifest.json',o.TRANSITION_PATH):
            with patch.object(o,'read',side_effect=AssertionError('must reject before metadata contents')):
                with self.subTest(name=name),self.assertRaises(ValueError):
                    o.inspect_superseded_CPU_batch(self.root,dict(root='project',path=name,sha256='a'*64))

    def test_false_actual_exit_bool_returncode_native_or_completed_cannot_be_superseded(self):
        original=deepcopy(self.e)
        for key,value in (('actual_tool_exit_observed',False),('returncode',True),('natural_completion',True),
                ('Native_started',True),('completed',1),('attempted',True),('all_child_processes_dead',False)):
            self.e=deepcopy(original);self.e[key]=value;self.er=put(self.root,o.SUPERSEDED_EXIT_PATH,self.e)
            with self.subTest(key=key),self.assertRaises(ValueError):self.inspect()

    def test_actual_session_or_source_outcome_or_child_returncode_drift_refused(self):
        original=deepcopy(self.actual)
        for key,value in (('actual_exec_session',28166),('source_outcomes_created',True),('Native_started',True),
                ('scientific_qualification_granted',True),('completed_native_cases',False)):
            self.actual=deepcopy(original);self.actual[key]=value;self.rewrite_actual()
            with self.subTest(key=key),self.assertRaises(ValueError):self.inspect()
        self.actual=deepcopy(original)
        phase=self.actual['CPU_phases'][0];t=json.loads((self.root/phase['terminal']['path']).read_text());t['returncode']=0
        phase['terminal']=put(self.root,phase['terminal']['path'],t);self.rewrite_actual()
        with self.assertRaises(ValueError):self.inspect()

    def test_live_owner_or_child_and_unlisted_native_launch_refused(self):
        with patch.object(o.readiness,'_not_live',return_value=False),self.assertRaises(ValueError):self.inspect()
        put(self.root,self.base+'/children/unexpected/native/launch.json',{'synthetic':True})
        with self.assertRaises(ValueError):self.inspect()

    def test_scientific_case_creation_or_native_runtime_cannot_be_hidden(self):
        put(self.root,self.jobs[0]['out']+'/case.json',{'synthetic':True})
        with self.assertRaises(ValueError):self.inspect()

    def test_missing_actual_tool_output_and_altered_status_are_not_empty_retry_permission(self):
        self.e['actual_tool_output']=self.lr;self.er=put(self.root,o.SUPERSEDED_EXIT_PATH,self.e)
        with self.assertRaises(ValueError):self.inspect()
        self.e['actual_tool_output']=self.ar
        self.status['failed'][0]['phase']='native'
        self.e['status']=put(self.root,self.sr['path'],self.status);self.er=put(self.root,o.SUPERSEDED_EXIT_PATH,self.e)
        with self.assertRaises(ValueError):self.inspect()


class RetiredPartialBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.base=o.RETIRED_BASE;self.owner=o.RETIRED_PROCESS
        self.gr=put(self.root,'artifacts/framework/multitask_r10/synthetic_retired/policy_graph.json',{'synthetic':True})
        js=jobs(118)
        for i,j in enumerate(js,74):j['job_id']=f'j{i:04d}'
        self.jobs={j['job_id']:j for j in js}
        self.plan=dict(schema='r10_parallel_fixed_TRAIN_suffix_plan_v1181',split='train',owner_runtime=self.base,
            graph=self.gr,preserved_prefix_count=74,global_expected_cases=192,expected_cases=118,jobs=js,
            automatic_retry=False,source_outcome_filtering=False,physical_policy_sha256=o.PHYSICAL_SHA)
        self.pr=put(self.root,self.base+'/plan.json',self.plan)
        self.lr=put(self.root,self.base+'/launch.json',dict(process=self.owner,plan=self.pr,graph=self.gr))
        self.completed=[dict(job_id='j0076',synthetic_original_development_record=True)]
        self.status=dict(stage='drained',active=None,attempted=4,completed=self.completed,automatic_retry=False,
            plan=self.pr,drain_reason='infrastructure_failure_drain',failed=[dict(job_id=j,phase=p,Native_failure_inferred=False)
                for j,p in (('j0074','CPU_check'),('j0078','CPU_check'),('j0075','native'))])
        self.sr=put(self.root,self.base+'/status.json',self.status)
        self.terminal=dict(complete=False,active=None,attempted=4,completed=1,infrastructure_failed=3,process=self.owner,
            plan=self.pr,automatic_retry=False,drain_reason='infrastructure_failure_drain',
            remaining_job_ids=[j['job_id'] for j in js if j['job_id']!='j0076'])
        self.tr=put(self.root,self.base+'/terminal.json',self.terminal)
        phases=[]
        expected=[('j0074','CPU_check',1),('j0075','CPU_check',0),('j0076','CPU_check',0),
            ('j0078','CPU_check',1),('j0075','native',1),('j0076','native',0)]
        for i,(jid,phase,code) in enumerate(expected):
            j=self.jobs[jid];outer=dict(pid=2147481000+i,start_ticks=1);inner=dict(pid=2147482000+i,start_ticks=1)
            base=self.base+'/children/'+j['task']+'/'+j['raw_episode_id']+'/'+j['method']+'/'+phase+'/'
            launch=put(self.root,base+'launch.json',dict(process=outer,check_only=phase=='CPU_check',graph=self.gr,
                resource_assignment=dict(job_id=jid,phase=phase),**{k:j[k] for k in ('task','raw_episode_id','method')}))
            terminal=put(self.root,base+'terminal.json',dict(process=outer,returncode=code,automatic_retry=False))
            err=self.root/base/'stderr.log';err.write_text('parent no longer holds its actual global FLOCK' if code else '')
            phases.append(dict(launch=launch,terminal=terminal,stderr=o.pointer(self.root,base+'stderr.log'),
                process=outer,returncode=code,check_only=phase=='CPU_check'))
            claim_dir=self.base+'/resources/claims/'+jid+'_'+phase+'/'
            put(self.root,claim_dir+'claim.json',dict(schema='r10_once_only_parallel_resource_claim_v1179',process=inner,phase=phase,job=j))
            put(self.root,claim_dir+'terminal.json',dict(process=inner,restored=True,leases_removed=True))
            put(self.root,j['runtime']+'/'+phase+'/launch.json',dict(process=inner))
            if code==0:put(self.root,j['runtime']+'/'+phase+'/worker_sidecar.json',dict(process=inner,complete=True))
            if phase=='native':
                witness=dict(graph=self.gr,physical_policy_sha256=o.PHYSICAL_SHA,configuration=j['configuration'],source_row_sha256=j['source_row_sha256'])
                identity=dict(graph_sha256=self.gr['sha256'],config_sha256=j['configuration']['sha256'],source_row_sha256=j['source_row_sha256'])
                identity['sha256']=o.readiness.digest(identity)
                put(self.root,j['out']+'/case.json',dict(process_id=inner['pid'],task=j['task'],method=j['method'],split='train',
                    source_raw_episode_id=j['raw_episode_id'],official_task_success=jid=='j0076',native_terminal_failure=False,
                    cleanup_completed=True,graph_witness=witness,identity=identity,execution_protocol={'synthetic':True}))
                put(self.root,j['out']+'/owned_hook_receipt.json',dict(actual_split='train',graph_witness=witness,
                    execution_protocol={'synthetic':True},reset=[],paths=[]))
        accepted_job=self.jobs['j0076'];accepted=self.completed[0]
        accepted['case']=o.pointer(self.root,accepted_job['out']+'/case.json')
        accepted['owned_hooks']=o.pointer(self.root,accepted_job['out']+'/owned_hook_receipt.json')
        for phase in ('CPU_check','native'):
            base=self.base+'/children/'+accepted_job['task']+'/'+accepted_job['raw_episode_id']+'/'+accepted_job['method']+'/'+phase+'/'
            accepted[phase]=dict(launch=o.pointer(self.root,base+'launch.json'),terminal=o.pointer(self.root,base+'terminal.json'),
                sidecar=o.pointer(self.root,accepted_job['runtime']+'/'+phase+'/worker_sidecar.json'))
        self.sr=put(self.root,self.base+'/status.json',self.status)
        self.e=dict(schema='r10_actual_partial_parallel_control_failure_exit_v1186',actual_tool_exit_observed=True,
            natural_completion=False,returncode=1,actual_exec_session=96946,Native_started=True,Native_killed=False,
            owner_dead=True,all_outer_children_dead=True,qualification_granted=False,automatic_retry=False,
            CPU_phase_attempts=4,Native_phase_attempts=2,attempted_cases=4,completed_accepted_cases=1,
            plan=self.pr,launch=self.lr,status=self.sr,terminal=self.tr,graph=self.gr,process=self.owner,phases=phases)
        self.er=put(self.root,o.RETIRED_EXIT_PATH,self.e)

    @contextmanager
    def pins(self):
        with patch.object(o,'RETIRED_EXIT_SHA',self.er['sha256']),patch.object(o,'RETIRED_PLAN_SHA',self.pr['sha256']), \
                patch.object(o,'RETIRED_GRAPH_SHA',self.gr['sha256']):yield

    def inspect(self):
        with self.pins():return o.inspect_retired_partial_batch(self.root,self.er)

    def rewrite(self):self.er=put(self.root,o.RETIRED_EXIT_PATH,self.e)

    def test_all_real_development_work_retained_and_not_reused_even_when_success_values_differ(self):
        r=self.inspect()
        self.assertEqual(len(r['phases']),6);self.assertEqual(len(r['Native_development_records']),2)
        self.assertEqual(r['completed_records'],self.completed)
        self.assertEqual({v['native_success'] for v in r['Native_development_records']},{False,True})
        self.assertFalse(r['accepted_into_new_cohort']);self.assertFalse(r['scientific_qualification_granted'])
        self.assertFalse(r['source_outcomes_used_for_rerun_or_version_selection'])
        self.assertTrue(r['historical_Native_reexecution_disclosed']);self.assertEqual(r['repeated_development_Native_cases'],2)

    def test_wrong_metadata_role_is_rejected_before_reading_VAL_TEST_or_other_exit(self):
        for name in ('artifacts/framework/protocol/validation_manifest.json','artifacts/framework/protocol/test_manifest.json',o.SUPERSEDED_EXIT_PATH):
            with patch.object(o,'read',side_effect=AssertionError('must classify before read')):
                with self.subTest(name=name),self.assertRaises(ValueError):
                    o.inspect_retired_partial_batch(self.root,dict(root='project',path=name,sha256='a'*64))

    def test_actual_native_work_cannot_be_relabelled_zero_or_natural_complete(self):
        original=deepcopy(self.e)
        for key,value in (('Native_started',False),('Native_phase_attempts',0),('completed_accepted_cases',0),
                ('natural_completion',True),('returncode',True),('actual_tool_exit_observed',False),('qualification_granted',True)):
            self.e=deepcopy(original);self.e[key]=value;self.rewrite()
            with self.subTest(key=key),self.assertRaises(ValueError):self.inspect()

    def test_missing_duplicate_live_or_extra_development_phase_cannot_be_hidden(self):
        original=deepcopy(self.e)
        self.e['phases'].pop();self.rewrite()
        with self.assertRaises(ValueError):self.inspect()
        self.e=deepcopy(original);self.e['phases'][0]=deepcopy(self.e['phases'][1]);self.rewrite()
        with self.assertRaises(ValueError):self.inspect()
        self.e=deepcopy(original);self.rewrite()
        with patch.object(o.readiness,'_not_live',return_value=False),self.assertRaises(ValueError):self.inspect()
        put(self.root,self.base+'/children/unrecorded/native/launch.json',{})
        with self.assertRaises(ValueError):self.inspect()

    def test_failed_phase_cannot_become_clean_and_native_case_cannot_disappear(self):
        j=self.jobs['j0075'];case=self.root/j['out']/'case.json';case.unlink()
        with self.assertRaises((ValueError,FileNotFoundError)):self.inspect()

    def test_retired_status_must_keep_context_and_all_three_infrastructure_errors(self):
        self.status['completed']=[];self.e['status']=put(self.root,self.sr['path'],self.status);self.rewrite()
        with self.assertRaises(ValueError):self.inspect()

    def test_retired_native_identity_graph_source_or_cleanup_drift_is_rejected(self):
        j=self.jobs['j0075'];name=j['out']+'/case.json';original=json.loads((self.root/name).read_text())
        for kind in ('graph','source','identity','cleanup','non_bool_outcome'):
            case=deepcopy(original)
            if kind=='graph':case['graph_witness']['graph']['sha256']='f'*64
            elif kind=='source':case['source_raw_episode_id']='foreign'
            elif kind=='identity':case['identity']['source_row_sha256']='f'*64
            elif kind=='cleanup':case['cleanup_completed']=False
            else:case['official_task_success']=1
            put(self.root,name,case)
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.inspect()

    def test_previously_accepted_development_record_original_checksums_must_still_match(self):
        j=self.jobs['j0076'];name=j['out']+'/case.json';case=json.loads((self.root/name).read_text())
        case['official_task_success']=False;put(self.root,name,case)
        with self.assertRaises(ValueError):self.inspect()


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
                plan=dict(split=split,owner_runtime=o.BASE+'/'+split+'/fixture',graph=gr,predecessor=predecessor,retired_partial_batch={'synthetic_retired':True},
                    preserved_parallel_segment={'synthetic_parallel':True},preserved_parallel_count=2,
                    preserved_prefix_count=191 if split=='train' else 0,validation_bundle=None)
                pr=put(root,plan['owner_runtime']+'/plan.json',plan)
                with patch.object(o,'_closed_owner',return_value=(plan,pr,graph,[],owner)):
                    fn=o.assemble_cohort if split=='train' else o.assemble_validation_cohort
                    reference=fn(root,pr,owner['exec_completion']);value,_=o.read(root,reference)
                self.assertEqual(value['schema'],'r10_mixed_control_TRAIN192_lineage_v1198' if split=='train' else o.VAL_COHORT_SCHEMA)
                self.assertTrue(value['complete']);self.assertEqual(value['graph'],gr)
                if split=='train':self.assertFalse(value['case_graphs_relabelled']);self.assertEqual(value['preserved_prefix_count'],191)
                else:self.assertEqual(value['expected_cases'],945)


def serial_env():return o.serial.ENV_SHA


if __name__=='__main__':unittest.main()
