"""Pure synthetic CPU contracts; never actual cases, leases or source manifests."""
from contextlib import contextmanager
from copy import deepcopy
import ast
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from scripts import run_public_held_repaired_owner_v1207_r10 as o


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


class PhaseClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.job=jobs(1)[0];self.job.update(out=o.ART+'/train/development_demo_only/cohorts/fixture/t/e/cv_geometry',runtime=o.RUNTIME+'/train/cohorts/fixture/t/e/cv_geometry')
        worker=put(self.root,o.WORKER,{'synthetic':True});preparer=put(self.root,o.PREPARER,{'synthetic':True})
        self.g=dict(physical_policy_sha256='a'*64,control_graph_sha256='b'*64,policy_sha256='c'*64,
            physical={'environment_manifest':{'sha256':serial_env()}},routes=[dict(task=self.job['task'],method=self.job['method'],GPU_required=False,
                train=dict(worker=worker,preparer=preparer))])
        guard=put(self.root,o.GUARD,{'synthetic_guard':True})
        parent=put(self.root,'our_mani/multitask/post_completed_TEST_resource_guard_v1106.py',{'synthetic_parent_guard':True})
        auditor=put(self.root,'our_mani/multitask/public_held_repaired_readiness_v1208.py',{'synthetic_auditor':True})
        self.g['control']=dict(components=dict(resource_guard=guard,train_auditor=auditor),sources_map=dict(parent=parent,guard=guard))
        affinity=affinity_fixture(self.root,self.g)
        self.gr=put(self.root,'artifacts/framework/multitask_r10/synthetic_graph/policy_graph.json',self.g)
        base=o.BASE+'/train/fixture';self.plan=dict(split='train',owner_runtime=base,graph=self.gr,profile=o.PROFILE,jobs=[self.job],CPU_affinity=affinity)
        self.owner=dict(pid=2147482000,start_ticks=1);self.outer=dict(pid=2147482001,start_ticks=1);self.inner=dict(pid=2147482002,start_ticks=1)
        from tests.test_public_held_repaired_readiness_v1208 import make_session
        self.session=make_session(self.root,base,self.gr,self.g,owner=self.owner,jobs=[self.job])
        put(self.root,base+'/launch.json',dict(process=self.owner,resource_session=self.session))
        a=dict(job_id=self.job['job_id'],phase='native',slot=0,gpu_slot=None)
        self.claim=dict(schema='r10_once_only_parallel_resource_claim_v1213',session=self.session,phase='native',process=self.inner,
            slot=0,gpu_slot=None,automatic_retry=False,test_only=False,job=self.job)
        self.claim_path=base+'/resources/claims/j0000_native/claim.json';cr=put(self.root,self.claim_path,self.claim)
        self.resource=dict(schema='r10_parallel_child_resource_receipt_v1213',body_completed=True,restored=True,leases_removed=True,
            errors=[],test_only=False,session=self.session,process=self.inner,actual_parent=self.owner,phase='native',job_id='j0000',
            slot=0,gpu_slot=None,parent_global_locks_verified=True,claim=cr,purpose='child_slot',Native_authorized=True,
            child_empty_inventory_claimed=False)
        saved_session=json.loads((self.root/self.session['path']).read_text())
        from tests.test_public_held_repaired_readiness_v1208 import make_kernel_lock_evidence
        for label,nonce in (('admission','a'*64),('final','b'*64)):
            self.resource['parent_broker_'+label]=dict(schema='r10_parent_lock_broker_verification_v1213',
                session=self.session,claim=cr,owner=self.owner,client=self.inner,boot_id=saved_session['boot_id'],nonce=nonce,
                checked_utc='2026-10-24T00:00:00+00:00',parent_FD_verified=True,kernel_FLOCK_verified=True,
                client_direct_parent_FD_read=False,server_authenticated_by_SO_PEERCRED=True,client_authenticated_by_SO_PEERCRED=True,
                server_uid=1000,server_gid=1000,client_uid=1000,client_gid=1000,lock_proof=saved_session['parent_global_locks'],
                guard=saved_session['guard'],parent_guard=saved_session['parent_guard'],job_id=self.job['job_id'],phase='native',slot=0,gpu_slot=None)
            self.resource['parent_broker_'+label].update(make_kernel_lock_evidence(saved_session))
        put(self.root,str(Path(self.claim_path).parent/'terminal.json'),self.resource)
        leaf=base+'/children/'+self.job['task']+'/'+self.job['raw_episode_id']+'/'+self.job['method']+'/native'
        self.sidecar=dict(schema='r10_parallel_graph_bound_public_held_repaired_case_v1206',complete=True,process=self.inner,
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




def serial_env():return o.serial.ENV_SHA


class FreshPlanningTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        from tests.test_public_held_repaired_readiness_v1208 import make_previous_closed_TRAIN_fixture
        self.history=make_previous_closed_TRAIN_fixture(self.root)
        from tests.test_public_held_repaired_readiness_v1208 import make_development_diagnostic_fixture
        self.development=make_development_diagnostic_fixture(self.root)
        self.sources=[dict(task=t,raw_episode_id='episode'+str(i),path='synthetic/'+t+'/'+str(i)) for t in o.graphs.TASKS for i in range(4)]
        worker=put(self.root,o.WORKER,{'synthetic':True});preparer=put(self.root,o.PREPARER,{'synthetic':True})
        self.graph=dict(physical_policy_sha256='a'*64,control_graph_sha256='b'*64,policy_sha256='c'*64,
            physical={'environment_manifest':{'sha256':serial_env()}},routes=[dict(task=t,method=m,GPU_required=m=='physmani',
                configuration={'root':'project','path':'synthetic/config.json','sha256':'a'*64},
                train=dict(worker=worker,preparer=preparer),evaluation=dict(worker=worker,preparer=preparer))
                for t in o.graphs.TASKS for m in o.graphs.METHODS])
        self.gr=put(self.root,'artifacts/framework/multitask_r10/new/policy_graph.json',self.graph)
        self.affinity=affinity_fixture(self.root,self.graph)
        self.addCleanup(patch.stopall)
        patch.object(o,'graph_for_owner',return_value=(self.graph,self.gr)).start()
        patch.object(o,'inspect_previous_closed_TRAIN',return_value=deepcopy(self.history)).start()
        patch.object(o,'inspect_development_diagnostic',return_value=deepcopy(self.development)).start()
        patch.object(o,'read_CPU_affinity',return_value=deepcopy(self.affinity)).start()
        patch.object(o,'train_rows',return_value=deepcopy(self.sources)).start()
    def review(self,split='train',count=192,namespace='fixture',bundle=None):
        return put(self.root,'artifacts/framework/multitask_r10/reviews/'+namespace+'.json',dict(
            schema=o.REVIEW_SCHEMA,reviewer='root',review_complete=True,release_fixed_cohort=True,
            split=split,namespace=namespace,expected_cases=count,automatic_retry=False,source_outcome_filtering=False,
            previous_TRAIN_readiness=self.history['readiness'],development_diagnostic=self.development,historical_Native_reexecution_disclosed=True,
            new_physical_policy_requires_complete_fresh_TRAIN=True,mixed_control_lineage=False,
            case_graphs_relabelled=False,reused_cases=[],validation_bundle=bundle,**o._adoption_fields(),**o._fresh_fields(),**o.stamp(self.gr,self.graph)))
    def prepare(self,namespace='fixture',**kw):
        return o.prepare_parallel_plan(self.root,self.gr,namespace=namespace,review_ref=self.review(namespace=namespace),
            previous_TRAIN_readiness_ref=self.history['readiness'],development_diagnostic_ref=self.development,**kw)
    def test_fresh192_has_all_globalIDs_new_namespaces_no_prefix_or_parallel_subtraction(self):
        ref=self.prepare();plan,_,_=o.load_plan(self.root,ref)
        self.assertEqual(len(plan['jobs']),192);self.assertEqual([j['job_id'] for j in plan['jobs']],[f'j{i:04d}' for i in range(192)])
        self.assertEqual(plan['preserved_prefix_count'],0);self.assertEqual(plan['preserved_parallel_count'],0)
        self.assertEqual(plan['control_segment_count'],1);self.assertFalse(plan['mixed_control_lineage'])
        self.assertIsNone(plan['predecessor']);self.assertEqual(plan['reused_cases'],[])
        self.assertTrue(all(j['out'].startswith(o.ART+'/train/development_demo_only/') for j in plan['jobs']))
    def test_prepare_refuses_any_existing_case_or_owner_namespace(self):
        case=o.build_jobs(self.graph,self.sources,'train','fixture')[0]
        p=self.root/case['out'];p.mkdir(parents=True)
        with self.assertRaises(ValueError):self.prepare()
        p.rmdir();ref=self.prepare()
        with self.assertRaises(ValueError):self.prepare()
    def test_load_rejects_mixed_history_retry_reordered_grid_or_wrong_profile(self):
        ref=self.prepare();original,_=o.read(self.root,ref)
        for field,value in (('preserved_prefix_count',74),('preserved_parallel_count',35),('mixed_control_lineage',True),
                ('automatic_retry',True),('expected_cases',83),('profile',dict(o.PROFILE,max_CPU_preflight_slots=2))):
            changed=deepcopy(original);changed[field]=value;rr=put(self.root,ref['path'],changed)
            with self.subTest(field=field),self.assertRaises(ValueError):o.load_plan(self.root,rr)
        changed=deepcopy(original);changed['jobs'][0],changed['jobs'][1]=changed['jobs'][1],changed['jobs'][0]
        with self.assertRaises(ValueError):o.load_plan(self.root,put(self.root,ref['path'],changed))
    def test_review_requires_complete_newphysical_and_honest_history(self):
        review=self.review();value,_=o.read(self.root,review)
        for key,changed in (('expected_cases',83),('new_physical_policy_requires_complete_fresh_TRAIN',False),
                ('historical_Native_reexecution_disclosed',False),('historical_results_counted_in_accepted192',True)):
            v=deepcopy(value);v[key]=changed;rr=put(self.root,review['path'],v)
            with self.subTest(key=key),self.assertRaises(ValueError):
                o.validate_review(self.root,rr,self.gr,self.graph,'train','fixture',192,self.history,None,self.development)
    def test_VAL945_requires_admitted_bundle_but_no_old_case_reuse(self):
        sources=[dict(task=o.graphs.TASKS[i%16],raw_episode_id='episode'+str(i),path='syntheticVAL/'+str(i)) for i in range(315)]
        bundle=put(self.root,o.BUNDLE_PREFIX+'fixture/bundle.json',{})
        review=self.review('validation',945,'validation',bundle)
        with patch.object(o,'validation_rows',return_value=(sources,bundle)):
            ref=o.prepare_parallel_plan(self.root,self.gr,split='validation',namespace='validation',review_ref=review,
                previous_TRAIN_readiness_ref=self.history['readiness'],development_diagnostic_ref=self.development,bundle_ref=bundle)
            plan,_,_=o.load_plan(self.root,ref,execute=True)
        self.assertEqual(len(plan['jobs']),945);self.assertIsNone(plan['predecessor'])
        self.assertEqual(plan['preserved_prefix_count'],0);self.assertEqual(plan['preserved_parallel_count'],0)
    def test_default_inert_and_wrong_plan_pin_before_any_guard(self):
        with patch.object(o,'execute_owner',side_effect=AssertionError('must stay inert')):
            self.assertEqual(o.main([])['Native_started'],False)
        ref=self.prepare()
        with patch.object(o,'_module',side_effect=AssertionError('no guard before pin')):
            with self.assertRaises(ValueError):o.execute_owner(self.root,ref,'0'*64)


class CollectorTests(unittest.TestCase):
    def test_requires_actual_strict_exit0_and_dead_identity(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);base=o.BASE+'/train/fixture';graph=dict(physical_policy_sha256='a'*64,control_graph_sha256='b'*64,policy_sha256='c'*64)
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
                self.assertEqual(o._closed_owner(root,pr,ar)[3],[row])
                for field,value in (('returncode',False),('natural_completion',False),('actual_tool_exit_observed',False)):
                    changed=dict(actual);changed[field]=value;ref=put(root,ar['path'],changed)
                    with self.subTest(field=field),self.assertRaises(ValueError):o._closed_owner(root,pr,ref)
                ar=put(root,ar['path'],actual)
                with patch.object(o.readiness,'_not_live',return_value=False),self.assertRaises(ValueError):o._closed_owner(root,pr,ar)
    def test_collectors_only_new_records_and_history_no_old_record_splice(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);graph=dict(physical_policy_sha256='a'*64,control_graph_sha256='b'*64,policy_sha256='c'*64)
            gr=put(root,'artifacts/framework/multitask_r10/g/policy_graph.json',graph)
            owner={k:put(root,'synthetic/'+k+'.json',{}) for k in ('launch','terminal','exec_completion')}
            for split in ('train','validation'):
                plan=dict(split=split,owner_runtime=o.BASE+'/'+split+'/fixture',graph=gr,
                    previous_closed_TRAIN={'synthetic_history':True},development_diagnostic={'synthetic_development':True},validation_bundle=None)
                pr=put(root,plan['owner_runtime']+'/plan.json',plan)
                with patch.object(o,'_closed_owner',return_value=(plan,pr,graph,[],owner)):
                    fn=o.assemble_cohort if split=='train' else o.assemble_validation_cohort
                    reference=fn(root,pr,owner['exec_completion']);value,_=o.read(root,reference)
                self.assertEqual(value['records'],[]);self.assertFalse(value['mixed_control_lineage'])
                self.assertEqual(value['preserved_prefix_count'],0);self.assertEqual(value['preserved_parallel_count'],0)
                self.assertFalse(value['historical_results_counted_in_accepted192'])
                self.assertEqual(value['schema'],'r10_fresh_public_held_TRAIN192_lineage_v1208' if split=='train' else o.VAL_COHORT_SCHEMA)


class HistoricalClosureTests(unittest.TestCase):
    def test_previous_full_reaudit_is_required_and_same_physical_reuse_is_refused(self):
        from our_mani.multitask import parallel_repaired_readiness_v1198 as previous
        from tests.test_public_held_repaired_readiness_v1208 import make_previous_closed_TRAIN_fixture
        with TemporaryDirectory() as directory:
            root=Path(directory);history=make_previous_closed_TRAIN_fixture(root)
            put(root,history['readiness']['path'],dict(graph=history['graph']))
            proof=o.pointer(root,history['readiness']['path'])
            put(root,history['lineage']['path'],dict(suffix_plan=history['owner_plan'],suffix_owner=history['owner']))
            lineage=o.pointer(root,history['lineage']['path'])
            actual=dict(lineage=lineage,retired_development_batch=history['retired_development_batch'])
            manifest=dict(root='project',path='metadata/TRAIN.json',sha256='d'*64)
            old=dict(physical_policy_sha256='a'*64,train_source_manifest=manifest)
            new=dict(physical_policy_sha256='b'*64,train_source_manifest=manifest)
            with patch.object(previous,'require_readiness_proof',return_value=actual) as reaudited,\
                    patch.object(o.graphs,'read_policy_graph',return_value=old):
                result=o.inspect_previous_closed_TRAIN(root,proof,new_graph=new)
                reaudited.assert_called_once_with(root,history['graph'],proof)
                self.assertFalse(result['prior_TRAIN_results_reused_for_new_qualification'])
                with self.assertRaises(ValueError):o.inspect_previous_closed_TRAIN(root,proof,new_graph=old)
                changed=dict(new,train_source_manifest=dict(manifest,sha256='e'*64))
                with self.assertRaises(ValueError):o.inspect_previous_closed_TRAIN(root,proof,new_graph=changed)
            with patch.object(previous,'require_readiness_proof',side_effect=ValueError('old owner not closed')):
                with self.assertRaises(ValueError):o.inspect_previous_closed_TRAIN(root,proof,new_graph=new)

    def test_scheduler_and_strict_closure_bodies_preserve_frozen_control_semantics(self):
        """New paths/protocol labels change; the mature execution algorithm does not."""
        from scripts import run_parallel_repaired_owner_v1197_r10 as old
        names={'ParallelScheduler','command','validate_live_parent_proofs','validate_phase',
            '_completed_case_base','completed_case','sample_resources','ProcessDriver','run_scheduler',
            'execute_owner','_closed_owner','_hardware_snapshot','read_CPU_affinity',
            'expected_CPU_cpus','parent_CPU_affinity'}
        before=Path(old.__file__).read_text()
        replacements={
            'our_mani/multitask/parallel_repaired_readiness_v1198.py':'our_mani/multitask/public_held_repaired_readiness_v1208.py',
            'r10_parallel_graph_bound_repaired_policy_case_v1196':'r10_parallel_graph_bound_public_held_repaired_case_v1206',
            'v1195':'v1213','actual1196':'actual1206',
            'parallel1197:':'public-held1207:'}
        for a,b in replacements.items():before=before.replace(a,b)
        def selected(source):
            return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(source).body
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names}
        self.assertEqual(selected(before),selected(Path(o.__file__).read_text()))


class DevelopmentDiagnosticFixture:
    """Complete synthetic six/12 phase metadata; no live worker or source read."""
    def __init__(self,root):
        from our_mani.multitask import train_public_held_block_development_v1204 as pilot
        from tests.test_repaired_readiness_v1144 import FreshFixture
        self.pilot=pilot;self.root=Path(root);original=FreshFixture(root)
        case_template=json.loads((self.root/next(r['case']['path'] for r in original.records
            if r['task'] in o.readiness.BLOCKS)).read_text())
        self.factory=dict(source=pilot.FACTORY_SOURCE,sha256=pilot.FACTORY_SHA,function='integrated_block_components')
        self.lock=put(self.root,pilot.SETUP_PREFIX+'source_overlay_lock.json',{'synthetic_lock':True})
        self.locked=dict(integration_complete=True,factory=self.factory)
        keys=['normal0_cv','normal0_ctx','high100_cv','high100_ctx','normal1_cv','normal1_ctx']
        planned=[dict(case_key=key,task=t,raw_episode_id=raw,method=m,
            out=pilot.ARTIFACT_PREFIX+'pilot6/synthetic/'+key,runtime=pilot.RUNTIME_PREFIX+'pilot6/synthetic/'+key)
            for key,(t,raw,m) in zip(keys,pilot.FIXED_CASES)]
        self.plan=put(root,o.DEVELOPMENT_BASE+'/plan.json',dict(schema='Root_preregistered_TRAIN_public_held_block_fixed6_v1204',
            automatic_retry=False,source_outcome_filtering=False,planned_cases=planned,source_overlay_lock=self.lock))
        records=[]
        for i,job in enumerate(planned):
            case=deepcopy(case_template);case.update(version=pilot.SCHEMA+'_case',task=job['task'],method=job['method'],
                split='train',source_raw_episode_id=job['raw_episode_id'],source_overlay_lock=self.lock,
                execution_protocol=dict(factory=self.factory))
            case_ref=put(root,job['out']+'/case.json',case)
            hooks=put(root,job['out']+'/owned_hook_receipt.json',dict(reset=[],paths=[]))
            raw_ref=put(root,job['out']+'/raw_execution_diagnostics.json',{})
            row=dict(case_index=i,**{k:job[k] for k in ('case_key','task','raw_episode_id','method')},case=case_ref,
                owned_hooks=hooks,raw_diagnostics=raw_ref,official_task_success=False,native_terminal_failure=False,
                counted_ticks=case['counted_clock']['ticks'],counted_seconds=case['counted_clock']['seconds'],motion_pass=False)
            for index,phase in enumerate(('CPU_check','native')):
                inner=dict(pid=2147482500+4*i+2*index,start_ticks=1,argv=['synthetic1204',
                    '--check' if phase=='CPU_check' else '--execute','--task',job['task'],
                    '--raw-episode-id',job['raw_episode_id'],'--method',job['method'],'--split','train',
                    '--out',job['out'],'--runtime',job['runtime'],'--overlay-lock',self.lock['path']])
                outer=dict(pid=inner['pid']+1,start_ticks=1,argv=['synthetic820'])
                execution=dict(complete=True,actual_split='train',mode='CPU_check',native_started=False,GPU_started=False,
                    physical_factory_constructed=False)
                if phase=='native':execution.update(mode='new_public_held_block_native_TRAIN',native_started=True,
                    native_false_preserved=True,native_success=False,native_failure=False,case=case_ref,
                    prior_CPU=row['CPU_check']['sidecar'],actual_native_interval=dict(start_monotonic=1.,end_monotonic=2.),
                    runtime_scope_restoration=dict(complete=True,restored=True,checks={'writer':True},extra_native_calls=0))
                launch=put(root,job['runtime']+'/'+phase+'/launch.json',dict(schema=pilot.SCHEMA,
                    source_overlay_lock=self.lock,selection=dict(task=job['task'],raw_episode_id=job['raw_episode_id'],
                        method=job['method'],fixed_development_index=i,development_only=True),process=inner))
                sidecar=put(root,job['runtime']+'/'+phase+'/worker_sidecar.json',dict(schema=pilot.SCHEMA,
                    source_overlay_lock=self.lock,selection=dict(task=job['task'],raw_episode_id=job['raw_episode_id'],
                        method=job['method'],fixed_development_index=i,development_only=True),process=inner,complete=True,resource_lease_scope_exited=True,
                    execution=execution,frozen_OS_environment=dict(observer_pid=inner['pid'],observer_start_ticks=1,
                        outer_wrapper_pid=outer['pid'],outer_wrapper_start_ticks=1)))
                session=1000+2*i+index
                observed=dict(schema='Root_actual_development_launch_observation_v1204',actual_tool_exit_observed=False,
                    actual_tool_session=session,phase=phase,selection=job,fixed_plan=self.plan,outer_wrapper=outer)
                root_launch=put(root,o.DEVELOPMENT_BASE+'/'+job['case_key']+'/'+phase+'_root_launch.json',observed)
                exit=put(root,o.DEVELOPMENT_BASE+'/'+job['case_key']+'/'+phase+'_root_exit.json',dict(
                    schema='Root_observed_development_phase_exit_v1204',actual_tool_exit_observed=True,natural_completion=True,
                    returncode=0,all_recorded_processes_dead=True,actual_tool_session=session,phase=phase,selection=job,
                    fixed_plan=self.plan,launch=launch,sidecar=sidecar,processes=[{k:inner[k] for k in ('pid','start_ticks')},{k:outer[k] for k in ('pid','start_ticks')}],
                    case=case_ref if phase=='native' else None,owned_hooks=hooks if phase=='native' else None))
                row[phase]=dict(root_launch=root_launch,root_exit=exit,launch=launch,sidecar=sidecar)
                if phase=='native':row[phase]['root_launch_session_correction']=None
            records.append(row)
        self.value=dict(schema='Root_fixed_TRAIN_public_held_block_development_summary_v1204',complete=True,
            development_only=True,formal_score_contribution=False,formal_TRAIN192_acceptance_granted=False,
            formal_VAL_TEST_qualification=False,borrowed_frozen_physical_graph_identity=False,automatic_retry=False,
            source_outcome_filtering=False,all_six_outcomes_retained=True,new_independent_dataset_claimed=False,
            fixed_plan=self.plan,source_overlay_lock=self.lock,planned_cases=6,CPU_attempted=6,CPU_completed=6,
            Native_attempted=6,Native_completed=6,infrastructure_failed=0,native_success_count=0,native_false_count=6,
            native_fatal_count=0,records=records)
        self.refresh()
    def refresh(self):self.ref=put(self.root,o.DEVELOPMENT_BASE+'/development_summary.json',self.value)
    def inspect(self):
        with patch.object(o,'DEVELOPMENT_PLAN_SHA',self.plan['sha256']),                patch.object(self.pilot,'read_source_lock',return_value=(self.locked,self.lock)):
            return o.inspect_development_diagnostic(self.root,self.ref)


class DevelopmentDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def test_all_six_false_are_retained_without_success_threshold(self):
        f=DevelopmentDiagnosticFixture(self.root)
        self.assertEqual(o.readiness.pointer(f.inspect()),o.readiness.pointer(f.ref))
    def test_missing_reordered_bool_count_false_exit_or_wrong_source_cannot_be_history(self):
        f=DevelopmentDiagnosticFixture(self.root);original=deepcopy(f.value)
        for kind in ('missing','reordered','bool_count','filtered','wrong_case','wrong_session'):
            f.value=deepcopy(original)
            if kind=='missing':f.value['records'].pop()
            elif kind=='reordered':f.value['records'].reverse()
            elif kind=='bool_count':f.value['Native_completed']=True
            elif kind=='filtered':f.value['all_six_outcomes_retained']=False
            elif kind=='wrong_case':f.value['records'][0]['raw_episode_id']='episode999'
            else:
                ref=f.value['records'][0]['native']['root_exit'];value,_=o.read(self.root,ref)
                value['actual_tool_session']+=1
                f.value['records'][0]['native']['root_exit']=put(self.root,ref['path'],value)
            f.refresh()
            with self.subTest(kind=kind),self.assertRaises(ValueError):f.inspect()
    def test_launch_correction_must_preserve_outer_and_match_observed_exit(self):
        f=DevelopmentDiagnosticFixture(self.root);row=f.value['records'][0];refs=row['native']
        launch,_=o.read(self.root,refs['root_launch']);actual=launch['actual_tool_session']
        launch['actual_tool_session']=999;refs['root_launch']=put(self.root,refs['root_launch']['path'],launch)
        correction=dict(schema='Root_actual_development_launch_session_correction_v1204',original_file_preserved=True,
            original_session_association_invalid=True,Native_reexecution=False,Native_restarted=False,
            actual_tool_exit_observed=False,phase='native',selection=launch['selection'],outer_wrapper=launch['outer_wrapper'],
            original_launch_receipt=refs['root_launch'],incorrect_recorded_session=999,correct_actual_tool_session=actual)
        name=o.DEVELOPMENT_BASE+'/normal0_cv/native_root_launch_session_correction.json'
        refs['root_launch_session_correction']=put(self.root,name,correction);f.refresh();f.inspect()
        correction['Native_restarted']=True;refs['root_launch_session_correction']=put(self.root,name,correction);f.refresh()
        with self.assertRaises(ValueError):f.inspect()


if __name__=='__main__':unittest.main()
