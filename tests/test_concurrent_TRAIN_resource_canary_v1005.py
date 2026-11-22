"""CPU-only synthetic metadata/schedulers. Never call production resource gates."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]/'scripts/run_concurrent_TRAIN_resource_canary_v1005_r10.py'
spec = importlib.util.spec_from_file_location('canary1005_cpu', SOURCE)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def pointer(name):
    return dict(path=name, sha256='a'*64)


def fixture(module=m):
    c = dict(author_outcome_protocol={'schema': 'fake_protocol'}, previous_CPU_receipts=[],
        training_bootstrap={'control': pointer('artifacts/train_control.json'), 'lock': pointer('artifacts/train_lock.json')})
    names = (module.BOOTSTRAP, 'our_mani/multitask/frozen_lookup_batch_v900.py',
        'scripts/run_formal_case_author_outcome_v751_r10.py', 'our_mani/multitask/formal_suite_admission_v728.py',
        'our_mani/multitask/rml_case_preflight_v742.py', module.WORKER,
        'scripts/run_task_executor_smoke_author_suite16_v744_r10.py')
    c['execution_implementation'] = module.implementation_from_sources({name:pointer(name) for name in names})
    cases, cpus, planned = [], {}, []
    for raw in module.RAWS:
        for method in module.METHODS:
            p = pointer('artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/'+raw+'_'+method+'_CPU_preflight.json')
            c['previous_CPU_receipts'].append(p)
            identity = dict(task=module.TASK, split='train', method=method, source_raw_episode_id=raw,
                source_row_sha256='b'*64, sha256='c'*64)
            cpus[p['path']] = dict(complete=True, native_started=False, GPU_started=False, actual_split='train',
                identity=identity, author_outcome_protocol=c['author_outcome_protocol'])
            argv = ['bash', 'scripts/run_aligned_gpu.sh' if method=='physmani' else 'scripts/run_local_sim.sh',
                module.WORKER, '--config','frozen.json','--manifest',module.MANIFEST,'--frozen-inputs',module.BASE_LOCK,
                '--task',module.TASK,'--raw-episode-id',raw,'--split','train','--method',method,
                '--development-demo','--out','old/development_demo_only/'+raw+'/'+method]
            cases.append(dict(task=module.TASK,raw_episode_id=raw,method=method,official_task_success=False,
                source_row_sha256='b'*64,case=pointer('old/'+raw+'/'+method+'/case.json'),
                provenance=dict(CPU_receipt=p, actual_argv=argv)))
            planned.append(dict(task=module.TASK,raw=raw,method=method,argv=deepcopy(argv)))
    plan = dict(planned=planned, source_outcome_filtering=False, automatic_retry=False)
    return dict(cases=cases), cpus, c, plan


def selected():
    inventory, cpus, control, plan = fixture()
    return m.selected_rows(inventory,cpus,control,plan),control


class SelectionTests(unittest.TestCase):
    def test_exact_four_original_sources_all_twelve_prior_false_retained(self):
        inv,cpu,c,plan = fixture(); rows=m.selected_rows(inv,cpu,c,plan)
        self.assertEqual(len(rows),12)
        self.assertEqual(m.RAWS,('episode0','episode1','episode105','episode106'))
        self.assertEqual({(r['raw'],r['method']) for r in rows},{(r,k) for r in m.RAWS for k in m.METHODS})
        self.assertTrue(all(r['official_task_success'] is False for r in inv['cases']))
        for row in rows:
            self.assertEqual(row['argv'][2],m.BOOTSTRAP)
            self.assertEqual(row['physical_argv'][2],m.WORKER)
            self.assertEqual(m.option(row['argv'],'--entry'),'train')
            self.assertEqual(m.option(row['physical_argv'],'--frozen-inputs'),m.BASE_LOCK)

    def test_missing_duplicate_changed_source_or_original_route_rejected(self):
        for kind in ('missing','duplicate','raw','plan','filter','method','TEST','CPU_identity'):
            inv,cpu,c,plan=fixture()
            if kind=='missing':inv['cases'].pop()
            elif kind=='duplicate':inv['cases'][-1]=deepcopy(inv['cases'][0])
            elif kind=='raw':inv['cases'][-1]['raw_episode_id']='episode2'
            elif kind=='plan':plan['planned'][0]['argv'][3]='--different'
            elif kind=='filter':plan['source_outcome_filtering']=True
            elif kind=='method':inv['cases'][0]['method']='Value'
            elif kind=='TEST':inv['cases'][0]['provenance']['actual_argv'][13]='test'
            else:next(iter(cpu.values()))['identity']={}
            with self.subTest(kind=kind),self.assertRaises(ValueError):m.selected_rows(inv,cpu,c,plan)

    def test_shared_implementation_excludes_either_split_controls(self):
        _,c=selected(); value=c['execution_implementation']
        self.assertEqual(len(value['sources']),7)
        self.assertNotIn('batch_control',value)
        with self.assertRaises(ValueError):m.implementation_from_sources(dict(value['sources'],extra=pointer('extra')))


class JointSchedulerTests(unittest.TestCase):
    def simulate(self, *, failure=False, stop=False, launch_error=False, never_ready=False, early_exit=False, timeout=False):
        rows,_=selected(); clock=[0]; handles=[]; launches=[]; interrupts=[]
        def launch(row,phase):
            if launch_error and len(launches)==3:raise RuntimeError('fake launch failure')
            h=dict(row=row,phase=phase,pid=100+len(handles),start_ticks=1000+len(handles),age=0,done=False,interrupted=False)
            handles.append(h);launches.append((phase,row['job_id']));return h
        def pause():
            clock[0]+=1
            for h in handles:
                h['age']+=1
                if h['age']>=(2 if h['phase']=='CPU' else (1 if early_exit else 10)):h['done']=True
        def finish(row,h):return dict(complete=not h['interrupted'] and not (failure and h['phase']=='CPU'),native_success=False)
        def interrupt(h):h.update(done=True,interrupted=True);interrupts.append(h['pid'])
        self.last=(launches,interrupts)
        def contexts():
            if never_ready or clock[0]<2:return None
            return dict(complete=True,job_ids=[h['row']['job_id'] for h in handles[:3]],monotonic_ns=clock[0])
        result=m.schedule_joint(rows,launch=lambda r:launch(r,'native'),launch_CPU=lambda r:launch(r,'CPU'),
            poll=lambda h:0 if h['done'] else None,finish=finish,finish_CPU=finish,
            identify=lambda h:{k:h[k] for k in ('pid','start_ticks')},interrupt=interrupt,pause=pause,
            should_stop=lambda:stop and len(launches)>=12,clock=lambda:clock[0]*10**9,
            contexts_ready=contexts,context_wait_timeout_seconds=1 if timeout else 600)
        return result,launches,interrupts

    def test_joint_eight_three_four_and_each_job_CPU_once_under_native(self):
        result,launches,interrupts=self.simulate()
        self.assertTrue(result['complete']);self.assertEqual(len(result['results']),12)
        self.assertEqual(len(result['late_CPU_results']),4)
        self.assertEqual([p for p,k in launches[:12]],['native']*8+['CPU']*4)
        self.assertEqual(len(launches),16);self.assertEqual(len(set(launches)),16)
        self.assertTrue(any((e['native_slots'],e['GPU_slots'],e['CPU_preflight_slots'])==(8,3,4) for e in result['events']))
        self.assertTrue(all(r['native_success'] is False for r in result['results'].values()))
        self.assertEqual(interrupts,[])

    def test_late_CPU_failure_drains_all_already_native_without_retry(self):
        result,launches,interrupts=self.simulate(failure=True)
        self.assertFalse(result['complete']);self.assertEqual(len(launches),12)
        self.assertEqual(len(result['results']),8);self.assertEqual(len(result['late_CPU_results']),4)
        self.assertEqual(interrupts,[])

    def test_STOP_interrupts_only_own_native_and_CPU(self):
        result,launches,interrupts=self.simulate(stop=True)
        self.assertFalse(result['complete']);self.assertEqual(len(launches),12)
        self.assertEqual(len(set(interrupts)),12)

    def test_exception_releases_owned_processes_without_unknown_process_signal(self):
        with self.assertRaisesRegex(RuntimeError,'fake launch failure'):self.simulate(launch_error=True)
        self.assertEqual(len(self.last[0]),3);self.assertEqual(len(self.last[1]),3)

    def test_delayed_three_contexts_before_geometry_and_CPU_wave(self):
        result,launches,_=self.simulate()
        self.assertEqual([p for p,k in launches[:3]],['native']*3)
        ready=[e for e in result['events'] if e['event']=='contexts_ready']
        self.assertEqual(len(ready),1);self.assertEqual(ready[0]['native_slots'],3)
        self.assertEqual(ready[0]['CPU_preflight_slots'],0)
        self.assertTrue(result['context_readiness_observed'])

    def test_initial_PHY_exit_and_readiness_timeout_preserve_first_block_reason(self):
        for kwargs,reason in (({'early_exit':True},'initial_PhysMani_exited_before_three_context_readiness'),
            ({'never_ready':True,'timeout':True},'three_GPU_context_readiness_timeout')):
            result,launches,interrupts=self.simulate(**kwargs)
            with self.subTest(kwargs=kwargs):
                self.assertFalse(result['complete']);self.assertEqual(len(launches),3)
                self.assertEqual(result['qualification_blocked_reason'],reason)
                self.assertEqual(interrupts,[])

    def test_first_wave_eight_CPU_use_four_workers_without_native(self):
        rows,c=selected();rows=m.ordered_native_rows(rows)[:8]
        barrier=threading.Barrier(4);lock=threading.Lock();state=dict(active=0,peak=0,finished=0)
        def launch(row,check,*_):
            self.assertTrue(check)
            with lock:state['active']+=1;state['peak']=max(state['peak'],state['active'])
            barrier.wait(timeout=3);return {}
        def finish(*_):
            with lock:state['active']-=1;state['finished']+=1
            return {'complete':True}
        with patch.object(m,'launch_child',launch),patch.object(m,'poll_child',return_value=0), \
             patch.object(m,'finish_child',finish),patch.object(m,'interrupt_child'),patch.object(m,'stopped',return_value=False):
            results=m.run_CPU(rows,c,threading.Event(),{})
        self.assertEqual((state['peak'],state['finished'],len(results)),(4,8,8))

    def test_incomplete_inventory_never_means_zero_and_fresh_writes_preserved(self):
        good=dict(host_visibility_available=True,same_uid_inventory_incomplete=False,
            live_project_native_processes=[],ancestor_chain_complete=True)
        m.require_empty_inventory(good)
        for k,v in [('ancestor_chain_complete',False),('live_project_native_processes',[{'pid':1}]),('same_uid_inventory_incomplete',True)]:
            with self.subTest(key=k),self.assertRaises(ValueError):m.require_empty_inventory(dict(good,**{k:v}))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'new.json';m.atomic_new(p,{'original':True})
            with self.assertRaises(FileExistsError):m.atomic_new(p,{'original':False})


if __name__=='__main__':unittest.main()
