"""Independent auditor CPU boundaries only: no real metadata/proc/GPU/cases."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/audit_concurrent_TRAIN_resource_canary_v904_r10.py'
spec = importlib.util.spec_from_file_location('resource904_fake_tests', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


PROFILE = dict(schema='r10_frozen_concurrent_native_resource_profile_v902',
    max_native_slots=4, max_GPU_slots=2, max_CPU_preflight_slots=2, ready_limit=8)


def fixture():
    methods = ['physmani', 'physmani', 'cv_geometry', 'context_geometry', 'cv_geometry', 'context_geometry']
    starts, ends = [10, 20, 30, 40, 100, 110], [300, 310, 80, 90, 180, 190]
    rows, launches, terminals, cases = [], {}, {}, {}
    for i, method in enumerate(methods):
        key = 'job' + str(i)
        rows.append(dict(job_id=key, method=method, native_success=False))
        launches[key] = dict(launcher_pid=2000 + i, launcher_start_ticks=3000 + i,
            started_monotonic_ns=starts[i])
        terminals[key] = dict(complete=True, returncode=0, timed_out=False, interrupted=False,
            automatic_retry=False, launcher_pid=2000 + i, launcher_start_ticks=3000 + i,
            started_monotonic_ns=starts[i], finished_monotonic_ns=ends[i], native_success=False)
        cases[key] = dict(process_id=1000 + i, official_task_success=False)
    def event(now, keys):
        values = [dict(job_id=key, method=next(r['method'] for r in rows if r['job_id'] == key),
            pid=launches[key]['launcher_pid'], start_ticks=launches[key]['launcher_start_ticks']) for key in keys]
        return dict(monotonic_ns=now, active=values, native_slots=len(values),
            GPU_slots=sum(value['method'] == 'physmani' for value in values))
    events = [event(50, ['job0', 'job1', 'job2', 'job3']), event(120, ['job0', 'job1', 'job4', 'job5']), event(320, [])]
    apps = []
    for i in (0, 1):
        key = 'job' + str(i)
        apps.append(dict(pid=1000 + i, pid_start_ticks=4000 + i, job_id=key,
            launcher_pid=2000 + i, launcher_start_ticks=3000 + i, used_memory_mib=1800,
            gpu_uuid='GPU-fake', ancestor_chain=[dict(pid=1000 + i, start_ticks=4000 + i, ppid=2000 + i, state='R'),
                dict(pid=2000 + i, start_ticks=3000 + i, ppid=1, state='S')]))
    samples = [dict(monotonic_ns=60, GPU_observation_complete=True,
        gpu=dict(uuid='GPU-fake', memory_used_mib=4096, memory_total_mib=16384, utilization_gpu_percent=85), compute_apps=apps)]
    return rows, launches, terminals, events, cases, samples


class IntervalTests(unittest.TestCase):
    def audit(self, value):
        rows, launches, terminals, events, _, _ = value
        return m.interval_audit(rows, launches, terminals, events, PROFILE)

    def test_six_real_pid_intervals_true_four_two_overlap_and_native_false_retained(self):
        value = fixture()
        result = self.audit(value)
        self.assertEqual((result['observed_native_peak'], result['observed_GPU_peak']), (4, 2))
        self.assertTrue(result['overlap_verified'])
        self.assertTrue(all(row['native_success'] is False for row in value[0]))
        self.assertTrue(all(case['official_task_success'] is False for case in value[4].values()))

    def test_partial_duplicate_or_wrong_launch_grid_rejected(self):
        for kind in ('missing_row', 'duplicate_row', 'missing_launch', 'missing_terminal'):
            value = fixture();rows,launches,terminals,*_ = value
            if kind == 'missing_row': rows.pop()
            elif kind == 'duplicate_row': rows[-1] = deepcopy(rows[0])
            elif kind == 'missing_launch': launches.pop('job5')
            else: terminals.pop('job5')
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.audit(value)

    def test_reused_pid_start_or_bool_identity_rejected(self):
        for kind in ('duplicate', 'bool_pid', 'bool_start', 'terminal_pid', 'terminal_start'):
            value=fixture();launches,terminals=value[1:3]
            if kind=='duplicate':launches['job1'].update({k:launches['job0'][k] for k in ('launcher_pid','launcher_start_ticks')})
            elif kind=='bool_pid':launches['job0']['launcher_pid']=True
            elif kind=='bool_start':launches['job0']['launcher_start_ticks']=True
            elif kind=='terminal_pid':terminals['job0']['launcher_pid']=9999
            else:terminals['job0']['launcher_start_ticks']=9999
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.audit(value)

    def test_claimed_overlap_outside_real_lifetime_is_rejected(self):
        value=fixture();value[3][0]['monotonic_ns']=95
        with self.assertRaisesRegex(ValueError,'lifetime'):self.audit(value)

    def test_pseudo_counts_without_four_unique_simultaneous_active_requests_rejected(self):
        for kind in ('wrong_count','duplicate_active','only_three','no_events','foreign'):
            value=fixture();events=value[3]
            if kind=='wrong_count':events[0]['native_slots']=3
            elif kind=='duplicate_active':events[0]['active'][1]=deepcopy(events[0]['active'][0])
            elif kind=='only_three':
                for e in events:
                    e['active']=[v for v in e['active'] if v['job_id']!='job1'];e['native_slots']=len(e['active']);e['GPU_slots']=sum(v['method']=='physmani' for v in e['active'])
            elif kind=='no_events':events.clear()
            else:events[0]['active'][0]['job_id']='foreign'
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.audit(value)

    def test_wrong_method_or_active_pid_start_is_rejected(self):
        for field,value in [('method','cv_geometry'),('pid',999),('start_ticks',999)]:
            data=fixture();data[3][0]['active'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.audit(data)

    def test_timeout_interrupt_nonzero_or_bool_returncode_are_infrastructure(self):
        for field,value in [('timed_out',True),('interrupted',True),('returncode',1),('returncode',False),('complete',False)]:
            data=fixture();data[2]['job0'][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.audit(data)

    def test_wrong_clock_order_and_bool_clock_rejected(self):
        for value in (True, 0, 40):
            data=fixture();data[3][1]['monotonic_ns']=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.audit(data)

    def test_quota_excess_cannot_be_hidden_by_valid_earlier_witness(self):
        data=fixture();event=deepcopy(data[3][0]);event['active'].append(dict(job_id='job4',method='cv_geometry',pid=2004,start_ticks=3004))
        event.update(monotonic_ns=120,native_slots=5)
        # Extend geometry0/1 lifetimes to make the five-request excess real.
        data[2]['job2']['finished_monotonic_ns']=180;data[2]['job3']['finished_monotonic_ns']=180
        data[3].insert(1,event)
        with self.assertRaisesRegex(ValueError,'quota'):self.audit(data)

    def test_actual_six_way_overlap_rejects_even_when_events_omit_two_live_intervals(self):
        data=fixture()
        data[2]['job2']['finished_monotonic_ns']=180
        data[2]['job3']['finished_monotonic_ns']=180
        # Leave the event at120 claiming only job0/1/4/5 unchanged.
        with self.assertRaisesRegex(ValueError,'actual child lifetime intervals.*quota'):
            self.audit(data)

    def test_equal_time_end_precedes_next_start_without_false_overallocation(self):
        data=fixture()
        data[2]['job2']['finished_monotonic_ns']=100
        data[2]['job3']['finished_monotonic_ns']=110
        result=self.audit(data)
        self.assertEqual(result['observed_native_peak'],4)
        self.assertTrue(result['independently_swept_all_child_intervals'])


class HardwareTests(unittest.TestCase):
    def audit(self,value):
        rows,launches,terminals,_,cases,samples=value
        return m.hardware_audit(samples,rows,launches,terminals,cases)

    def test_two_case_python_pids_have_stable_parent_chain_and_real_headroom(self):
        result=self.audit(fixture())
        self.assertTrue(result['actual_two_compute_contexts_observed'])
        self.assertEqual(set(result['witness']['jobs']),{'job0','job1'})
        self.assertEqual(result['minimum_sampled_headroom_mib'],12288)
        self.assertFalse(result['continuous_hardware_bound_claimed'])

    def test_exact_two_gib_margin_passes_one_mib_less_rejected(self):
        value=fixture();value[5][0]['gpu'].update(memory_total_mib=6144,memory_used_mib=4096)
        self.assertEqual(self.audit(value)['minimum_sampled_headroom_mib'],2048)
        value[5][0]['gpu']['memory_used_mib']=4097
        with self.assertRaisesRegex(ValueError,'headroom'):self.audit(value)

    def test_all_unknown_or_missing_samples_cannot_supply_two_context_witness(self):
        for samples in ([],[dict(monotonic_ns=50,GPU_observation_complete=False,gpu=None,compute_apps=[])]):
            value=fixture();value[5][:]=samples
            with self.subTest(samples=samples),self.assertRaises(ValueError):self.audit(value)

    def test_unknown_samples_are_retained_without_becoming_zero(self):
        value=fixture();value[5].insert(0,dict(monotonic_ns=5,GPU_observation_complete=False,gpu=None,compute_apps=[]))
        result=self.audit(value)
        self.assertEqual(result['unknown_samples'],1)
        self.assertEqual(result['valid_samples'],1)
        self.assertEqual(result['minimum_sampled_headroom_mib'],12288)

    def test_unknown_pid_or_single_job_does_not_supply_second_context(self):
        for mutation in ('unknown','one_app','both_unknown'):
            value=fixture();apps=value[5][0]['compute_apps']
            if mutation=='one_app':apps.pop()
            elif mutation=='unknown':apps[1]['job_id']=None
            else:
                for app in apps:app['job_id']=None
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,'two-case'):self.audit(value)

    def test_foreign_job_wrong_case_pid_or_wrong_gpu_rejected(self):
        for field,value in [('job_id','foreign'),('job_id','job2'),('pid',99999),('gpu_uuid','other'),('launcher_pid',99),('launcher_start_ticks',99)]:
            data=fixture();data[5][0]['compute_apps'][0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.audit(data)

    def test_bad_ancestry_edge_start_or_dead_state_is_rejected(self):
        for mutation in ('edge','main_start','launcher_start','dead','duplicate_node','empty'):
            value=fixture();chain=value[5][0]['compute_apps'][0]['ancestor_chain']
            if mutation=='edge':chain[0]['ppid']=999
            elif mutation=='main_start':chain[0]['start_ticks']=999
            elif mutation=='launcher_start':chain[-1]['start_ticks']=999
            elif mutation=='dead':chain[0]['state']='Z'
            elif mutation=='duplicate_node':chain.append(deepcopy(chain[-1]))
            else:chain.clear()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.audit(value)

    def test_bool_hardware_numbers_and_out_of_interval_time_rejected(self):
        for kind in ('pid','start','used','total','utilization','clock','outside'):
            value=fixture();sample=value[5][0]
            if kind=='pid':sample['compute_apps'][0]['pid']=True
            elif kind=='start':sample['compute_apps'][0]['pid_start_ticks']=True
            elif kind=='used':sample['gpu']['memory_used_mib']=True
            elif kind=='total':sample['gpu']['memory_total_mib']=True
            elif kind=='utilization':sample['gpu']['utilization_gpu_percent']=True
            elif kind=='clock':sample['monotonic_ns']=True
            else:sample['monotonic_ns']=315
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.audit(value)

    def test_duplicate_actual_pid_cannot_count_twice(self):
        value=fixture();value[5][0]['compute_apps'][1]['pid']=1000
        with self.assertRaisesRegex(ValueError,'unique'):self.audit(value)

    def test_low_later_headroom_not_erased_by_earlier_good_witness(self):
        value=fixture();late=deepcopy(value[5][0]);late['monotonic_ns']=70;late['gpu']['memory_used_mib']=15000
        value[5].append(late)
        with self.assertRaisesRegex(ValueError,'headroom'):self.audit(value)

    def test_extra_owned_GPU_child_is_diagnostic_not_a_second_qualifying_job(self):
        value=fixture();child=deepcopy(value[5][0]['compute_apps'][0]);child.update(pid=1100,pid_start_ticks=4100)
        child['ancestor_chain'].insert(0,dict(pid=1100,start_ticks=4100,ppid=1000,state='R'))
        value[5][0]['compute_apps'].append(child)
        result=self.audit(value)
        self.assertEqual(set(result['witness']['jobs']),{'job0','job1'})
        self.assertEqual(len(result['witness']['all_owned_compute_processes']['job0']),2)
        value[5][0]['compute_apps']=[value[5][0]['compute_apps'][0],child]
        with self.assertRaisesRegex(ValueError,'two-case'):self.audit(value)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.path=self.root/'leaf.json';self.path.write_bytes(b'{"value":1}')
        self.pointer=dict(path='leaf.json',sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_exact_pointer_reads_recorded_bytes_and_rechecks(self):
        evidence=m.Evidence(self.root)
        self.assertEqual(evidence.pointer(self.pointer,'leaf.json'),{'value':1})
        self.assertEqual(evidence.inputs,{'leaf.json':self.pointer['sha256']})
        evidence.recheck()

    def test_missing_None_uppercase_or_wrong_sha_refuses(self):
        pointers=[{'path':'leaf.json'},dict(self.pointer,sha256=None),dict(self.pointer,sha256='A'*64),dict(self.pointer,sha256='0'*64)]
        for pointer in pointers:
            with self.subTest(pointer=pointer),self.assertRaises(ValueError):m.Evidence(self.root).pointer(pointer,'leaf.json')

    def test_allowlist_path_mismatch_refuses_before_open(self):
        original=Path.read_bytes
        def forbidden(path):
            self.fail('disallowed payload opened: '+str(path))
        with patch.object(Path,'read_bytes',forbidden),self.assertRaisesRegex(ValueError,'allowlist'):
            m.Evidence(self.root).pointer(self.pointer,'exact-other.json')

    def test_symlink_alias_and_escape_refuse(self):
        alias=self.root/'alias.json';alias.symlink_to(self.path)
        with self.assertRaises(ValueError):m.Evidence(self.root).pointer(dict(self.pointer,path='alias.json'),'alias.json')
        for path in ('../leaf.json','/tmp/leaf.json'):
            with self.subTest(path=path),self.assertRaises(ValueError):m.Evidence(self.root).pointer(dict(self.pointer,path=path),path)

    def test_change_after_first_consumption_is_detected_by_recheck(self):
        evidence=m.Evidence(self.root);evidence.pointer(self.pointer,'leaf.json');self.path.write_bytes(b'{"value":2}')
        with self.assertRaisesRegex(ValueError,'SHA changed'):evidence.recheck()


class MetricsTests(unittest.TestCase):
    def test_complete_map_and_used_path_counters_required(self):
        good=dict(schema='r10_scoped_frozen_lookup_metrics_v900',closed=True,complete=True,
            maps=2,declared_files=4,watched_directories=2,pointer_checks=6,full_graph_boundary_checks=4,
            used_path_end_checks=3,used_external=2,actual_SHA_checks=9,strong_stat_reuses=7,exact_JSON_payloads=1)
        m.require_batch_metrics(good)
        for field,value in [('closed',False),('maps',True),('full_graph_boundary_checks',3),('actual_SHA_checks',8),('used_external',7)]:
            with self.subTest(field=field),self.assertRaises(ValueError):m.require_batch_metrics(dict(good,**{field:value}))


class CompleteFakeAuditTests(unittest.TestCase):
    """Full writer-shape integration; existing scientific validators are spies.

    Their established semantics are not retested here; this verifies the new
    auditor actually applies each validator six times to the exact identity /
    protocol/profile while preserving every False and checking real byte links.
    """
    def build(self,root,*,cleanup=True,driver_sha=True,self_signed=False):
        rows,launches,terminals,events,cases,samples=fixture()
        runtime='runtime/framework/fake_canary903';source='scripts/fake_canary903.py'
        def write(name,value,raw=False):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(value if raw else json.dumps(value,sort_keys=True).encode())
            return dict(path=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        def point(name):
            name=str(name);return dict(path=name,sha256=hashlib.sha256((root/name).read_bytes()).hexdigest())
        write(source,b'# fake frozen runner\n',True);write(m.SOURCE,b'# fake frozen independent auditor\n',True)
        binding=dict(control=write('artifacts/fake_control.json',{}),lock=write('artifacts/fake_lock.json',{}),files={})
        control=dict(inventory=write('runtime/fake_TRAIN_metadata.json',{}),previous_CPU_receipts=[],
            resource_profile=PROFILE,execution_implementation={'schema':'fake frozen implementation','sources':{}},policy_sha256='a'*64,
            author_outcome_protocol={'schema':'fake frozen author protocol'},common_RML_profile_sha256='b'*64,
            training_bootstrap=dict(control=write('artifacts/fake_TRAIN_bootstrap_control.json',{}),lock=write('artifacts/fake_TRAIN_bootstrap_lock.json',{})))
        for i,row in enumerate(rows):
            row.update(raw='episode'+str(int(i in (1,4,5))),out='artifacts/fake_development_demo_only/'+row['job_id'],
                argv=['bash','fake_shell.sh','scripts/fake901.py','--split','train','--out','fake/'+row['job_id']],
                physical_argv=['bash','fake_shell.sh','scripts/fake743.py','--split','train','--out','fake/'+row['job_id']],
                expected_identity=dict(sha256=str(i)*64,split='train',method=row['method']))
        plan=dict(schema='fake_plan_v903',rows=rows,inputs=binding,resource_profile=PROFILE,
            **{key:control[key] for key in ('execution_implementation','policy_sha256','author_outcome_protocol','common_RML_profile_sha256','training_bootstrap')})
        plan_pointer=write(runtime+'/plan.json',plan)
        owners={}
        for phase,pid in (('CPU',100),('native',200)):
            env=dict(observer_pid=pid,current_mount_namespace='mnt:[fake]')
            argv=[source,'--control',binding['control']['path'],'--lock',binding['lock']['path'],'--lock-sha256',binding['lock']['sha256']]
            if phase=='native':argv.append('--execute-native')
            owner=dict(pid=pid,start_ticks=pid+1,argv=argv,source_sha256=point(source)['sha256'],
                control=binding['control'],lock=binding['lock'],plan=plan_pointer,
                resource_inventory={'empty':True},frozen_environment=env)
            owners[phase]=owner;write(runtime+'/'+phase+'_owner_launch.json',owner)
        metrics=dict(schema='r10_scoped_frozen_lookup_metrics_v900',closed=True,complete=True,
            maps=1,declared_files=1,watched_directories=1,pointer_checks=2,full_graph_boundary_checks=2,
            used_path_end_checks=1,used_external=0,actual_SHA_checks=3,strong_stat_reuses=1,exact_JSON_payloads=0)
        cpu_results,native_results,native_pointers={},{},{}
        for row in rows:
            key=row['job_id']
            for phase in ('CPU','native'):
                check=phase=='CPU';suffix=['--check'] if check else [];leaf=runtime+'/'+phase+'/'+key
                launch=dict(launches[key],argv=row['argv']+suffix,physical_argv=row['physical_argv']+suffix,
                    owner_pid=owners[phase]['pid'],owner_start_ticks=owners[phase]['start_ticks'],check_only=check,
                    native_slots=int(not check),GPU_slots=int(not check and row['method']=='physmani'),
                    inherited_frozen_environment=owners[phase]['frozen_environment'],current_mount_namespace='mnt:[fake]')
                lp=write(leaf+'/launch.json',launch)
                terminal=dict(terminals[key],launch=lp)
                batch=dict(schema='r10_scoped_frozen_lookup_bootstrap_receipt_v901',complete=True,entry='train',check_only=check,
                    actual_outer_argv=row['argv'][2:]+suffix,forwarded_original_argv=row['physical_argv'][2:]+suffix,
                    control=control['training_bootstrap']['control'],lock=control['training_bootstrap']['lock'],
                    sources={},scientific_policy_changed=False,full_verify_inputs_calls_retained=True,
                    phases=[dict(phase='original742_prepare',boundary_recheck_complete=True,metrics=metrics)])
                if check:
                    proof=dict(complete=True,native_started=False,GPU_started=False,identity=row['expected_identity'],batch_execution=batch)
                    terminal['CPU_preflight']=write(leaf+'/CPU_preflight.json',proof)
                    write(leaf+'/stdout.log',proof);cpu_results[key]=terminal
                else:
                    case=dict(cases[key],identity=row['expected_identity'],split='train',cleanup_completed=cleanup,
                        component_domain={'purpose':'train_qualification'},common_author_RML_parameters={'fixture':'original binding'})
                    cp=write(row['out']+'/case.json',case);batch['case']=cp;terminal['case']=cp
                    terminal['batch_driver_receipt']=write(row['out']+'/batch_driver_receipt.json',batch)
                    if not driver_sha:terminal['batch_driver_receipt']['sha256']='0'*64
                    write(leaf+'/stdout.log',b'RML diagnostics\nCoppeliaSim done\n',True)
                    native_results[key]=terminal
                write(leaf+'/stderr.log',b'',True)
                tp=write(leaf+'/terminal.json',terminal)
                if not check:native_pointers[key]=tp
        write(runtime+'/CPU_owner_terminal.json',dict(complete=True,native_started=False,GPU_started=False,results=cpu_results))
        ep=write(runtime+'/native_resource_events.json',dict(events=events))
        sp=write(runtime+'/native_resource_samples.jsonl',b'\n'.join(json.dumps(s).encode() for s in samples)+b'\n',True)
        summary=dict(schema='fake_summary_v903',complete=True,independent_audit_complete=self_signed,resource_bounds_verified=False,
            TEST_read=False,formal_score_contribution=False,training_weights_updated=False,infrastructure_failed=0,
            not_launched_job_ids=[],case_count=6,cases=native_pointers,results=native_results,resource_profile=PROFILE,
            plan=plan_pointer,owner=point(runtime+'/native_owner_launch.json'),resource_events=ep,resource_samples=sp,
            final_resource_inventory={'empty':True},**{key:plan[key] for key in ('execution_implementation','policy_sha256','author_outcome_protocol','common_RML_profile_sha256')})
        summary_pointer=write(runtime+'/run_summary.json',summary)
        write(runtime+'/native_owner_terminal.json',dict(complete=True,independent_audit_complete=False,
            pid=owners['native']['pid'],start_ticks=owners['native']['start_ticks'],launch=point(runtime+'/native_owner_launch.json'),
            run_summary=summary_pointer,resource_events=ep,resource_samples=sp))
        def inventory_check(value):
            m.require(value=={'empty':True},'fake resource inventory is not empty')
        def cpu_check(row,receipt,scope):
            m.require(receipt.get('identity')==row['expected_identity'] and receipt.get('native_started') is False
                and receipt.get('GPU_started') is False,'fake CPU receipt binding changed')
        runner=SimpleNamespace(RUNTIME=runtime,SOURCE=source,PLAN_SCHEMA='fake_plan_v903',SUMMARY_SCHEMA='fake_summary_v903',PROFILE=PROFILE,
            selected_rows=lambda *args:rows,read_pointer=lambda p:json.loads((root/p['path']).read_bytes()),point=point,
            alive=lambda value:False,option=lambda argv,key:argv[argv.index(key)+1],require_empty_inventory=inventory_check,
            require_CPU_receipt=cpu_check)
        return control,binding,runner,rows

    def test_full_six_case_writer_chain_preserves_all_false_and_calls_existing_validators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);control,binding,runner,rows=self.build(root)
            with patch.object(m,'ROOT',root),patch('our_mani.multitask.evaluation.validate_completed_case') as case_check, \
                    patch('our_mani.multitask.author_outcome_receipt_v749.require_sampling_case') as sampling, \
                    patch('our_mani.multitask.common_rml_protocol_v530.require_attestation') as rml:
                result=m.audit(control,binding,runner)
            self.assertTrue(result['complete']);self.assertTrue(result['independent_audit_complete'])
            self.assertEqual(result['native_successes_diagnostic'],0)
            self.assertTrue(result['native_failures_retained'])
            self.assertFalse(result['formal_score_contribution']);self.assertFalse(result['TEST_read'])
            self.assertEqual((case_check.call_count,sampling.call_count,rml.call_count),(6,6,6))
            self.assertEqual({call.args[1]['sha256'] for call in case_check.call_args_list},{row['expected_identity']['sha256'] for row in rows})
            self.assertTrue(all(call.kwargs=={'block':True} and call.args[1]==control['author_outcome_protocol'] for call in sampling.call_args_list))
            self.assertTrue(all(call.args[1]==control['common_RML_profile_sha256'] for call in rml.call_args_list))

    def test_full_chain_refuses_self_signed_cleanup_or_wrong_driver_sha(self):
        for kwargs in ({'self_signed':True},{'cleanup':False},{'driver_sha':False}):
            with self.subTest(kwargs=kwargs),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);control,binding,runner,_=self.build(root,**kwargs)
                with patch.object(m,'ROOT',root),patch('our_mani.multitask.evaluation.validate_completed_case'), \
                        patch('our_mani.multitask.author_outcome_receipt_v749.require_sampling_case'), \
                        patch('our_mani.multitask.common_rml_protocol_v530.require_attestation'),self.assertRaises(ValueError):
                    m.audit(control,binding,runner)


if __name__=='__main__':
    unittest.main()
